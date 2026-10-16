#!/usr/bin/env python3
# ============================================================================
# tasks/anicond/cli.py — DTI -> anisotropic conductivity tensor, from the CLI
# ----------------------------------------------------------------------------
# The same call the web page's Run button makes (service.run_job()), reached
# without a browser -- for a single subject, or a whole cohort at once. Not
# wired into run_web.py; run straight from here, per the README's own layout.
#
#   One subject:
#       python tasks/anicond/cli.py --dwi dwi.nii.gz --bval dwi.bval \
#           --bvec dwi.bvec --seg seg.nii.gz
#
#   A cohort:
#       python tasks/anicond/cli.py --manifest subjects.csv
#
#   Manifest columns (a header row, comma-separated):
#       subject_id, dwi, bval, bvec, seg
#       [, model, t1, tracts, conductivity_csv, freq, aniso_ids,
#          iso_ref_ids, seg_registered]
#     `dwi` and `model` may hold several values in one cell, ';'-separated
#     (dwi: the per-direction 3D volumes, in order; model: run more than one
#     model per subject). `seg_registered` accepts true/false/1/0/yes/no and
#     defaults to true, same as the web page's checkbox. Relative paths
#     resolve against the MANIFEST's own folder, not whatever directory the
#     command happens to be run from.
#
# Progress prints to the terminal exactly as the web page's console box would
# show it, and is ALSO written into the run's own run.log -- a CLI-driven run
# and a web-driven run are indistinguishable on disk. See app/jobs.py.
#
# Batch mode is resumable. Each subject's outcome is written to a results CSV
# the moment it finishes, so a crash mid-batch loses nothing already done; a
# subject marked "done" there is skipped on the next invocation (--force
# redoes everyone). One subject's failure is logged and does not stop the
# rest -- the process exits 1 at the end if anything did fail.
#
# Subjects run ONE AT A TIME. Whole-brain tractography is the single most
# CPU-heavy step this app runs; two subjects' tractography fighting over the
# same cores would only make both slower, not finish sooner.
# ============================================================================

# --- repo root + vendor/anicond on sys.path ---------------------------------
# tasks/anicond/service.py already does this bootstrap for itself when
# imported (see its own header), but argparse has to run before that import
# happens, so the same lines are repeated here.
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
for _p in (_root, _os.path.join(_root, "vendor", "anicond")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from app import jobs
from tasks.anicond import service
from tasks.anicond import settings as task_settings

TASK_SLUG = "anicond"

MANIFEST_REQUIRED_COLUMNS = ("subject_id", "dwi", "bval", "bvec", "seg")
RESULTS_COLUMNS = ("subject_id", "status", "run_id", "elapsed_s", "zip_path",
                   "error")


def _print(line: str) -> None:
    print(line, flush=True)


def _parse_bool(text: str, default: bool) -> bool:
    text = (text or "").strip().lower()
    if not text:
        return default
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    raise ValueError(f"not a yes/no value: {text!r}")


# ============================================================================
# One subject
# ----------------------------------------------------------------------------
# Everything web.py's _run() does around service.run_job() -- prune old runs,
# open a fresh run folder, wire a Transcript through the shared queue -- minus
# the Gradio-specific streaming, since a terminal does not need polling.
# ============================================================================
def run_one(*, dwi_paths: Sequence, bval_path, bvec_path, seg_path,
           model_names: Sequence[str], case_name: Optional[str],
           seg_registered: bool, t1_path=None, tracts_path=None,
           conductivity_csv=None, freq: Optional[str] = None,
           aniso_ids_text: Optional[str] = None,
           iso_ref_ids_text: Optional[str] = None) -> service.JobResult:
    jobs.prune_old_runs(TASK_SLUG)
    run = jobs.new_run(TASK_SLUG)
    transcript = jobs.Transcript(log_file=run.log_file, sink=_print)
    with jobs.capture_logs(transcript):
        with jobs.QUEUE.slot(report=transcript.report):
            return service.run_job(
                dwi_paths=dwi_paths, bval_path=bval_path, bvec_path=bvec_path,
                seg_path=seg_path, model_names=model_names,
                aniso_ids_text=aniso_ids_text,
                iso_ref_ids_text=iso_ref_ids_text,
                conductivity_csv=conductivity_csv, freq=freq,
                seg_registered=seg_registered, t1_path=t1_path,
                tracts_path=tracts_path, case_name=case_name,
                run=run, report=transcript.report)


def _result_summary(result: service.JobResult) -> Dict:
    """The JSON-safe, useful half of a JobResult."""
    return {
        "job_id": result.run.job_id,
        "case_name": result.ctx.case_name,
        "model_names": list(result.ctx.model_names),
        "tensors_6x1": [str(p) for p in result.tensors_6x1],
        "zip_path": str(result.zip_path) if result.zip_path else None,
        "elapsed_s": result.elapsed_s,
        "output_count": len(result.all_outputs),
        "run_folder": str(result.run.root),
    }


# ============================================================================
# Manifest (batch) mode
# ============================================================================
def _resolve(path_text: str, base: Path) -> str:
    """A manifest cell as a path, relative to the manifest's OWN folder."""
    p = Path(path_text)
    return str(p if p.is_absolute() else (base / p))


def _resolve_optional(row: Dict[str, str], column: str, base: Path
                      ) -> Optional[str]:
    text = (row.get(column) or "").strip()
    return _resolve(text, base) if text else None


def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    """Every row, after checking the header has what run_one() will need."""
    with open(manifest_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        missing = [c for c in MANIFEST_REQUIRED_COLUMNS if c not in fields]
        if missing:
            raise ValueError(f"{manifest_path}: missing required column(s) "
                             f"{missing}; found {fields}")
        rows = list(reader)
    for i, row in enumerate(rows, start=2):        # row 1 is the header
        if not (row.get("subject_id") or "").strip():
            raise ValueError(f"{manifest_path}: row {i} has no subject_id")
    return rows


def load_results(results_csv: Path) -> Dict[str, Dict[str, str]]:
    if not results_csv.is_file():
        return {}
    with open(results_csv, newline="") as fh:
        return {row["subject_id"]: row for row in csv.DictReader(fh)}


def write_results(results_csv: Path, rows: Dict[str, Dict[str, str]]) -> None:
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({k: row.get(k, "") for k in RESULTS_COLUMNS})


def run_batch(manifest_path: Path, *, model_default: Sequence[str],
             results_csv: Optional[Path], force: bool) -> int:
    manifest_path = manifest_path.resolve()
    manifest_dir = manifest_path.parent
    try:
        rows = read_manifest(manifest_path)
    except ValueError as exc:
        # A malformed manifest is a setup mistake, not a subject's
        # runtime failure -- one clean line, no traceback, no rows touched.
        _print(f"FAILED — {exc}")
        return 1
    results_csv = (results_csv or
                   manifest_path.with_name(manifest_path.stem + "_results.csv"))
    done = {} if force else load_results(results_csv)

    _print(f"Manifest: {len(rows)} subject(s) from {manifest_path}")
    _print(f"Results:  {results_csv}" + (
        "" if force else
        " (resuming — subjects already marked 'done' there are skipped)"))

    n_ok = n_failed = n_skipped = 0
    for i, row in enumerate(rows, start=1):
        subject_id = row["subject_id"].strip()
        if not force and done.get(subject_id, {}).get("status") == "done":
            _print(f"[{i}/{len(rows)}] {subject_id}: already done, skipping")
            n_skipped += 1
            continue

        try:
            dwi_paths = [_resolve(p, manifest_dir)
                        for p in row["dwi"].split(";") if p.strip()]
            if not dwi_paths:
                raise ValueError("empty 'dwi' cell")
            bval_path = _resolve(row["bval"], manifest_dir)
            bvec_path = _resolve(row["bvec"], manifest_dir)
            seg_path = _resolve(row["seg"], manifest_dir)
            model_text = (row.get("model") or "").strip()
            model_names = ([m.strip() for m in model_text.split(";") if m.strip()]
                           or list(model_default))
            t1_path = _resolve_optional(row, "t1", manifest_dir)
            tracts_path = _resolve_optional(row, "tracts", manifest_dir)
            conductivity_csv = _resolve_optional(row, "conductivity_csv",
                                                 manifest_dir)
            freq = (row.get("freq") or "").strip() or None
            aniso_ids_text = (row.get("aniso_ids") or "").strip() or None
            iso_ref_ids_text = (row.get("iso_ref_ids") or "").strip() or None
            seg_registered = _parse_bool(row.get("seg_registered", ""),
                                         default=True)
            if not seg_registered and not t1_path:
                raise ValueError("seg_registered is false but no 't1' column "
                                 "value was given")

            _print(f"[{i}/{len(rows)}] {subject_id}: starting "
                  f"({', '.join(model_names)}) …")
            result = run_one(
                dwi_paths=dwi_paths, bval_path=bval_path, bvec_path=bvec_path,
                seg_path=seg_path, model_names=model_names,
                case_name=subject_id, seg_registered=seg_registered,
                t1_path=t1_path, tracts_path=tracts_path,
                conductivity_csv=conductivity_csv, freq=freq,
                aniso_ids_text=aniso_ids_text,
                iso_ref_ids_text=iso_ref_ids_text)
        except Exception as exc:            # noqa: BLE001 - recorded, not fatal to the batch
            _print(f"[{i}/{len(rows)}] {subject_id}: FAILED — "
                  f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
            done[subject_id] = {
                "subject_id": subject_id, "status": "failed", "run_id": "",
                "elapsed_s": "", "zip_path": "",
                "error": f"{type(exc).__name__}: {exc}"}
            n_failed += 1
        else:
            _print(f"[{i}/{len(rows)}] {subject_id}: done in "
                  f"{result.elapsed_s:.0f}s")
            done[subject_id] = {
                "subject_id": subject_id, "status": "done",
                "run_id": result.run.job_id,
                "elapsed_s": f"{result.elapsed_s:.1f}",
                "zip_path": str(result.zip_path or ""), "error": ""}
            n_ok += 1
        write_results(results_csv, done)     # after EVERY subject, not just at the end

    _print(f"\nBatch finished: {n_ok} done, {n_failed} failed, {n_skipped} "
          f"skipped (of {len(rows)}). Results: {results_csv}")
    return 1 if n_failed else 0


# ============================================================================
# Argument parsing
# ============================================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="tasks/anicond/cli.py",
        description="DTI -> anisotropic conductivity tensor (the /anicond "
                    "pipeline), from the command line: one subject, or a "
                    "manifest of many.")

    single = p.add_argument_group("single subject")
    single.add_argument("--dwi", action="append",
                        help="DWI volume: one 4D file, or repeat --dwi for "
                             "each 3D direction, in acquisition order")
    single.add_argument("--bval", help="b-values, one per DWI volume")
    single.add_argument("--bvec", help="gradient directions (3xN or Nx3)")
    single.add_argument("--seg", help="tissue segmentation")
    single.add_argument("--case-name",
                        help="subject id / output prefix (default: from the "
                             "segmentation's filename)")
    single.add_argument("--out-json",
                        help="write this run's result summary as JSON")

    batch = p.add_argument_group("batch (a manifest of subjects)")
    batch.add_argument("--manifest",
                       help="CSV with columns: subject_id, dwi, bval, bvec, "
                            "seg[, model, t1, tracts, conductivity_csv, "
                            "freq, aniso_ids, iso_ref_ids, seg_registered]")
    batch.add_argument("--results-csv",
                       help="where batch progress is recorded (default: "
                            "<manifest>_results.csv); a subject marked "
                            "'done' there is skipped next time")
    batch.add_argument("--force", action="store_true",
                       help="re-run subjects already marked done")

    common = p.add_argument_group("common")
    common.add_argument("--model", action="append",
                        choices=task_settings.ENABLED_MODELS,
                        help=f"conductivity model, repeat to run several "
                             f"(default: {task_settings.DEFAULT_MODEL}); a "
                             f"manifest's own 'model' column overrides this "
                             f"per subject")
    common.add_argument("--seg-not-registered", dest="seg_registered",
                        action="store_false", default=True,
                        help="the segmentation is NOT already in the DWI's "
                             "space; requires --t1 so it can be registered "
                             "through a rigid T1 -> b0 fit (a manifest's own "
                             "'seg_registered' column overrides this per "
                             "subject)")
    common.add_argument("--t1",
                        help="the T1 the segmentation was drawn on (only "
                             "used with --seg-not-registered)")
    common.add_argument("--tracts",
                        help="a whole-brain tractogram of this DWI to reuse "
                             "instead of tracking (.vtk)")
    common.add_argument("--conductivity-csv",
                        help="target conductivity table (default: the "
                             "shipped default table)")
    common.add_argument("--freq",
                        help="which conductivity_iso_<freq> column of that "
                             "table to use (default: the table's own default)")
    common.add_argument("--aniso-ids",
                        help='anisotropic tissue id(s), e.g. "13,14,15" '
                             "(default: the table's WM-related ids)")
    common.add_argument("--iso-ref-ids",
                        help="reference tissue id(s) for directmapping's "
                             "global scale (default: the table's GM-related "
                             "ids)")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    single_only = (args.dwi, args.bval, args.bvec, args.seg, args.case_name,
                   args.out_json)
    if args.manifest and any(single_only):
        parser.error("--manifest cannot be combined with the single-subject "
                     "options (--dwi/--bval/--bvec/--seg/--case-name/"
                     "--out-json)")
    if not args.manifest and not (args.dwi and args.bval and args.bvec
                                  and args.seg):
        parser.error("either --manifest, or --dwi/--bval/--bvec/--seg "
                     "together, are required")
    if not args.manifest and not args.seg_registered and not args.t1:
        parser.error("--seg-not-registered needs --t1")

    model_default = tuple(args.model) if args.model \
        else (task_settings.DEFAULT_MODEL,)

    if args.manifest:
        return run_batch(
            Path(args.manifest), model_default=model_default,
            results_csv=Path(args.results_csv) if args.results_csv else None,
            force=args.force)

    _print(f"Subject: single run ({', '.join(model_default)})")
    try:
        result = run_one(
            dwi_paths=args.dwi, bval_path=args.bval, bvec_path=args.bvec,
            seg_path=args.seg, model_names=list(model_default),
            case_name=args.case_name, seg_registered=args.seg_registered,
            t1_path=args.t1, tracts_path=args.tracts,
            conductivity_csv=args.conductivity_csv, freq=args.freq,
            aniso_ids_text=args.aniso_ids, iso_ref_ids_text=args.iso_ref_ids)
    except Exception as exc:                # noqa: BLE001 - reported, then re-raised as an exit code
        _print(f"FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _print(f"\nDone in {result.elapsed_s:.0f}s — "
          f"{', '.join(result.ctx.model_names)}")
    _print(f"Archive: {result.zip_path}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(_result_summary(result),
                                                   indent=2))
        _print(f"Summary: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
