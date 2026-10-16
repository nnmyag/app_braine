#!/usr/bin/env python3
# ============================================================================
# tasks/isocond/cli.py — T1/T2 -> conductivity map, from the command line
# ----------------------------------------------------------------------------
# The same call the web page's Run button makes (service.run_job()), reached
# without a browser -- for a single subject, or a whole cohort at once. Not
# wired into run_web.py; run straight from here, per the README's own layout.
#
#   One subject:
#       python tasks/isocond/cli.py --t1 T1.nii.gz --t2 T2.nii.gz
#
#   A cohort:
#       python tasks/isocond/cli.py --manifest subjects.csv
#
#   Manifest columns (a header row, comma-separated):
#       subject_id, t1, t2[, model]
#     Relative paths resolve against the MANIFEST's own folder, not whatever
#     directory the command happens to be run from, so a manifest and the
#     files it names can be moved around together.
#
# Progress prints to the terminal exactly as the web page's console box would
# show it, and is ALSO written into the run's own run.log -- a CLI-driven run
# and a web-driven run are indistinguishable on disk (same data/runs/isocond/
# folder, same file names, same log). See app/jobs.py for why.
#
# Batch mode is resumable. Each subject's outcome is written to a results CSV
# the moment it finishes, not just at the end, so a crash mid-batch loses
# nothing already done; a subject marked "done" there is skipped on the next
# invocation (pass --force to redo everyone). One subject failing is logged
# and does not stop the rest -- the process exits 1 at the end if anything
# did fail, so a cron job or CI step still notices.
#
# Subjects run ONE AT A TIME, deliberately. This is the same code path
# service.run_job() already validates for a single web request; running two
# at once would mean two Python processes fighting over the one loaded model
# and, if CACHE_LOADED_MODELS is on, the SAME shared session object it is not
# built to be used from two threads at once. If a GPU is genuinely idle
# between subjects, that is worth revisiting -- not assumed here.
# ============================================================================

# --- repo root on sys.path --------------------------------------------------
# tasks/isocond/service.py already does this bootstrap for itself when
# imported (see its own header), but argparse has to run before that import
# happens, so the same three lines are repeated here.
import os as _os
import sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

from app import jobs
from tasks.isocond import service
from tasks.isocond import settings as task_settings

TASK_SLUG = "isocond"

MANIFEST_REQUIRED_COLUMNS = ("subject_id", "t1", "t2")
RESULTS_COLUMNS = ("subject_id", "status", "run_id", "elapsed_s", "zip_path",
                   "error")


def _print(line: str) -> None:
    print(line, flush=True)


# ============================================================================
# One subject
# ----------------------------------------------------------------------------
# Everything web.py's _run() does around service.run_job() -- prune old runs,
# open a fresh run folder, wire a Transcript through the shared queue -- minus
# the Gradio-specific streaming, since a terminal does not need polling.
# ============================================================================
def run_one(t1_path, t2_path, *, model: str, save_intermediate: bool,
           case_name: Optional[str]) -> service.JobResult:
    jobs.prune_old_runs(TASK_SLUG)
    run = jobs.new_run(TASK_SLUG)
    transcript = jobs.Transcript(log_file=run.log_file, sink=_print)
    with jobs.capture_logs(transcript):
        with jobs.QUEUE.slot(report=transcript.report):
            return service.run_job(
                t1_path=t1_path, t2_path=t2_path, model_key=model,
                save_intermediate=save_intermediate, case_name=case_name,
                run=run, report=transcript.report)


def _result_summary(result: service.JobResult) -> Dict:
    """The JSON-safe, useful half of a JobResult.

    `session_result` is deliberately left out: it is an in-memory handle the
    canvas uses to redraw without re-running anything, not something that
    survives being written to disk and read back.
    """
    return {
        "job_id": result.run.job_id,
        "model_key": result.model_key,
        "model_name": result.model_name,
        "conductivity_path": (str(result.conductivity_path)
                              if result.conductivity_path else None),
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


def read_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    """Every row, after checking the header has what run_one() will need.

    Checked up front, before touching any subject, so a typo'd column name
    fails once with a clear message instead of once per row it happens to hit.
    """
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
    """{subject_id: row} from a previous run of this same manifest, or {}."""
    if not results_csv.is_file():
        return {}
    with open(results_csv, newline="") as fh:
        return {row["subject_id"]: row for row in csv.DictReader(fh)}


def write_results(results_csv: Path, rows: Dict[str, Dict[str, str]]) -> None:
    """The whole results table, rewritten. Cheap at batch scale (tens to a
    few hundred rows) and far simpler than appending under a lock."""
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(results_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULTS_COLUMNS)
        writer.writeheader()
        for row in rows.values():
            writer.writerow({k: row.get(k, "") for k in RESULTS_COLUMNS})


def run_batch(manifest_path: Path, *, model_default: str,
             save_intermediate: bool, results_csv: Optional[Path],
             force: bool) -> int:
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

        t1 = _resolve(row["t1"], manifest_dir)
        t2 = _resolve(row["t2"], manifest_dir)
        model = (row.get("model") or "").strip() or model_default
        _print(f"[{i}/{len(rows)}] {subject_id}: starting ({model}) …")
        try:
            result = run_one(t1, t2, model=model,
                             save_intermediate=save_intermediate,
                             case_name=subject_id)
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
        prog="tasks/isocond/cli.py",
        description="T1/T2 -> conductivity map (the /isocond pipeline), from "
                    "the command line: one subject, or a manifest of many.")

    single = p.add_argument_group("single subject")
    single.add_argument("--t1", help="T1 volume (.nii.gz)")
    single.add_argument("--t2", help="T2 volume (.nii.gz)")
    single.add_argument("--case-name",
                        help="output filename prefix (default: from the T1's "
                             "filename)")
    single.add_argument("--out-json",
                        help="write this run's result summary as JSON")

    batch = p.add_argument_group("batch (a manifest of subjects)")
    batch.add_argument("--manifest",
                       help="CSV with columns: subject_id, t1, t2[, model]. "
                            "Relative paths resolve against the manifest's "
                            "own folder.")
    batch.add_argument("--results-csv",
                       help="where batch progress is recorded (default: "
                            "<manifest>_results.csv); a subject marked "
                            "'done' there is skipped next time")
    batch.add_argument("--force", action="store_true",
                       help="re-run subjects already marked done")

    common = p.add_argument_group("common")
    # Choices = what is in vendor/isocond/models right now (one CondNet-TART
    # per frequency found, plus CondNet).
    common.add_argument("--model", choices=service.available_model_keys(),
                        default=service.default_model_key(),
                        help=f"conductivity model (default: "
                             f"{service.default_model_key()}); a manifest's "
                             f"own 'model' column overrides this per subject")
    common.add_argument("--no-intermediate", action="store_true",
                        help="skip the segmentation, its legend and the "
                             "normalized volumes (default: save them)")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    single_only = (args.t1, args.t2, args.case_name, args.out_json)
    if args.manifest and any(single_only):
        parser.error("--manifest cannot be combined with --t1/--t2/"
                     "--case-name/--out-json")
    if not args.manifest and not (args.t1 and args.t2):
        parser.error("either --manifest, or both --t1 and --t2, are required")

    save_intermediate = not args.no_intermediate

    if args.manifest:
        return run_batch(
            Path(args.manifest), model_default=args.model,
            save_intermediate=save_intermediate,
            results_csv=Path(args.results_csv) if args.results_csv else None,
            force=args.force)

    _print(f"Subject: single run ({args.model})")
    try:
        result = run_one(args.t1, args.t2, model=args.model,
                         save_intermediate=save_intermediate,
                         case_name=args.case_name)
    except Exception as exc:                # noqa: BLE001 - reported, then re-raised as an exit code
        _print(f"FAILED — {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1

    _print(f"\nDone in {result.elapsed_s:.0f}s — {result.model_name}")
    _print(f"Archive: {result.zip_path}")
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(_result_summary(result),
                                                   indent=2))
        _print(f"Summary: {args.out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
