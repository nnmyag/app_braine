# ============================================================================
# tasks/tract_simplify/service.py — tractogram -> tracts that look right
# ----------------------------------------------------------------------------
# THE seam. UI-agnostic: run_job() takes paths and values and returns paths
# and values, so the web page and a later cli.py produce the same run folder.
#
# For VISUALIZATION ONLY. Task 2's whole-brain tractogram is honest about
# tractography: long tracts broken into pieces, thousands of near-identical
# streamlines per bundle, short fragments everywhere. This task makes a copy
# that reads well in a viewer and must never go back into task 2's
# tract-based model (a thinned tractogram would bias it). A healed copy is a
# fair input for task 5, which does the same join outward from its selection.
#
# What one job does, in two halves
#   Run (queued, once per upload)
#     1. load      the tractogram against the reference image's grid
#     2. heal      join the pieces of a broken tract: endpoints within the
#                  gap whose exit tangents are head-to-tail, matched
#                  one-to-one best score first (heal_breaks)
#     3. cluster   QuickBundles by shape, once -- the thinning slider only
#                  chooses how much of every cluster to keep
#   Sliders (apply(), milliseconds, no queue)
#     4. cut       drop streamlines shorter than the length (after healing)
#     5. thin      keep a fraction of each cluster, never less than one
#     6. smooth    a Gaussian along each kept streamline, sigma in mm
#     7. write     <case>_tracts_viz.vtk (LPS, coloured by direction, with
#                  per-streamline pieces / cluster / length) and a .trk twin
#                  (TrackVis, RAS mm through the reference's header) for the
#                  browser viewer
#
# Steps 2-6 are vendor/tractviz, shared: this file loads, keeps the state
# between slider releases, writes and draws.
#
# Space: the streamlines are read in the reference image's RAS mm, so any
# NIfTI in the tractogram's space serves (the DWI's b0, or task 3's T1).
# ============================================================================
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from app import jobs
from app import mosaic as mosaic_rule
from tasks.tract_efield import preview
from tasks.tract_efield import service as tract_efield_service
from tasks.tract_simplify import settings as task_settings
from vendor import tractviz

logger = logging.getLogger(__name__)

TASK_SLUG = "tract_simplify"

TRACT_TYPES = tract_efield_service.TRACT_TYPES
IMAGE_TYPES = (".nii", ".nii.gz", ".gz")


# ============================================================================
# What a run produces
# ============================================================================
FILE_MEANING: Dict[str, str] = {
    "tracts_vtk": "revised streamlines, coloured by direction, with "
                  "pieces/cluster/length per streamline — LPS, opens in "
                  "3D Slicer",
    "tracts_trk": "same streamlines as TrackVis .trk — what the 3D viewer "
                  "loads; also opens in Slicer, DSI Studio, DIPY",
}
FILE_ORDER = ("tracts_vtk", "tracts_trk")


@dataclass
class JobResult:
    """One loaded tractogram and the current state of the sliders. The
    arrays stay in memory for the session, so a slider release costs
    milliseconds; the paths are what the page hands out."""
    run: jobs.RunDir
    subject_id: str
    tractogram_name: str = ""
    reference_name: str = ""
    reference_path: Optional[Path] = None
    # -- the Run half: settings and what it found ---------------------------
    heal: bool = task_settings.DEFAULT_HEAL
    heal_gap_mm: float = task_settings.DEFAULT_HEAL_GAP_MM
    heal_angle_deg: float = task_settings.DEFAULT_HEAL_ANGLE_DEG
    n_original: int = 0
    n_healed: int = 0                  # streamlines after healing
    n_joins: int = 0                   # endpoint pairs joined
    n_clusters: int = 0
    load_s: float = 0.0
    heal_s: float = 0.0
    cluster_s: float = 0.0
    # -- the arrays (RAS mm), one entry per HEALED streamline ---------------
    original_mm: List[np.ndarray] = field(default_factory=list, repr=False)
    healed_mm: List[np.ndarray] = field(default_factory=list, repr=False)
    pieces: Optional[np.ndarray] = field(default=None, repr=False)
    lengths_mm: Optional[np.ndarray] = field(default=None, repr=False)
    cluster_id: Optional[np.ndarray] = field(default=None, repr=False)
    prepared: Optional[tractviz.Prepared] = field(default=None, repr=False)
    # -- the slider half: current values and what they keep -----------------
    min_length_mm: float = task_settings.DEFAULT_MIN_LENGTH_MM
    keep_percent: float = task_settings.DEFAULT_KEEP_PERCENT
    smooth_mm: float = task_settings.DEFAULT_SMOOTH_MM
    n_long_enough: int = 0
    n_kept: int = 0
    kept: Optional[np.ndarray] = field(default=None, repr=False)   # bool
    kept_mm: List[np.ndarray] = field(default_factory=list, repr=False)
    apply_s: float = 0.0
    apply_count: int = 0
    files: Dict[str, Path] = field(default_factory=dict)
    elapsed_s: float = 0.0
    completed_at: Optional[datetime] = None
    summary: str = ""
    # <subject>_tract_simplify___output.zip, the two files; rewritten with
    # them on every slider release, so the link is always current.
    zip_path: Optional[Path] = None

    @property
    def settings_line(self) -> str:
        heal = (f"healed (gap ≤ {self.heal_gap_mm:g} mm, ≤ "
                f"{self.heal_angle_deg:g}°)" if self.heal else "not healed")
        smooth = (f"smoothed σ = {self.smooth_mm:g} mm" if self.smooth_mm > 0
                  else "not smoothed")
        return (f"{heal} · length ≥ {self.min_length_mm:g} mm · "
                f"{self.keep_percent:g} % of each cluster · {smooth}")

    def guide(self, roles: Sequence[str] = FILE_ORDER
              ) -> List[Tuple[str, Path, str]]:
        return [(role, self.files[role], FILE_MEANING[role])
                for role in roles
                if role in self.files and Path(self.files[role]).exists()]


# ============================================================================
# Small helpers
# ============================================================================
def say(report: jobs.Reporter, message: str) -> None:
    logger.info(message)
    if report is not None:
        report(message)


def default_case_name(tract_path) -> str:
    """The first CASE_NAME_CHARS of the tractogram's name (task 2's rule)."""
    name = Path(str(tract_path)).name
    for ext in (".nii.gz", ".vtk", ".vtp", ".trk", ".tck"):
        if name.lower().endswith(ext):
            name = name[:-len(ext)]
            break
    return name[:task_settings.CASE_NAME_CHARS] or "case"


def is_image(path) -> bool:
    name = Path(str(path)).name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def load_tracts(path, ref_img) -> List[np.ndarray]:
    """Streamlines in the reference image's RAS mm (task 5's loader, which
    puts task 2's voxel-space read straight back into mm)."""
    return tract_efield_service.load_tracts(path, ref_img)


# The algorithms -- healing, clustering, the cut, thinning, smoothing --
# are vendor/tractviz (shared with the simplified tractograms tasks 2 and 5
# export). The names are kept here for callers and the README.
streamline_lengths = tractviz.streamline_lengths
heal_breaks = tractviz.heal_breaks
cluster_streamlines = tractviz.cluster_streamlines
thin_mask = tractviz.thin_mask
smooth_streamlines = tractviz.smooth_streamlines


# ============================================================================
# Writing
# ============================================================================
def write_trk(path: Path, streamlines_mm: Sequence[np.ndarray],
              reference_path) -> Path:
    """The streamlines as TrackVis .trk (nibabel). A .trk stores voxel-mm
    coordinates on a named grid, so the header is filled from the reference
    image (dimensions, voxel sizes, voxel-to-RAS affine); given RAS mm
    streamlines and that header nibabel does the conversion, and reading
    the file back with nibabel -- what tractio.load_tractogram does -- gives
    the same RAS mm. NiiVue reads it the same way."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(str(reference_path))
    header = nib.streamlines.trk.TrkFile.create_empty_header()
    header["voxel_to_rasmm"] = ref_img.affine.astype(np.float32)
    header["voxel_sizes"] = np.asarray(ref_img.header.get_zooms()[:3],
                                       dtype=np.float32)
    header["dimensions"] = np.asarray(ref_img.shape[:3], dtype=np.int16)
    header["voxel_order"] = "".join(nib.aff2axcodes(ref_img.affine))
    tractogram = nib.streamlines.Tractogram(
        [np.asarray(s, dtype=np.float32) for s in streamlines_mm],
        affine_to_rasmm=np.eye(4))
    nib.streamlines.save(tractogram, str(path), header=header)
    return path


def count_kept(result: JobResult, min_length_mm: float,
               keep_percent: float) -> int:
    """How many streamlines the length cut and thinning WOULD keep at
    these values. The same selection apply() makes, without the smoothing,
    the files or any change to `result`; a few milliseconds, so the page
    can show it while a slider is still being dragged."""
    if result.prepared is None:
        return 0
    return tractviz.count_kept(
        result.prepared,
        float(np.clip(min_length_mm, task_settings.MIN_LENGTH_MIN,
                      task_settings.MIN_LENGTH_MAX)),
        float(np.clip(keep_percent, task_settings.KEEP_PERCENT_MIN,
                      task_settings.KEEP_PERCENT_MAX)))


def apply(result: JobResult, min_length_mm: float, keep_percent: float,
          smooth_mm: float, report: jobs.Reporter = None) -> JobResult:
    """Cut, thin, smooth and write with the values given. Milliseconds of
    selection plus the write; no queue, this is what the sliders call."""
    t0 = time.time()
    result.min_length_mm = float(np.clip(min_length_mm,
                                         task_settings.MIN_LENGTH_MIN,
                                         task_settings.MIN_LENGTH_MAX))
    result.keep_percent = float(np.clip(keep_percent,
                                        task_settings.KEEP_PERCENT_MIN,
                                        task_settings.KEEP_PERCENT_MAX))
    result.smooth_mm = float(np.clip(smooth_mm, task_settings.SMOOTH_MIN_MM,
                                     task_settings.SMOOTH_MAX_MM))
    selection = tractviz.select(result.prepared, result.min_length_mm,
                                result.keep_percent, result.smooth_mm)
    kept = selection.kept
    result.n_long_enough = selection.n_long_enough
    result.kept = kept
    result.n_kept = selection.n_kept
    result.kept_mm = selection.streamlines
    result.apply_count += 1

    run = result.run
    files = result.files
    if result.kept_mm:
        directions = tract_efield_service.point_directions(result.kept_mm)
        files["tracts_vtk"] = tract_efield_service.write_tracts_vtk(
            run.output(f"{result.subject_id}_tracts_viz.vtk"),
            result.kept_mm, directions,
            cell_arrays=[("pieces", result.pieces[kept]),
                         ("cluster", result.cluster_id[kept])],
            cell_float_arrays=[("length_mm", result.lengths_mm[kept])])
        files["tracts_trk"] = write_trk(
            run.output(f"{result.subject_id}_tracts_viz.trk"), result.kept_mm,
            result.reference_path)
    else:
        for role in ("tracts_vtk", "tracts_trk"):
            path = files.pop(role, None)
            if path and Path(path).exists():
                Path(path).unlink()
    result.zip_path = (run.zip_outputs(subject_id=result.subject_id)
                       if result.kept_mm else None)
    result.apply_s = time.time() - t0
    result.summary = _summarize(result)
    say(report, f"Sliders: length ≥ {result.min_length_mm:g} mm keeps "
                f"{result.n_long_enough:,} of {result.n_healed:,}; "
                f"{result.keep_percent:g} % of each cluster keeps "
                f"{result.n_kept:,}"
                + (f"; smoothed σ = {result.smooth_mm:g} mm"
                   if result.smooth_mm > 0 else "")
                + f" ({result.apply_s:.1f} s)")
    return result


# ============================================================================
# The job
# ============================================================================
def run_job(tractogram, reference,
            heal: bool = task_settings.DEFAULT_HEAL,
            heal_gap_mm: float = task_settings.DEFAULT_HEAL_GAP_MM,
            heal_angle_deg: float = task_settings.DEFAULT_HEAL_ANGLE_DEG,
            min_length_mm: float = task_settings.DEFAULT_MIN_LENGTH_MM,
            keep_percent: float = task_settings.DEFAULT_KEEP_PERCENT,
            smooth_mm: float = task_settings.DEFAULT_SMOOTH_MM,
            subject_id: Optional[str] = None,
            run: Optional[jobs.RunDir] = None,
            report: jobs.Reporter = None) -> JobResult:
    """Load, heal and cluster one tractogram, then apply the sliders once."""
    t_start = time.time()
    if not tractogram:
        raise ValueError("A tractogram (.vtk / .vtp / .trk / .tck) is "
                         "required.")
    if Path(str(tractogram)).suffix.lower() not in TRACT_TYPES:
        raise ValueError(f"Tractogram: {Path(str(tractogram)).name} is not "
                         f"one of {', '.join(TRACT_TYPES)}")
    if not reference:
        raise ValueError("A reference image in the tractogram's space "
                         "(the DWI's b0, .nii.gz) is required: it gives the "
                         "grid and the background.")
    if not is_image(reference):
        raise ValueError(f"Reference: {Path(str(reference)).name} is not a "
                         f".nii / .nii.gz")
    heal_gap_mm = float(np.clip(heal_gap_mm, task_settings.HEAL_GAP_MIN,
                                task_settings.HEAL_GAP_MAX))
    heal_angle_deg = float(np.clip(heal_angle_deg,
                                   task_settings.HEAL_ANGLE_MIN,
                                   task_settings.HEAL_ANGLE_MAX))

    run = run or jobs.new_run(TASK_SLUG)
    subject_id = (subject_id or "").strip() or default_case_name(tractogram)
    say(report, f"Job {run.job_id} · subject {subject_id}")

    # -- 1. load ------------------------------------------------------------
    ref_name = Path(str(reference)).name
    ref_staged = jobs.stage_upload(reference, run, f"{subject_id}_reference"
                                   + (".nii.gz" if ref_name.lower().endswith(
                                       ".nii.gz") else ".nii"))
    ref_img = nib.load(str(ref_staged))
    staged = jobs.stage_upload(tractogram, run, f"{subject_id}_tracts"
                               f"{Path(str(tractogram)).suffix.lower()}")
    say(report, f"Tracts: loading {Path(str(tractogram)).name} against "
                f"{ref_name} ({'×'.join(str(d) for d in ref_img.shape[:3])}) …")
    t0 = time.time()
    original = load_tracts(staged, ref_img)
    if not original:
        raise ValueError(f"{Path(str(tractogram)).name} contains no "
                         f"streamline")
    result = JobResult(run=run, subject_id=subject_id,
                       tractogram_name=Path(str(tractogram)).name,
                       reference_name=ref_name, reference_path=ref_staged,
                       heal=bool(heal), heal_gap_mm=heal_gap_mm,
                       heal_angle_deg=heal_angle_deg,
                       original_mm=original, n_original=len(original),
                       load_s=time.time() - t0)
    lengths0 = streamline_lengths(original)
    say(report, f"Tracts: {len(original):,} streamlines, length "
                f"{np.median(lengths0):.0f} mm median "
                f"({np.percentile(lengths0, 5):.0f}–"
                f"{np.percentile(lengths0, 95):.0f} mm, 5th–95th), "
                f"{result.load_s:.1f} s")

    # -- 2-3. heal + cluster, once (vendor/tractviz) ------------------------
    t0 = time.time()
    prepared = tractviz.prepare(original, heal=bool(heal),
                                heal_gap_mm=heal_gap_mm,
                                heal_angle_deg=heal_angle_deg,
                                report=lambda line: say(report, line))
    result.prepared = prepared
    result.healed_mm, result.pieces = prepared.healed, prepared.pieces
    result.n_joins, result.n_healed = prepared.n_joins, prepared.n_healed
    result.lengths_mm = prepared.lengths_mm
    result.cluster_id, result.n_clusters = prepared.cluster_id, prepared.n_clusters
    result.heal_s = time.time() - t0          # healing and clustering together

    # -- 4-7. the sliders, once with the values given -----------------------
    apply(result, min_length_mm, keep_percent, smooth_mm, report)
    result.completed_at = datetime.now()
    result.elapsed_s = time.time() - t_start
    result.summary = _summarize(result)
    return result


def _summarize(result: JobResult) -> str:
    lines = [f"{result.n_kept:,} of {result.n_original:,} streamlines kept — "
             f"{result.settings_line}"]
    if result.heal:
        lines.append(f"healing: {result.n_joins:,} joins, "
                     f"{result.n_original:,} → {result.n_healed:,} "
                     f"streamlines")
    lines.append(f"length cut: {result.n_long_enough:,} of "
                 f"{result.n_healed:,} ≥ {result.min_length_mm:g} mm; "
                 f"thinning: {result.n_kept:,} kept across "
                 f"{result.n_clusters:,} clusters")
    if "tracts_vtk" in result.files:
        path = Path(result.files["tracts_vtk"])
        lines.append(f"Written: {path.name}, "
                     f"{path.stat().st_size / 1e6:.1f} MB"
                     + (f"; archive {Path(result.zip_path).name}"
                        if result.zip_path else ""))
    return "\n".join(lines)


# ============================================================================
# The 3D viewer (optional, JavaScript only)
# ============================================================================
FILE_ROUTE = "/gradio_api/file="


def viewer_status() -> Tuple[bool, str]:
    """Whether the browser viewer can run: its page and the NiiVue bundle
    are both in assets/. Nothing Python-side depends on it."""
    if not task_settings.VIEWER_PAGE.is_file():
        return False, f"{task_settings.VIEWER_PAGE.name} is missing from assets/"
    if not task_settings.VIEWER_BUNDLE.is_file():
        return False, (f"{task_settings.VIEWER_BUNDLE.name} is not installed "
                       f"in assets/ (see tasks/tract_simplify/settings.py)")
    return True, ""


def viewer_url(result: Optional[JobResult], reload: int = 0) -> str:
    """The iframe URL for this result's current .trk and reference, or ''.
    `reload` is a number the page bumps on each press of its Refresh button,
    so the same files load again even when nothing changed."""
    ok, _reason = viewer_status()
    if not ok or result is None or "tracts_trk" not in result.files:
        return ""
    from urllib.parse import urlencode
    page = f"{FILE_ROUTE}{task_settings.VIEWER_PAGE.resolve()}"
    query = urlencode({
        "vol": f"{FILE_ROUTE}{Path(result.reference_path).resolve()}",
        "tracts": f"{FILE_ROUTE}{Path(result.files['tracts_trk']).resolve()}",
        "n": result.n_kept,
        "v": result.apply_count,           # defeats the browser cache
        "r": int(reload),                  # and makes a manual refresh reload
    })
    return f"{page}?{query}"


# ============================================================================
# The canvas: one figure, original (left) vs. revised (right)
# ============================================================================
def _drawn(streamlines: Sequence[np.ndarray]) -> List[np.ndarray]:
    limit = task_settings.MAX_DRAWN_STREAMLINES
    if len(streamlines) <= limit:
        return list(streamlines)
    rng = np.random.default_rng(task_settings.DRAW_SEED)
    keep = np.sort(rng.choice(len(streamlines), size=limit, replace=False))
    return [streamlines[i] for i in keep]


def middle_slice(background: np.ndarray) -> int:
    """The axial index the canvas shows: the middle of the slices that
    carry anatomy, which depends on the reference image alone."""
    lo, hi = mosaic_rule.occupied_span(background)
    return int(round(0.5 * (lo + hi)))


def render_canvas(result: Optional[JobResult]):
    """The canvas: one figure, the original tractogram on the left and the
    current revision on the right, both at the same middle slice. Redrawn
    on Run and on every slider release; nothing to pick, so there is no
    view control on the page."""
    if result is None:
        return None
    try:
        background, affine = preview.load_canonical(result.reference_path)
        idx = middle_slice(background)
        title = (f"{result.subject_id} Processed Tracts\n"
                 f"Min length: {result.min_length_mm:g} mm; Thinning: "
                 f"{result.keep_percent:g} % ({result.n_kept:,}/"
                 f"{result.n_original:,} tracts)\n"
                 f"Shown slice: #{idx}; Color-encoding: red L-R, green "
                 f"A-P, blue S-I.")
        return preview.tract_pair_view(
            background, affine, _drawn(result.original_mm),
            _drawn(result.kept_mm), idx, title,
            tile_in=task_settings.CANVAS_TILE_IN,
            linewidth=task_settings.TRACT_LINEWIDTH_CANVAS,
            dpi=task_settings.CANVAS_DPI)
    except Exception:
        logger.exception("Could not render the canvas")
        return None
