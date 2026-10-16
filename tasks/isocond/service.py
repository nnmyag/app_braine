# ============================================================================
# tasks/isocond/service.py — T1/T2 -> conductivity map
# ----------------------------------------------------------------------------
# THE seam. UI-agnostic: no gradio, no Qt. Every front-end calls run_job() and
# gets back paths and values.
#
# Thinner than tasks/anicond/service.py on purpose. The vendored snapshot
# already HAS a UI-agnostic layer -- `vendor/isocond/pipeline/session.py`, which
# imports no toolkit and was written to back both a Qt and a web front-end. This
# file does not reimplement it; it adapts it to APP_BRAINE's conventions:
#
#   * the job runs inside a data/runs/isocond/<job id>/ folder rather than a
#     configured OUTPUT_DIR, so a served run leaves the same trail as any other
#   * uploads are staged into inputs/, so the run folder is a complete record
#   * progress arrives as lines on a Reporter instead of a (fraction, message)
#     callback
#   * the results are zipped
#
# What is NOT here: registration, preprocessing, batching, inference and output
# writing. Those stay in the snapshot, where the study can keep improving them.
# ============================================================================

# --- repo root on sys.path, so `vendor.isocond` and `app` import from any
# --- working directory (same bootstrap convention as PROJ_DCONDNET)
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import csv
import hashlib
import logging
import os
import shutil
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from matplotlib.colors import Normalize

import settings as app_settings
from app import jobs
from app import mosaic as mosaic_rule
from app.web import slice_viewer as sv
from tasks.isocond import preview
from tasks.isocond import settings as task_settings
from vendor.isocond import settings as isocond_settings
from vendor.isocond.nets import condnettart
from vendor.isocond.nets import registry as isocond_registry
from vendor.isocond.pipeline import preprocess as isocond_preprocess
from vendor.isocond.pipeline import session as isocond_session
from vendor.isocond.pipeline.imaging import (_SAVE_KWARGS, _build_mosaic_figure,
                                             discrete_label_cmap, label_display_range,
                                             load_image_as_array, render_mosaic)
from vendor.isocond.pipeline.runner import InferenceRunner
from vendor.isocond.pipeline.session import (DEFAULT_VIEW, RESULT_VIEW,
                                             VIEW_LABELS, InferenceSession)
from vendor.isocond.pipeline.tissues import (load_tissue_conductivity,
                                             load_tissue_labels)

logger = logging.getLogger(__name__)

# Keys of `written_paths` that are review images rather than results.
_PREVIEW_SUFFIX = "_mosaic_path"


@dataclass
class JobResult:
    """What one completed job produced. Paths only -- no widgets, no arrays."""
    run: jobs.RunDir
    model_key: str
    model_name: str
    conductivity_path: Optional[Path] = None       # the deliverable
    all_outputs: List[Path] = field(default_factory=list)
    previews: List[Path] = field(default_factory=list)
    zip_path: Optional[Path] = None
    elapsed_s: float = 0.0
    summary: str = ""

    # {role: path} exactly as the snapshot wrote them, e.g.
    # "conductivity_volume_path" / "segmentation_path". A front-end groups its
    # download list by these roles rather than by guessing from filenames.
    written_paths: Dict[str, Path] = field(default_factory=dict)
    # The "<case>_<model>_" every output filename starts with, so a list of
    # this run's files can drop the prefix every row repeats.
    prefix: str = ""
    # Opaque handle for render_view(). It carries the prediction arrays, which
    # is why the canvas can switch views without re-running anything.
    session_result: object = None
    # What the slice viewer needs beyond the prediction (see ViewerData).
    viewer: object = None

    def short_name(self, path) -> str:
        name = Path(path).name
        return (name[len(self.prefix):]
                if self.prefix and name.startswith(self.prefix) else name)


# ============================================================================
# The shared session
# ----------------------------------------------------------------------------
# One InferenceSession per process, holding the loaded checkpoints. CondNet-TART
# is 107 MB, so loading it per job would cost more than the inference does.
#
# Sharing mutable state between jobs is only safe because the queue in
# app/jobs.py runs one at a time; the lock below guards the LOAD, so two pages
# opening at once cannot start building the same model twice.
# ============================================================================
_session: Optional[InferenceSession] = None
_session_lock = threading.Lock()


def get_session() -> InferenceSession:
    """The process-wide inference session, built on first use."""
    global _session
    if not task_settings.CACHE_LOADED_MODELS:
        return InferenceSession(device=task_settings.DEVICE)
    with _session_lock:
        if _session is None:
            logger.info("Creating the shared inference session on %s",
                        task_settings.DEVICE)
            _session = InferenceSession(device=task_settings.DEVICE)
        return _session


def model_labels() -> List[tuple]:
    """[(label, key)] for a model picker: the enabled families in
    ENABLED_MODELS order, each CondNet-TART frequency found in the model
    folder as its own entry (lowest frequency first). Rescans the folder, so
    a checkpoint pair added since the last call shows up. Uploaded models are
    not listed: each belongs to the page session that uploaded it."""
    isocond_registry.refresh_models()
    names = dict(isocond_session.model_choices())        # key -> display name
    return [(names[key], key)
            for family in task_settings.ENABLED_MODELS
            for key in names if isocond_registry.model_family(key) == family
            and not isocond_registry.is_extra(key)]


# ============================================================================
# Uploaded CondNet-TART pairs
# ----------------------------------------------------------------------------
# A user may run their own checkpoints: the two files named exactly as the
# model folder's (condnettart_<freq>_stage0.pth / _stage1.pth), plus the
# model card condnettart_<freq>.json when the frequency's output scale is not
# built in. Kept under data/user_models/isocond/<freq>_<hash>/ (same content
# = same folder and key), registered under the key
# condnettart_<freq>_upload<hash>, re-registered after a restart, and deleted
# with the runs after settings.KEEP_RUNS_DAYS.
# ============================================================================
UPLOAD_OPTION = "__upload__"
UPLOAD_LABEL = "Upload CondNet-TART model files"
UPLOAD_DIR = app_settings.DATA_DIR / "user_models" / "isocond"


def _register_upload_folder(folder: Path) -> Optional[type]:
    found = condnettart.discover_checkpoints(folder)
    if not found:
        return None
    ck = found[0]
    base = condnettart.model_class_for(ck)
    key = f"{condnettart.MODEL_FAMILY}_{ck.frequency}_upload{folder.name.rsplit('_', 1)[-1]}"
    cls = type(base.__name__ + "_upload", (base,), {
        "KEY": key, "NAME": f"{condnettart.MODEL_NAME} ({condnettart.frequency_label(ck.frequency)}, uploaded)"})
    isocond_registry.register_extra(cls)
    return cls


def _restore_uploads() -> None:
    """Re-register the uploads still on disk (after a restart); drop old ones."""
    if not UPLOAD_DIR.is_dir():
        return
    cutoff = time.time() - app_settings.KEEP_RUNS_DAYS * 86400
    for folder in sorted(UPLOAD_DIR.iterdir()):
        if not folder.is_dir():
            continue
        if folder.stat().st_mtime < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            continue
        try:
            _register_upload_folder(folder)
        except Exception:                       # noqa: BLE001 - one bad folder
            logger.exception("Uploaded model %s could not be registered", folder)


def register_upload(paths) -> tuple:
    """(label, key) for an uploaded pair, or ValueError saying what is wrong."""
    pth, cards = {}, []
    for path in map(Path, paths or []):
        name = path.name
        m = condnettart.CHECKPOINT_RE.match(name)
        if m:
            pth[(m["freq"], m["stage"])] = path
        elif name.startswith(condnettart.MODEL_FAMILY + "_") and name.endswith(".json"):
            cards.append(path)
        else:
            raise ValueError(f"{name}: expected condnettart_<freq>_stage0.pth, "
                             f"condnettart_<freq>_stage1.pth (and condnettart_<freq>.json)")
    freqs = {f for f, _ in pth}
    if len(freqs) != 1 or len(pth) != 2:
        raise ValueError("Upload both files of ONE frequency: condnettart_<freq>_stage0.pth "
                         "and condnettart_<freq>_stage1.pth (e.g. condnettart_10kHz_...)")
    freq = freqs.pop()
    card = next((c for c in cards if c.name == f"{condnettart.MODEL_FAMILY}_{freq}.json"), None)
    if cards and card is None:
        raise ValueError(f"The model card must be named condnettart_{freq}.json")
    digest = hashlib.sha256()
    for stage in ("0", "1"):
        with open(pth[(freq, stage)], "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
    folder = UPLOAD_DIR / f"{freq}_{digest.hexdigest()[:8]}"
    folder.mkdir(parents=True, exist_ok=True)
    for (_f, stage), src in pth.items():
        shutil.copyfile(src, folder / f"{condnettart.MODEL_FAMILY}_{freq}_stage{stage}.pth")
    if card is not None:
        shutil.copyfile(card, folder / card.name)
    os.utime(folder)                            # "recently used" for the clean-up
    try:
        cls = _register_upload_folder(folder)
        if cls is None:
            raise ValueError(f"The output scale at {freq} is not built in: also upload "
                             f"condnettart_{freq}.json (the model card)")
        cls(device="cpu")                       # the weights must fit the network
    except ValueError:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    except Exception as exc:                    # noqa: BLE001 - a bad .pth
        shutil.rmtree(folder, ignore_errors=True)
        raise ValueError(f"These files are not a CondNet-TART pair: "
                         f"{type(exc).__name__}: {exc}") from None
    logger.info("Uploaded model registered: %s (%s)", cls.KEY, folder)
    return cls.NAME, cls.KEY


def available_model_keys() -> List[str]:
    return [key for _label, key in model_labels()]


def is_runnable(model_key: Optional[str]) -> bool:
    """A listed model, or an uploaded one."""
    return bool(model_key) and (model_key in available_model_keys()
                                or isocond_registry.is_extra(model_key))


def default_model_key() -> Optional[str]:
    """task_settings.DEFAULT_MODEL if available, else the first model listed."""
    keys = available_model_keys()
    if task_settings.DEFAULT_MODEL in keys:
        return task_settings.DEFAULT_MODEL
    return keys[0] if keys else None


def model_frequency(model_key: str) -> Optional[str]:
    """'condnettart_100MHz' -> '100MHz'; None for a model with no frequency tag."""
    return getattr(isocond_registry.get_model_class(model_key), "FREQUENCY", None)


# ============================================================================
# The tissue table
# ----------------------------------------------------------------------------
# A segmentation is a volume of integers; on its own it says nothing about what
# label 8 is or what conductivity the pipeline gives it. This writes that
# mapping out beside the results, at the frequency the run actually used, so a
# downloaded segmentation is self-explanatory months later.
#
# It reads the same two functions the runner reads (tissues.py), so the file can
# never disagree with what the pipeline applied.
# ============================================================================
TISSUE_TABLE_KEY = "tissue_table_path"
HISTOGRAM_KEY = "value_histogram_path"
# A view of this task's own, after the snapshot's: the histogram above.
HISTOGRAM_VIEW = "Conductivity histogram"


def anisotropic_tissue_ids() -> List[int]:
    """Ids of ANISOTROPIC_TISSUE_NAMES in the snapshot's tissue legend."""
    wanted = tuple(n.lower() for n in task_settings.ANISOTROPIC_TISSUE_NAMES)
    return sorted(tid for tid, name in load_tissue_labels().items()
                  if any(w in str(name).lower() for w in wanted))


def write_value_histogram(session_result, out_dir, prefix: str, unit: str):
    """The conductivity histogram, anisotropic tissues vs the rest, as a PNG.

    Returns the path, or None when the model gave no segmentation (there is
    then nothing to split the map by) or no anisotropic id is present.
    """
    if not session_result.has_segmentation:
        return None
    seg = np.asarray(session_result.prediction["segmentation"])
    cond = np.asarray(session_result.conductivity)
    if seg.shape != cond.shape:
        logger.warning("Histogram skipped: segmentation %s and map %s differ",
                       seg.shape, cond.shape)
        return None
    ids = anisotropic_tissue_ids()
    aniso = np.isin(seg, ids)
    if not aniso.any():
        return None
    masks = {"anisotropic": aniso, "isotropic": (seg > 0) & ~aniso}
    path = Path(out_dir) / f"{prefix}conductivity_histogram.png"
    names = load_tissue_labels()
    preview.save_png(
        preview.value_hist(cond, masks,
                           f"{prefix.rstrip('_')} — anisotropic = "
                           f"{', '.join(str(names.get(i, i)) for i in ids)}",
                           unit=unit),
        path)
    return path


def write_tissue_table(out_dir, prefix: str = "",
                       frequency: Optional[str] = None) -> Optional[Path]:
    """Write `tissue_id, tissue_name, conductivity_s_per_m` as a CSV.

    Returns the path, or None if the tissue table could not be read -- a
    finished run must not be lost over its documentation.
    """
    column = frequency or isocond_settings.CONDUCTIVITY_FREQUENCY
    try:
        names = load_tissue_labels()
        conductivity = load_tissue_conductivity(column=column)
    except (KeyError, OSError) as exc:
        logger.warning("Could not write the tissue table: %s", exc)
        return None
    if not names:
        logger.warning("Tissue table is empty; not writing it")
        return None

    path = Path(out_dir) / f"{prefix}tissue_conductivity_{column}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tissue_id", "tissue_name", "conductivity_s_per_m"])
        for tissue_id in sorted(names):
            # A label absent from the conductivity column maps to 0.0, which is
            # exactly what segmentation_to_conductivity does with it.
            writer.writerow([tissue_id, names[tissue_id],
                             conductivity.get(tissue_id, 0.0)])
    return path


# ============================================================================
# The job
# ============================================================================
def _staged_name(path, case_name: Optional[str], modality: str) -> Optional[str]:
    """`<case>_T1.nii.gz` for an uploaded T1, keeping the original suffix.

    None leaves the upload's own filename alone. Worth getting right because
    the snapshot reads the case id off the T1 filename, so this string becomes
    the prefix of every output file.
    """
    if not case_name:
        return None
    name = Path(str(path)).name
    suffix = ".nii.gz" if name.lower().endswith(".nii.gz") else Path(name).suffix
    return f"{case_name}_{modality}{suffix}"


def run_job(t1_path,
            t2_path,
            model_key: Optional[str] = None,
            save_intermediate: bool = task_settings.SAVE_INTERMEDIATE,
            case_name: Optional[str] = None,
            run: Optional[jobs.RunDir] = None,
            report: jobs.Reporter = None) -> JobResult:
    """Predict a conductivity map from one co-registered T1/T2 pair.

    Params:
        t1_path, t2_path  : NIfTI volumes (or 2D images; the snapshot handles
                            both, and the output follows the input's dimensions)
        model_key         : a key from model_labels(), e.g. 'condnettart_10kHz'
                            or 'condnet'; None -> default_model_key()
        save_intermediate : also write the segmentation, its legend, and the
                            normalized volumes the model actually saw
        case_name         : output filename prefix; None -> from the T1 name
        run               : an existing run folder; None -> a fresh one
        report            : callable taking one progress line

    Returns a JobResult. Raises ValueError / FileNotFoundError for bad input.
    """
    t_start = time.time()
    run = run or jobs.new_run("isocond")

    model_key = model_key or default_model_key()
    if not is_runnable(model_key):
        raise ValueError(f"Unknown model {model_key!r}; available: "
                         f"{', '.join(available_model_keys())}")

    def say(message: str) -> None:
        logger.info(message)
        if report is not None:
            report(message)

    say(f"Job {run.job_id} — model {model_key}")

    # Stage the upload, so the run folder records what was computed from. The
    # snapshot derives the case id from the T1 FILENAME, so the staged name is
    # what ends up on every output -- renaming here renames the results. The
    # original suffix is kept, since a 2D .png input must stay a .png.
    t1_staged = jobs.stage_upload(t1_path, run, name=_staged_name(t1_path, case_name, "T1"))
    t2_staged = jobs.stage_upload(t2_path, run, name=_staged_name(t2_path, case_name, "T2"))

    session = get_session()
    say(f"Preparing {model_key} (checkpoints are cached between jobs)")

    # session.run reports (fraction, message); the shell wants lines.
    def progress(_fraction: float, message: str) -> None:
        if report is not None:
            report(message)

    with _capture_registration() as captured:
        result = session.run(
            t1_path=str(t1_staged),
            t2_path=str(t2_staged),
            model_key=model_key,
            save_intermediate=save_intermediate,
            output_dir=str(run.outputs),
            progress=progress,
        )
    try:
        viewer = build_viewer(result, captured, model_key)
    except Exception:                      # the viewer must never cost the run
        logger.exception("Slice viewer data could not be prepared")
        viewer = None

    written: Dict[str, str] = dict(result.written_paths)

    # The snapshot's review mosaic is scaled to the model's ceiling (2 S/m);
    # the page shows the map capped (CONDUCTIVITY_VMAX). Replace the file with
    # the page's picture, so what is downloaded is what was looked at.
    mosaic_path = written.get("conductivity_mosaic_path")
    if mosaic_path:
        capped = capped_conductivity_mosaic(result)
        if capped is not None:
            capped.save(str(mosaic_path))
            say(f"Conductivity mosaic: rewritten at the page's scale "
                f"(0–{task_settings.CONDUCTIVITY_VMAX:g} S/m)")

    # One of the intermediates: it explains the segmentation's integers, so it
    # travels with them rather than with the conductivity map.
    if save_intermediate:
        # The table's column is the model's own frequency (tissue_labels.csv
        # columns are lower-case: 10khz, 100mhz, ...); a model without a
        # frequency tag (CondNet) keeps the snapshot's default column.
        freq = model_frequency(model_key)
        column = freq.lower() if freq else isocond_settings.CONDUCTIVITY_FREQUENCY
        table = write_tissue_table(run.outputs, result.prefix, frequency=column)
        if table is not None:
            written[TISSUE_TABLE_KEY] = str(table)
            say(f"Tissue table: {table.name} ({column})")
        else:
            say(f"Tissue table: skipped -- tissue_labels.csv has no '{column}' column")
        if task_settings.SAVE_VALUE_HISTOGRAM:
            unit = getattr(session.get_runner(model_key).model,
                           "CONDUCTIVITY_UNIT", "S/m")
            hist = write_value_histogram(result, run.outputs, result.prefix,
                                         unit)
            if hist is not None:
                written[HISTOGRAM_KEY] = str(hist)
                say(f"Histogram: {hist.name} (anisotropic tissues vs the rest)")
            else:
                say("Histogram: skipped — this model gives no segmentation to "
                    "split the map by")

    outputs = [Path(p) for p in written.values()]
    previews = [Path(p) for key, p in written.items()
                if key.endswith(_PREVIEW_SUFFIX)]
    # The conductivity mosaic first -- it is the result; the rest are context.
    previews.sort(key=lambda p: ("conductivity" not in p.name, p.name))

    out = JobResult(
        run=run,
        model_key=result.model_key,
        model_name=result.model_name,
        conductivity_path=(Path(result.conductivity_path)
                           if result.conductivity_path else None),
        all_outputs=outputs,
        previews=previews,
        elapsed_s=time.time() - t_start,
        written_paths={key: Path(p) for key, p in written.items()},
        prefix=result.prefix,
        session_result=result,
        viewer=viewer,
    )
    out.zip_path = run.zip_outputs(f"{result.prefix.rstrip('_') or 'isocond'}")
    out.summary = result.summary()
    say(f"Finished in {out.elapsed_s:.0f}s — {len(outputs)} file(s), zipped as "
        f"{out.zip_path.name}")
    return out


# ============================================================================
# Rendering the canvas
# ----------------------------------------------------------------------------
# Both helpers return a PIL image, never a widget or a file, so a Qt front-end
# could show the same thing. The snapshot already draws every view; these only
# choose which of its two paths to take:
#
#   before a run  -> the uploaded volume itself, so a file can be eyeballed
#                    before spending minutes on inference
#   after a run   -> the prediction, straight out of the JobResult
#
# `InferenceSession.render` would also serve the first case through its own
# `previews` dict, but that dict lives on the PROCESS-WIDE session -- two tabs
# would overwrite each other's preview. Loading the file here keeps a preview
# local to the caller that asked for it.
# ============================================================================
def view_labels() -> List[str]:
    """The selectable views, in display order: the slice viewer (the default
    after a run), the snapshot's mosaics (handy for exporting pictures), then
    ours."""
    return [VIEWER_VIEW] + list(VIEW_LABELS) + [HISTOGRAM_VIEW]


def is_input_view(view_label: str) -> Optional[str]:
    """'T1 loaded' -> 't1', 'T2 loaded' -> 't2', anything else -> None."""
    return {"t1_loaded": "t1", "t2_loaded": "t2"}.get(
        _view_key(view_label) or "")


def _view_key(view_label: str) -> Optional[str]:
    # Cheap: InferenceSession.__init__ loads no weights, it only holds them.
    return get_session().view_key_for(view_label)


def _layout_for(volume: np.ndarray):
    """The grid for a volume the snapshot draws: it shows volume[k, :, :], so
    the drawn width is shape[2] and the height shape[1]. One rule for every
    task -- see app/mosaic.py."""
    return mosaic_rule.layout_for_slice(volume.shape[2], volume.shape[1])


def _for_mosaic(volume: np.ndarray) -> np.ndarray:
    """The volume trimmed to the slices that carry signal (app/mosaic's rule).

    The snapshot's renderer spreads its tiles over the WHOLE z extent, which
    on a raw scan spends a row on noise-only slices; vendor/ is frozen, so the
    trim is done here, before the volume is handed over. Only the mosaic sees
    the trimmed copy."""
    volume = np.asarray(volume)
    if volume.ndim == 2:
        volume = volume[None, ...]
    lo, hi = mosaic_rule.occupied_span(volume, axis=0)
    return volume[lo:hi + 1]


def render_input_preview(path, label: str = "T1"):
    """Mosaic of an uploaded volume, windowed the way the snapshot windows raw
    scanner intensities (no fixed scale, so each volume gets its own)."""
    if not path:
        return None
    volume, _affine = load_image_as_array(str(path))
    volume = _for_mosaic(volume)               # a single slice still mosaics
    vmin, vmax = InferenceRunner._display_window(volume)
    return render_mosaic(
        volume, vmin=vmin, vmax=vmax,
        layout=_layout_for(volume),
        cmap=isocond_settings.ANATOMY_CMAP,
        cbar_label=f"{label} raw intensity (a.u.)",
    )


def render_result_view(view_label: str, result: Optional[JobResult]):
    """Mosaic of one view of a finished run, or None if it has no such view.

    The conductivity view is drawn HERE rather than by the snapshot when
    task_settings.CONDUCTIVITY_VMAX is set: the snapshot scales it to the
    model's own ceiling (2 S/m), this caps it at blood so GM and WM separate.
    Only for a model that outputs S/m -- a normalized output keeps the
    snapshot's [0, 1]. The review mosaic the run writes to disk is the
    snapshot's and keeps its scale.
    """
    if result is None or result.session_result is None:
        return None
    if view_label == HISTOGRAM_VIEW:
        path = result.written_paths.get(HISTOGRAM_KEY)
        return str(path) if path and Path(path).is_file() else None
    session = get_session()
    sr = result.session_result
    if session.view_key_for(view_label) == "conductivity":
        capped = capped_conductivity_mosaic(sr)
        if capped is not None:
            return capped
    return _render_session_view(session, view_label, sr)


def _render_session_view(session, view_label: str, sr):
    """The snapshot's session.render(), with the grid chosen by app/mosaic
    instead of its fixed MOSAIC_LAYOUT.

    Mirrors vendor/isocond/pipeline/session.py's render() line for line --
    the view lookup, the input-preview-over-result precedence, the window and
    the bar label all come from the session -- so the two differ in nothing
    but the grid. Done here rather than in the snapshot because vendor/ is
    frozen; the review mosaics the snapshot writes to disk keep its layout.
    """
    view = isocond_session._VIEWS_BY_LABEL.get(view_label)
    if view is None:
        raise KeyError(f"Unknown view {view_label!r}")
    volume = session.previews.get(view.key)
    if volume is None and sr is not None:
        volume = sr.prediction.get(view.key)
    if volume is None:
        return None
    volume = _for_mosaic(volume)
    vmin, vmax = session._window(view, volume, sr)
    return render_mosaic(
        volume, vmin=vmin, vmax=vmax,
        layout=_layout_for(volume),
        cmap=view.cmap,
        cbar_label=session._cbar_label(view, sr),
    )


def capped_conductivity_mosaic(session_result):
    """The conductivity map as the page shows it: [0, CONDUCTIVITY_VMAX] with
    the top tick reading ">= cap". None when the cap is off or the model's
    output is not in S/m (a normalized map keeps the snapshot's scale).

    One function for the canvas AND the exported review image, so the mosaic
    a user downloads is the one they looked at.
    """
    cap = task_settings.CONDUCTIVITY_VMAX
    if cap is None:
        return None
    model = get_session().get_runner(session_result.model_key).model
    unit = getattr(model, "CONDUCTIVITY_UNIT", "S/m")
    if unit != "S/m":
        return None
    volume = _for_mosaic(session_result.conductivity)
    cap = float(cap)
    return _capped_mosaic(np.clip(volume, 0.0, cap), cap,
                          f"Conductivity ({unit})")


def _capped_mosaic(volume: np.ndarray, cap: float, cbar_label: str):
    """The snapshot's mosaic with the colour bar's top tick reading ">= cap".

    Goes through the snapshot's own figure builder rather than render_mosaic
    so the bar can be relabelled before the figure is rasterised; saved the
    same way render_mosaic saves, so the crop matches the other views.
    """
    import io
    import matplotlib.pyplot as plt
    from PIL import Image

    fig = _build_mosaic_figure(volume, vmin=0.0, vmax=cap,
                               layout=_layout_for(volume),
                               cmap=isocond_settings.CONDUCTIVITY_CMAP,
                               cbar_label=cbar_label)
    cax = fig.axes[-1]                       # the colour bar, added last
    ticks = list(cax.get_yticks())
    labels = [t.get_text() for t in cax.get_yticklabels()]
    if ticks and labels:
        labels[-1] = f"≥{cap:g}"
        cax.set_yticks(ticks)
        cax.set_yticklabels(labels)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", **_SAVE_KWARGS)
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


# ============================================================================
# The slice viewer
# ----------------------------------------------------------------------------
# One slice of four volumes in a 2 x 2 grid -- T1, T2, conductivity,
# segmentation -- that you hover over to read values (app/web/slice_viewer).
# All four share the model's 256 x 256 grid, the grid of the saved
# conductivity map, so one cursor position is the same point in each:
#
#   T1  the ORIGINAL scanner intensity of the uploaded T1 at that point
#   T2  the ORIGINAL scanner intensity of the uploaded T2, after the pipeline
#       registered it onto T1 -- i.e. at the point the model actually used
#   conductivity / segmentation  the run's own output
#
# The model grid is T1 resampled per slice (preprocess._fit_slice_isotropic:
# per-axis resize to TARGET_SPACING, then a centred pad/crop), so each model
# pixel maps back to one T1 voxel; padding outside the scan reads "—". The
# registered T2 sits on T1's grid, so the same mapping applies. It is caught
# from the pipeline's own registration call (_capture_registration) rather
# than recomputed, so it is exactly what the model saw.
# ============================================================================
VIEWER_VIEW = "Per-Slice Check on the Output"
# Classes covering fewer voxels than this are left out of the tissue table.
VIEWER_MIN_TISSUE_VOXELS = 100
# Tissue-table rows shown first, in this order (by tissue name); the rest
# follow in tissue-id order.
VIEWER_TOP_TISSUES = ("csf", "gray matter", "white matter")
# The slider starts at the middle of the slices holding this tissue
# (tissue_labels.csv name), see viewer_default_slice().
VIEWER_START_TISSUE = "white matter"
# A median or mean above this multiple of the expected value is shown in red;
# a mean below VIEWER_LOW_RATIO x expected in blue.
VIEWER_FLAG_RATIO = 2.0
VIEWER_LOW_RATIO = 0.5


@dataclass
class ViewerData:
    t1: np.ndarray                  # uploaded T1, canonical (Z, X, Y), raw
    t2: Optional[np.ndarray]        # uploaded T2 registered onto T1's grid, raw
    rows: np.ndarray                # model row -> T1 row (int), per slice
    cols: np.ndarray                # model col -> T1 col
    inside: np.ndarray              # (H, W) model pixels that come from the scan
    window1: tuple
    window2: tuple
    original: bool                  # False: T1/T2 fall back to the model inputs
    tissue_rows: List[list]
    tissue_caption: str
    unit: str
    frequency: Optional[str]


@contextmanager
def _capture_registration():
    """Keep the pipeline's own registered T2 (on T1's grid, raw intensities)
    and T1's affine while a job runs, without touching
    preprocess.make_pair_from_inputs (it looks the function up by name)."""
    original = isocond_preprocess.register_volume_t2_to_t1_3d
    box = {}

    def keep(t1_volume, t2_volume, affine1, affine2, *args, **kwargs):
        out = original(t1_volume, t2_volume, affine1, affine2, *args, **kwargs)
        box["t2_registered"] = out[0] if isinstance(out, tuple) else out
        box["affine1"] = affine1
        return out

    isocond_preprocess.register_volume_t2_to_t1_3d = keep
    try:
        yield box
    finally:
        isocond_preprocess.register_volume_t2_to_t1_3d = original


def model_grid_to_native(src: int, out: int, mm: Optional[float]) -> tuple:
    """Per axis: (native index of each model pixel, inside-the-scan mask),
    inverting preprocess._fit_slice_isotropic (or the plain resize used when
    the spacing is unknown): `src` native pixels of `mm` mm -> `out` model
    pixels. Nearest voxel. Also used by tasks/isocond_debug."""
    idx = np.arange(out, dtype=np.float64)
    if mm is None:
        native = (idx + 0.5) * src / out - 0.5
        return np.clip(np.rint(native), 0, src - 1).astype(int), np.ones(out, bool)
    new = max(1, int(round(src * mm / isocond_settings.TARGET_SPACING)))
    resized = idx - max(0, (out - new) // 2) + max(0, (new - out) // 2)
    native = (resized + 0.5) * src / new - 0.5
    return (np.clip(np.rint(native), 0, src - 1).astype(int),
            (resized >= 0) & (resized < new))


def tissue_headers(unit: str) -> List[str]:
    """Column headers for tissue_summary()'s rows."""
    return ["Tissue", "Voxels", f"Median ({unit})", f"Mean ({unit})", f"Expected ({unit})"]


def tissue_summary(conductivity, segmentation, frequency, unit,
                   scope: str = "Entire volume</b> (not only this slice)",
                   min_voxels: int = VIEWER_MIN_TISSUE_VOXELS,
                   csv_path=None, column: Optional[str] = None):
    """(rows, caption HTML) for the tissue table over `conductivity` -- the
    entire volume here; one slice on tasks/isocond_debug. Per class of
    `segmentation`: [name, voxel count, median, mean, expected] (headers:
    tissue_headers()); a median or mean above VIEWER_FLAG_RATIO x expected
    is a (text, "sv-bad") cell, drawn red; a mean below VIEWER_LOW_RATIO x
    expected a (text, "sv-low") cell, drawn blue.
    `scope` opens the caption, in bold up to its </b>. Names and expected
    values come from `csv_path` (default: the app's tissue_labels.csv),
    column `column` (default: the frequency, lower-cased, e.g. 10khz)."""
    if segmentation is None:
        return [], "No tissue table: this model gives no segmentation."
    names = load_tissue_labels(csv_path)
    csv_name = Path(csv_path).name if csv_path else "tissue_labels.csv"
    column = column or (frequency.lower() if frequency else None)
    try:
        expected = load_tissue_conductivity(column, csv_path) if column else {}
    except KeyError:
        expected = {}
    seg = np.rint(segmentation).astype(int)
    order = {name: i for i, name in enumerate(VIEWER_TOP_TISSUES)}
    ids = sorted((k for k in names if k != 0),           # 0 = background
                 key=lambda k: (order.get(names[k].strip().lower(), len(order)), k))
    rows = []
    for k in ids:
        mask = seg == k
        if mask.sum() < min_voxels:
            continue
        values = conductivity[mask]
        ref = expected.get(k)

        def cell(x, low=False):         # red above FLAG x expected; blue (mean) below LOW x
            text = f"{x:.3f}"
            if ref and x > VIEWER_FLAG_RATIO * ref:
                return text, "sv-bad"
            if ref and low and x < VIEWER_LOW_RATIO * ref:
                return text, "sv-low"
            return text

        rows.append([names[k], f"{values.size:,}", cell(float(np.median(values))),
                     cell(float(np.mean(values)), low=True),
                     f"{ref:.3f}" if ref is not None else "—"])
    freq_text = frequency_label(frequency) if frequency else "this model's frequency"
    caption = (f"<b>{scope}: median and mean predicted "
               f"conductivity of each predicted tissue vs the expected value at "
               f"{freq_text} ({csv_name}"
               + (f", column {column}); <span class=\"sv-bad\">red</span> = more than "
                  f"{VIEWER_FLAG_RATIO:g}× expected; <span class=\"sv-low\">blue</span> = "
                  f"mean less than {VIEWER_LOW_RATIO:g}× expected."
                  if expected else f" has no column for {freq_text}, so '—')."))
    return rows, caption


def frequency_label(tag: str) -> str:
    from vendor.isocond.nets.condnettart import frequency_label as _label
    return _label(tag)


def build_viewer(session_result, captured: dict, model_key: str) -> ViewerData:
    pred = session_result.prediction
    model = get_session().get_runner(model_key).model
    unit = getattr(model, "CONDUCTIVITY_UNIT", "S/m")
    frequency = getattr(model, "FREQUENCY", None)
    cond = np.asarray(pred["conductivity"])
    seg = pred.get("segmentation")
    rows_t, caption = tissue_summary(cond, seg, frequency, unit)

    t1 = np.asarray(pred["t1_loaded"])
    t2 = captured.get("t2_registered")
    affine1 = captured.get("affine1")
    out_h, out_w = cond.shape[-2:]
    original = (t2 is not None and t1.ndim == 3 and t1.shape[0] == cond.shape[0]
                and np.shape(t2) == t1.shape)
    if original:
        spacing = isocond_preprocess.in_plane_spacing_from_affine(affine1)
        r_mm, c_mm = spacing if spacing else (None, None)
        rows, r_in = model_grid_to_native(t1.shape[1], out_h, r_mm)
        cols, c_in = model_grid_to_native(t1.shape[2], out_w, c_mm)
        inside = np.outer(r_in, c_in)
        t2 = np.asarray(t2, dtype=np.float32)
        win1 = tuple(map(float, InferenceRunner._display_window(t1)))
        win2 = tuple(map(float, InferenceRunner._display_window(t2)))
    else:                       # 2D input or no affine: show the model inputs
        rows = np.arange(out_h); cols = np.arange(out_w)
        inside = np.ones((out_h, out_w), bool)
        t1, t2 = np.asarray(pred["t1_input"]), np.asarray(pred["t2_input"])
        win1 = win2 = (0.0, 1.0)
    return ViewerData(t1, t2, rows, cols, inside, win1, win2, bool(original),
                      rows_t, caption, unit, frequency)


def viewer_depth(result: Optional[JobResult]) -> int:
    if result is None or result.session_result is None:
        return 0
    cond = np.asarray(result.session_result.conductivity)
    return cond.shape[0] if cond.ndim == 3 else 1


def viewer_default_slice(result: Optional[JobResult]) -> int:
    """Where the slider starts after a run: the middle of the slices that hold
    VIEWER_START_TISSUE in the segmentation (first to last), or the middle
    slice of the volume when there is no segmentation or no such tissue."""
    cond = np.asarray(result.session_result.conductivity)
    if cond.ndim != 3:
        return 0
    seg = result.session_result.prediction.get("segmentation")
    ids = [k for k, name in load_tissue_labels().items()
           if name.strip().lower() == VIEWER_START_TISSUE]
    if seg is not None and np.ndim(seg) == 3 and ids:
        present = np.flatnonzero(np.isin(np.rint(seg).astype(int), ids).any(axis=(1, 2)))
        if present.size:
            return int((present[0] + present[-1]) // 2)
    return cond.shape[0] // 2


def _on_model_grid(volume: np.ndarray, v: ViewerData, k: int) -> np.ndarray:
    sl = volume[k][np.ix_(v.rows, v.cols)].astype(np.float32)
    sl[~v.inside] = np.nan
    return sl


def render_viewer(result: Optional[JobResult], k) -> str:
    """The 2 x 2 viewer for slice k, as HTML for a gr.HTML with
    slice_viewer.HOVER_JS."""
    if result is None or result.session_result is None:
        return sv.placeholder_html("Run the model first: the per-slice check shows "
                                   "its result.")
    v = result.viewer
    if v is None:
        return sv.placeholder_html("The per-slice check is not available for this run "
                                   "(see the console).")
    depth = viewer_depth(result)
    k = int(np.clip(int(k), 0, depth - 1))
    pred = result.session_result.prediction
    cond = np.asarray(pred["conductivity"])
    cond_k = cond[k] if cond.ndim == 3 else cond
    seg = pred.get("segmentation")

    from matplotlib import colormaps
    gray = colormaps[isocond_settings.ANATOMY_CMAP]
    if v.original:
        t1_k, t2_k = _on_model_grid(v.t1, v, k), _on_model_grid(v.t2, v, k)
        what1, what2, digits = "original intensity", "original intensity, registered onto T1", 1
    else:
        t1_k, t2_k = v.t1[k], v.t2[k]
        what1 = what2 = "model input (normalized)"; digits = 3
    n1, n2 = Normalize(*v.window1), Normalize(*v.window2)

    if v.unit == "S/m" and task_settings.CONDUCTIVITY_VMAX:
        cap = float(task_settings.CONDUCTIVITY_VMAX); top = f"≥{cap:g}"
    else:
        cap = float(np.nanmax(cond)) or 1.0; top = None
    nc = Normalize(0.0, cap)
    jet = colormaps[isocond_settings.CONDUCTIVITY_CMAP]
    side = cond_k.shape[1]
    hover_unit = v.unit if v.unit == "S/m" else ""     # "normalized, ..." is the bar's job

    panels = [
        sv.panel_html(t1_k, n1, gray, sv.channel(t1_k, digits), "T1",
                      footer=sv.bar_html(n1, gray, f"T1 {what1}", side)),
        sv.panel_html(t2_k, n2, gray, sv.channel(t2_k, digits), "T2",
                      footer=sv.bar_html(n2, gray, f"T2 {what2}", side)),
    ]
    if seg is not None:
        seg_k = np.asarray(seg[k] if np.ndim(seg) == 3 else seg)
        num = int(getattr(get_session().get_runner(result.model_key).model,
                          "NUM_SEG_CLASSES", 15) or 15)
        panels.append(sv.panel_html(cond_k, nc, jet, sv.channel(cond_k, 3, hover_unit), "Cond",
                                    footer=sv.bar_html(nc, jet, f"Conductivity ({v.unit})",
                                                       side, top)))
        panels.append(sv.panel_html(seg_k, Normalize(*label_display_range(num)),
                                    discrete_label_cmap(num),
                                    sv.channel(seg_k, names=load_tissue_labels()), "Seg",
                                    footer='<div class="sv-bar">Tissue: hover for its name</div>'))
    else:                       # conductivity spans both cells, drawn at 2x
        panels.append(sv.panel_html(cond_k, nc, jet, sv.channel(cond_k, 3, hover_unit), "Cond",
                                    zoom=2, wide=True,
                                    footer=sv.bar_html(nc, jet, f"Conductivity ({v.unit})",
                                                       2 * side, top)))
    table = (sv.table_html(tissue_headers(v.unit),
                           v.tissue_rows, v.tissue_caption)
             if v.tissue_rows else f'<div class="sv-note">{v.tissue_caption}</div>')
    note = (f"Slice {k} of 0–{depth - 1} · {result.model_name}. Hover over any image: "
            f"every panel shows its value at that point.")
    return sv.canvas_html(panels, below=table, note=note)


_restore_uploads()
