# ============================================================================
# tasks/isocond_debug/service.py — check a conductivity result, slice by slice
# ----------------------------------------------------------------------------
# What the isocond debug page does, minus the widgets (web.py). Unlisted page
# for checking a result that ALREADY EXISTS -- typically task 1's (isocond)
# conductivity_map.nii.gz and segmentation.nii.gz -- against the scans it
# came from. Nothing is estimated here.
#
#   prepare()      T1 (+ optional T2, Cond, optional Seg + its tissue CSV):
#                  T2 registered onto T1 with task 1's own 3D registration;
#                  Cond/Seg placed on their grid (place_on_grid); model and
#                  frequency read from the Cond (or Seg) file name
#                  (parse_file_name); the CSV gives the tissue names and the
#                  expected conductivity column (read_tissue_csv).
#   render_slice() one slice of every volume in a 2 x 2 (or 4 x 1) grid that
#                  you hover over to read values (app/web/slice_viewer), and
#                  a per-tissue table FOR THAT SLICE below it.
#   inspect_architecture()  the checkpoints of the model the file name
#                  names, if they are in vendor/isocond/models: module
#                  printouts and one diagram of the two stages.
#
# Where Cond/Seg sit (the first rule that fits, else an error):
#   model grid   (T1 slices, 256, 256) as stored: task 1's output. Its grid
#                is T1 resampled per slice to 1 mm and centre-fitted to 256
#                (vendor preprocess._fit_slice_isotropic); T1/T2 are mapped
#                onto it, nearest voxel, exactly as task 1's "Per-Slice
#                Check on the Output" does (NaN = padding outside the scan).
#   T1 grid      the same shape as T1 (as stored, or after the canonical
#                reorientation T1 itself gets): everything shown on T1's own
#                voxels.
# ============================================================================

# --- repo root on sys.path, as tasks/isocond/service.py does ----------------
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import csv
import html
import io
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
from matplotlib import colormaps
from matplotlib.colors import Normalize

from app import mosaic as mosaic_rule
from app.web import slice_viewer as sv
from tasks.isocond.service import model_grid_to_native, tissue_headers, tissue_summary
from tasks.isocond_debug import settings as task_settings
from vendor.isocond import settings as isocond_settings
from vendor.isocond.nets import condnettart
from vendor.isocond.nets import registry as isocond_registry
from vendor.isocond.nets.registry import build_model, get_model_class
from vendor.isocond.pipeline.imaging import (discrete_label_cmap,
                                             label_display_range,
                                             load_image_as_array)
from vendor.isocond.pipeline.preprocess import (in_plane_spacing_from_affine,
                                                register_volume_t2_to_t1_3d)
from vendor.isocond.pipeline.runner import InferenceRunner
from vendor.isocond.pipeline.session import _case_id
from vendor.isocond.pipeline.tissues import (load_tissue_conductivity,
                                             load_tissue_labels)

logger = logging.getLogger(__name__)

TASK_SLUG = "isocond_debug"

Reporter = Optional[Callable[[str], None]]

NUM_CLASSES = condnettart.NUM_SEG_CLASSES

# Checkpoint role (the model's CHECKPOINTS key, which is also the .pth file's
# name) -> the attribute of the loaded CondNetTart holding that network, and
# what it is. NOTE the off-by-one the checkpoint files carry: stage0.pth is
# the paper's Stage 1, stage1.pth its Stage 2 (nets/condnettart.py, "stage
# numbering is 0-based").
STAGES = {
    "stage0": ("stage1_model", "Stage 1 · UNet_2D",
               "T1 + T2 → decoder features + 15-class segmentation"),
    "stage1": ("stage2_model", "Stage 2 · CondNet_Transfer",
               "Stage 1 features → normalized conductivity L^v"),
}


def _say(report: Reporter, message: str) -> None:
    logger.info(message)
    if report:
        report(message)


# ============================================================================
# What a file name says: model and frequency
# ----------------------------------------------------------------------------
# Task 1 names its outputs "<case>_<model key>_conductivity_map.nii.gz", e.g.
# IXI258_condnettart_10kHz_conductivity_map.nii.gz or IXI258_condnet_...; any
# name with a frequency in it ("..._100MHz_...") works too.
# ============================================================================
_FREQ_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*(ghz|mhz|khz|hz)(?![A-Za-z])",
                      re.IGNORECASE)
_FREQ_UNITS = {"hz": "Hz", "khz": "kHz", "mhz": "MHz", "ghz": "GHz"}


@dataclass
class FileInfo:
    model: Optional[str]          # "CondNet-TART", "CondNet", or None
    frequency: Optional[str]      # e.g. "10kHz" (the tissue table's column, lower-cased)
    unit: str                     # "S/m", or "normalized" for CondNet
    model_key: Optional[str]      # a vendor/isocond model with checkpoints, if any

    def describe(self) -> str:
        freq = condnettart.frequency_label(self.frequency) if self.frequency else "frequency unknown"
        return f"{self.model or 'model unknown'} · {freq} · {self.unit}"


def parse_file_name(*paths) -> FileInfo:
    """Model and frequency from the first of `paths` that names them."""
    model = frequency = None
    for path in paths:
        if not path:
            continue
        name = Path(path).name
        low = name.lower()
        if model is None:
            if "condnettart" in low or "condnet-tart" in low or "condnet_tart" in low:
                model = condnettart.MODEL_NAME
            elif "condnet" in low:
                model = "CondNet"
        if frequency is None:
            m = _FREQ_RE.search(name)
            if m:
                frequency = m.group(1) + _FREQ_UNITS[m.group(2).lower()]
    if model == "CondNet" and frequency is None:
        frequency = "900MHz"                # CondNet's only frequency
    unit = "normalized" if model == "CondNet" else "S/m"
    key = None
    if model == condnettart.MODEL_NAME and frequency:
        isocond_registry.refresh_models()
        candidate = f"{condnettart.MODEL_FAMILY}_{frequency}"
        key = candidate if candidate in isocond_registry.MODEL_REGISTRY else None
    return FileInfo(model, frequency, unit, key)


# ============================================================================
# The tissue CSV that goes with a Seg
# ----------------------------------------------------------------------------
# Same format as vendor/isocond's tissue_labels.csv:
#     tissue_id,tissue_name,10khz,1hz
#     0,background,0,0
#     8,gray matter,0.1,0.2
# One conductivity column per frequency (S/m); the one used is named after
# the frequency in the Cond file name (lower-case, e.g. 10khz), or the only
# conductivity column there is.
# ============================================================================
@dataclass
class TissueTable:
    path: str
    names: Dict[int, str]           # tissue_id -> tissue_name
    column: Optional[str]           # conductivity column used, or None
    note: str                       # how the column was chosen


def read_tissue_csv(path, frequency: Optional[str]) -> TissueTable:
    names = load_tissue_labels(str(path))
    if not names:
        raise ValueError(f"{Path(path).name}: no rows with a numeric tissue_id "
                         f"(expected columns tissue_id, tissue_name, then one "
                         f"conductivity column per frequency, e.g. 10khz)")
    with open(path, encoding="utf-8", newline="") as fh:
        header = [c.strip().lower() for c in next(csv.reader(fh), [])]
    if header[:2] != ["tissue_id", "tissue_name"]:
        raise ValueError(f"{Path(path).name}: the header must start with "
                         f"tissue_id,tissue_name (got {','.join(header[:2])})")
    columns = header[2:]
    want = frequency.lower() if frequency else None
    if want and want in columns:
        column, note = want, f"column {want} (the frequency in the Cond file name)"
    elif len(columns) == 1:
        column = columns[0]
        note = (f"column {column} (its only conductivity column"
                + (f"; there is no {want} column)" if want else ")"))
    else:
        column = None
        note = (f"none, shown as '—' (no {want or 'frequency'} column among "
                f"{', '.join(columns) or 'none'})")
    if column:
        values = load_tissue_conductivity(column, str(path))
        if not values:
            raise ValueError(f"{Path(path).name}: column {column} has no numbers")
    return TissueTable(str(path), names, column, note)


# ============================================================================
# A loaded case
# ============================================================================
@dataclass
class Case:
    case_id: str
    info: FileInfo
    grid: str                       # "model grid" or "T1 grid"
    t1: np.ndarray                  # canonical (Z, X, Y), raw
    t2: Optional[np.ndarray]        # registered onto T1's grid, raw
    cond: np.ndarray                # (Z, H, W) on the display grid
    seg: Optional[np.ndarray]       # (Z, H, W) class ids, or None
    tissues: Optional[TissueTable]  # with Seg: names + expected values
    rows: np.ndarray                # display row -> T1 row
    cols: np.ndarray                # display col -> T1 col
    inside: np.ndarray              # (H, W) display pixels that come from the scan
    window1: Tuple[float, float]
    window2: Tuple[float, float]

    @property
    def depth(self) -> int:
        return int(self.cond.shape[0])

    def default_slice(self) -> int:
        """The middle of the slices that carry a conductivity."""
        lo, hi = mosaic_rule.occupied_span(self.cond, axis=0)
        return (lo + hi) // 2

    def on_grid(self, volume: np.ndarray, k: int) -> np.ndarray:
        """Slice k of a T1-grid volume on the display grid, NaN outside."""
        sl = volume[k][np.ix_(self.rows, self.cols)].astype(np.float32)
        sl[~self.inside] = np.nan
        return sl


def _load_stored(path) -> Tuple[np.ndarray, np.ndarray]:
    """(array as stored, array as T1 is loaded: canonical (Z, X, Y))."""
    stored = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    canonical, _affine = load_image_as_array(str(path))
    return stored, np.asarray(canonical, dtype=np.float32)


def place_on_grid(path, label: str, t1_shape: tuple) -> Tuple[str, np.ndarray]:
    """(grid, volume) for a Cond/Seg file -- see the module comment."""
    stored, canonical = _load_stored(path)
    size = tuple(isocond_settings.TARGET_SIZE[::-1])          # (H, W)
    if stored.ndim == 3 and stored.shape[0] == t1_shape[0] and stored.shape[1:] == size:
        return "model grid", stored
    # Reoriented exactly as T1 is first: right whenever the file carries T1's
    # header. As stored only as the fallback (a header-less array in T1's
    # canonical layout).
    for arr in (canonical, stored):
        if arr.shape == tuple(t1_shape):
            return "T1 grid", arr
    raise ValueError(
        f"{label} {Path(path).name} has shape {stored.shape}; expected task 1's "
        f"model grid ({t1_shape[0]}, {size[0]}, {size[1]}) -- T1's {t1_shape[0]} "
        f"slices at 256 x 256 -- or T1's own shape {tuple(t1_shape)}. Is it from "
        f"this T1?")


def prepare(t1_path, t2_path, cond_path, seg_path, csv_path,
            report: Reporter = None) -> Case:
    """Load, register T2 onto T1 (if given), place Cond/Seg; no estimation."""
    for label, path in (("T1", t1_path), ("Cond", cond_path)):
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f"No {label} file")
    if seg_path and not csv_path:
        raise ValueError("A Seg needs its tissue CSV (tissue_id, tissue_name and "
                         "one conductivity column per frequency): upload it too")
    info = parse_file_name(cond_path, seg_path)
    _say(report, f"From the file name: {info.describe()}"
                 + (f" (checkpoints: {info.model_key})" if info.model_key else ""))
    tissues = None
    if seg_path:
        tissues = read_tissue_csv(csv_path, info.frequency)
        _say(report, f"Tissue CSV {Path(csv_path).name}: {len(tissues.names)} "
                     f"tissues; expected values: {tissues.note}")
    elif csv_path:
        _say(report, "Tissue CSV ignored: it is used only with a Seg")

    _say(report, f"Loading T1: {Path(t1_path).name}")
    t1, affine1 = load_image_as_array(str(t1_path))
    t1 = np.asarray(t1, dtype=np.float32)
    if t1.ndim != 3:
        raise ValueError(f"T1 must be a volume, got shape {t1.shape}")

    _say(report, f"Loading Cond: {Path(cond_path).name}")
    grid, cond = place_on_grid(cond_path, "Cond", t1.shape)
    seg = None
    if seg_path:
        _say(report, f"Loading Seg: {Path(seg_path).name}")
        seg_grid, seg = place_on_grid(seg_path, "Seg", t1.shape)
        if seg_grid != grid:
            raise ValueError(f"Cond is on the {grid}, Seg on the {seg_grid}: "
                             f"they must share one grid")
        seg = np.rint(seg).astype(np.int16)
        unknown = sorted(set(np.unique(seg).tolist()) - set(tissues.names))
        if unknown:
            _say(report, f"WARNING: Seg labels not in the CSV: {unknown} "
                         f"(shown as 'class <id>')")
    _say(report, f"Cond{' and Seg' if seg is not None else ''} on the {grid}: "
                 f"{cond.shape}; T1 {t1.shape}")

    t2 = None
    if t2_path:
        _say(report, f"Loading T2: {Path(t2_path).name}")
        t2_raw, affine2 = load_image_as_array(str(t2_path))
        if affine1 is None or affine2 is None:
            raise ValueError("T2 registration needs NIfTI headers on both T1 and T2")
        _say(report, f"T2 {np.shape(t2_raw)} — registering onto T1 "
                     f"(task 1's 3D registration, ~15 s)...")
        t0 = time.time()
        t2 = np.asarray(register_volume_t2_to_t1_3d(t1, t2_raw, affine1, affine2),
                        dtype=np.float32)
        _say(report, f"Registered in {time.time() - t0:.0f} s")

    if grid == "model grid":
        spacing = in_plane_spacing_from_affine(affine1)
        r_mm, c_mm = spacing if spacing else (None, None)
        rows, r_in = model_grid_to_native(t1.shape[1], cond.shape[1], r_mm)
        cols, c_in = model_grid_to_native(t1.shape[2], cond.shape[2], c_mm)
        inside = np.outer(r_in, c_in)
    else:
        rows, cols = np.arange(t1.shape[1]), np.arange(t1.shape[2])
        inside = np.ones(t1.shape[1:], bool)

    window = lambda v: tuple(map(float, InferenceRunner._display_window(v)))
    case = Case(case_id=_case_id(str(t1_path)), info=info, grid=grid, t1=t1, t2=t2,
                cond=cond, seg=seg, tissues=tissues, rows=rows, cols=cols, inside=inside,
                window1=window(t1), window2=window(t2) if t2 is not None else (0.0, 1.0))
    _say(report, f"Ready: {case.depth} slices. Conductivity "
                 f"{np.nanmin(cond):.3f}..{np.nanmax(cond):.3f} {info.unit}.")
    return case


# ============================================================================
# One slice: the hover grid and its per-tissue table
# ============================================================================
def placeholder_html(message: str) -> str:
    return sv.placeholder_html(message)


def _slice_table(case: Case, k: int, cond_k: np.ndarray, t1_k, t2_k) -> str:
    """Per tissue on slice k (with Seg); else the value range of each panel
    over the slice's in-head pixels: σ above HEAD_FRACTION of the map's
    maximum (a normalized map is not exactly 0 in air)."""
    if case.seg is not None:
        rows, caption = tissue_summary(
            cond_k, case.seg[k], case.info.frequency, case.info.unit,
            scope=f"Slice {k}</b> (this slice only)",
            min_voxels=task_settings.MIN_TISSUE_PIXELS,
            csv_path=case.tissues.path, column=case.tissues.column)
        if not rows:
            return f'<div class="sv-note">No tissue covers {task_settings.MIN_TISSUE_PIXELS}+ pixels on slice {k}.</div>'
        unit = case.info.unit
        return sv.table_html(tissue_headers(unit),
                             rows, caption)
    level = task_settings.HEAD_FRACTION * float(np.nanmax(case.cond))
    head = cond_k > level
    out = []
    for name, values, digits in (("T1", t1_k, 1), ("T2", t2_k, 1), ("Cond", cond_k, 3)):
        if values is None:
            continue
        v = values[head & np.isfinite(values)]
        out.append([name] + ([f"{np.min(v):.{digits}f}", f"{np.median(v):.{digits}f}",
                              f"{np.max(v):.{digits}f}"] if v.size else ["—"] * 3))
    return sv.table_html(["", "Min", "Median", "Max"], out,
                         f"<b>Slice {k}</b> (this slice only), pixels with σ &gt; "
                         f"{level:.3g} ({task_settings.HEAD_FRACTION:.0%} of the map's maximum). "
                         f"No Seg given, so no per-tissue table.")


def render_slice(case: Optional[Case], k) -> str:
    if case is None:
        return placeholder_html("Upload T1 and Cond (T2 and Seg are optional), "
                                "then press Run.")
    k = int(np.clip(int(k), 0, case.depth - 1))
    gray = colormaps[task_settings.ANATOMY_CMAP]
    jet = colormaps[task_settings.CONDUCTIVITY_CMAP]
    cond_k = case.cond[k]
    side = cond_k.shape[1]
    t1_k = case.on_grid(case.t1, k)
    t2_k = case.on_grid(case.t2, k) if case.t2 is not None else None

    if case.info.unit == "S/m":
        cap = float(task_settings.CONDUCTIVITY_VMAX); top = f"≥{cap:g}"
    else:
        cap = float(np.nanmax(case.cond)) or 1.0; top = None
    nc = Normalize(0.0, cap)
    hover_unit = "S/m" if case.info.unit == "S/m" else ""
    cond_label = f"Conductivity ({case.info.unit})"

    n1 = Normalize(*case.window1)
    panels = [sv.panel_html(t1_k, n1, gray, sv.channel(t1_k, 1), "T1",
                            footer=sv.bar_html(n1, gray, "T1 original intensity", side))]
    if t2_k is not None:
        n2 = Normalize(*case.window2)
        panels.append(sv.panel_html(t2_k, n2, gray, sv.channel(t2_k, 1), "T2",
                                    footer=sv.bar_html(n2, gray, "T2 original intensity, "
                                                       "registered onto T1", side)))
    # Cond spans two cells (at 2x) only when it would otherwise leave the
    # 2 x 2 grid's last cell empty: T1, T2 and Cond, no Seg.
    wide = t2_k is not None and case.seg is None
    zoom = 2 if wide else 1
    panels.append(sv.panel_html(cond_k, nc, jet, sv.channel(cond_k, 3, hover_unit), "Cond",
                                zoom=zoom, wide=wide,
                                footer=sv.bar_html(nc, jet, cond_label, zoom * side, top)))
    if case.seg is not None:
        seg_k = case.seg[k]
        num = max(NUM_CLASSES, max(case.tissues.names) + 1, int(case.seg.max()) + 1)
        panels.append(sv.panel_html(seg_k, Normalize(*label_display_range(num)),
                                    discrete_label_cmap(num),
                                    sv.channel(seg_k, names=case.tissues.names), "Seg",
                                    footer='<div class="sv-bar">Tissue: hover for its name</div>'))
    where = ("task 1's model grid (1 mm, 256 × 256); T1/T2 show the nearest original voxel, "
             "— = padding outside the scan" if case.grid == "model grid"
             else "T1's own voxels")
    note = (f"Slice {k} of 0–{case.depth - 1} · {case.info.describe()} · on {where}. "
            f"Hover over any image: every panel shows its value at that point.")
    return sv.canvas_html(panels, below=_slice_table(case, k, cond_k, t1_k, t2_k),
                          note=note)


# ============================================================================
# The CondNet-TART checkpoints, for the architecture panel
# ----------------------------------------------------------------------------
# Loaded only to be inspected (never run on the case). Each is loaded once
# and kept until its checkpoint files change.
# ============================================================================
_model_lock = threading.Lock()
_models: Dict[str, tuple] = {}          # key -> (checkpoint signature, model)


def default_model_key() -> Optional[str]:
    isocond_registry.refresh_models()
    keys = [key for key in isocond_registry.MODEL_REGISTRY
            if isocond_registry.model_family(key) == condnettart.MODEL_FAMILY]
    if task_settings.DEFAULT_MODEL_KEY in keys:
        return task_settings.DEFAULT_MODEL_KEY
    return keys[0] if keys else None


def _signature(model_class) -> tuple:
    sig = []
    for role, path in sorted(model_class.CHECKPOINTS.items()):
        st = Path(path).resolve().stat()
        sig.append((role, str(Path(path).resolve()), st.st_mtime, st.st_size))
    return tuple(sig)


def get_model(key: str):
    model_class = get_model_class(key)
    sig = _signature(model_class)
    with _model_lock:
        cached = _models.get(key)
        if cached is None or cached[0] != sig:
            _models[key] = (sig, build_model(key, device=task_settings.DEVICE))
        return _models[key][1]


def checkpoint_choices() -> List[Tuple[str, str]]:
    """(label, role) of the two checkpoint files of every CondNet-TART."""
    return [(f"{role}.pth — {title}", role)
            for role, (_attr, title, _what) in STAGES.items()]


# ============================================================================
# The architecture panel: module printouts + one diagram of both stages
# ----------------------------------------------------------------------------
# The diagram is drawn from a TRACE, not by hand: one zero slice is pushed
# through the network with a hook on every top-level layer, so the layers,
# their order and their output shapes are the network's own. Each layer is a
# row, indented by its spatial resolution (256 -> 128 -> ... -> 16 px), so a
# U-Net reads as a "U" turned on its side; Stage 2 follows Stage 1 in the
# same column. What a trace cannot see -- the skip connections, and which
# Stage 1 output feeds which Stage 2 layer -- is read off the two forward()
# methods in vendor/isocond/nets/condnettart.py, below.
# ============================================================================
_inspect_cache: Dict[tuple, dict] = {}
_inspect_lock = threading.Lock()

# (encoder layer, decoder layer it is concatenated into), per checkpoint.
SKIPS = {
    "stage0": [("TCB4", "TCB6"), ("TCB3", "TCB7"), ("TCB2", "TCB8"), ("TCB1", "TCB9")],
    "stage1": [],
}
# Stage 2 layer -> the Stage 1 feature it takes, and the Stage 1 layer whose
# output that feature is (UNet_2D.forward: t = the Transformer stack's output
# reshaped, t6..t9 = TCB6..TCB9's).
STAGE1_FEATURES = {"UCRCM1": "t", "att1": "t6", "att2": "t7", "att3": "t8", "att4": "t9"}
STAGE1_FEATURE_SOURCES = {"t": "Trans", "t6": "TCB6", "t7": "TCB7", "t8": "TCB8", "t9": "TCB9"}

DIAGRAM_ROW, DIAGRAM_HEADER, DIAGRAM_BOX_H, DIAGRAM_BOX_W = 28, 36, 22, 300
DIAGRAM_INDENT, DIAGRAM_LEFT = 28, 64
# Approximate glyph widths (px) at the diagram's 11 px type, for fitting the
# layer type between the name and the shape.
_CHAR_PX, _MONO_PX, _KIND_X = 6.0, 6.8, 78
DIAGRAM_FILLS = ("#e8f1fd", "#d6e6fb", "#c2d9f8", "#adcbf5", "#97bdf2", "#82aeef")


@dataclass
class _Layer:
    name: str
    kind: str
    shape: tuple          # output shape, batch first
    params: int


def _n_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def _trace(model, role: str) -> Tuple[List[_Layer], List[str]]:
    """(layers in the order forward() calls them, names never called). A
    layer called twice (maxpool) appears twice; consecutive blocks of one
    ModuleList (the 6 Transformer blocks) are merged into one row."""
    attr = STAGES[role][0]
    net = getattr(model, attr)
    targets = []
    for name, child in net.named_children():
        if isinstance(child, torch.nn.ModuleList):
            targets += [(f"{name}.{i}", m) for i, m in enumerate(child)]
        else:
            targets.append((name, child))

    calls, handles = [], []
    for name, mod in targets:
        def hook(m, _inputs, output, name=name):
            if isinstance(output, torch.Tensor):
                calls.append(_Layer(name, type(m).__name__, tuple(output.shape),
                                    _n_params(m)))
        handles.append(mod.register_forward_hook(hook))
    x = torch.zeros(task_settings.TRACE_INPUT_SHAPE, device=model.device)
    try:
        with torch.no_grad():
            if attr == "stage1_model":
                model.stage1_model(x, x)
            else:
                # Only Stage 2's layers carry hooks, so only they are recorded.
                model.stage2_model(model.stage1_model(x, x))
    finally:
        for h in handles:
            h.remove()

    merged: List[_Layer] = []
    for layer in calls:
        prev = merged[-1] if merged else None
        base = layer.name.split(".")[0]
        if prev and "." in layer.name and prev.name.split(".")[0] == base \
                and prev.kind.split(" ×")[0] == layer.kind:
            n = int(prev.kind.split(" ×")[1]) + 1 if " ×" in prev.kind else 2
            first = prev.name.split("–")[0]
            merged[-1] = _Layer(f"{first}–{layer.name.split('.')[1]}",
                                f"{layer.kind} ×{n}", layer.shape,
                                prev.params + layer.params)
        else:
            merged.append(layer)
    called = {l.name.split(".")[0] for l in calls}
    unused = [f"{n} ({type(m).__name__})" for n, m in targets
              if n.split(".")[0] not in called]
    return merged, unused


def _level_and_label(shape: tuple, prev_level: int) -> Tuple[int, str]:
    """Indent level from the spatial size (256 px = 0, each halving +1), and
    the shape as printed; a token sequence keeps the previous level."""
    side = task_settings.TRACE_INPUT_SHAPE[-1]
    if len(shape) == 4:
        level = int(round(np.log2(side / shape[2])))
        return level, f"{shape[1]}×{shape[2]}×{shape[3]}"
    if len(shape) == 3:
        return prev_level, f"{shape[1]} tokens × {shape[2]}"
    return prev_level, "×".join(map(str, shape[1:]))


def _diagram_svg(traces: Dict[str, Tuple[List[_Layer], List[str]]],
                 params: Dict[str, int], checkpoints: Dict[str, str]) -> str:
    """Both stages as ONE diagram: Stage 1's rows, then Stage 2's, each under
    a shaded header; flow arrows within a stage, skip connections as arcs on
    the left, and Stage 1's features (t, t6-t9) as arcs on the right into the
    Stage 2 layers that take them."""
    esc = html.escape
    rows: List[dict] = []

    def header(role):
        _attr, title, what = STAGES[role]
        rows.append(dict(kind="header", stage=role, title=(
            f"{title} — {Path(checkpoints[role]).name} · "
            f"{params[role]:,} parameters"), sub=what))

    def io(role, name, text, shape, level):
        rows.append(dict(kind="io", stage=role, name=name, type=text, shape=shape,
                         level=level, tip=""))

    def layer_rows(role, level):
        for layer in traces[role][0]:
            level, label = _level_and_label(layer.shape, level)
            rows.append(dict(kind="layer", stage=role, name=layer.name,
                             type=layer.kind, shape=label, level=level,
                             tip=f"{layer.name} · {layer.kind} · output {label} · "
                                 f"{layer.params:,} parameters"))
        return level

    header("stage0")
    io("stage0", "input", "T1 + T2", "2×256×256", 0)
    layer_rows("stage0", 0)
    io("stage0", "output", "segmentation, 15 classes", "", 0)
    header("stage1")
    layer_rows("stage1", 4)
    io("stage1", "output", "L^v → σ = L^v × 2.0/0.9", "1×256×256", 0)

    # vertical positions
    top, y = 46, 46
    for r in rows:
        r["y"] = y + (8 if r["kind"] == "header" and y > top else 0)
        y = r["y"] + (DIAGRAM_HEADER if r["kind"] == "header" else DIAGRAM_ROW)
    unused = [u for role in traces for u in traces[role][1]]
    height = y + (26 if unused else 8)

    max_level = max(r["level"] for r in rows if r["kind"] != "header")
    box_right = DIAGRAM_LEFT + max_level * DIAGRAM_INDENT + DIAGRAM_BOX_W
    features = list(STAGE1_FEATURES.items())          # (target, feature), in order
    width = box_right + 26 + 13 * len(features) + 26

    def bx(r):
        return DIAGRAM_LEFT + r["level"] * DIAGRAM_INDENT

    def mid(r):
        return r["y"] + DIAGRAM_BOX_H / 2

    parts = [
        f'<text x="8" y="16" class="arch-title">CondNet-TART — both checkpoints, '
        f'{sum(params.values()):,} parameters</text>',
        f'<text x="8" y="31" class="arch-sub">Rows in forward() order, indented by '
        f'resolution (256 → 16 px); hover a box for details.</text>',
    ]
    # stage bands with their headers
    for role in STAGES:
        stage_rows = [r for r in rows if r["stage"] == role]
        h0, last = stage_rows[0], stage_rows[-1]
        parts.append(f'<rect x="2" y="{h0["y"]}" width="{width - 4}" '
                     f'height="{last["y"] + DIAGRAM_BOX_H + 8 - h0["y"]}" rx="6" '
                     f'class="arch-band"/>')
        parts.append(f'<text x="10" y="{h0["y"] + 15}" class="arch-stage">'
                     f'{esc(h0["title"])}</text>'
                     f'<text x="10" y="{h0["y"] + 28}" class="arch-sub">'
                     f'{esc(h0["sub"])}</text>')
    boxes = [r for r in rows if r["kind"] != "header"]
    # flow arrows: consecutive boxes of the same stage
    for a, b in zip(boxes, boxes[1:]):
        if a["stage"] != b["stage"]:
            continue
        parts.append(f'<line x1="{bx(a) + DIAGRAM_BOX_W / 2}" y1="{a["y"] + DIAGRAM_BOX_H}" '
                     f'x2="{bx(b) + DIAGRAM_BOX_W / 2}" y2="{b["y"] - 1}" '
                     f'class="arch-flow" marker-end="url(#arch-arrow)"/>')

    def find(stage, name):
        return next((r for r in boxes if r["stage"] == stage
                     and r["name"].split(".")[0] == name.split(".")[0]), None)

    # skip connections (left)
    for n, (src, dst) in enumerate(SKIPS["stage0"]):
        a, b = find("stage0", src), find("stage0", dst)
        if a and b:
            bulge = 22 + 10 * n
            parts.append(f'<path d="M {bx(a)} {mid(a)} C {bx(a) - bulge} {mid(a)}, '
                         f'{bx(b) - bulge} {mid(b)}, {bx(b)} {mid(b)}" class="arch-skip" '
                         f'marker-end="url(#arch-arrow-skip)"/>')
    first_skip = find("stage0", SKIPS["stage0"][-1][0]) if SKIPS["stage0"] else None
    if first_skip:
        parts.append(f'<text x="8" y="{first_skip["y"] + 15}" class="arch-skip-label">skip</text>')
    # Stage 1 features into Stage 2 (right). The first is the outermost arc,
    # so the nested arcs never cross.
    for n, (target, feature) in enumerate(features):
        a = find("stage0", STAGE1_FEATURE_SOURCES[feature])
        b = find("stage1", target)
        if not (a and b):
            continue
        X = box_right + 22 + 13 * (len(features) - 1 - n)
        xa, xb = bx(a) + DIAGRAM_BOX_W, bx(b) + DIAGRAM_BOX_W
        parts.append(f'<path d="M {xa} {mid(a)} H {X - 6} Q {X} {mid(a)}, {X} {mid(a) + 6} '
                     f'V {mid(b) - 6} Q {X} {mid(b)}, {X - 6} {mid(b)} H {xb + 1}" '
                     f'class="arch-feat" marker-end="url(#arch-arrow-feat)"/>')
        parts.append(f'<text x="{xa + 5}" y="{mid(a) - 3}" class="arch-feat-label">'
                     f'{feature}</text>')
    # boxes on top
    for r in boxes:
        x, y = bx(r), r["y"]
        cls = "arch-box arch-io" if r["kind"] == "io" else "arch-box"
        fill = ("" if r["kind"] == "io"
                else f' style="fill:{DIAGRAM_FILLS[min(r["level"], 5)]}"')
        tip = f'<title>{esc(r["tip"])}</title>' if r["tip"] else ""
        # The type gets what is left between name and shape; a longer one is
        # cut with an ellipsis (the tooltip has it in full).
        room = int((DIAGRAM_BOX_W - _KIND_X - 14 - len(r["shape"]) * _MONO_PX) / _CHAR_PX)
        kind = r["type"] if len(r["type"]) <= room else r["type"][:max(room - 1, 1)] + "…"
        parts.append(
            f'<g>{tip}<rect x="{x}" y="{y}" width="{DIAGRAM_BOX_W}" '
            f'height="{DIAGRAM_BOX_H}" rx="4" class="{cls}"{fill}/>'
            f'<text x="{x + 7}" y="{y + 15}" class="arch-name">{esc(r["name"])}</text>'
            f'<text x="{x + _KIND_X}" y="{y + 15}" class="arch-kind">{esc(kind)}</text>'
            f'<text x="{x + DIAGRAM_BOX_W - 7}" y="{y + 15}" class="arch-shape" '
            f'text-anchor="end">{esc(r["shape"])}</text></g>')
    if unused:
        parts.append(f'<text x="8" y="{height - 8}" class="arch-sub">'
                     f'Not called by forward(): {esc(", ".join(unused))}</text>')

    def marker(mid_, cls):
        return (f'<marker id="{mid_}" viewBox="0 0 8 8" refX="7" refY="4" '
                f'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
                f'<path d="M0,0 L8,4 L0,8 z" class="{cls}"/></marker>')
    defs = ("<defs>" + marker("arch-arrow", "arch-head")
            + marker("arch-arrow-skip", "arch-head-skip")
            + marker("arch-arrow-feat", "arch-head-feat") + "</defs>")
    return (f'<div class="arch-diagram"><svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'role="img" aria-label="CondNet-TART architecture">{defs}'
            f'{"".join(parts)}</svg></div>')


def inspect_architecture(model_key: Optional[str] = None) -> dict:
    """{"diagram": HTML of both stages as one diagram, "printouts": {role:
    module printout}} for one CondNet-TART checkpoint pair. Cached until a
    checkpoint file changes."""
    model_key = model_key or default_model_key()
    model_class = get_model_class(model_key)
    roles = [role for role in STAGES if role in model_class.CHECKPOINTS]
    key = _signature(model_class)
    with _inspect_lock:
        if key in _inspect_cache:
            return _inspect_cache[key]
    model = get_model(model_key)
    nets = {role: getattr(model, STAGES[role][0]) for role in roles}
    traces = {role: _trace(model, role) for role in roles}
    result = {"diagram": _diagram_svg(traces, {r: _n_params(n) for r, n in nets.items()},
                                      model_class.CHECKPOINTS),
              "printouts": {role: str(net) for role, net in nets.items()}}
    with _inspect_lock:
        _inspect_cache[key] = result
    return result
