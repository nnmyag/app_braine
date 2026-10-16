# ============================================================================
# tasks/registration/service.py — register a moving NIfTI onto a fixed one
# ----------------------------------------------------------------------------
# What the page does, minus the widgets (web.py):
#
#   register()     fixed + moving (+ the moving image's segmentation) ->
#                  (SimpleITK; method and settings in settings.py)
#                  a rigid or rigid-then-affine transform in physical (mm)
#                  space, the moving image resampled onto the FIXED image's
#                  voxel grid (linear) and the segmentation with the same
#                  transform (nearest neighbour, labels and type kept). The
#                  outputs carry the fixed image's header: they overlay it
#                  voxel for voxel.
#   render_slice() one axial slice: fixed (red) + registered moving (green)
#                  as one RGB image, and the registered moving image, with
#                  hover values (app/web/slice_viewer); below, the
#                  displacement co-histogram (cohistogram_png).
#   cohistogram_png()  joint histogram of the moving image before vs after
#                  registration, voxel by voxel on the fixed grid (jet), with
#                  both histograms as marginals, the mutual information, the
#                  transform and the image centre before and after.
#
# Registration runs in physical space through each file's own header, so the
# segmentation need not share the moving image's voxel grid -- only its space.
# ============================================================================
# --- repo root on sys.path, as the other tasks do ---------------------------
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
if _root not in _sys.path:
    _sys.path.insert(0, _root)

import base64
import io
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from matplotlib import colormaps
from matplotlib.colors import Normalize

from app import jobs
from app import mosaic as mosaic_rule
from app.web import slice_viewer as sv
from tasks.registration import settings as task_settings

logger = logging.getLogger(__name__)

TASK_SLUG = "registration"
Reporter = Optional[Callable[[str], None]]


def _say(report: Reporter, message: str) -> None:
    logger.info(message)
    if report:
        report(message)


def _stem(path) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", Path(path).name, flags=re.I)


def _read(path, label: str, pixel_type=None) -> sitk.Image:
    img = sitk.ReadImage(str(path), pixel_type) if pixel_type is not None \
        else sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"{label} {Path(path).name} is {img.GetDimension()}-D; "
                         f"a 3-D volume is needed")
    return img


# ============================================================================
# Registration
# ============================================================================
@dataclass
class Result:
    run_dir: str
    files: List[str]                 # registered image (+ segmentation)
    fixed: np.ndarray                # canonical (X, Y, Z) on the fixed grid
    moving: np.ndarray               # registered moving, same grid
    summary: str
    cohistogram: str                 # PNG data URI, see cohistogram_png()

    @property
    def depth(self) -> int:
        return int(self.fixed.shape[2])

    def default_slice(self) -> int:
        lo, hi = mosaic_rule.occupied_span(self.fixed, axis=2)
        return (lo + hi) // 2


def _method(initial_transform) -> sitk.ImageRegistrationMethod:
    """Mattes MI, regular-step gradient descent, coarse-to-fine (settings.py)."""
    st = task_settings
    m = sitk.ImageRegistrationMethod()
    m.SetMetricAsMattesMutualInformation(numberOfHistogramBins=st.HISTOGRAM_BINS)
    m.SetMetricSamplingStrategy(m.RANDOM)
    m.SetMetricSamplingPercentage(st.SAMPLING_PERCENTAGE, seed=st.SAMPLING_SEED)
    m.SetInterpolator(sitk.sitkLinear)
    m.SetOptimizerAsRegularStepGradientDescent(
        learningRate=st.LEARNING_RATE, minStep=st.MIN_STEP,
        numberOfIterations=st.MAX_ITERATIONS, gradientMagnitudeTolerance=1e-8)
    m.SetOptimizerScalesFromPhysicalShift()
    m.SetShrinkFactorsPerLevel(list(st.SHRINK_FACTORS))
    m.SetSmoothingSigmasPerLevel(list(st.SMOOTHING_SIGMAS_MM))
    m.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    m.SetInitialTransform(initial_transform, inPlace=False)
    return m


def _unwrap(transform):
    """Execute() may wrap the result in a CompositeTransform."""
    if isinstance(transform, sitk.CompositeTransform):
        return transform.GetNthTransform(transform.GetNumberOfTransforms() - 1)
    return transform


def _stage(fixed, moving, transform, name, report):
    method = _method(transform)
    t0 = time.time()
    try:
        out = _unwrap(method.Execute(fixed, moving))
    except RuntimeError as exc:
        _say(report, f"{name} stage failed ({exc}); keeping the previous transform")
        return transform, None
    _say(report, f"{name}: {method.GetOptimizerStopConditionDescription()} "
                 f"({time.time() - t0:.0f} s, metric {method.GetMetricValue():.4f})")
    return out, method.GetMetricValue()


def _metric(fixed, moving, transform) -> float:
    """Mattes MI of the pair under `transform` (lower = better match)."""
    return _method(transform).MetricEvaluate(fixed, moving)


def register(fixed_path, moving_path, seg_path, mode: str,
             report: Reporter = None) -> Result:
    stages = task_settings.MODES[mode]
    fixed = _read(fixed_path, "Fixed", sitk.sitkFloat32)
    moving = _read(moving_path, "Moving", sitk.sitkFloat32)
    seg = _read(seg_path, "Segmentation") if seg_path else None
    _say(report, f"Fixed {Path(fixed_path).name}: {fixed.GetSize()} voxels of "
                 f"{tuple(round(s, 3) for s in fixed.GetSpacing())} mm")
    _say(report, f"Moving {Path(moving_path).name}: {moving.GetSize()} voxels of "
                 f"{tuple(round(s, 3) for s in moving.GetSpacing())} mm")
    if seg is not None and seg.GetSize() != moving.GetSize():
        _say(report, "Note: the segmentation's grid differs from the moving image's; "
                     "it is mapped through its own header (same space assumed)")

    transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    before = _metric(fixed, moving, transform)
    _say(report, f"Start: image centres aligned (metric {before:.4f})")
    if "rigid" in stages:
        transform, _ = _stage(fixed, moving, transform, "Rigid", report)
    if "affine" in stages:
        affine = sitk.AffineTransform(3)
        affine.SetMatrix(transform.GetMatrix())
        affine.SetTranslation(transform.GetTranslation())
        affine.SetCenter(transform.GetCenter())
        transform, _ = _stage(fixed, moving, affine, "Affine", report)
    after = _metric(fixed, moving, transform)

    run = jobs.new_run(TASK_SLUG)
    out_img = sitk.Resample(moving, fixed, transform, sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    files = [run.output(f"{_stem(moving_path)}_reg.nii.gz")]
    sitk.WriteImage(out_img, str(files[0]))
    if seg is not None:
        out_seg = sitk.Resample(seg, fixed, transform, sitk.sitkNearestNeighbor, 0,
                                seg.GetPixelID())
        files.append(run.output(f"{_stem(seg_path)}_reg.nii.gz"))
        sitk.WriteImage(out_seg, str(files[1]))
    m = np.asarray(transform.GetMatrix()).reshape(3, 3)
    t = np.asarray(transform.GetTranslation())
    summary = (f"{mode}: metric {before:.4f} → {after:.4f} (Mattes mutual information; "
               f"lower = better match). Matrix rows {np.round(m, 4).tolist()}, "
               f"translation {np.round(t, 2).tolist()} mm.")
    _say(report, summary)

    # The moving image before registration on the same grid: placed by its
    # header only (identity in physical space).
    before = sitk.Resample(moving, fixed, sitk.Transform(3, sitk.sitkIdentity),
                           sitk.sitkLinear, 0.0, sitk.sitkFloat32)
    figure = cohistogram_png(sitk.GetArrayFromImage(before), sitk.GetArrayFromImage(out_img),
                             transform, moving, mode)

    canon = lambda p: np.asarray(nib.as_closest_canonical(nib.load(str(p))).dataobj)
    return Result(str(run.root), [str(f) for f in files],
                  canon(fixed_path).astype(np.float32), canon(files[0]).astype(np.float32),
                  summary, figure)


# ============================================================================
# The displacement co-histogram
# ============================================================================
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])       # ITK works in LPS, NIfTI viewers in RAS


def moving_to_fixed_ras(transform) -> np.ndarray:
    """4x4 matrix taking a point of the moving image to where it lands in the
    fixed image, in RAS mm (the inverse of ITK's fixed -> moving transform,
    converted from ITK's LPS)."""
    m = np.asarray(transform.GetMatrix()).reshape(3, 3)
    c = np.asarray(transform.GetCenter()); t = np.asarray(transform.GetTranslation())
    a = np.eye(4); a[:3, :3] = m; a[:3, 3] = c + t - m @ c          # fixed -> moving, LPS
    inv = np.linalg.inv(a)
    d = np.diag([-1.0, -1.0, 1.0, 1.0])
    return d @ inv @ d


def centre_of_mass_ras(image: sitk.Image) -> np.ndarray:
    """Intensity-weighted centre of an image (voxels > 0), in RAS mm."""
    arr = sitk.GetArrayFromImage(image).astype(np.float64)        # (z, y, x)
    w = np.where(arr > 0, arr, 0.0)
    idx = np.indices(arr.shape).reshape(3, -1)
    zyx = (idx * w.ravel()).sum(axis=1) / max(w.sum(), 1e-12)
    lps = image.TransformContinuousIndexToPhysicalPoint([float(zyx[2]), float(zyx[1]), float(zyx[0])])
    return _LPS_TO_RAS @ np.asarray(lps)


def mutual_information(joint: np.ndarray) -> float:
    """MI (bits) of a joint histogram of counts."""
    p = joint / max(joint.sum(), 1)
    px, py = p.sum(axis=1, keepdims=True), p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def cohistogram_png(before: np.ndarray, after: np.ndarray, transform,
                    moving: sitk.Image, mode: str) -> str:
    """The moving image before (x) vs after (y) registration, voxel by voxel
    on the fixed grid: counts in jet (log scale), the two histograms as
    marginals, and the MI, transform and centres printed on the figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = task_settings
    keep = (before > 0) | (after > 0)                  # drop shared background
    x, y = before[keep].astype(np.float64), after[keep].astype(np.float64)
    top = float(np.percentile(np.concatenate([x[x > 0], y[y > 0]]), st.COHIST_TOP_PERCENTILE)) or 1.0
    edges = np.linspace(0.0, top, st.COHIST_BINS + 1)
    joint, _, _ = np.histogram2d(np.clip(x, 0, top), np.clip(y, 0, top), bins=[edges, edges])
    mi = mutual_information(joint)

    a = moving_to_fixed_ras(transform)
    c0 = centre_of_mass_ras(moving)
    c1 = (a @ np.append(c0, 1.0))[:3]
    shift = c1 - c0

    fig = plt.figure(figsize=st.COHIST_FIGSIZE, dpi=st.COHIST_DPI)
    gs = fig.add_gridspec(2, 3, width_ratios=[4, 1, 0.18], height_ratios=[1, 4],
                          wspace=0.06, hspace=0.06, left=0.11, right=0.92, top=0.95, bottom=0.31)
    ax = fig.add_subplot(gs[1, 0]); ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, 1], sharey=ax); cax = fig.add_subplot(gs[1, 2])
    shown = np.ma.masked_equal(joint.T, 0)
    im = ax.imshow(np.ma.log10(shown), origin="lower", cmap="jet", aspect="auto",
                   extent=[0, top, 0, top], interpolation="nearest")
    ax.plot([0, top], [0, top], color="white", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("moving image BEFORE registration (intensity)")
    ax.set_ylabel("moving image AFTER registration (intensity)")
    fig.colorbar(im, cax=cax).set_label("log10 voxel count")
    centres = 0.5 * (edges[:-1] + edges[1:])
    ax_top.bar(centres, joint.sum(axis=1), width=edges[1] - edges[0], color="#4b5563")
    ax_right.barh(centres, joint.sum(axis=0), height=edges[1] - edges[0], color="#4b5563")
    for a_ in (ax_top, ax_right):
        a_.set_yscale("log") if a_ is ax_top else a_.set_xscale("log")
        a_.tick_params(labelbottom=False, labelleft=False, labelsize=7)
    ax_top.set_title("Displacement co-histogram of the moving image", fontsize=10)

    rows = "\n".join("  [" + "  ".join(f"{v:9.4f}" for v in r) + " ]" for r in a)
    fmt = lambda v: "(" + ", ".join(f"{u:7.1f}" for u in v) + ")"
    text = (f"Mutual information (before vs after): {mi:.3f} bits\n"
            f"Transform ({mode.split(' (')[0].lower()}; moving -> fixed, RAS mm, 4x4):\n{rows}\n"
            f"Image centre (centre of mass, RAS mm):\n"
            f"  before {fmt(c0)}\n  after  {fmt(c1)}\n"
            f"  shift  {fmt(shift)}   |shift| = {np.linalg.norm(shift):.1f} mm")
    fig.text(0.04, 0.015, text, family="monospace", fontsize=8, va="bottom")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=st.COHIST_DPI)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ============================================================================
# One axial slice, for the hover check
# ============================================================================
def _axial(volume: np.ndarray, k: int) -> np.ndarray:
    """Canonical (X, Y, Z) slice k -> rows anterior..posterior, columns left..right."""
    return np.rot90(volume[:, :, k])


def _window(volume: np.ndarray) -> Normalize:
    v = volume[np.isfinite(volume) & (volume != 0)]
    lo, hi = np.percentile(v, task_settings.WINDOW_PERCENTILES) if v.size else (0.0, 1.0)
    return Normalize(float(lo), float(hi) if hi > lo else float(lo) + 1.0)


def render_slice(result: Optional[Result], k) -> str:
    if result is None:
        return sv.placeholder_html("Upload a fixed and a moving image, then press Run.")
    k = int(np.clip(int(k), 0, result.depth - 1))
    gray = colormaps["gray"]
    f, m = _axial(result.fixed, k), _axial(result.moving, k)
    h, w = f.shape
    zoom = max(1, round(task_settings.VIEW_TARGET_PX / max(h, w)))
    nf, nm = _window(result.fixed), _window(result.moving)
    # Fixed in red, registered moving in green: aligned structures turn yellow.
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[..., 0] = (np.clip(nf(f), 0, 1) * 255).astype(np.uint8)
    rgb[..., 1] = (np.clip(nm(m), 0, 1) * 255).astype(np.uint8)
    panels = [
        sv.panel_html(f, nf, gray, sv.channel(f, 1, second=m, labels=("F", "M")),
                      "Fixed (red) + moving (green)", zoom=zoom, rgb=rgb,
                      footer='<div class="sv-bar">yellow = both bright · hover: '
                             'F fixed, M registered moving</div>'),
        sv.panel_html(m, nm, gray, sv.channel(m, 1), "Moving (registered)", zoom=zoom,
                      footer=sv.bar_html(nm, gray, "moving image, on the fixed grid", w * zoom)),
    ]
    note = (f"Axial slice {k} of 0–{result.depth - 1} on the fixed image's grid "
            f"(anterior up, subject's left on the left). Hover over either image: both "
            f"show their values at that point.")
    below = (f'<div style="margin-top:12px"><img src="{result.cohistogram}" '
             f'style="max-width:100%;height:auto" alt="displacement co-histogram"/></div>')
    return sv.canvas_html(panels, below=below, note=note)
