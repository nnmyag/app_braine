# ============================================================================
# tasks/resample/service.py — resample a volume to an isotropic voxel size
# ----------------------------------------------------------------------------
# What the page does, minus the widgets (web.py):
#
#   describe()   a volume's grid: voxels, voxel size, field of view,
#                orientation, data type, value range, file size.
#   resample()   (optionally) reorient to RAS -- nibabel's
#                as_closest_canonical, as isocond loads images -- then
#                resample to an isotropic voxel size with SimpleITK, keeping
#                the field of view: same physical extent, same outer corner,
#                same axis directions, so the output overlays the input.
#                Labels: nearest neighbour; images: linear, smoothed first
#                (Gaussian, per axis) where the voxels grow, so shrinking
#                does not alias.
#   resample_many()  the same for several files in one run (one shared
#                voxel size and RAS setting; "Auto" still decides per file),
#                plus a zip of all outputs.
#   mosaic_png() a row of axial slices of the result.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import nibabel as nib
import numpy as np
import SimpleITK as sitk

from app import jobs
from app import mosaic as mosaic_rule
from app.web import slice_viewer as sv
from tasks.resample import settings as task_settings
from vendor.isocond.pipeline.imaging import discrete_label_cmap

logger = logging.getLogger(__name__)

TASK_SLUG = "resample"
Reporter = Optional[Callable[[str], None]]

_INTERPOLATORS = {"Nearest": sitk.sitkNearestNeighbor,
                  "Linear": sitk.sitkLinear, "B-spline": sitk.sitkBSpline}


def _say(report: Reporter, message: str) -> None:
    logger.info(message)
    if report:
        report(message)


def _stem(path) -> str:
    return re.sub(r"\.nii(\.gz)?$", "", Path(path).name, flags=re.I)


def _size_text(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024


# ============================================================================
# What a volume is
# ============================================================================
@dataclass
class Info:
    name: str
    file_size: str
    shape: tuple                 # voxels per axis, as stored
    voxel_mm: tuple
    fov_mm: tuple
    orientation: str             # e.g. "RAS", "LPS"
    dtype: str
    value_range: str

    def short_line(self) -> str:
        """For the upload: grid, orientation, file size."""
        return (f"**{self.name}**: {' × '.join(map(str, self.shape))} voxels of "
                f"{' × '.join(num(v) for v in self.voxel_mm)} mm, {self.orientation}, "
                f"{self.file_size}")

    def line(self) -> str:
        f = lambda t, d: " × ".join(f"{v:.{d}f}" for v in t)
        return (f"**{self.name}**: {' × '.join(map(str, self.shape))} voxels of "
                f"{f(self.voxel_mm, 3)} mm ({f(self.fov_mm, 1)} mm), {self.orientation}, "
                f"{self.dtype}, {self.file_size}")


def num(x, digits: int = 3) -> str:
    """A number without meaningless trailing zeros: 1.000 -> 1, 191.0 -> 191,
    0.800 -> 0.8."""
    text = f"{float(x):.{digits}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def _sig(x: float) -> str:
    """4 significant digits, no exponent below 1e5: 1811.0895 -> 1811, 0.012345 -> 0.01235."""
    x = float(x)
    if x == int(x) or abs(x) >= 1e4:
        return f"{x:.0f}"
    return f"{x:.4g}"


def describe(path) -> Info:
    img = nib.load(str(path))
    if img.ndim != 3:
        raise ValueError(f"{Path(path).name} is {img.ndim}-D; a 3-D volume is needed")
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    data = np.asanyarray(img.dataobj)
    lo, hi = (float(np.nanmin(data)), float(np.nanmax(data))) if data.size else (0.0, 0.0)
    return Info(Path(path).name, _size_text(Path(path).stat().st_size), tuple(img.shape),
                zooms, tuple(n * z for n, z in zip(img.shape, zooms)),
                "".join(nib.aff2axcodes(img.affine)), str(img.get_data_dtype()),
                f"{_sig(lo)} … {_sig(hi)}")


def is_labels(path) -> bool:
    """Integer data type with at most MAX_LABELS distinct values."""
    img = nib.load(str(path))
    if not np.issubdtype(img.get_data_dtype(), np.integer):
        return False
    return np.unique(np.asanyarray(img.dataobj)).size <= task_settings.MAX_LABELS


# ============================================================================
# Resampling
# ============================================================================
@dataclass
class Result:
    path: str
    before: Info
    after: Info
    method: str                  # the interpolation actually used
    labels: bool
    mosaic: str                  # PNG data URI


def _resample(img: sitk.Image, voxel: float, method: str, labels: bool,
              report: Reporter) -> sitk.Image:
    old_sp = np.asarray(img.GetSpacing()); old_n = np.asarray(img.GetSize())
    new_sp = np.full(3, float(voxel))
    new_n = np.maximum(1, np.round(old_n * old_sp / new_sp)).astype(int)
    direction = np.asarray(img.GetDirection()).reshape(3, 3)
    # Keep the outer corner of the field of view: the first voxel's centre
    # moves by half the change in voxel size along each axis.
    origin = np.asarray(img.GetOrigin()) + direction @ ((new_sp - old_sp) / 2)
    if not labels and method != "Nearest":
        grow = new_sp > old_sp
        if grow.any():
            # Anti-alias: Gaussian whose width makes up the difference
            # between the old and the new voxel (FWHM = sqrt(new^2 - old^2)).
            sigma = np.where(grow, np.sqrt(np.maximum(new_sp ** 2 - old_sp ** 2, 0)) / 2.3548, 0)
            img = sitk.DiscreteGaussian(sitk.Cast(img, sitk.sitkFloat32),
                                        (sigma ** 2).tolist(), 32, 0.01, True)
            _say(report, "Anti-alias smoothing before shrinking: sigma "
                         + " × ".join(f"{s:.2f}" for s in sigma) + " mm")
    out_type = img.GetPixelID() if labels else sitk.sitkFloat32
    return sitk.Resample(img, new_n.tolist(), sitk.Transform(3, sitk.sitkIdentity),
                         _INTERPOLATORS[method], origin.tolist(), new_sp.tolist(),
                         direction.ravel().tolist(), 0, out_type)


def _check_voxel(voxel_mm) -> float:
    voxel = float(voxel_mm)
    if not task_settings.MIN_VOXEL_MM <= voxel <= task_settings.MAX_VOXEL_MM:
        raise ValueError(f"Voxel size must be {task_settings.MIN_VOXEL_MM:g}–"
                         f"{task_settings.MAX_VOXEL_MM:g} mm")
    return voxel


def resample(path, voxel_mm: float, method: str = "Auto", reorient_ras: bool = True,
             report: Reporter = None, run: Optional[jobs.RunDir] = None) -> Result:
    voxel = _check_voxel(voxel_mm)
    before = describe(path)
    labels = is_labels(path)
    used = method if method != "Auto" else ("Nearest" if labels else "Linear")
    _say(report, f"Input: {before.line().replace('**', '')}")
    _say(report, f"Content: {'labels' if labels else 'image'} → {used}"
                 + (" (auto)" if method == "Auto" else ""))

    run = run or jobs.new_run(TASK_SLUG)
    src = Path(path)
    if reorient_ras and before.orientation != "RAS":
        img = nib.as_closest_canonical(nib.load(str(path)))
        src = run.root / f"{_stem(path)}__ras.nii.gz"      # beside outputs/, not in it
        nib.save(img, str(src))
        _say(report, f"Reoriented {before.orientation} → RAS")
    elif reorient_ras:
        _say(report, "Already RAS")
    out_img = _resample(sitk.ReadImage(str(src)), voxel, used, labels, report)
    tag = f"{voxel:g}".replace(".", "p")
    out = run.output(f"{_stem(path)}_{tag}mm.nii.gz")
    sitk.WriteImage(out_img, str(out))
    after = describe(out)
    _say(report, f"Output: {after.line().replace('**', '')}")
    return Result(str(out), before, after, used, labels, mosaic_png(out, labels))


@dataclass
class Batch:
    results: List[Result]
    zip_path: Optional[str]      # all outputs, when there is more than one

    def names(self) -> List[str]:
        return [r.before.name for r in self.results]

    def by_name(self, name) -> Optional[Result]:
        return next((r for r in self.results if r.before.name == name),
                    self.results[0] if self.results else None)


def resample_many(paths, voxel_mm: float, method: str = "Auto", reorient_ras: bool = True,
                  report: Reporter = None) -> Batch:
    """Every file with the same settings, into one run folder."""
    paths = [p for p in (paths or []) if p]
    if not paths:
        raise ValueError("No volume uploaded")
    _check_voxel(voxel_mm)
    names = [Path(p).name for p in paths]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Two uploads share a file name: {', '.join(dup)}")
    run = jobs.new_run(TASK_SLUG)
    results = []
    for i, p in enumerate(paths, 1):
        _say(report, f"--- {i}/{len(paths)}: {Path(p).name}")
        results.append(resample(p, voxel_mm, method, reorient_ras, report, run=run))
    zip_path = str(run.zip_outputs()) if len(results) > 1 else None
    return Batch(results, zip_path)


# ============================================================================
# Preview
# ============================================================================
def mosaic_png(path, labels: bool) -> str:
    """MOSAIC_SLICES axial slices in a row, anterior up, subject's left on the
    left (canonical RAS view, whatever the file's own orientation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = task_settings
    img = nib.as_closest_canonical(nib.load(str(path)))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    lo, hi = mosaic_rule.occupied_span(vol, axis=2)
    ks = np.unique(np.linspace(lo, hi, st.MOSAIC_SLICES + 2).round().astype(int)[1:-1])
    if labels:
        top = max(int(vol.max()), 1)
        cmap, kw = discrete_label_cmap(top + 1), dict(vmin=-0.5, vmax=top + 0.5, interpolation="nearest")
    else:
        v = vol[np.isfinite(vol) & (vol != 0)]
        a, b = np.percentile(v, st.WINDOW_PERCENTILES) if v.size else (0, 1)
        cmap, kw = "gray", dict(vmin=a, vmax=b, interpolation="nearest")
    zoom = img.header.get_zooms()[:3]
    h_in = st.MOSAIC_SLICE_INCHES * (vol.shape[1] * zoom[1]) / (vol.shape[0] * zoom[0])
    fig, axes = plt.subplots(1, len(ks), figsize=(st.MOSAIC_SLICE_INCHES * len(ks), h_in + 0.35),
                             dpi=st.MOSAIC_DPI, squeeze=False)
    for ax, k in zip(axes[0], ks):
        z_mm = float((img.affine @ [0, 0, k, 1])[2])
        ax.imshow(np.rot90(vol[:, :, k]), cmap=cmap, aspect=zoom[1] / zoom[0], **kw)
        ax.set_title(f"slice {k}  (z = {z_mm:.0f} mm)", fontsize=12,
                     color="black" if labels else "white")
        ax.axis("off")
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="black" if not labels else "white")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


TABLE_ROWS = (("File", "name"), ("File size", "file_size"), ("Dimension (voxels)", "shape"),
              ("Voxel size (mm)", "voxel_mm"), ("Dimension (mm)", "fov_mm"),
              ("Orientation", "orientation"), ("Data type", "dtype"), ("Value range", "value_range"))


def _cell(v, key: str = "") -> str:
    if isinstance(v, tuple):
        return " × ".join(num(x, 1 if key == "fov_mm" else 3) for x in v)
    return str(v)


def render(result: Optional[Result]) -> str:
    if result is None:
        return sv.placeholder_html("Upload a volume, pick a voxel size, then press Run.")
    rows = [[label, _cell(getattr(result.before, key), key), _cell(getattr(result.after, key), key)]
            for label, key in TABLE_ROWS]
    table = ('<div class="rs-table">' + sv.table_html(["", "Before", "After"], rows,
                                                      "<b>Before and after resampling</b>")
             + "</div>")
    return (f'<div><img src="{result.mosaic}" style="max-width:100%;height:auto" '
            f'alt="axial slices of the result"/></div>'
            f'<div class="sv-note">Axial slices of the result (anterior up, '
            f'subject\'s left on the left).</div>{table}')
