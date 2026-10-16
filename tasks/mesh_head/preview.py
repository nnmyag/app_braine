# ============================================================================
# tasks/mesh_head/preview.py — the pictures this task shows
# ----------------------------------------------------------------------------
# UI-agnostic, like service.py: every function returns a PIL image or writes a
# PNG, never a widget, so a cli.py can save exactly what the page displays.
#
#   anatomy_mosaic()    axial mosaic of a scalar volume (T1, T2, b0)
#   label_mosaic()      the same for charm's final_tissues, in SimNIBS's own
#                       colours (final_tissues_LUT.txt), with a legend
#   surface_mosaic()    cortical surfaces (gifti) cut by each axial slice and
#                       drawn as contours over the T1 they were built from
#   alignment_mosaic()  the registered T1 as contours over the b0 -- the one
#                       picture that shows whether the header rewrite put the
#                       head where the DWI says it is
#
# Every volume is reoriented to RAS before drawing (load_canonical), so "axial"
# means axial whatever the scanner wrote: the IXI T1s are stored sagittal-first.
# Surfaces are in world (RAS mm) coordinates -- the same space as the T1's
# affine -- so they are mapped into voxel space through that affine and cut on
# voxel planes; a slightly oblique acquisition then still draws correctly.
#
# Rendered on demand from the files a run wrote, not held in memory: a finished
# job is a folder, so the canvas keeps working after the page is reloaded.
# ============================================================================
import io
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
# Gotcha #1 of the repo: on a headless server matplotlib otherwise reaches for
# Qt through the plugins cv2 bundles and the job dies drawing its first figure.
matplotlib.use("Agg")

import matplotlib.pyplot as plt                                     # noqa: E402
import nibabel as nib                                               # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib.collections import LineCollection                   # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap          # noqa: E402
from matplotlib.patches import Patch                                # noqa: E402
from PIL import Image                                               # noqa: E402

from app import mosaic as mosaic_rule

# Inches per tile. Small on purpose: these are read at ~600 px on a page.
TILE_IN = 2.2
DPI = 110

# Contour colours: one per surface, chosen to read against a grey T1.
SURFACE_COLORS = {"central": "#ff3b30", "pial": "#34c759", "white": "#0a84ff"}


# ============================================================================
# Loading
# ============================================================================
def load_canonical(path) -> Tuple[np.ndarray, np.ndarray]:
    """(volume, affine) of a NIfTI reoriented to RAS, squeezed to 3D.

    A 4D file (charm writes final_tissues with a trailing singleton) keeps its
    first frame; the affine returned is the REORIENTED one, so world <-> voxel
    through it matches the array that comes back.
    """
    img = nib.as_closest_canonical(nib.load(str(path)))
    data = np.asanyarray(img.dataobj)
    while data.ndim > 3:
        data = data[..., 0]
    return np.asarray(data, dtype=np.float32), img.affine


def load_gifti_surface(path) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices [n,3] world mm, faces [m,3]) from a SimNIBS/FreeSurfer gifti."""
    g = nib.load(str(path))
    coords = faces = None
    for arr in g.darrays:
        if arr.intent == 1008:          # NIFTI_INTENT_POINTSET
            coords = np.asarray(arr.data, dtype=np.float64)
        elif arr.intent == 1009:        # NIFTI_INTENT_TRIANGLE
            faces = np.asarray(arr.data, dtype=np.int64)
    if coords is None or faces is None:
        raise ValueError(f"{path}: no pointset/triangle arrays in gifti")
    return coords, faces


def read_lut(path) -> Dict[int, Tuple[str, Tuple[float, float, float]]]:
    """{id: (name, rgb 0-1)} from a FreeSurfer-style colour LUT (charm's
    final_tissues_LUT.txt). Blank lines and '#' comments are skipped."""
    lut = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].startswith("#"):
            continue
        try:
            tid = int(parts[0])
            rgb = tuple(int(c) / 255.0 for c in parts[2:5])
        except ValueError:
            continue
        lut[tid] = (parts[1].replace("_", " "), rgb)
    return lut


# ============================================================================
# Shared helpers
# ============================================================================
def _to_png(fig, dpi: Optional[float] = None) -> Image.Image:
    """Figure -> PIL image. `dpi` is for callers whose figure is SAVED and
    zoomed into rather than read at browser size -- task 4's do that."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi or DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def save_png(image: Image.Image, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path), format="PNG")


def _slice_indices(volume: np.ndarray, n_slices: int) -> np.ndarray:
    """Evenly spaced axial indices spanning only the slices that carry data --
    app/mosaic's rule, shared with the other tasks' mosaics."""
    return mosaic_rule.slice_indices(volume, n_slices)


def _window(volume: np.ndarray) -> Tuple[float, float]:
    """1st-99th percentile of the non-zero voxels: one hot voxel must not push
    the whole head into the bottom of the grey scale."""
    values = volume[np.isfinite(volume) & (volume != 0)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, (1.0, 99.0))
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _layout(shape, n_slices: Optional[int], n_cols: Optional[int]
            ) -> Tuple[int, int]:
    """(n_slices, n_cols) to draw: the caller's, or app/mosaic's rule for a
    slice of this volume. Slices are drawn as data[:, :, k].T, so the drawn
    width is shape[0] and the height shape[1]."""
    rows, cols = mosaic_rule.layout_for_slice(shape[0], shape[1])
    return (n_slices or rows * cols), (n_cols or cols)


def _grid(n_tiles: int, n_cols: int, shape, tile_in: float = TILE_IN):
    n_rows = math.ceil(n_tiles / n_cols)
    aspect = shape[1] / max(shape[0], 1)        # drawn transposed, so y over x
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(tile_in * n_cols, tile_in * aspect * n_rows))
    return fig, axes.flatten()


def _finish(fig, axes, title: str, shape, rect=(0, 0, 1, 0.965),
            dpi: Optional[float] = None) -> Image.Image:
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{title}\n{shape[0]}×{shape[1]}×{shape[2]}",
                 fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=rect)
    return _to_png(fig, dpi)


# ============================================================================
# Mosaics
# ============================================================================
def anatomy_mosaic(volume: np.ndarray, title: str, cbar_label: str = "",
                   n_slices: Optional[int] = None,
                   n_cols: Optional[int] = None,
                   cmap: str = "gray",
                   dpi: Optional[float] = None,
                   slice_fontsize: float = 7,
                   mask: Optional[np.ndarray] = None,
                   vmin: Optional[float] = None,
                   vmax: Optional[float] = None) -> Image.Image:
    """Axial mosaic of a scalar volume with one colour bar.

    Grey by default, which is how anatomy is read. `cmap` is for the volumes
    that are a derived QUANTITY rather than anatomy -- task 4's E-field on a
    NIfTI grid, say -- where the colour is the message.
    """
    n_slices, n_cols = _layout(volume.shape, n_slices, n_cols)
    # `mask` keeps the drawing to one tissue (task 4's white-matter view)
    # without changing what the colour window is fitted to.
    shown = volume if mask is None else np.where(mask, volume, np.nan)
    if vmin is None or vmax is None:
        auto_lo, auto_hi = _window(volume if mask is None else volume[mask])
        vmin = auto_lo if vmin is None else vmin
        vmax = auto_hi if vmax is None else vmax
    indices = _slice_indices(volume if mask is None else shown, n_slices)
    fig, axes = _grid(len(indices), n_cols, volume.shape)
    im = None
    for ax, idx in zip(axes, indices):
        im = ax.imshow(np.ma.masked_invalid(shown[:, :, idx].T), cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       origin="lower", interpolation="nearest")
        ax.set_title(str(idx), fontsize=slice_fontsize, color="#888", pad=2)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{title}\n{volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}",
                 fontsize=10, y=0.995, va="top")
    # A horizontal bar under the grid (task 2's placement): a vertical one
    # beside a wide grid is carved out of the last column and overlaps it.
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    if im is not None:
        cax = fig.add_axes((0.25, 0.035, 0.5, 0.018))
        bar = fig.colorbar(im, cax=cax, orientation="horizontal")
        bar.set_label(cbar_label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    return _to_png(fig, dpi)


def label_mosaic(labels: np.ndarray, title: str,
                 lut: Optional[Dict[int, Tuple[str, tuple]]] = None,
                 n_slices: Optional[int] = None,
                 n_cols: Optional[int] = None) -> Image.Image:
    """Axial mosaic of an integer label map, coloured by `lut`, with a legend.

    The ids are nominal, so each present id gets its own colour -- SimNIBS's
    when the LUT names it, tab20 otherwise -- rather than a continuous map that
    would imply id 9 is "more" than id 8.
    """
    n_slices, n_cols = _layout(labels.shape, n_slices, n_cols)
    labels = np.rint(np.asarray(labels)).astype(np.int32)
    lut = lut or {}
    present = [int(v) for v in np.unique(labels) if int(v) != 0] or [1]

    fallback = plt.get_cmap("tab20")
    colors = ["#101010"] + [
        lut[tid][1] if tid in lut else fallback(i % 20)
        for i, tid in enumerate(present)]
    cmap = ListedColormap(colors)
    rank = {tid: i + 1 for i, tid in enumerate(present)}
    ranked = np.zeros_like(labels)
    for tid, i in rank.items():
        ranked[labels == tid] = i
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors))

    indices = _slice_indices(labels, n_slices)
    fig, axes = _grid(len(indices), n_cols, labels.shape)
    for ax, idx in zip(axes, indices):
        ax.imshow(ranked[:, :, idx].T, cmap=cmap, norm=norm, origin="lower",
                  interpolation="nearest")
        ax.set_title(str(idx), fontsize=7, color="#888", pad=2)
    handles = [Patch(facecolor=colors[rank[tid]],
                     label=f"{tid} {lut[tid][0] if tid in lut else ''}".strip())
               for tid in present]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 6), fontsize=7, frameon=False)
    return _finish(fig, axes, title, labels.shape, rect=(0, 0.06, 1, 0.965))


# ============================================================================
# Surfaces over the T1
# ============================================================================
def _plane_segments(vox: np.ndarray, faces: np.ndarray, k: float) -> np.ndarray:
    """Line segments [s, 2, 2] where the mesh crosses the voxel plane z = k.

    Each triangle with vertices on both sides of the plane is cut along its two
    crossing edges; the segment between the two crossing points is one piece
    of the contour. Vectorised over all triangles at once.
    """
    z = vox[:, 2][faces] - k                          # [m, 3] signed distance
    side = z > 0
    crossing = np.any(side, axis=1) & ~np.all(side, axis=1)
    if not crossing.any():
        return np.empty((0, 2, 2))
    tri = faces[crossing]
    zc = z[crossing]
    pts = []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        za, zb = zc[:, a], zc[:, b]
        cross = (za > 0) != (zb > 0)
        t = np.where(cross, za / np.where(za - zb == 0, 1e-12, za - zb), np.nan)
        pa, pb = vox[tri[:, a]], vox[tri[:, b]]
        pts.append(pa[:, :2] + t[:, None] * (pb[:, :2] - pa[:, :2]))
    pts = np.stack(pts, axis=1)                       # [m, 3 edges, 2]
    # Exactly two of the three edges cross; take those two, in edge order.
    valid = ~np.isnan(pts[:, :, 0])
    order = np.argsort(~valid, axis=1, kind="stable")[:, :2]
    segs = np.take_along_axis(pts, order[:, :, None], axis=1)
    return segs[np.all(~np.isnan(segs[:, :, 0]), axis=1)]


def surface_mosaic(volume: np.ndarray, affine: np.ndarray,
                   surfaces: Dict[str, Sequence[Tuple[np.ndarray, np.ndarray]]],
                   title: str,
                   n_slices: Optional[int] = None,
                   n_cols: Optional[int] = None,
                   linewidth: float = 0.5) -> Image.Image:
    """The T1 in grey with each surface's cut drawn on every slice.

    `surfaces` maps a name ("central", "pial", …) to the (vertices, faces) of
    each hemisphere; every mesh under one name shares one colour. `linewidth`
    is thin for the canvas (three surfaces a millimetre apart must stay
    distinguishable) and thicker for the saved file, which is read zoomed in.
    """
    n_slices, n_cols = _layout(volume.shape, n_slices, n_cols)
    inv = np.linalg.inv(affine)
    in_vox = {
        name: [((inv @ np.c_[v, np.ones(len(v))].T).T[:, :3], f)
               for v, f in meshes]
        for name, meshes in surfaces.items()}

    # Span the slices the SURFACES occupy, not the whole head: the cortex is
    # what this picture is about.
    all_z = np.concatenate([m[0][:, 2] for ms in in_vox.values() for m in ms])
    lo, hi = int(np.floor(all_z.min())), int(np.ceil(all_z.max()))
    lo, hi = max(lo, 0), min(hi, volume.shape[2] - 1)
    indices = np.unique(np.linspace(lo, hi, n_slices, dtype=int))

    vmin, vmax = _window(volume)
    fig, axes = _grid(len(indices), n_cols, volume.shape)
    for ax, idx in zip(axes, indices):
        ax.imshow(volume[:, :, idx].T, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", interpolation="nearest")
        for name, meshes in in_vox.items():
            color = SURFACE_COLORS.get(name, "#ffcc00")
            for vox, faces in meshes:
                segs = _plane_segments(vox, faces, float(idx))
                if len(segs):
                    ax.add_collection(LineCollection(
                        segs, colors=color, linewidths=linewidth))
        ax.set_xlim(-0.5, volume.shape[0] - 0.5)
        ax.set_ylim(-0.5, volume.shape[1] - 0.5)
        ax.set_title(str(idx), fontsize=7, color="#888", pad=2)
    handles = [Patch(facecolor=SURFACE_COLORS.get(n, "#ffcc00"), label=n)
               for n in in_vox]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=7, frameon=False)
    return _finish(fig, axes, title, volume.shape, rect=(0, 0.04, 1, 0.965))


# ============================================================================
# Alignment check
# ============================================================================
def alignment_mosaic(b0: np.ndarray, t1_on_b0: np.ndarray, title: str,
                     n_slices: Optional[int] = None,
                     n_cols: Optional[int] = None) -> Image.Image:
    """b0 in grey, the registered T1 (already on the b0 grid) as red contours.

    Contours rather than a colour blend: what the eye checks is whether the
    T1's skull and ventricle outlines sit on the b0's, and a blend hides small
    offsets that contours show.
    """
    n_slices, n_cols = _layout(b0.shape, n_slices, n_cols)
    vmin, vmax = _window(b0)
    t_lo, t_hi = _window(t1_on_b0)
    levels = np.linspace(t_lo, t_hi, 6)[1:-1]
    indices = _slice_indices(b0, n_slices)
    fig, axes = _grid(len(indices), n_cols, b0.shape)
    for ax, idx in zip(axes, indices):
        ax.imshow(b0[:, :, idx].T, cmap="gray", vmin=vmin, vmax=vmax,
                  origin="lower", interpolation="nearest")
        sl = t1_on_b0[:, :, idx].T
        if np.any(sl > t_lo):
            ax.contour(sl, levels=levels, colors="#ff3b30", linewidths=0.35,
                       alpha=0.9)
        ax.set_title(str(idx), fontsize=7, color="#888", pad=2)
    return _finish(fig, axes, title, b0.shape)
