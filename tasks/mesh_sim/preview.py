# ============================================================================
# tasks/mesh_sim/preview.py — the pictures this task shows
# ----------------------------------------------------------------------------
# UI-agnostic, like service.py: every function returns a PIL image or writes a
# PNG, never a widget, so a cli.py can save exactly what the page displays.
#
#   field_surface_mosaic()  a cortical surface cut by each axial slice and
#                           coloured by the field along it -- optionally with
#                           the coil drawn on top, and with the per-tissue
#                           percentile table annotated in the corner
#   field_volume_mosaic()   the field on its NIfTI grid over the T1, with the
#                           white surface's cut drawn on top as the gray/white
#                           boundary
#   coil_position_slice()   where the coil sat: the one axial slice through
#                           its centre, with the centre marked and axis drawn
#   sample_volume_at()      the field volume read at surface vertices -- the
#                           fallback for a surface the worker wrote no overlay
#                           for (it writes central, pial and white)
#
# Volume mosaics (T1, the field on a NIfTI grid, tissue labels) are
# mesh_head's already: this module borrows its loaders, its grid helpers and
# its drawing rather than writing a second copy. What is genuinely new here is
# a surface contour whose COLOUR carries a value, and the coil overlay.
#
# ONE function serves both the canvas and the saved file. They differ only in
# arguments -- `dpi` and `linewidth`, higher for a file that gets zoomed into,
# lower for something read at ~600 px in a browser -- so the picture on the
# page and the picture in the archive cannot drift apart.
#
# Why contours rather than a rendered 3D surface: the whole app draws axial
# mosaics, and a field read off the cortex is only meaningful next to the
# anatomy it sits on. The 3D view is what the exported .vtk files are for.
# ============================================================================
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
# Gotcha #1 of the repo: on a headless server matplotlib otherwise reaches for
# Qt through the plugins cv2 bundles and the job dies drawing its first figure.
matplotlib.use("Agg")

import matplotlib.pyplot as plt                                      # noqa: E402
import nibabel as nib                                                # noqa: E402
import numpy as np                                                   # noqa: E402
from matplotlib.collections import LineCollection                    # noqa: E402
from matplotlib.colors import Normalize                              # noqa: E402
from PIL import Image                                                # noqa: E402

from app import mosaic as mosaic_rule                               # noqa: E402
# Deliberate reuse: same loaders, same tile geometry, same finishing, so a
# mesh_sim mosaic and a mesh_head mosaic sit side by side without a seam.
from tasks.mesh_head.preview import (_grid, _plane_segments, _to_png,  # noqa: E402
                                     _window, anatomy_mosaic,
                                     label_mosaic, load_canonical,
                                     load_gifti_surface, read_lut, save_png)
from tasks.mesh_sim import settings as task_settings                # noqa: E402

__all__ = ["anatomy_mosaic", "label_mosaic", "load_canonical",
           "load_gifti_surface", "read_lut", "save_png",
           "field_surface_mosaic", "field_volume_mosaic",
           "coil_position_slice", "tensor_orientation_mosaic",
           "read_curv_values", "sample_volume_at", "resample_to_grid",
           "load_tensor6", "tensor_eigen", "percentile_table",
           "anisotropy_table",
           "FIELD_CMAP", "CANVAS_LINEWIDTH", "SAVED_LINEWIDTH",
           "BOUNDARY_COLOR"]

# The field's colour ramp, from settings so the study can change it in one
# place. 'jet' by request: it is what the project's own figures use.
FIELD_CMAP = task_settings.FIELD_CMAP

# Percentiles the colour scale spans. The top of a TMS field is a small,
# sharp peak; scaling to the true maximum leaves the rest of the cortex dark,
# so the window is fitted to the bulk and the peak is allowed to saturate.
FIELD_PCTL = (2.0, 99.5)

# Widths for the coloured surface cut: thin on screen (the two hemispheres'
# folds are close together), thicker in the saved copy, which is read zoomed.
CANVAS_LINEWIDTH = 1.2
SAVED_LINEWIDTH = 1.8

# How much brighter than its own 99th percentile the anatomy is windowed to,
# i.e. how far it is pushed toward black behind the coloured cut. 1.0 leaves
# it as-is; task 2's overlay mosaic has the same knob for the same reason.
BACKGROUND_DIM = 1.7

# The gray/white boundary on the volume mosaic: white, by request, and drawn
# thin -- it is a reference line over a colour map, not a second data layer.
BOUNDARY_COLOR = "white"
BOUNDARY_LINEWIDTH = 0.6

COIL_COLOR = task_settings.COIL_COLOR
COIL_LINEWIDTH = task_settings.COIL_LINEWIDTH
COIL_MARKER_EDGEWIDTH = task_settings.COIL_MARKER_EDGEWIDTH


# ============================================================================
# Per-vertex values
# ============================================================================
def read_curv_values(path, n_expected: Optional[int] = None) -> np.ndarray:
    """One value per vertex from a FreeSurfer curv file.

    SimNIBS's map_to_surf writes its cortex overlays in this format (see
    subject_overlays/ after a run) -- the same format charm uses for the
    per-vertex `.sigma` uncertainty, which is why vendor/gii2vtk can read
    both with one function.
    """
    import nibabel as nib
    values = np.asarray(nib.freesurfer.io.read_morph_data(str(path)),
                        dtype=np.float64)
    if n_expected is not None and values.shape[0] != n_expected:
        raise ValueError(f"{Path(path).name} has {values.shape[0]} values but "
                         f"the surface has {n_expected} vertices")
    return values


def sample_volume_at(volume: np.ndarray, affine: np.ndarray,
                     points: np.ndarray, order: int = 1) -> np.ndarray:
    """The volume read at world-mm `points`, trilinearly.

    The fallback for a surface with no overlay of its own (the worker writes
    central, pial and white; an older run has only central): sampling the
    field volume at its vertices gives it one. Only meaningful because the
    volume carries white matter as well as gray (worker: tissues_in_niftis),
    otherwise every white-surface vertex would read zero.
    """
    from scipy.ndimage import map_coordinates
    inverse = np.linalg.inv(affine)
    voxel = (inverse @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    return map_coordinates(np.asarray(volume, dtype=float), voxel.T,
                           order=order, mode="nearest")


def resample_to_grid(volume: np.ndarray, affine: np.ndarray,
                     dst_shape: Tuple[int, int, int],
                     dst_affine: np.ndarray, order: int = 1) -> np.ndarray:
    """`volume` (on `affine`) resampled onto a grid of `dst_shape` cells at
    `dst_affine` -- every voxel of the destination grid, not a handful of
    points, so it is `sample_volume_at` driven by a whole grid's own voxel
    centres rather than a surface's vertices.

    Needed because the conductivity tensor arrives on the DWI/b0 grid
    (task 2's own space), never the head model's T1 grid: the tensor QC map
    is drawn in the tensor's own (usually coarser) space, and it is the T1
    and the tissue mask that get pulled onto it, rather than the much larger
    other way round.
    """
    ijk = np.indices(dst_shape).reshape(3, -1).T.astype(np.float64)
    world = nib.affines.apply_affine(dst_affine, ijk)
    return sample_volume_at(volume, affine, world, order=order).reshape(dst_shape)


# ============================================================================
# The conductivity tensor: loaded, eigen-decomposed, summarised
# ----------------------------------------------------------------------------
# This is the ONE place either the QC mosaic or the anisotropy stats reads
# the tensor from, so a change to how the eigenvectors or the anisotropy
# index are computed cannot make the two disagree.
# ============================================================================
def _tensor_matrices(tensor6: np.ndarray) -> np.ndarray:
    """[x,y,z,6] upper-triangular components -> [x,y,z,3,3] symmetric."""
    xx, xy, xz, yy, yz, zz = (tensor6[..., i] for i in range(6))
    mats = np.zeros(tensor6.shape[:3] + (3, 3), dtype=np.float64)
    mats[..., 0, 0], mats[..., 1, 1], mats[..., 2, 2] = xx, yy, zz
    mats[..., 0, 1] = mats[..., 1, 0] = xy
    mats[..., 0, 2] = mats[..., 2, 0] = xz
    mats[..., 1, 2] = mats[..., 2, 1] = yz
    return mats


def _fsl_world_rotation(affine: np.ndarray) -> np.ndarray:
    """Voxel frame -> world RAS for TENSOR COMPONENTS, SimNIBS's own rule.

    Copied deliberately from simnibs.utils.cond_utils.cond2elmdata's
    `correct_FSL` branch, so this module and the solver read one tensor file
    the same way: the affine's normalised direction cosines, with the x axis
    flipped when the determinant is positive (FSL's convention, which is what
    anicond writes -- its eigenvectors come straight out of dipy's fit, in
    the DWI's own voxel frame).
    """
    M = affine[:3, :3] / np.linalg.norm(affine[:3, :3], axis=0)[:, None]
    R = np.eye(3)
    if np.linalg.det(M) > 0:
        R[0, 0] = -1
    return M.dot(R)


def _rotate_tensor6(tensor6: np.ndarray, rotation: np.ndarray) -> np.ndarray:
    """Each voxel's tensor carried into another frame: T -> M T M'."""
    mats = _tensor_matrices(tensor6)
    rotated = np.einsum("ij,...jk,lk->...il", rotation, mats, rotation)
    return np.stack([rotated[..., 0, 0], rotated[..., 0, 1], rotated[..., 0, 2],
                     rotated[..., 1, 1], rotated[..., 1, 2], rotated[..., 2, 2]],
                    axis=-1)


def load_tensor6(path) -> Tuple[np.ndarray, np.ndarray]:
    """(tensor, affine) of a 6-component conductivity tensor NIfTI, with the
    components rotated into world RAS.

    Component order is anicond's own, FSL's dtifit layout: xx, xy, xz, yy,
    yz, zz. Two separate things have to happen and it is easy to do only the
    first: `as_closest_canonical` reorients the three spatial axes of the
    ARRAY to RAS, but the six numbers in each voxel are a matrix in the
    file's own voxel frame and nibabel neither knows nor touches that. So the
    array is reoriented AND the components are rotated (`_fsl_world_rotation`,
    SimNIBS's own rule), and what comes back is in the frame the solver will
    actually use. The two operations commute -- one moves voxels, the other
    rewrites what is stored in them.

    Without the rotation the map is still right wherever the acquisition
    happens to be axis-aligned (the pure axis flip cancels under the `abs()`
    a direction-encoded colour takes), and drifts with the obliquity
    otherwise -- which is a QC picture that is trustworthy only when it is
    not needed.
    """
    img = nib.load(str(path))
    rotation = _fsl_world_rotation(img.affine)
    canonical = nib.as_closest_canonical(img)
    data = np.asanyarray(canonical.dataobj, dtype=np.float32)
    if data.ndim != 4 or data.shape[-1] != 6:
        raise ValueError(f"{path}: expected a 6-component tensor volume, "
                         f"got shape {data.shape}")
    return _rotate_tensor6(data, rotation), canonical.affine


def tensor_eigen(tensor6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(principal eigenvector [x,y,z,3], anisotropy index [x,y,z]) of a 6x1
    conductivity tensor volume.

    The anisotropy index is diffusion MRI's fractional-anisotropy formula
    applied to the conductivity tensor's own eigenvalues -- 0 for a
    perfectly isotropic tensor, towards 1 the more the conductivity
    concentrates along one axis. It is not FA (the tensor is conductivity,
    not diffusivity) so it is never called that on a figure, but the
    construction -- and the 0..1 reading of it -- is the same.

    A background voxel the tensor never touched is a zero matrix: its
    eigenvalues are all zero, so the ratio below is 0/0. `nan_to_num` sends
    that to 0 (fully isotropic reading, weight it out with a mask instead of
    trusting this number alone) rather than letting it poison the figure.
    """
    evals, evecs = np.linalg.eigh(_tensor_matrices(tensor6))  # ascending
    principal = evecs[..., :, -1]               # eigenvector of the largest
    l = evals[..., ::-1]                        # descending, for the formula
    mean_l = l.mean(axis=-1, keepdims=True)
    denom = np.sqrt((l ** 2).sum(axis=-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        anisotropy = (np.sqrt(1.5)
                     * np.sqrt(((l - mean_l) ** 2).sum(axis=-1)) / denom)
    anisotropy = np.nan_to_num(anisotropy, nan=0.0, posinf=0.0, neginf=0.0)
    return principal, np.clip(anisotropy, 0.0, 1.0)


def tensor_tissue_stats(anisotropy: np.ndarray, mask: np.ndarray) -> Dict:
    """{mean, median, p95, n_voxels} of the anisotropy index inside `mask`.

    Voxel-based and unweighted, unlike worker.tissue_field_stats: this is a
    regular NIfTI grid, where every voxel is the same volume, not a
    tetrahedral mesh with wildly varying element sizes -- there is nothing
    here for a volume weight to correct for.
    """
    values = anisotropy[mask]
    if values.size == 0:
        return {}
    return {"mean": float(values.mean()),
           "median": float(np.median(values)),
           "p95": float(np.percentile(values, 95.0)),
           "n_voxels": int(values.size)}


# ============================================================================
# The percentile table drawn on a figure
# ============================================================================
def percentile_table(stats: Dict, percentiles: Sequence[float]) -> str:
    """The per-tissue percentile block, as monospaced text for a figure.

    `stats` is what worker.tissue_field_stats produced, keyed by tissue. Kept
    here rather than in service.py because it is presentation: the same text
    goes on the canvas and into the saved PNG.
    """
    if not stats:
        return ""
    columns = [f"p{p:g}" for p in percentiles]
    header = "|E| (V/m)".ljust(14) + "".join(c.rjust(9) for c in columns)
    lines = [header, "-" * len(header)]
    for tissue in ("gray_matter", "white_matter"):
        entry = stats.get(tissue)
        if not entry:
            continue
        row = tissue.replace("_", " ").ljust(14)
        for p in percentiles:
            value = entry.get("percentiles", {}).get(f"{p:g}", {})
            threshold = value.get("threshold")
            row += ("—" if threshold is None else f"{threshold:.3g}").rjust(9)
        row += f"   max {entry.get('max', float('nan')):.3g}"
        lines.append(row)
    return "\n".join(lines)


def anisotropy_table(stats: Dict) -> str:
    """The per-tissue anisotropy-index block, as monospaced text.

    `stats` is `{tissue: tensor_tissue_stats(...)}`. Companion to
    `percentile_table`, same layout convention, for the tensor QC mosaic.
    """
    if not stats:
        return ""
    header = "anisotropy".ljust(14) + "".join(
        c.rjust(9) for c in ("mean", "median", "p95"))
    lines = [header, "-" * len(header)]
    for tissue in ("gray_matter", "white_matter"):
        entry = stats.get(tissue)
        if not entry:
            continue
        row = tissue.replace("_", " ").ljust(14)
        for key in ("mean", "median", "p95"):
            row += (f"{entry[key]:.3g}" if key in entry else "—").rjust(9)
        lines.append(row)
    return "\n".join(lines)


def _annotate(fig, text: str, fontsize: float = 7.5) -> None:
    """A monospaced block in the figure's bottom-left corner.

    Bottom-left, because the mosaic's colour bar owns the middle of that strip
    and the grid itself never reaches the corner.
    """
    if not text:
        return
    fig.text(0.012, 0.012, text, family="monospace", fontsize=fontsize,
             va="bottom", ha="left", color="#222",
             bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                       edgecolor="#bbb", alpha=0.85))


# ============================================================================
# The coil, drawn on a slice
# ============================================================================
def _draw_coil(ax, centre_vox, direction_vox, alpha: float,
               linewidth: Optional[float] = None) -> None:
    """The coil centre and its axis on one slice, in one colour.

    One colour throughout (settings.COIL_COLOR) rather than a separate hue
    for the axis: two colours on a busy field mosaic read as two different
    things, when they are one object.
    """
    width = linewidth if linewidth is not None else COIL_LINEWIDTH
    if direction_vox is not None:
        ax.plot([centre_vox[0], centre_vox[0] + direction_vox[0]],
                [centre_vox[1], centre_vox[1] + direction_vox[1]],
                color=COIL_COLOR, lw=width, alpha=alpha * 0.9,
                solid_capstyle="round")
    # The ring grows with the stroke: at double thickness on an 8 pt marker
    # the ring and the cross fill in and read as one green dot, which is not
    # a coil marker any more.
    scale = width / COIL_LINEWIDTH
    ax.plot(centre_vox[0], centre_vox[1], marker="o", markersize=14 * scale,
            markerfacecolor="none", markeredgecolor=COIL_COLOR,
            markeredgewidth=width * (COIL_MARKER_EDGEWIDTH / COIL_LINEWIDTH),
            alpha=alpha)
    ax.plot(centre_vox[0], centre_vox[1], marker="+", markersize=18 * scale,
            color=COIL_COLOR, markeredgewidth=width, alpha=alpha)


def _coil_in_voxels(affine, coil_centre, coil_normal, length_mm: float = 40.0):
    """(centre, direction) of the coil in voxel coordinates."""
    inverse = np.linalg.inv(affine)
    centre = np.asarray(coil_centre, dtype=float)
    centre_vox = (inverse @ np.append(centre, 1.0))[:3]
    direction_vox = None
    if coil_normal is not None:
        tip = centre + np.asarray(coil_normal, dtype=float) * length_mm
        direction_vox = (inverse @ np.append(tip, 1.0))[:3] - centre_vox
    return centre_vox, direction_vox


# ============================================================================
# The cortical surface, coloured by the field
# ============================================================================
def _plane_segments_valued(vox: np.ndarray, faces: np.ndarray,
                           values: np.ndarray, k: float
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """Cut the mesh on the voxel plane z = k, carrying `values` along.

    mesh_head's _plane_segments, plus the scalar: each crossing point sits a
    fraction t along one edge, so the value there is the same linear blend of
    the edge's two vertex values. Returns the segments [s, 2, 2] and one
    value per segment (the mean of its two ends).
    """
    z = vox[:, 2][faces] - k                          # [m, 3] signed distance
    side = z > 0
    crossing = np.any(side, axis=1) & ~np.all(side, axis=1)
    if not crossing.any():
        return np.empty((0, 2, 2)), np.empty((0,))
    tri = faces[crossing]
    zc = z[crossing]
    pts, vals = [], []
    for a, b in ((0, 1), (1, 2), (2, 0)):
        za, zb = zc[:, a], zc[:, b]
        cross = (za > 0) != (zb > 0)
        denominator = np.where(za - zb == 0, 1e-12, za - zb)
        t = np.where(cross, za / denominator, np.nan)
        pa, pb = vox[tri[:, a]], vox[tri[:, b]]
        pts.append(pa[:, :2] + t[:, None] * (pb[:, :2] - pa[:, :2]))
        va, vb = values[tri[:, a]], values[tri[:, b]]
        vals.append(va + t * (vb - va))
    pts = np.stack(pts, axis=1)                       # [m, 3 edges, 2]
    vals = np.stack(vals, axis=1)                     # [m, 3 edges]
    # Exactly two of the three edges cross; take those two, in edge order.
    valid = ~np.isnan(pts[:, :, 0])
    order = np.argsort(~valid, axis=1, kind="stable")[:, :2]
    segments = np.take_along_axis(pts, order[:, :, None], axis=1)
    seg_values = np.take_along_axis(vals, order, axis=1)
    keep = np.all(~np.isnan(segments[:, :, 0]), axis=1)
    return segments[keep], np.nanmean(seg_values[keep], axis=1)


def field_surface_mosaic(volume: np.ndarray, affine: np.ndarray,
                         meshes: Sequence[Tuple[np.ndarray, np.ndarray,
                                                np.ndarray]],
                         title: str, cbar_label: str,
                         n_slices: Optional[int] = None,
                         n_cols: Optional[int] = None,
                         linewidth: float = CANVAS_LINEWIDTH,
                         cmap: str = None,
                         vmin: Optional[float] = None,
                         vmax: Optional[float] = None,
                         symmetric: bool = False,
                         dpi: Optional[float] = None,
                         slice_fontsize: float = 7,
                         annotation: str = "",
                         coil_centre: Optional[Sequence[float]] = None,
                         coil_normal: Optional[Sequence[float]] = None,
                         subtitle: str = "") -> Image.Image:
    """The T1 in grey with a surface drawn on it, coloured by the field.

    `meshes` is one (vertices world-mm, faces, per-vertex values) triple per
    hemisphere -- the central surface for the cortex views, the white surface
    for the white-matter one. `symmetric` centres the colour scale on zero,
    for a signed quantity (the normal component, inward on one bank of a
    sulcus and outward on the other).

    Passing `coil_centre` draws the coil on every slice as well, fading with
    distance from it: that is the "field and coil together" figure, and it is
    the same function rather than a second one so the two cannot diverge.
    """
    cmap = cmap or FIELD_CMAP
    inv = np.linalg.inv(affine)
    in_vox = [((inv @ np.c_[v, np.ones(len(v))].T).T[:, :3], f, s)
              for v, f, s in meshes]

    all_values = np.concatenate([s for _v, _f, s in in_vox])
    finite = all_values[np.isfinite(all_values)]
    if vmin is None or vmax is None:
        if symmetric:
            top = (float(np.percentile(np.abs(finite), FIELD_PCTL[1]))
                   if finite.size else 1.0)
            vmin, vmax = -top, top
        else:
            lo, hi = (np.percentile(finite, FIELD_PCTL)
                      if finite.size else (0.0, 1.0))
            vmin, vmax = float(lo), float(hi)
    if not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(finite)) if finite.size else 0.0
        vmax = float(np.nanmax(finite)) if finite.size else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    all_z = np.concatenate([v[:, 2] for v, _f, _s in in_vox])
    lo = max(int(np.floor(all_z.min())), 0)
    hi = min(int(np.ceil(all_z.max())), volume.shape[2] - 1)
    rows, cols = mosaic_rule.layout_for_slice(volume.shape[0], volume.shape[1])
    n_slices = n_slices or rows * cols
    n_cols = n_cols or cols
    indices = np.unique(np.linspace(lo, hi, n_slices, dtype=int))

    # Dim the anatomy so the coloured cut carries the picture. Without this
    # a diverging scale's mid-range -- the near-zero field on a signed
    # quantity -- is pale on a bright T1 and simply disappears.
    bg_lo, bg_hi = _window(volume)
    bg_hi = bg_lo + (bg_hi - bg_lo) * BACKGROUND_DIM

    centre_vox = direction_vox = None
    if coil_centre is not None:
        centre_vox, direction_vox = _coil_in_voxels(affine, coil_centre,
                                                    coil_normal)

    fig, axes = _grid(len(indices), n_cols, volume.shape)
    collection = None
    for ax, idx in zip(axes, indices):
        ax.imshow(volume[:, :, idx].T, cmap="gray", vmin=bg_lo, vmax=bg_hi,
                  origin="lower", interpolation="nearest")
        for vox, faces, values in in_vox:
            segments, seg_values = _plane_segments_valued(
                vox, faces, values, float(idx))
            if len(segments):
                collection = LineCollection(segments, cmap=cmap, norm=norm,
                                            linewidths=linewidth)
                collection.set_array(seg_values)
                ax.add_collection(collection)
        if centre_vox is not None:
            offset = abs(idx - centre_vox[2])
            alpha = float(np.clip(1.0 - offset / 25.0, 0.0, 1.0))
            if alpha > 0.05:
                _draw_coil(ax, centre_vox, direction_vox, alpha,
                           linewidth=max(linewidth * 1.6, COIL_LINEWIDTH))
        ax.set_xlim(-0.5, volume.shape[0] - 0.5)
        ax.set_ylim(-0.5, volume.shape[1] - 0.5)
        ax.set_title(str(idx), fontsize=slice_fontsize, color="#888", pad=2)

    for ax in axes:
        ax.axis("off")
    second = subtitle or (f"{volume.shape[0]}×{volume.shape[1]}"
                          f"×{volume.shape[2]}")
    fig.suptitle(f"{title}\n{second}", fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    if collection is not None:
        cax = fig.add_axes((0.35, 0.032, 0.42, 0.016))
        bar = fig.colorbar(collection, cax=cax, orientation="horizontal")
        bar.set_label(cbar_label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    _annotate(fig, annotation)
    return _to_png(fig, dpi)


# ============================================================================
# A field volume, over the anatomy
# ============================================================================
def field_volume_mosaic(field: np.ndarray, background: np.ndarray,
                        title: str, cbar_label: str,
                        mask: Optional[np.ndarray] = None,
                        cmap: str = None,
                        n_slices: Optional[int] = None,
                        n_cols: Optional[int] = None,
                        dpi: Optional[float] = None,
                        slice_fontsize: float = 7,
                        annotation: str = "",
                        subtitle: str = "",
                        boundaries: Sequence[Tuple[np.ndarray, np.ndarray]] = (),
                        affine: Optional[np.ndarray] = None,
                        boundary_linewidth: float = BOUNDARY_LINEWIDTH,
                        vmin: Optional[float] = None,
                        vmax: Optional[float] = None,
                        cbar_extend: str = "neither") -> Image.Image:
    """The field drawn over the T1, inside `mask` only.

    `vmin` / `vmax` pin the colour window (default: FIELD_PCTL of the shown
    values) -- for a ratio that starts at 1, say; `cbar_extend` marks a
    pinned end on the colour bar ("max", "min", "both").

    An overlay rather than the field alone: a white-matter field floating on
    a blank page says nothing about WHERE in the head it is. Task 2's own
    conductivity mosaics are read the same way, for the same reason.

    `boundaries` is a list of (vertices world-mm, faces) meshes -- the white
    surface per hemisphere -- whose cut through each slice is drawn in
    BOUNDARY_COLOR on top of the field, so gray and white matter can be told
    apart on a map that colours both. Needs `affine` (the field grid's) to
    bring the vertices into voxels.
    """
    cmap = cmap or FIELD_CMAP
    shown = np.asarray(field, dtype=float)
    if mask is not None:
        shown = np.where(mask, shown, np.nan)
    shown = np.where(np.isfinite(shown) & (shown != 0), shown, np.nan)

    finite = shown[np.isfinite(shown)]
    auto_lo, auto_hi = ((float(np.nanpercentile(finite, FIELD_PCTL[0])),
                         float(np.nanpercentile(finite, FIELD_PCTL[1])))
                        if finite.size else (0.0, 1.0))
    vmin = auto_lo if vmin is None else float(vmin)
    vmax = auto_hi if vmax is None else float(vmax)
    if vmax <= vmin:
        vmax = vmin + 1.0

    rows, cols = mosaic_rule.layout_for_slice(background.shape[0],
                                              background.shape[1])
    n_slices = n_slices or rows * cols
    n_cols = n_cols or cols
    indices = mosaic_rule.slice_indices(
        np.where(np.isfinite(shown), shown, 0.0), n_slices)

    in_vox = []
    if boundaries:
        if affine is None:
            raise ValueError("boundaries need the field grid's affine")
        inv = np.linalg.inv(affine)
        in_vox = [((inv @ np.c_[v, np.ones(len(v))].T).T[:, :3], f)
                  for v, f in boundaries]

    bg_lo, bg_hi = _window(background)
    bg_hi = bg_lo + (bg_hi - bg_lo) * BACKGROUND_DIM
    fig, axes = _grid(len(indices), n_cols, background.shape)
    image = None
    for ax, idx in zip(axes, indices):
        ax.imshow(background[:, :, idx].T, cmap="gray", vmin=bg_lo,
                  vmax=bg_hi, origin="lower", interpolation="nearest")
        image = ax.imshow(np.ma.masked_invalid(shown[:, :, idx].T), cmap=cmap,
                          vmin=vmin, vmax=vmax, origin="lower",
                          interpolation="nearest")
        for vox, faces in in_vox:
            segments = _plane_segments(vox, faces, float(idx))
            if len(segments):
                ax.add_collection(LineCollection(
                    segments, colors=BOUNDARY_COLOR,
                    linewidths=boundary_linewidth))
        ax.set_xlim(-0.5, background.shape[0] - 0.5)
        ax.set_ylim(-0.5, background.shape[1] - 0.5)
        ax.set_title(str(idx), fontsize=slice_fontsize, color="#888", pad=2)

    for ax in axes:
        ax.axis("off")
    second = subtitle or (f"{background.shape[0]}×{background.shape[1]}"
                          f"×{background.shape[2]}")
    fig.suptitle(f"{title}\n{second}", fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    if image is not None:
        cax = fig.add_axes((0.35, 0.032, 0.42, 0.016))
        bar = fig.colorbar(image, cax=cax, orientation="horizontal",
                           extend=cbar_extend)
        bar.set_label(cbar_label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
    _annotate(fig, annotation)
    return _to_png(fig, dpi)


# ============================================================================
# The conductivity tensor's orientation, over the T1
# ============================================================================
def tensor_orientation_mosaic(principal: np.ndarray, anisotropy: np.ndarray,
                              background: np.ndarray,
                              mask: Optional[np.ndarray],
                              title: str,
                              n_slices: Optional[int] = None,
                              n_cols: Optional[int] = None,
                              dpi: Optional[float] = None,
                              slice_fontsize: float = 7,
                              annotation: str = "",
                              subtitle: str = "") -> Image.Image:
    """The tensor's principal axis, direction-encoded, over the T1.

    Colour is the (unsigned) principal eigenvector in R/A/S -- red for
    left-right, green for anterior-posterior, blue for superior-inferior,
    the convention every DTI viewer uses, spelled out in `subtitle` since
    there is no colourbar to carry it. Opacity is the anisotropy index
    itself: near an isotropic voxel the eigenvector direction is numerical
    noise, so it fades into the T1 rather than being drawn at full strength.

    This is a sanity check on the UPLOADED tensor, not on the simulation: a
    flipped axis or a mislabelled tissue mask upstream shows up here as fibre
    directions that do not look like fibres (the corpus callosum should read
    red, the corticospinal tract blue), long before it would be visible in
    the E-field result.
    """
    rgb = np.abs(np.nan_to_num(principal, nan=0.0))
    alpha = np.clip(np.nan_to_num(anisotropy, nan=0.0), 0.0, 1.0)
    if mask is not None:
        alpha = np.where(mask, alpha, 0.0)

    rows, cols = mosaic_rule.layout_for_slice(background.shape[0],
                                              background.shape[1])
    n_slices = n_slices or rows * cols
    n_cols = n_cols or cols
    indices = mosaic_rule.slice_indices(alpha, n_slices)

    bg_lo, bg_hi = _window(background)
    bg_hi = bg_lo + (bg_hi - bg_lo) * BACKGROUND_DIM
    fig, axes = _grid(len(indices), n_cols, background.shape)
    for ax, idx in zip(axes, indices):
        ax.imshow(background[:, :, idx].T, cmap="gray", vmin=bg_lo,
                  vmax=bg_hi, origin="lower", interpolation="nearest")
        slab = np.transpose(rgb[:, :, idx, :], (1, 0, 2))
        a = alpha[:, :, idx].T
        ax.imshow(np.concatenate([slab, a[..., None]], axis=-1),
                  origin="lower", interpolation="nearest")
        ax.set_title(str(idx), fontsize=slice_fontsize, color="#888", pad=2)

    for ax in axes:
        ax.axis("off")
    second = subtitle or ("principal eigenvector: R=left–right, "
                          "G=anterior–posterior, B=superior–inferior · "
                          "opacity = anisotropy index")
    fig.suptitle(f"{title}\n{second}", fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0.045, 1, 0.965))
    _annotate(fig, annotation)
    return _to_png(fig, dpi)


# ============================================================================
# Where the coil sat
# ============================================================================
# One tile, not a mosaic: figure sized for a single slice read on its own,
# plus the band above it for the two-line title and the slice label.
SLICE_FIGURE_IN = 5.0
SLICE_TITLE_IN = 0.75


def coil_position_slice(volume: np.ndarray, affine: np.ndarray,
                        coil_centre: Sequence[float],
                        coil_normal: Optional[Sequence[float]] = None,
                        title: str = "Coil position",
                        subtitle: str = "",
                        dpi: Optional[float] = None,
                        slice_fontsize: float = 7,
                        linewidth: Optional[float] = None) -> Image.Image:
    """The one axial slice through the coil centre, with the centre marked
    and the coil's axis drawn into the head.

    A TMS coil is a flat thing held against the scalp; on an axial slice it
    is one point, so what is worth showing is WHERE on the head that point
    sits and which way the coil faces -- and the slice it actually sits on
    says that by itself. The slices around it (the mosaic this replaced)
    only repeated the same marker fading out.
    """
    centre = np.asarray(coil_centre, dtype=float)
    centre_vox, direction_vox = _coil_in_voxels(affine, coil_centre,
                                                coil_normal)
    idx = int(np.clip(round(centre_vox[2]), 0, volume.shape[2] - 1))

    lo, hi = _window(volume)
    aspect = volume.shape[1] / max(volume.shape[0], 1)
    # Sized by hand: the slice at SLICE_FIGURE_IN wide, plus a fixed band for
    # the two-line title and the slice label. tight_layout over-reserves
    # for a lone axes under a suptitle and leaves a white gap between them.
    height = SLICE_FIGURE_IN * aspect + SLICE_TITLE_IN
    fig, ax = plt.subplots(figsize=(SLICE_FIGURE_IN, height))
    fig.subplots_adjust(left=0, right=1, bottom=0,
                        top=1 - SLICE_TITLE_IN / height)
    ax.imshow(volume[:, :, idx].T, cmap="gray", vmin=lo, vmax=hi,
              origin="lower", interpolation="nearest")
    _draw_coil(ax, centre_vox, direction_vox, 1.0, linewidth=linewidth)
    ax.set_xlim(-0.5, volume.shape[0] - 0.5)
    ax.set_ylim(-0.5, volume.shape[1] - 0.5)
    ax.set_title(f"axial slice {idx}", fontsize=slice_fontsize, color="#888",
                 pad=2)
    ax.axis("off")
    second = subtitle or (f"coil centre at ({centre[0]:.0f}, {centre[1]:.0f}, "
                          f"{centre[2]:.0f}) mm")
    fig.suptitle(f"{title}\n{second}", fontsize=10, y=0.995, va="top")
    return _to_png(fig, dpi)
