# ============================================================================
# tasks/tract_efield/preview.py — the pictures this task shows
# ----------------------------------------------------------------------------
# UI-agnostic: returns PIL images, never widgets. One drawing routine,
# tract_mosaic(): axial slices of a darkened T1, the white surface's cut on
# each slice as a thin grey line (the hot spot in white), and the selected
# streamlines where they pass through a slab around the slice, each
# segment coloured by its direction and fading with its distance from the
# slice plane -- the only colour on the picture, so a hue can only mean a
# direction. No field map: that is mesh_sim's white-surface mosaic. The canvas and
# the saved PNG are the same call with different `dpi` / line widths.
#
# Everything geometric is done in VOXEL coordinates of the drawn grid: the
# streamlines and the surface arrive in RAS mm and are put through the
# grid's inverse affine once, up front. The direction colour is taken in mm
# (before that transform), so it means the same on any grid.
# ============================================================================
import textwrap
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")                    # headless server, see HANDOFF gotcha 1

import numpy as np                                                   # noqa: E402
from matplotlib.cm import ScalarMappable                             # noqa: E402
from matplotlib.collections import LineCollection                    # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize   # noqa: E402
from PIL import Image                                                # noqa: E402

from app import mosaic as mosaic_rule                               # noqa: E402
from tasks.mesh_head.preview import (_grid, _to_png, _window,        # noqa: E402
                                     anatomy_mosaic, load_canonical,
                                     save_png)
from tasks.mesh_sim.preview import (_plane_segments_valued,          # noqa: E402
                                    resample_to_grid, sample_volume_at)
from tasks.tract_efield import settings as task_settings            # noqa: E402

__all__ = ["anatomy_mosaic", "load_canonical", "save_png", "tract_mosaic",
           "tract_pair_view", "resample_to_grid", "sample_volume_at",
           "to_voxels", "segment_index", "direction_colors", "ACTIVATION_CMAP"]

# Diverging, dark in the middle (settings.ACTIVATION_RAMP): blue -> grey -> red.
ACTIVATION_CMAP = LinearSegmentedColormap.from_list(
    "activation", list(task_settings.ACTIVATION_RAMP))


# Characters per subtitle line, about the width of the six-column mosaic at
# the title's font size.
SUBTITLE_WRAP = 150
LEGEND_WRAP = 165                # the legend's monospace font is smaller
LEGEND_LINES = 3                 # strip height reserved, in lines
# Height of one 10 pt title line with its leading, inches; the same for
# one 7.5 pt monospace legend line.
TITLE_LINE_IN = 0.16
LEGEND_LINE_IN = 0.13


# ============================================================================
# Streamlines as one flat array
# ----------------------------------------------------------------------------
# A whole selection is drawn as ONE LineCollection per slice: all the points
# stacked, plus an index of consecutive-point pairs that never spans two
# streamlines. Slicing is then a boolean mask over that index.
# ============================================================================
def segment_index(lengths: Sequence[int]) -> np.ndarray:
    """[M, 2] indices (a, b) of every consecutive point pair, given the
    streamline lengths in stacking order."""
    lengths = np.asarray(lengths, dtype=np.int64)
    if lengths.size == 0 or lengths.max() < 2:
        return np.empty((0, 2), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    a = np.concatenate([np.arange(s, s + n - 1)
                        for s, n in zip(starts, lengths) if n > 1])
    return np.stack([a, a + 1], axis=1)


def to_voxels(affine: np.ndarray, points_mm: np.ndarray) -> np.ndarray:
    inverse = np.linalg.inv(affine)
    return (inverse @ np.c_[points_mm, np.ones(len(points_mm))].T).T[:, :3]


def direction_colors(points_mm: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    """[M, 3] RGB per segment: the unsigned unit direction in RAS mm, the
    DTI viewers' encoding (R = left-right, G = anterior-posterior,
    B = superior-inferior)."""
    if pairs.size == 0:
        return np.empty((0, 3))
    d = points_mm[pairs[:, 1]] - points_mm[pairs[:, 0]]
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    return np.abs(d / np.where(norm == 0, 1.0, norm))


# ============================================================================
# The mosaic
# ============================================================================
def tract_mosaic(background: np.ndarray, affine: np.ndarray,
                 streamlines_mm: Sequence[np.ndarray],
                 surfaces: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                 title: str,
                 threshold: float,
                 hops: Optional[Sequence[int]] = None,
                 slab_mm: float = task_settings.SLAB_MM,
                 n_slices: Optional[int] = None,
                 n_cols: Optional[int] = None,
                 slices: Optional[Sequence[int]] = None,
                 tile_in: Optional[float] = None,
                 linewidth: float = task_settings.TRACT_LINEWIDTH_CANVAS,
                 surface_linewidth: float = task_settings.SURFACE_LINEWIDTH_CANVAS,
                 dpi: Optional[float] = None,
                 slice_fontsize: float = 7,
                 legend: str = "",
                 subtitle: str = "",
                 point_values: Optional[Sequence[np.ndarray]] = None,
                 value_cmap=task_settings.FIELD_CMAP,
                 value_range: Optional[Tuple[float, float]] = None,
                 endpoints_mm: Optional[np.ndarray] = None,
                 endpoint_values: Optional[np.ndarray] = None
                 ) -> Image.Image:
    """Axial slices spanning the streamlines: the darkened T1, the white
    surface's cut as a thin neutral line (`surfaces`: (vertices mm, faces,
    |E|)), the hot spot -- |E| at or above `threshold` -- as the same line
    in white and a little thicker, and the streamlines' in-slab segments
    on top, fading with distance from the slice plane. `hops`, one per
    streamline, thins the pieces reached by following a break. `legend`
    goes in a strip UNDER the tiles, where it covers nothing.

    Segment colour: by direction (the default), or -- given `point_values`,
    one array per streamline in the same order -- by the value at the
    segment through `value_cmap` (a name or a Colormap) over
    `value_range`. `endpoints_mm` / `endpoint_values` add a dot at each
    given point, coloured on the same scale, on the slices it falls in
    (the terminal field at fibre ends). No colour bar either way: the
    pictures are meant to be flipped between, so their geometry is
    identical, and the value window is stated in `legend`.

    `slices` names the slice indices to draw outright (tract_simplify draws one,
    the middle), in place of `n_slices` spread over the streamlines' span;
    `tile_in` is each tile's width in inches (the shared default when
    None), for a single tile that should fill the canvas.
    """
    lengths = [len(s) for s in streamlines_mm]
    if lengths:
        points_mm = np.vstack(streamlines_mm)
        points = to_voxels(affine, points_mm)
    else:
        points_mm = points = np.empty((0, 3))
    pairs = segment_index(lengths)
    mappable = None
    if point_values is not None and pairs.size:
        values = np.concatenate([np.asarray(v, dtype=float)
                                 for v in point_values])
        if values.shape[0] != points.shape[0]:
            raise ValueError("point_values must have one value per point")
        lo, hi = value_range or (float(np.nanmin(values)),
                                 float(np.nanmax(values)))
        if hi <= lo:
            hi = lo + 1.0
        mappable = ScalarMappable(norm=Normalize(lo, hi, clip=True),
                                  cmap=value_cmap)
        colors = mappable.to_rgba(
            np.nan_to_num(0.5 * (values[pairs[:, 0]] + values[pairs[:, 1]]),
                          nan=lo))[:, :3]
    else:
        colors = direction_colors(points_mm, pairs)
    ends = ends_rgb = None
    if (endpoints_mm is not None and endpoint_values is not None
            and mappable is not None and len(endpoints_mm)):
        ends = to_voxels(affine, np.asarray(endpoints_mm, dtype=float))
        ends_rgb = mappable.to_rgba(np.nan_to_num(
            np.asarray(endpoint_values, dtype=float), nan=lo))
    if hops is not None and len(lengths) and pairs.size:
        per_point = np.repeat(np.asarray(hops) >= 1, lengths)
        widths = np.where(per_point[pairs[:, 0]],
                          linewidth * task_settings.FOLLOWED_LINEWIDTH_FACTOR,
                          linewidth)
    else:
        widths = np.full(len(pairs), linewidth)
    in_vox = [(to_voxels(affine, v), f, s) for v, f, s in surfaces]

    dz = float(np.linalg.norm(affine[:3, 2])) or 1.0
    half = 0.5 * slab_mm / dz

    if len(points):
        lo, hi = np.percentile(points[:, 2], (1.0, 99.0))
        lo = int(np.clip(np.floor(lo), 0, background.shape[2] - 1))
        hi = int(np.clip(np.ceil(hi), 0, background.shape[2] - 1))
    else:
        lo, hi = mosaic_rule.occupied_span(background)
    rows, cols = mosaic_rule.layout_for_slice(background.shape[0],
                                              background.shape[1])
    n_slices = n_slices or rows * cols
    n_cols = n_cols or cols
    if slices is not None:
        indices = np.unique(np.clip(np.asarray(list(slices), dtype=int), 0,
                                    background.shape[2] - 1))
        n_cols = min(n_cols, len(indices)) or 1
    else:
        indices = np.unique(np.linspace(lo, hi, n_slices, dtype=int))

    bg_lo, bg_hi = _window(background)
    bg_hi = bg_lo + (bg_hi - bg_lo) * task_settings.BACKGROUND_DIM
    fig, axes = _grid(len(indices), n_cols, background.shape,
                      **({"tile_in": tile_in} if tile_in else {}))
    for ax, idx in zip(axes, indices):
        ax.imshow(background[:, :, idx].T, cmap="gray", vmin=bg_lo,
                  vmax=bg_hi, origin="lower", interpolation="nearest")
        for vox, faces, vals in in_vox:
            segments, seg_values = _plane_segments_valued(vox, faces, vals,
                                                          float(idx))
            if not len(segments):
                continue
            hot = seg_values >= threshold
            ax.add_collection(LineCollection(
                segments[~hot], colors=task_settings.SURFACE_COLOR,
                linewidths=surface_linewidth))
            if hot.any():
                ax.add_collection(LineCollection(
                    segments[hot], colors=task_settings.HOTSPOT_COLOR,
                    linewidths=surface_linewidth
                    * task_settings.HOTSPOT_LINEWIDTH_FACTOR,
                    capstyle="round"))
        if pairs.size:
            z = points[:, 2]
            depth = 0.5 * (np.abs(z[pairs[:, 0]] - idx)
                           + np.abs(z[pairs[:, 1]] - idx))
            keep = ((np.abs(z[pairs[:, 0]] - idx) <= half)
                    & (np.abs(z[pairs[:, 1]] - idx) <= half))
            if keep.any():
                segments = np.stack([points[pairs[keep, 0], :2],
                                     points[pairs[keep, 1], :2]], axis=1)
                fade = 1.0 - np.clip(depth[keep] / half, 0.0, 1.0)
                alpha = (task_settings.TRACT_ALPHA_FAR
                         + fade * (task_settings.TRACT_ALPHA_NEAR
                                   - task_settings.TRACT_ALPHA_FAR))
                rgba = np.c_[colors[keep], alpha]
                # Near segments drawn last, so a fibre on the plane is not
                # hidden under a faded one from the slab's edge.
                order = np.argsort(alpha)
                ax.add_collection(LineCollection(
                    segments[order], colors=rgba[order],
                    linewidths=widths[keep][order], capstyle="round",
                    joinstyle="round"))
        if ends is not None:
            near = np.abs(ends[:, 2] - idx) <= half
            if near.any():
                ax.scatter(ends[near, 0], ends[near, 1], s=linewidth * 3.5,
                           c=ends_rgb[near], marker="o", linewidths=0,
                           zorder=5)
        ax.set_xlim(-0.5, background.shape[0] - 0.5)
        ax.set_ylim(-0.5, background.shape[1] - 0.5)
        ax.set_title(str(idx), fontsize=slice_fontsize, color="#888", pad=2)

    for ax in axes:
        ax.axis("off")
    second = subtitle or (f"{background.shape[0]}×{background.shape[1]}"
                          f"×{background.shape[2]}")
    # The subtitle is wrapped to the mosaic's own width: a long single line
    # would widen the saved picture's tight bounding box far beyond the
    # tiles and leave them small in the middle of a wide white page.
    second = "\n".join(textwrap.wrap(second, SUBTITLE_WRAP)) or second
    n_lines = 2 + second.count("\n")
    fig.suptitle(f"{title}\n{second}", fontsize=10, y=0.995, va="top")
    # The legend is wrapped like the subtitle, for the same reason.
    if legend:
        legend = "\n".join(
            line for block in legend.split("\n")
            for line in (textwrap.wrap(block, LEGEND_WRAP) or [""]))
    # Title above, legend below: each reserved by its own height, so
    # neither overlaps the tiles.
    height_in = fig.get_size_inches()[1]
    title_in = n_lines * TITLE_LINE_IN
    # A fixed strip whatever the legend's length (up to LEGEND_LINES), so
    # the mosaics of one run keep the same tile geometry and flip cleanly.
    legend_lines = max(legend.count("\n") + 1, LEGEND_LINES) if legend else 0
    legend_in = legend_lines * LEGEND_LINE_IN + (0.08 if legend else 0.0)
    fig.tight_layout(rect=(0, legend_in / height_in, 1,
                           1 - title_in / height_in))
    if legend:
        fig.text(0.5, 0.01, legend, family="monospace", fontsize=7.5,
                 ha="center", va="bottom", color="#222")
    return _to_png(fig, dpi)


def tract_pair_view(background: np.ndarray, affine: np.ndarray,
                    left_streamlines_mm: Sequence[np.ndarray],
                    right_streamlines_mm: Sequence[np.ndarray],
                    slice_index: int, title: str,
                    slab_mm: float = task_settings.SLAB_MM,
                    tile_in: Optional[float] = None,
                    linewidth: float = task_settings.TRACT_LINEWIDTH_CANVAS,
                    dpi: Optional[float] = None) -> Image.Image:
    """One figure, two axial tiles at the SAME `slice_index`:
    `left_streamlines_mm` on the left, `right_streamlines_mm` on the
    right, each coloured by direction and fading with distance from the
    slice plane -- the same look as one `tract_mosaic` tile, without its
    per-tile slice number or a separate legend strip (both belong in
    `title`, composed by the caller and printed once for the whole figure).

    Not built by calling `tract_mosaic` twice: the two sides need two
    different streamline sets sharing one figure and one title, which
    `tract_mosaic`'s one-streamlines-array, one-title-per-call contract
    does not fit. Used by task 6 (tract_simplify) for its original-vs-revised
    canvas.
    """
    dz = float(np.linalg.norm(affine[:3, 2])) or 1.0
    half = 0.5 * slab_mm / dz
    bg_lo, bg_hi = _window(background)
    bg_hi = bg_lo + (bg_hi - bg_lo) * task_settings.BACKGROUND_DIM
    fig, axes = _grid(2, 2, background.shape,
                      **({"tile_in": tile_in} if tile_in else {}))
    for ax, streamlines_mm in zip(axes, (left_streamlines_mm,
                                         right_streamlines_mm)):
        lengths = [len(s) for s in streamlines_mm]
        if lengths:
            points_mm = np.vstack(streamlines_mm)
            points = to_voxels(affine, points_mm)
        else:
            points_mm = points = np.empty((0, 3))
        pairs = segment_index(lengths)
        colors = direction_colors(points_mm, pairs)
        ax.imshow(background[:, :, slice_index].T, cmap="gray", vmin=bg_lo,
                  vmax=bg_hi, origin="lower", interpolation="nearest")
        if pairs.size:
            z = points[:, 2]
            depth = 0.5 * (np.abs(z[pairs[:, 0]] - slice_index)
                           + np.abs(z[pairs[:, 1]] - slice_index))
            keep = ((np.abs(z[pairs[:, 0]] - slice_index) <= half)
                    & (np.abs(z[pairs[:, 1]] - slice_index) <= half))
            if keep.any():
                segments = np.stack([points[pairs[keep, 0], :2],
                                     points[pairs[keep, 1], :2]], axis=1)
                fade = 1.0 - np.clip(depth[keep] / half, 0.0, 1.0)
                alpha = (task_settings.TRACT_ALPHA_FAR
                         + fade * (task_settings.TRACT_ALPHA_NEAR
                                   - task_settings.TRACT_ALPHA_FAR))
                rgba = np.c_[colors[keep], alpha]
                order = np.argsort(alpha)          # near segments drawn last
                ax.add_collection(LineCollection(
                    segments[order], colors=rgba[order],
                    linewidths=linewidth, capstyle="round",
                    joinstyle="round"))
        ax.set_xlim(-0.5, background.shape[0] - 0.5)
        ax.set_ylim(-0.5, background.shape[1] - 0.5)
        ax.axis("off")

    n_lines = title.count("\n") + 1
    fig.suptitle(title, fontsize=10, y=0.995, va="top")
    height_in = fig.get_size_inches()[1]
    title_in = n_lines * TITLE_LINE_IN
    fig.tight_layout(rect=(0, 0, 1, 1 - title_in / height_in))
    return _to_png(fig, dpi)
