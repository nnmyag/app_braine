# ============================================================================
# tasks/anicond/preview.py — the pictures this task shows
# ----------------------------------------------------------------------------
# UI-agnostic, like service.py: every function returns a PIL image or writes a
# PNG, never a widget, so a cli.py can save exactly what the page displays.
#
#   mosaic()            axial mosaic of a scalar volume (b0, FA, conductivity)
#   label_mosaic()      the same for an integer label map, with a legend
#   eigenvalue_hist()   sigma_1 / sigma_2 / sigma_3 inside the anisotropic
#                       tissue, the per-subject form of
#                       PROJ_DCONDNET/scripts/p2_condmodel/c6
#
# Rendered on demand from the files a run wrote, not held in memory: a finished
# job is a folder, so the canvas keeps working after the page is reloaded and
# two tabs cannot overwrite each other's preview.
# ============================================================================
import io
import math
from typing import Dict, Optional, Sequence, Tuple

import matplotlib
# Gotcha #1 of the repo: on a headless server matplotlib otherwise reaches for
# Qt through the plugins cv2 bundles and the job dies drawing its first figure.
matplotlib.use("Agg")

import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm  # noqa: E402
from matplotlib.patches import Patch                                # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator             # noqa: E402
from PIL import Image                                               # noqa: E402

from app import mosaic as mosaic_rule                               # noqa: E402

# --- mosaic geometry --------------------------------------------------------
# Inches per tile. Small on purpose: these are read at ~600 px on a page, and
# the study's own generate_axial_mosaic draws 6 in tiles at 55 pt titles, which
# takes seconds per figure and is meant for a printed A0 sheet.
TILE_IN = 2.2
DPI = 110

# --- histogram --------------------------------------------------------------
# The three colours c4/c6 give the three eigenvalues, so a served figure and a
# study figure are read together. No legend: the components are ordered
# (sigma_1 >= sigma_2 >= sigma_3), so left-to-right IS the labelling.
COMP_LABELS = (r"$\sigma_1$", r"$\sigma_2$", r"$\sigma_3$")
COMP_COLORS = ("#e41a1c", "#4daf4a", "#377eb8")
COMP_EDGES = ("#b2182b", "#1b7837", "#2166ac")
HIST_BINS = 200
PCTL_LO, PCTL_HI = 1.0, 99.0        # the span the drawn x-window is fitted to
XLIM_PAD = 1.15                     # multiplicative margin on each end


# ============================================================================
# Shared helpers
# ============================================================================
def _to_png(fig) -> Image.Image:
    """Figure -> PIL image without touching the disk."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _deoverlap_xticklabels(fig, ax, pad_px: float = 3.0) -> None:
    """Hide x tick labels that would collide, MEASURED on the rendered figure.

    c6's helper, kept because the problem is the same: which sub-decade labels
    fit depends on the axes width and how many decades are on screen, both of
    which change with the model and the case, so choosing them by hand only
    ever works for one figure. Decade labels are reserved first -- where a
    decade and a sub-decade compete, the decade is the one worth keeping.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    x_lo, x_hi = sorted(ax.get_xlim())

    def boxes(ticks):
        out = []
        for tick in ticks:
            label = tick.label1
            if not label.get_visible() or not label.get_text():
                continue
            if not (x_lo <= tick.get_loc() <= x_hi):
                continue                    # artists exist off the axis too
            out.append((label.get_window_extent(rend), label))
        return sorted(out, key=lambda r: r[0].x0)

    kept = [bb for bb, _ in boxes(ax.xaxis.get_major_ticks())]
    for bb, label in boxes(ax.xaxis.get_minor_ticks()):
        if any(bb.x0 - pad_px < k.x1 and k.x0 - pad_px < bb.x1 for k in kept):
            label.set_visible(False)
        else:
            kept.append(bb)


def _slice_indices(volume: np.ndarray, n_slices: int) -> np.ndarray:
    """Evenly spaced axial indices spanning only the slices that carry data --
    app/mosaic's rule, shared with the other tasks' mosaics."""
    return mosaic_rule.slice_indices(volume, n_slices)


def _window(volume: np.ndarray, mask: Optional[np.ndarray] = None
            ) -> Tuple[float, float]:
    """Display window from the 1st-99th percentile of the non-zero voxels.

    Percentiles rather than min/max: one hot voxel in a conductivity map (or a
    ringing artefact in a b0) otherwise pushes the whole brain into the bottom
    of the colour scale.
    """
    values = volume[mask] if mask is not None else volume
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values, (1.0, 99.0))
    if not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.min(values)), float(np.max(values))
    return float(lo), float(hi if hi > lo else lo + 1.0)


def _layout(shape, n_slices: Optional[int], n_cols: Optional[int]
            ) -> Tuple[int, int]:
    """(n_slices, n_cols) to draw: the caller's, or app/mosaic's rule for a
    slice of this volume. Slices are drawn as data[:, :, k].T, so the drawn
    width is shape[0] and the height shape[1]."""
    rows, cols = mosaic_rule.layout_for_slice(shape[0], shape[1])
    return (n_slices or rows * cols), (n_cols or cols)


def _grid(n_tiles: int, n_cols: int, shape) -> Tuple[plt.Figure, np.ndarray]:
    n_rows = math.ceil(n_tiles / n_cols)
    aspect = shape[1] / max(shape[0], 1)        # drawn transposed, so y over x
    fig, axes = plt.subplots(
        n_rows, n_cols, squeeze=False,
        figsize=(TILE_IN * n_cols, TILE_IN * aspect * n_rows))
    return fig, axes.flatten()


# ============================================================================
# Mosaics
# ============================================================================
def mosaic(volume: np.ndarray, title: str, cbar_label: str = "",
           cmap: str = "jet", n_slices: Optional[int] = None,
           n_cols: Optional[int] = None,
           vmin: Optional[float] = None, vmax: Optional[float] = None,
           mask: Optional[np.ndarray] = None,
           background: Optional[np.ndarray] = None,
           log: bool = False,
           contour: Optional[np.ndarray] = None,
           cap_top: bool = False,
           cap_bottom: bool = False,
           subtitle: Optional[str] = None,
           background_dim: float = 1.0) -> Image.Image:
    """Axial mosaic of a scalar volume, with one colour bar for the whole grid.

    Without `background`, every voxel is drawn and `mask` only narrows the
    display window. With `background` (a b0, say), the picture is an OVERLAY:
    `background` is drawn in grey and `volume` on top of it only where `mask`
    is True -- the way PROJ_DCONDNET's c4 draws a score over the b0, so a
    voxel outside the mask reads as anatomy, never as a value.

    `log` puts the colour scale on a log axis: for a map spanning decades
    (conductivity, 0.02 S/m bone to 2 S/m CSF) a linear scale shows one
    tissue and saturates the rest.

    `contour` (a bool volume) outlines its boundary in thin white on every
    slice -- the anisotropic tissue, say, so the WM/GM border is seen even
    where the two colours are close. `cap_top` / `cap_bottom` label the
    colour bar's end ticks ">= vmax" / "<= vmin", saying that the map was
    clipped there. `subtitle`
    replaces the second title line (by default the grid size).
    `background_dim` > 1 darkens the overlay's backdrop by that factor, so
    the colours on top of it stand out.
    """
    volume = np.asarray(volume, dtype=np.float64)
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {volume.shape}")

    if vmin is None or vmax is None:
        vmin, vmax = _window(volume, mask)
    if log:
        if vmin <= 0:
            positive = volume[np.isfinite(volume) & (volume > 0)]
            vmin = float(positive.min()) if positive.size else 1e-3
        norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 1.001))
        scale = dict(norm=norm)
    else:
        scale = dict(vmin=vmin, vmax=vmax)

    overlay = background is not None
    if overlay:
        if mask is None:
            raise ValueError("An overlay needs `mask`: where to draw `volume`")
        background = np.asarray(background, dtype=np.float64)
        bg_lo, bg_hi = _window(background)
        bg_hi = bg_lo + (bg_hi - bg_lo) * max(float(background_dim), 1e-6)
        shown = np.where(mask.astype(bool), volume, np.nan)
    else:
        shown = volume

    n_slices, n_cols = _layout(volume.shape, n_slices, n_cols)
    indices = _slice_indices(shown if overlay else volume, n_slices)
    fig, axes = _grid(len(indices), n_cols, volume.shape)
    image = None
    for ax, idx in zip(axes, indices):
        # data[:, :, k].T with origin='lower' -- the orientation every mosaic in
        # PROJ_DCONDNET is drawn in (core/med/lib_mri.generate_axial_mosaic).
        if overlay:
            ax.imshow(background[:, :, idx].T, cmap="gray", origin="lower",
                      vmin=bg_lo, vmax=bg_hi, alpha=0.8,
                      interpolation="nearest")
        image = ax.imshow(np.ma.masked_invalid(shown[:, :, idx].T), cmap=cmap,
                          origin="lower", alpha=0.9 if overlay else 1.0,
                          interpolation="nearest", **scale)
        if contour is not None:
            outline = np.asarray(contour[:, :, idx], dtype=float).T
            if outline.any() and not outline.all():
                ax.contour(outline, levels=[0.5], colors="white",
                           linewidths=0.4, alpha=0.85)
        ax.set_title(str(idx), fontsize=7, color="#888", pad=2)
    for ax in axes:
        ax.axis("off")

    second = (subtitle if subtitle is not None
              else f"{volume.shape[0]}×{volume.shape[1]}×{volume.shape[2]}")
    fig.suptitle(f"{title}\n{second}", fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0.055, 1, 0.965))
    if image is not None:
        cax = fig.add_axes((0.25, 0.035, 0.5, 0.018))
        bar = fig.colorbar(image, cax=cax, orientation="horizontal")
        bar.set_label(cbar_label, fontsize=8)
        bar.ax.tick_params(labelsize=7)
        if (cap_top or cap_bottom) and not log:
            # An end of the scale is a cap, not an extreme: say so on the
            # tick. Auto ticks rarely land on the ends, so they are added.
            span = vmax - vmin
            # Inner ticks within a tenth of the span of an end would collide
            # with the end label, which is the wider "<=" / ">=" one.
            inner = [float(t) for t in bar.get_ticks()
                     if vmin + 0.10 * span < t < vmax - 0.10 * span]
            ticks = [float(vmin)] + inner + [float(vmax)]
            labels = [f"{t:g}" for t in ticks]
            if cap_bottom:
                labels[0] = f"≤{vmin:.3g}"
            if cap_top:
                labels[-1] = f"≥{vmax:.3g}"
            bar.set_ticks(ticks)
            bar.set_ticklabels(labels)
    return _to_png(fig)


def label_mosaic(labels: np.ndarray, title: str,
                 names: Optional[Dict[int, str]] = None,
                 highlight: Optional[Sequence[int]] = None,
                 n_slices: Optional[int] = None,
                 n_cols: Optional[int] = None) -> Image.Image:
    """Axial mosaic of an integer label map, with a legend instead of a bar.

    A continuous colour map on tissue ids would imply that id 14 is "more" than
    id 13, and neighbouring tissues would come out in neighbouring colours; the
    labels are nominal, so each gets its own tab20 colour and its name.

    `highlight` restricts the drawing to those ids (the anisotropic-tissue
    view), which is the same picture the job masks its tensors with.
    """
    labels = np.rint(np.asarray(labels)).astype(np.int32)
    if highlight is not None:
        labels = np.where(np.isin(labels, list(highlight)), labels, 0)

    present = [int(v) for v in np.unique(labels) if int(v) != 0]
    if not present:
        present = [1]                       # empty map: keep the figure valid

    palette = plt.get_cmap("tab20")
    colors = ["#101010"] + [palette(i % 20) for i in range(len(present))]
    cmap = ListedColormap(colors)
    # One bin per id, in the order they are drawn: the ids are not contiguous
    # (a job may carry 13, 14, 15 only), so the norm maps rank, not value.
    rank = {tid: i + 1 for i, tid in enumerate(present)}
    ranked = np.zeros_like(labels)
    for tid, i in rank.items():
        ranked[labels == tid] = i
    norm = BoundaryNorm(np.arange(len(colors) + 1) - 0.5, len(colors))

    n_slices, n_cols = _layout(labels.shape, n_slices, n_cols)
    indices = _slice_indices((labels != 0).astype(float), n_slices)
    fig, axes = _grid(len(indices), n_cols, labels.shape)
    for ax, idx in zip(axes, indices):
        ax.imshow(ranked[:, :, idx].T, cmap=cmap, norm=norm, origin="lower",
                  interpolation="nearest")
        ax.set_title(str(idx), fontsize=7, color="#888", pad=2)
    for ax in axes:
        ax.axis("off")

    names = names or {}
    handles = [Patch(facecolor=colors[rank[tid]],
                     label=f"{tid} {names.get(tid, '')}".strip())
               for tid in present]
    fig.suptitle(f"{title}\n{labels.shape[0]}×{labels.shape[1]}×{labels.shape[2]}",
                 fontsize=10, y=0.995, va="top")
    fig.tight_layout(rect=(0, 0.06, 1, 0.965))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6),
               fontsize=7, frameon=False)
    return _to_png(fig)


# ============================================================================
# Eigenvalue histogram
# ----------------------------------------------------------------------------
# The per-subject form of c6_visualize_cohort_eigenvalue_distribution.py. Same
# question -- where do the three conductivity eigenvalues sit inside the
# anisotropic tissue -- asked of one brain, so c6's constant-memory machinery
# (a fixed log grid, running weighted sums, percentiles read off that grid) is
# not needed: one subject's voxels fit in memory and every statistic here is
# exact. What is kept is what makes the figure readable and comparable: log
# bins, one normalisation per component, medians drawn, statistics in the title.
# ============================================================================
def eigenvalue_hist(c_evals: np.ndarray, mask: np.ndarray, title: str,
                    eff_cond: Optional[np.ndarray] = None) -> Image.Image:
    """Histogram of sigma_1 / sigma_2 / sigma_3 over `mask`.

    Params:
        c_evals  : [X, Y, Z, 3] conductivity eigenvalues, descending
        mask     : bool [X, Y, Z] -- the anisotropic tissue
        eff_cond : optional G_eff map; its median is reported in the title
    """
    c_evals = np.asarray(c_evals, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    comps = []
    for i in range(3):
        v = c_evals[..., i][mask]
        v = v[np.isfinite(v)]
        comps.append(v[v > 0])              # a log axis has no place for <= 0

    n_dropped = sum(int((~np.isfinite(c_evals[..., i][mask])).sum()
                        + (c_evals[..., i][mask] <= 0).sum()) for i in range(3))
    if all(v.size == 0 for v in comps):
        ax.text(0.5, 0.5, "no positive eigenvalue in the anisotropic tissue",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _to_png(fig)

    # x-window from the p1-p99 span of every component, padded: a handful of
    # collapsed sigma_3 or runaway sigma_1 voxels otherwise stretch the axis
    # over decades that hold almost nothing (c6's reasoning, its numbers).
    p_lo = [np.percentile(v, PCTL_LO) for v in comps if v.size]
    p_hi = [np.percentile(v, PCTL_HI) for v in comps if v.size]
    lo, hi = min(p_lo) / XLIM_PAD, max(p_hi) * XLIM_PAD
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = 1e-6, 1e1
    edges = np.logspace(np.log10(lo), np.log10(hi), HIST_BINS + 1)

    for i, v in enumerate(comps):
        if v.size == 0:
            continue
        # Clipped into the end bins rather than dropped, so the bars still
        # account for every voxel the statistics below were computed from.
        counts, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
        counts = counts / max(counts.sum(), 1)
        ax.stairs(counts, edges, fill=True, alpha=0.40, color=COMP_COLORS[i])
        ax.stairs(counts, edges, fill=False, linewidth=0.7, color=COMP_EDGES[i])

    # Medians on top of the fills, each label at its own height: sigma_2 and
    # sigma_3 can land within a few percent of each other.
    for i, (v, y) in enumerate(zip(comps, (0.97, 0.66, 0.35))):
        if v.size == 0:
            continue
        med = float(np.median(v))
        ax.axvline(med, color=COMP_EDGES[i], linestyle="--", linewidth=1.6)
        ax.text(med, y, f" {med:.3g} ", rotation=90, ha="right", va="top",
                transform=ax.get_xaxis_transform(), color=COMP_EDGES[i],
                fontsize=8,
                bbox=dict(facecolor="white", alpha=0.45, edgecolor="none",
                          pad=1.0))

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    # The distributions often live inside one decade, where the default log
    # locator labels a single tick and the figure stops being readable.
    decades = np.log10(hi / lo)
    subs = () if decades > 4 else ((0.2, 0.3, 0.5) if decades > 1.5
                                   else tuple(np.arange(2, 9) * 0.1))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=subs or (1.0,)))
    ax.xaxis.set_minor_formatter(
        FuncFormatter((lambda v, _: f"{v:g}") if subs else (lambda v, _: "")))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Conductivity [S/m]")
    ax.set_ylabel("Fractional count\n(each of $\\sigma_1$, $\\sigma_2$, "
                  "$\\sigma_3$ sums to 1)", fontsize=9)

    lines = [f"{title} — {int(mask.sum())} voxels"]
    for i, v in enumerate(comps):
        if v.size == 0:
            lines.append(f"{COMP_LABELS[i]}: no positive voxel")
            continue
        lines.append(
            f"{COMP_LABELS[i]}: min–max [{v.min():.3g}, {v.max():.3g}]   "
            f"p{PCTL_LO:g}–p{PCTL_HI:g} [{np.percentile(v, PCTL_LO):.3g}, "
            f"{np.percentile(v, PCTL_HI):.3g}]   mean {v.mean():.3g}   "
            f"SD {v.std():.3g}  S/m")
    if eff_cond is not None:
        g = np.asarray(eff_cond)[mask]
        g = g[np.isfinite(g)]
        if g.size:
            lines.append(f"median $G_{{eff}}$ {np.median(g):.3g} S/m")
    if n_dropped:
        lines.append(f"{n_dropped} non-finite or non-positive eigenvalue(s) "
                     f"excluded")
    ax.set_title("\n".join(lines), fontsize=8, loc="left")
    fig.tight_layout()
    # Last, once the layout has settled: the label positions this measures are
    # only the final ones after the axes box stops moving.
    _deoverlap_xticklabels(fig, ax)
    return _to_png(fig)


def save_png(image: Image.Image, path) -> None:
    """Write a rendered figure where the run keeps its files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
