# ============================================================================
# tasks/isocond/preview.py — figures this task draws beyond the snapshot's
# ----------------------------------------------------------------------------
# UI-agnostic: returns PIL images or writes PNGs, never widgets. The mosaics
# are the snapshot's (vendor/isocond/pipeline/imaging.py); what lives here is
# what the snapshot does not draw.
#
#   value_hist()   the predicted conductivity inside the anisotropic tissues
#                  (WM, brainstem, thalamus) against the isotropic rest --
#                  styled like task 2's eigenvalue histogram, so the two pages
#                  are read the same way
# ============================================================================
import io
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")                    # headless server; see HANDOFF gotcha 1

import matplotlib.pyplot as plt                                     # noqa: E402
import numpy as np                                                  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator             # noqa: E402
from PIL import Image                                               # noqa: E402

DPI = 110
HIST_BINS = 200
PCTL_LO, PCTL_HI = 1.0, 99.0        # the span the drawn x-window is fitted to
XLIM_PAD = 1.15

# Anisotropic tissue in task 2's sigma_1 red, the isotropic rest in its
# sigma_3 blue: the same two ends of the same palette.
GROUP_COLORS = {"anisotropic": "#e41a1c", "isotropic": "#377eb8"}
GROUP_EDGES = {"anisotropic": "#b2182b", "isotropic": "#2166ac"}


def _to_png(fig) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=DPI, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def _deoverlap_xticklabels(fig, ax, pad_px: float = 3.0) -> None:
    """Hide minor x tick labels that would collide, measured on the rendered
    figure; decade labels are reserved first (task 2's helper)."""
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
                continue
            out.append((label.get_window_extent(rend), label))
        return sorted(out, key=lambda r: r[0].x0)

    kept = [bb for bb, _ in boxes(ax.xaxis.get_major_ticks())]
    for bb, label in boxes(ax.xaxis.get_minor_ticks()):
        if any(bb.x0 - pad_px < k.x1 and k.x0 - pad_px < bb.x1 for k in kept):
            label.set_visible(False)
        else:
            kept.append(bb)


def value_hist(volume: np.ndarray, masks: Dict[str, np.ndarray], title: str,
               unit: str = "S/m") -> Image.Image:
    """Histogram of `volume` over each mask in `masks` ({group: bool array}).

    Log bins over the p1-p99 span of every group, medians drawn and labelled,
    and every statistic in the title -- task 2's conventions, so a served
    isotropic map and a served tensor are compared the same way. Unlike task
    2's, the y axis is the VOXEL COUNT, not a fraction: the two regions here
    are one map's tissues, and how much of the head each one is matters.
    """
    volume = np.asarray(volume, dtype=np.float64)
    groups = {}
    n_dropped = 0
    for name, mask in masks.items():
        v = volume[np.asarray(mask, dtype=bool)]
        finite = v[np.isfinite(v)]
        n_dropped += int(v.size - finite.size + (finite <= 0).sum())
        groups[name] = finite[finite > 0]            # a log axis needs > 0

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    if all(v.size == 0 for v in groups.values()):
        ax.text(0.5, 0.5, "no positive value in any region",
                ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return _to_png(fig)

    p_lo = [np.percentile(v, PCTL_LO) for v in groups.values() if v.size]
    p_hi = [np.percentile(v, PCTL_HI) for v in groups.values() if v.size]
    lo, hi = min(p_lo) / XLIM_PAD, max(p_hi) * XLIM_PAD
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        lo, hi = 1e-3, 1e1
    edges = np.logspace(np.log10(lo), np.log10(hi), HIST_BINS + 1)

    for name, v in groups.items():
        if v.size == 0:
            continue
        counts, _ = np.histogram(np.clip(v, lo, hi), bins=edges)
        ax.stairs(counts, edges, fill=True, alpha=0.40,
                  color=GROUP_COLORS.get(name, "#777777"), label=name)
        ax.stairs(counts, edges, fill=False, linewidth=0.7,
                  color=GROUP_EDGES.get(name, "#444444"))

    for (name, v), y in zip(groups.items(), (0.97, 0.66, 0.35)):
        if v.size == 0:
            continue
        med = float(np.median(v))
        color = GROUP_EDGES.get(name, "#444444")
        ax.axvline(med, color=color, linestyle="--", linewidth=1.6)
        ax.text(med, y, f" {med:.3g} ", rotation=90, ha="right", va="top",
                transform=ax.get_xaxis_transform(), color=color, fontsize=8,
                bbox=dict(facecolor="white", alpha=0.45, edgecolor="none",
                          pad=1.0))

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(LogLocator(base=10.0))
    decades = np.log10(hi / lo)
    subs = () if decades > 4 else ((0.2, 0.3, 0.5) if decades > 1.5
                                   else tuple(np.arange(2, 9) * 0.1))
    ax.xaxis.set_minor_locator(LogLocator(base=10.0, subs=subs or (1.0,)))
    ax.xaxis.set_minor_formatter(
        FuncFormatter((lambda v, _: f"{v:g}") if subs else (lambda v, _: "")))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel(f"Conductivity [{unit}]")
    ax.set_ylabel("Voxel count", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)

    lines = [title]
    for name, v in groups.items():
        if v.size == 0:
            lines.append(f"{name}: no positive voxel")
            continue
        lines.append(
            f"{name} ({v.size} voxels): min–max [{v.min():.3g}, {v.max():.3g}]   "
            f"p{PCTL_LO:g}–p{PCTL_HI:g} [{np.percentile(v, PCTL_LO):.3g}, "
            f"{np.percentile(v, PCTL_HI):.3g}]   mean {v.mean():.3g}   "
            f"SD {v.std():.3g}  {unit}")
    if n_dropped:
        lines.append(f"{n_dropped} non-finite or non-positive value(s) excluded")
    ax.set_title("\n".join(lines), fontsize=8, loc="left")
    fig.tight_layout()
    _deoverlap_xticklabels(fig, ax)
    return _to_png(fig)


def save_png(image: Image.Image, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(path))
