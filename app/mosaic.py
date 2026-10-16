# ============================================================================
# app/mosaic.py — the one rule for a review mosaic's grid
# ----------------------------------------------------------------------------
# Every task draws axial mosaics, each with its own drawing code (task 1 uses
# the vendored renderer, tasks 2 and 3 their own preview modules), but they
# should all tile the same way, so the rule for HOW MANY rows and columns
# lives here and nowhere else. It depends only on the shape of one slice as
# drawn -- pixels, not millimetres, because the renderers draw square pixels.
#
#     slice taller than wide (portrait, e.g. a sagittal-first T1 seen
#     axially: 196 x 301)                         -> 3 rows x 6 columns
#     slice wider than tall (landscape)           -> 4 rows x 5 columns
#     roughly square (within SQUARE_TOLERANCE)    -> 3 rows x 5 columns
#
# Portrait tiles are narrow, so more of them fit across a page; landscape
# tiles are wide, so fewer do and an extra row makes up the count. A square
# slice sits between the two.
# ============================================================================
from typing import Tuple

import numpy as np

# A slice whose width is within this fraction of its height counts as square:
# a 128 x 128 b0 obviously, but also 256 x 240 -- the eye does not see 6 %.
SQUARE_TOLERANCE = 0.10

PORTRAIT_LAYOUT = (3, 6)
LANDSCAPE_LAYOUT = (4, 5)
SQUARE_LAYOUT = (3, 5)


def layout_for_slice(width: int, height: int) -> Tuple[int, int]:
    """(rows, cols) for slices drawn `width` pixels wide, `height` high."""
    width, height = max(int(width), 1), max(int(height), 1)
    ratio = width / height
    if ratio < 1.0 - SQUARE_TOLERANCE:
        return PORTRAIT_LAYOUT
    if ratio > 1.0 + SQUARE_TOLERANCE:
        return LANDSCAPE_LAYOUT
    return SQUARE_LAYOUT


def n_tiles(layout: Tuple[int, int]) -> int:
    return layout[0] * layout[1]


# ============================================================================
# Which slices to draw
# ----------------------------------------------------------------------------
# A grid spans the slices that carry anatomy, not the whole array: a raw T1
# has noise in every slice and a DWI has empty ones top and bottom, and either
# way spreading the tiles over the full extent spends a row on black.
# ============================================================================
# A voxel counts as signal above this fraction of the display window (1st to
# 99th percentile of the non-zero voxels); a slice counts as occupied when
# more than OCCUPIED_FRACTION of its voxels are signal. Both are small: the
# point is to skip noise-only slices, not to crop the neck.
SIGNAL_LEVEL = 0.05
OCCUPIED_FRACTION = 0.002


def occupied_span(volume: np.ndarray, axis: int = 2) -> Tuple[int, int]:
    """(first, last) index along `axis` of the slices that carry signal; the
    whole extent if none qualifies."""
    volume = np.asarray(volume, dtype=np.float64)
    finite = np.where(np.isfinite(volume), volume, 0.0)
    values = finite[finite != 0]
    n = volume.shape[axis]
    if values.size:
        lo, hi = np.percentile(values, (1.0, 99.0))
        level = lo + SIGNAL_LEVEL * max(hi - lo, 0.0)
        signal = np.moveaxis(finite > level, axis, 0).reshape(n, -1)
        occupied = np.flatnonzero(signal.mean(axis=1) > OCCUPIED_FRACTION)
        if occupied.size:
            return int(occupied[0]), int(occupied[-1])
    return 0, n - 1


def slice_indices(volume: np.ndarray, n_slices: int, axis: int = 2) -> np.ndarray:
    """`n_slices` evenly spaced indices along `axis`, spanning only the
    occupied slices. Unique, so a thin volume yields fewer tiles rather than
    repeats."""
    lo, hi = occupied_span(volume, axis)
    return np.unique(np.linspace(lo, hi, n_slices, dtype=int))
