# -*- coding: utf-8 -*-
"""
Tract reference: what the tracts say about fibre geometry, per voxel.
    build_voxel_index(streamlines, wm_mask)      : WM voxel -> tracts through it
    structure_tensor(center, sids, streamlines,) : orientation tensor T of one window
    window_geometry(window, win_tract_len_minlim): W -> (h, radius, min traversal)
    compute_tract_reference(streamlines, wm_mask): the per-scan (e1, c_x, valid) triple

    The tract reference is TRACT GEOMETRY: it says where the fibres run and how
    coherently. Two consumers need it:
        cond_models/tractbased_[g|e].py  turn the window's T into a conductivity tensor
        core/cond/model_scoring.py scores ANY model against e1 / c_x / valid

Depends on: numpy + core.general only.
Coordinates: streamlines are in VOX space (core.cond.lib_cond.load_streamlines).
"""
import numpy as np
from core import general as gnr

# =============================
# Window geometry
# =============================
def window_geometry(window=None, win_tract_len_minlim=None):
    """Derive the window geometry from the edge length W.

    Returns (window, h, radius, win_tract_len_minlim): tracts are clipped to the
    ball of `radius` = h + 0.5 centred on the voxel, so the ball DIAMETER equals W.
    """
    w = int(window if window is not None else gnr.tract_window)
    if w < 1 or w % 2 == 0:
        raise ValueError("window must be a positive odd integer (e.g. 1, 3, 5)")
    h = (w - 1) // 2
    radius = float(h) + 0.5
    minlim = (float(win_tract_len_minlim) if win_tract_len_minlim is not None
              else max(1.0, float(h)))
    return w, h, radius, minlim


# =============================
# Indexing: WM voxel (i, j, k) -> streamline indices passing it
# =============================
def build_voxel_index(streamlines, wm_mask):
    """Return voxel2sl, mapping a WM (i, j, k) tuple to streamline ids that round into it.
    Selection is strict-through-center (a point must round into x); 
    the spherical in-window clipping is applied later on the sub-voxel points.
    """
    voxel2sl = {}
    for sid, sl in enumerate(streamlines):
        ivox = np.rint(np.asarray(sl)).astype(np.int64)
        visited = {tuple(v) for v in ivox}   # unique voxels (avoid double count)
        for v in visited:
            i, j, k = v
            if (0 <= i < wm_mask.shape[0] and 0 <= j < wm_mask.shape[1]
                    and 0 <= k < wm_mask.shape[2] and wm_mask[i, j, k]):
                voxel2sl.setdefault(v, []).append(sid)
    return voxel2sl


# =============================
# Orientation structure tensor for one window
# =============================
def structure_tensor(center, sids, streamlines, radius, min_tangent_tracts,
                     win_tract_len_minlim, eps=1e-12):
    """Accumulate the length-weighted scatter matrix T for the spherical window. 
    Returns (T, n_used) or (None, n_used).

    `sids` (from voxel2sl[center]) are clipped to the ball before their tangent
    is taken. A spherical window is symmetric, so the in-window arc length 
    depends only on a tract's impact, not its direction. Returns None (the model caller 
    falls back to volume-normalized;  the reference caller marks the voxel invalid) if 
    traversals are shorter than win_tract_len_minlim or fewer than min_tangent_tracts.
    """
    T = np.zeros((3, 3), dtype=np.float64)   # tensor accumulator
    Lsum = 0.0
    n_used = 0                               # count of in-window valid tracts
    c = np.asarray(center, dtype=np.float64)
    r2 = radius ** 2
    for sid in sids:
        pts_all = np.asarray(streamlines[sid], dtype=np.float64)
        sel = np.sum((pts_all - c) ** 2, axis=1) <= r2
        if np.count_nonzero(sel) < 2:        # nearly all points outside window
            continue
        pts = pts_all[sel]
        d = pts[-1] - pts[0]
        norm = np.linalg.norm(d)
        if norm < eps:
            continue
        # Unit tangent keeps T a true orientation structure tensor (trace 1);
        # reliability is the length WEIGHT l below.
        t = d / norm
        l = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        if l < win_tract_len_minlim:
            continue
        T += l * np.outer(t, t)
        Lsum += l
        n_used += 1
    if n_used < min_tangent_tracts or Lsum < eps:
        return None, n_used
    T /= Lsum
    tr = np.trace(T)                         # keep trace exactly 1
    if tr > eps:
        T /= tr
    return T, n_used


# =============================
# The per-scan tract reference
# =============================
def compute_tract_reference(streamlines, wm_mask, window=None,
                            min_center_tracts=None, min_tangent_tracts=None,
                            win_tract_len_minlim=None, eps=1e-12):
    """
    Build the per-scan TRACT REFERENCE from whole-brain tracts: the direction field
    e1, the coherence weight c_x, and the validity mask.

    Params
        streamlines : list of (P,3); Whole-brain tracts in VOX space
        wm_mask : (X,Y,Z); WM mask
        window : Window edge length W (voxels); ball diameter = W.
        min_center_tracts : Min tracts through the voxel centre (pre-clipping).
        min_tangent_tracts : Min tracts contributing an in-window tangent. 
        win_tract_len_minlim : Min in-window traversal length.

    Returns
        tract_dirs : (X,Y,Z,3); Unit e1 per WM voxel, zeros where undefined
        coh_map : (X,Y,Z); Tract coherence c = lambda_1 - lambda_3 in [0,1], 0 where undefined
        tract_valid : (X,Y,Z) bool; Voxels backed by enough local tracts for a
                      trustworthy T -- at least min_center_tracts through the centre AND
                      a usable T from them (sparse or disrupted windows are False).
    """
    _w, _h, radius, minlim = window_geometry(window, win_tract_len_minlim)
    n_center = int(min_center_tracts if min_center_tracts is not None
                   else gnr.tract_min_center_tracts)
    n_tangent = int(min_tangent_tracts if min_tangent_tracts is not None
                    else gnr.tract_min_tangent_tracts)

    voxel2sl = build_voxel_index(streamlines, wm_mask)
    tract_dirs = np.zeros(wm_mask.shape + (3,), dtype=np.float32)
    coh_map = np.zeros(wm_mask.shape, dtype=np.float32)
    tract_valid = np.zeros(wm_mask.shape, dtype=bool)

    for (i, j, k) in np.argwhere(wm_mask):
        sids = voxel2sl.get((i, j, k), ())
        if len(sids) < n_center:                  # sparse window -> no reference
            continue

        T, _ = structure_tensor((i, j, k), sids, streamlines,
                                radius=radius, min_tangent_tracts=n_tangent,
                                win_tract_len_minlim=minlim, eps=eps)
        if T is None:
            continue

        # eigendecomposition (ascending) -> reorder to descending lambda_1..3
        evals, evecs = np.linalg.eigh(T)
        order = np.argsort(evals)[::-1]
        lam = np.clip(evals[order], 0.0, None)

        tract_dirs[i, j, k] = evecs[:, order[0]]
        coh_map[i, j, k] = lam[0] - lam[2]
        tract_valid[i, j, k] = True

    return tract_dirs, coh_map, tract_valid
