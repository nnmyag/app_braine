# -*- coding: utf-8 -*-
"""
Tract-based conductivity model: shared implementation (TBg / TBe).

`TractBasedModelBase` holds everything the two tract-based variants have in
common -- window geometry, the per-WM-voxel loop, the volume-normalized
fallback, tensor reassembly and the colored-tract export. The ONLY thing a
variant supplies is `_eigen_conductivity`, the step that turns one window's
structure-tensor spectrum (lambda_i, coherence c) into the three conductivity
eigenvalues sigma_i:

    cond_models/tractbased_g.py -> TBg, sigma calibrated so the GEOMETRIC MEAN
                                   of sigma_i equals sigma_iso
    cond_models/tractbased_e.py -> TBe, sigma calibrated so the SPHERICAL AVERAGE
                                   of the effective conductivity equals sigma_iso

The base lives here, not in cond_models/, on purpose: `lib_cond.load_cond_model`
scans cond_models/*.py and requires exactly ONE ConductivityModel subclass
DEFINED per file, so an abstract base sitting next to the variants would be
picked up as a model of its own.

Inputs:
    d_evals    : [X, Y, Z, 3]      diffusion eigenvalues 
    d_evecs    : [X, Y, Z, 3, 3]   diffusion eigenvectors
    segs       : [X, Y, Z]         tissue segmentation
    streamlines: list of (P, 3)    whole-brain tracts in VOX space

Outputs:
    C       : [X, Y, Z, 3, 3]  conductivity tensor
    c_evals : [X, Y, Z, 3]     eigenvalues (descending; [...,0]=along)
    c_evecs : [X, Y, Z, 3, 3]  eigenvectors ([..., i, k] = i-th comp. of k-th evec)

Optionally exports a VTK of WM tracts colored by the anisotropy ratio
sigma_1/sigma_3 (= r, in [1, r_max]; 1 = isotropic).

VENDORED, TRIMMED (see vendor/VERSION.md): the original also inherited
`TractBasedDebugMixin` (core/cond/lib_tractbased_debug.py), which supplies the
per-voxel inspection figures and the tri-planar projection renderer used by c1b
and the DL dataset export. None of it runs in a served job, and it pulls in
lib_cond_viz plus a DirectMappingModel import, so the mixin is dropped and the
class inherits ConductivityModel alone. Nothing else changed: __call__, the
window handling, the fallback and _export_colored_tracts are the original.
"""

import time
import numpy as np
from typing import Tuple, Optional, List

from cond_models.base import ConductivityModel
from cond_models.volumenormalized import VolumeNormalizedModel
from core.cond import lib_tractref as ltref
from core import general as gnr
from core.med import lib_tract as ltract


def eff_cond_spherical_mean(s_para, s_perp, eps=1e-12):
    """Spherical average of the effective conductivity. 
    
        sigma_eff(alpha) = sqrt((sigma_para cos a)^2 + (sigma_perp sin a)^2)
        K = (1 / 4pi) * Integral sigma_eff dOmega        (doi:10.1088/1361-6560/adfe31)

    Evaluated in closed form. With u = cos(alpha),
        K = Int_0^1 sqrt(sigma_perp^2 + (sigma_para^2 - sigma_perp^2) u^2) du
          = sigma_para/2 + (sigma_perp^2 / 2a) * ln((sigma_para + a) / sigma_perp),
        with a = sqrt(sigma_para^2 - sigma_perp^2)

    Three regimes, all handled:
      - prolate (sigma_para > sigma_perp, the physical case here): the log form above.
      - a -> 0 (isotropic): a removable singularity, so the limit K -> sigma is
        substituted rather than computed.
      - oblate (sigma_para < sigma_perp): a is imaginary and the same integral
        continues as arcsin((sqrt(sigma_perp^2 - sigma_para^2)) / sigma_perp).
        Not reachable from the tract-based models (their s_1 >= mean(s_2, s_3)),
        but the branch keeps the helper valid for any tensor handed to it.

    Params:
        s_para, s_perp : scalars or broadcastable arrays; along- / cross-fiber
                         conductivity. Must be positive.
        eps            : relative cutoff selecting the degenerate branch.
    Returns: array (or 0-d array) of the same broadcast shape.
    """
    sp = np.asarray(s_para, dtype=np.float64)
    sq = np.asarray(s_perp, dtype=np.float64)

    d2 = sp ** 2 - sq ** 2
    a = np.sqrt(np.abs(d2))
    degenerate = a <= eps * np.maximum(np.abs(sp), eps)

    # Keep every branch finite: the guarded values are discarded by np.where.
    a_safe = np.where(degenerate, 1.0, a)
    sq_safe = np.where(sq > eps, sq, 1.0)

    # ln((sigma_para + a) / sigma_perp) == arcsinh(a / sigma_perp) when
    # sigma_para = sqrt(a^2 + sigma_perp^2); arcsinh stays accurate for small a.
    prolate = 0.5 * sp + (sq ** 2 / (2.0 * a_safe)) * np.arcsinh(a_safe / sq_safe)
    oblate = 0.5 * sp + (sq ** 2 / (2.0 * a_safe)) * np.arcsin(
        np.clip(a_safe / sq_safe, -1.0, 1.0))

    return np.where(degenerate, 0.5 * (sp + sq),
                    np.where(d2 > 0.0, prolate, oblate))


class TractBasedModelBase(ConductivityModel):
    """Shared tract-based model. """
    name: str = "tractbased_base"
    short_name: str = "TB"

    def __init__(
        self,
        dict_cond: dict = None,
        window: int = None,
        r_max: float = 6.0,
        wm_id: List[int] = None,
        min_center_tracts: int = None,
        min_tangent_tracts: int = None,
        win_tract_len_minlim: float = None,
        eps: float = 1e-12,
        coh_eps: float = 1e-6,
    ):
        """
        Params:
            dict_cond      : Mapping tissue_id -> isotropic conductivity
            window         : Window edge length W (voxels); W in {1, 3, 5}. 
            r_max          : Upper limit for the along/cross conductivity ratio
            wm_id          : WM label id(s) (defaults to gnr.ls_wm_seg)
            min_center_tracts : Voxels with N_x < min_center_tracts fall back to volume-normalized.
                             Defaults to gnr.tract_min_center_tracts.
            min_tangent_tracts: below this the structure tensor is untrustworthy.
                             Defaults to gnr.tract_min_tangent_tracts.
            win_tract_len_minlim: Min in-window traversal length (voxels). Default h (>=1.0).
            coh_eps        : Coherence below this -> isotropic
        """
        self.dict_cond = dict_cond if dict_cond is not None else gnr.dict_cond_in_use
        # Window geometry comes from lib_tractref
        self.window, self.h, self.radius, self.win_tract_len_minlim = \
            ltref.window_geometry(window, win_tract_len_minlim)
        self.r_max = float(r_max)
        self.wm_id = wm_id if wm_id is not None else gnr.ls_wm_seg
        if np.isscalar(self.wm_id):
            self.wm_id = [int(self.wm_id)]
        self.min_center_tracts = int(
            min_center_tracts if min_center_tracts is not None
            else gnr.tract_min_center_tracts)
        self.min_tangent_tracts = int(
            min_tangent_tracts if min_tangent_tracts is not None
            else gnr.tract_min_tangent_tracts)
        self.eps = eps
        self.coh_eps = coh_eps
        # timing accumulators (mean structure-tensor extraction time, spherical window)
        self._struct_time_total = 0.0
        self._struct_time_count = 0

    # ------------------------------------------------------------------ #
    # Variant hook: spectrum of T -> conductivity eigenvalues            #
    # ------------------------------------------------------------------ #
    def _shape_factors(self, lam: np.ndarray, c: float) -> Tuple[np.ndarray, float]:
        """Unscaled conductivity shape s_i = r ** ((3*lambda_i - 1) / (3*c)).
        """
        r = 1.0 + (self.r_max - 1.0) * c
        expo = (3.0 * lam - 1.0) / (3.0 * c)
        return r ** expo, r

    def _eigen_conductivity(self, lam: np.ndarray, c: float,
                            sigma_iso: float) -> np.ndarray:
        """Conductivity eigenvalues sigma_1>=sigma_2>=sigma_3 for one window.

        Params:
            lam       : (3,) structure-tensor eigenvalues, descending, summing to 1
            c         : coherence lambda_1 - lambda_3, guaranteed >= self.coh_eps
            sigma_iso : isotropic WM conductivity the variant calibrates against
        Returns: (3,) float64 conductivity eigenvalues.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Tract geometry: delegated to core.cond.lib_tractref                #
    #                                                                    #
    # These two primitives are NOT model logic -- they are what the tracts #
    # say about a window, and the scoring reference needs the       #
    # same answer. They live in lib_tractref so this model and that      #
    # reference read one implementation instead of the reference having to #
    # instantiate this model to reach them. Kept as thin methods because #
    # the debug/dataprep helpers call them through a model instance.     #
    # ------------------------------------------------------------------ #
    def _build_voxel_index(self, streamlines, wm_mask):
        """WM voxel (i,j,k) -> ids of the tracts through it. See
        lib_tractref.build_voxel_index."""
        return ltref.build_voxel_index(streamlines, wm_mask)

    def _structure_tensor(self, center, sids, streamlines):
        """Orientation structure tensor T of this model's window around `center`.
        Returns (T, n_used); T is None -> caller falls back to volume-normalized.
        See lib_tractref.structure_tensor."""
        return ltref.structure_tensor(
            center, sids, streamlines,
            radius=self.radius,
            min_tangent_tracts=self.min_tangent_tracts,
            win_tract_len_minlim=self.win_tract_len_minlim,
            eps=self.eps,
        )

    def __call__(
        self,
        d_evals: np.ndarray,
        d_evecs: np.ndarray,
        segs: Optional[np.ndarray] = None,
        streamlines: Optional[list] = None,
        affine: Optional[np.ndarray] = None,
        # --- optional colored-tract export ---
        export_vtk: bool = False,
        ref_img=None,
        out_fdpath: Optional[str] = None,
        out_fname: Optional[str] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build the conductivity tensor field from diffusion tracts.
        Voxels are filled with the isotropic conductivity; WM voxels are refined.
        """
        self.validate_input(d_evals, d_evecs)
        if segs is None:
            raise ValueError("segmentation is required")
        if streamlines is None:
            raise ValueError("whole-brain tracts (VOX space) are required")

        X, Y, Z, _ = d_evals.shape
        shape = (X, Y, Z)

        # 0. Initialise every voxel with isotropic conductivity (c_iso * I).
        c_iso = np.zeros(shape, dtype=np.float64)
        for seg_id, cond in self.dict_cond.items():
            c_iso[segs == seg_id] = cond
        c_evals = np.repeat(c_iso[..., None], 3, axis=-1).astype(np.float32)
        c_evecs = np.zeros((X, Y, Z, 3, 3), dtype=np.float32)
        for i in range(3):
            c_evecs[..., i, i] = 1.0

        # 1. WM mask and its isotropic conductivity (sigma_iso).
        wm_mask = np.isin(segs, self.wm_id)
        if not np.any(wm_mask):
            raise ValueError("WM mask is empty for the given wm_id")
        sigma_iso = float(self.dict_cond.get(self.wm_id[0], gnr.WM_ISO_COND))

        # 2. Volume-normalized fallback (diffusion-based), computed once.
        vn_model = VolumeNormalizedModel(dict_cond=self.dict_cond)
        _, vn_evals, vn_evecs = vn_model(d_evals, d_evecs, segs=segs)

        # 3. Index tracts and count N_x (tracts through each WM voxel).
        voxel2sl = self._build_voxel_index(streamlines, wm_mask)
        wm_voxels = np.argwhere(wm_mask)
        N_map = np.zeros(shape, dtype=np.int64)
        for (i, j, k) in wm_voxels:
            N_map[i, j, k] = len(voxel2sl.get((i, j, k), ()))

        # 4. Per-WM-voxel anisotropic assignment (timing the T extraction).
        t_struct, n_struct = 0.0, 0
        for (i, j, k) in wm_voxels:
            N_x = N_map[i, j, k]

            if N_x < self.min_center_tracts:     # sparse -> volume-normalized fallback
                c_evals[i, j, k] = vn_evals[i, j, k]
                c_evecs[i, j, k] = vn_evecs[i, j, k]
                continue

            sids = voxel2sl.get((i, j, k), ())
            _t0 = time.perf_counter()
            T, n_used = self._structure_tensor((i, j, k), sids, streamlines)
            t_struct += time.perf_counter() - _t0
            n_struct += 1
            if T is None:
                c_evals[i, j, k] = vn_evals[i, j, k]
                c_evecs[i, j, k] = vn_evecs[i, j, k]
                continue

            # eigendecomposition (ascending) -> reorder to descending lambda_1..3
            evals, evecs = np.linalg.eigh(T)
            order = np.argsort(evals)[::-1]
            lam = np.clip(evals[order], 0.0, None)
            vecs = evecs[:, order]

            c = float(lam[0] - lam[2])           # coherence in [0, 1]
            if c < self.coh_eps:
                sig = np.full(3, sigma_iso, dtype=np.float64)   # isotropic limit
            else:
                sig = self._eigen_conductivity(lam, c, sigma_iso)

            c_evals[i, j, k] = sig.astype(np.float32)
            c_evecs[i, j, k] = vecs.astype(np.float32)

        # timing: mean structure-tensor (spherical window) extraction time.
        self._struct_time_total = t_struct
        self._struct_time_count = n_struct
        if n_struct:
            print(f"[{self.short_name}] structure-tensor T (spherical window): mean "
                  f"{1e3 * t_struct / n_struct:.3f} ms/voxel over {n_struct} voxels")

        # 5. Reconstruct full conductivity tensor from evals/evecs.
        C = np.zeros((X, Y, Z, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                C[..., i, j] = (
                    c_evals[..., 0] * c_evecs[..., i, 0] * c_evecs[..., j, 0] +
                    c_evals[..., 1] * c_evecs[..., i, 1] * c_evecs[..., j, 1] +
                    c_evals[..., 2] * c_evecs[..., i, 2] * c_evecs[..., j, 2]
                )

        # 6. Optional: WM tracts colored by the anisotropy ratio sigma_1/sigma_3.
        if export_vtk:
            if ref_img is None or out_fdpath is None or out_fname is None:
                raise ValueError("export_vtk requires ref_img, out_fdpath and out_fname")
            ratio_map = c_evals[..., 0] / np.maximum(c_evals[..., 2], self.eps)
            self._export_colored_tracts(
                streamlines, ratio_map, wm_mask, ref_img, out_fdpath, out_fname)

        return C, c_evals, c_evecs

    # ------------------------------------------------------------------ #
    # Optional colored-tract export (invoked from __call__)              #
    # ------------------------------------------------------------------ #
    def _export_colored_tracts(
        self, streamlines, ratio_map, wm_mask,
        ref_img, out_fdpath, out_fname,
    ):
        """Crop whole-brain tracts to WM, sample the anisotropy ratio
        sigma_1/sigma_3 per point and export a VTK colored by it (viridis pinned to
        [1, r_max]; 1 = isotropic, r_max = max anisotropy)."""
        wm_streamlines = ltract.crop_streamlines_to_mask(
            streamlines, wm_mask, affine=np.eye(4), tract_len_thr=0.0)

        X, Y, Z = ratio_map.shape
        point_scalars = []
        for sl in wm_streamlines:
            ivox = np.rint(np.asarray(sl)).astype(np.int64)
            ivox[:, 0] = np.clip(ivox[:, 0], 0, X - 1)
            ivox[:, 1] = np.clip(ivox[:, 1], 0, Y - 1)
            ivox[:, 2] = np.clip(ivox[:, 2], 0, Z - 1)
            point_scalars.append(ratio_map[ivox[:, 0], ivox[:, 1], ivox[:, 2]])

        affine = np.asarray(ref_img.affine) if hasattr(ref_img, "affine") else None
        ltract.export_streamlines_colored_to_vtk(
            wm_streamlines, point_scalars, out_fdpath, out_fname,
            iso_val=None, cmap='viridis', vmin=1.0, vmax=self.r_max,
            scalar_name="aniso_ratio", affine=affine,
        )
