"""
Diffusion-to-conductivity mapping using a global scaling model.

Method:
-------
1. For each tissue (WM/GM), compute a geometric diffusion coefficient (k_t):
       k_t = cbrt(mean(λ1 * λ2 * λ3))

2. Combine tissue coefficient with isotropic conductivities to obtain a global factor:
       k = (k_WM*σ_WM + k_GM*σ_GM) / (k_WM² + k_GM²)

3. Map diffusion eigenvalues to conductivity eigenvalues:
       σ_i = k * λ_i

4. Reconstruct conductivity tensor from eigenvalues and eigenvectors.

Inputs:
- d_evals: diffusion eigenvalues [X,Y,Z,3]
- d_evecs: diffusion eigenvectors [X,Y,Z,3,3]
- segs: tissue labels [X,Y,Z]

Outputs:
- C: conductivity tensor field
- c_evals: conductivity eigenvalues
- c_evecs: eigenvectors (unchanged)

Notes:
- Uses WM/GM tissue statistics for calibration.
"""

import numpy as np
from typing import Tuple, Optional, List
from cond_models.base import ConductivityModel
from core import general as gnr


class DirectMappingModel(ConductivityModel):
    name: str = "directmapping"
    short_name: str = "DM"

    def __init__(
        self,
        dict_cond: dict = None,
        WM_segs: List[int] = None,
        GM_segs: List[int] = None,
        clip_min: float = 0.0,
        eps: float = 1e-12,
    ):
        """
        Params:
            dict_cond: Mapping tissue_id -> isotropic conductivity
            WM_segs, GM_segs   : list of WM/GM label IDs
            clip_min : Minimum conductivity (stability constraint)
            eps      : Numerical stability constant
        """
        self.dict_cond = dict_cond if dict_cond is not None else gnr.dict_cond_in_use
        self.WM_segs = WM_segs if WM_segs is not None else gnr.ls_wm_seg #[13, 14, 15]
        self.GM_segs = GM_segs if GM_segs is not None else gnr.ls_gm_seg #[8]
        self.clip_min = clip_min
        self.eps = eps

    def global_scale(self, d_evals: np.ndarray, segs: np.ndarray):
        """Whole-brain global scale factor k (sigma_i = k * lambda_i) and the
        WM/GM geometric diffusion coefficients it is built from.

        Shared by __call__ and reused by other models (e.g. the tract-based models'
        debug overlays) so the direct-mapping calibration lives in one place.

        Returns (k, k_WM, k_GM). Raises ValueError if the WM or GM mask is empty.
        """
        wm_mask = np.isin(segs, self.WM_segs)
        gm_mask = np.isin(segs, self.GM_segs)
        if not np.any(wm_mask):
            raise ValueError("WM mask is empty")
        if not np.any(gm_mask):
            raise ValueError("GM mask is empty")

        # Geometric diffusion coefficient per tissue: cbrt(mean(prod of eigenvalues)).
        k_WM = float(np.cbrt(np.mean(np.prod(d_evals[wm_mask], axis=1)) + self.eps))
        k_GM = float(np.cbrt(np.mean(np.prod(d_evals[gm_mask], axis=1)) + self.eps))

        iso_WM = self.dict_cond.get(self.WM_segs[0], 0.0)
        iso_GM = self.dict_cond.get(self.GM_segs[0], 0.0)
        k = float(
            (k_WM * iso_WM + k_GM * iso_GM) /
            (k_WM ** 2 + k_GM ** 2 + self.eps)
        )
        return k, k_WM, k_GM

    def __call__(
        self,
        d_evals: np.ndarray,
        d_evecs: np.ndarray,
        segs: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        if segs is None:
            raise ValueError("segmentation is required for tissue-aware scaling")

        X, Y, Z, _ = d_evals.shape
        c_evecs = d_evecs.copy()

        # global scaling factor from WM/GM diffusion statistics
        k, _, _ = self.global_scale(d_evals, segs)

        # ------------------------------------------
        # eigenvalue mapping & tensor reconstruction
        # ------------------------------------------
        c_evals = d_evals * k
        c_evals = np.maximum(c_evals, self.clip_min)
        
        C = np.zeros((X, Y, Z, 3, 3), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                C[..., i, j] = (
                    c_evals[..., 0] * d_evecs[..., i, 0] * d_evecs[..., j, 0] +
                    c_evals[..., 1] * d_evecs[..., i, 1] * d_evecs[..., j, 1] +
                    c_evals[..., 2] * d_evecs[..., i, 2] * d_evecs[..., j, 2]
                )
        

        return C, c_evals, c_evecs