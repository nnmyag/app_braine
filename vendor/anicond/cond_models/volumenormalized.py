import numpy as np
from typing import Tuple, Optional
from cond_models.base import ConductivityModel

from core.general import dict_cond_in_use #maping tissue_id --> conductivity


class VolumeNormalizedModel(ConductivityModel):
    """
    Volume-normalized conductivity mapping (Güllmar et al., 2010)

    Key idea:
        det(c_aniso​)=det(c_iso​)
        with det() is the product of three eigenvalues, or the tensor volume.

    Voxel-wise mapping:
        c_evals = alpha * d_evals
        alpha = c_iso / cbrt(prod(d_evals))

    This preserves anisotropy while matching isotropic conductivity locally.
    """

    name: str = "volumenormalized"
    short_name: str = "VN"
    def __init__(
        self,
        dict_cond: dict = None,
        clip_min: float = 0.0,
        eps: float = 1e-12,
    ):
        """
        Params:
            dict_cond: Mapping tissue_id -> isotropic conductivity
            clip_min : Minimum conductivity (stability constraint)
            eps      : Numerical stability constant
        """
        self.dict_cond = dict_cond if dict_cond is not None else dict_cond_in_use
        self.clip_min = clip_min
        self.eps = eps

    def __call__(
        self,
        d_evals: np.ndarray,
        d_evecs: np.ndarray,
        segs: Optional[np.ndarray] = None,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        self.validate_input(d_evals, d_evecs)

        if segs is None:
            raise ValueError("segmentation is required for tissue-wise conductivity")

        X, Y, Z, _ = d_evals.shape
        c_evecs = d_evecs.copy()

        # Isotropic conductivity map (use instance mapping if provided)
        c_iso = np.zeros((X, Y, Z), dtype=np.float32)
        for seg_id, cond in self.dict_cond.items():
            c_iso[segs == seg_id] = cond

        # ----------------------------------
        # Tensor volume (TV = λ1 * λ2 * λ3)
        # ----------------------------------
        tv = np.prod(d_evals, axis=-1)  # product, return shape (X,Y,Z)        
        tv_cbr = np.cbrt(tv + self.eps) # cubic root

        # ----------------------------------
        # Voxel-wise scaling (α)
        # ----------------------------------
        alpha = c_iso / (tv_cbr + self.eps)  # (X, Y, Z)
        alpha_expanded = alpha[..., np.newaxis] # (X, Y, Z, n)

        # ----------------------------------
        # Diffusion-to-conductivity mapping
        # ----------------------------------
        c_evals = d_evals * alpha_expanded
        c_evals = np.maximum(c_evals, self.clip_min)

        # -----------------------------
        # Tensor reconstruction
        # Note:
        #      d_evecs[..., i, k]: i-th component of k-th eigenvector
        # -----------------------------
        C = np.zeros((X, Y, Z, 3, 3), dtype=np.float32)

        for i in range(3):
            for j in range(3):
                C[..., i, j] = (
                    c_evals[..., 0] * d_evecs[..., i, 0] * d_evecs[..., j, 0] +
                    c_evals[..., 1] * d_evecs[..., i, 1] * d_evecs[..., j, 1] +
                    c_evals[..., 2] * d_evecs[..., i, 2] * d_evecs[..., j, 2]
                )

        return C, c_evals, c_evecs