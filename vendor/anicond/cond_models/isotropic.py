# -*- coding: utf-8 -*-
"""
Isotropic conductivity model (Cole–Cole-based lookup).

- Assign scalar conductivity per voxel based on tissue label
- Construct isotropic conductivity tensor: c_iso * I
- Eigenvectors: identity
- Eigenvalues: [c_iso, c_iso, c_iso]

Dependencies:
- core.general.dict_cond_10khz
"""
import numpy as np
from typing import Tuple
from cond_models.base import ConductivityModel
from core.general import dict_cond_in_use #maping tissue_id --> conductivity


class IsotropicModel(ConductivityModel):
    name: str = "isotropic"
    short_name: str = "Iso"

    def __call__(
        self,
        d_evals: np.ndarray,
        d_evecs: np.ndarray,
        **kwargs
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Params:
            d_evals : Dummy input (not used)
            d_evecs : Dummy input (not used)
            kwargs:
                tissue_label : np.ndarray [X, Y, Z], Integer segmentation map
        """
        segs = kwargs.get("segs", None) 
        X, Y, Z = segs.shape

        # scalar cond. map
        c_iso = np.zeros((X, Y, Z), dtype=np.float32)
        for seg_id, cond in dict_cond_in_use.items():
            c_iso[segs == seg_id] = cond


        # construct isotropic tensor: c_iso * I
        C = np.zeros((X, Y, Z, 3, 3), dtype=np.float32)
        for i in range(3):
            C[..., i, i] = c_iso

        # eigenvalues: [c_iso, c_iso, c_iso]
        c_evals = np.repeat(c_iso[..., None], 3, axis=-1)

        # eigenvectors: identity
        c_evecs = np.zeros((X, Y, Z, 3, 3), dtype=np.float32)
        c_evecs[..., 0, 0] = 1
        c_evecs[..., 1, 1] = 1
        c_evecs[..., 2, 2] = 1

        return C, c_evals, c_evecs