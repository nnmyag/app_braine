# -*- coding: utf-8 -*-
"""
Tract-Based Conductivity Model, geometric-mean variant (TBg)
============================================================
Builds anisotropic WM conductivity tensors from local fiber tracts instead of
the single-voxel DTI tensor. All the machinery -- spherical window, tract
tangents, length-weighted structure tensor T, volume-normalized fallback --
lives in `TractBasedModelBase` (core/cond/lib_tractbased_core.py); read its
module docstring for the pipeline, the inputs and the outputs.

TBg is the ORIGINAL calibration, unchanged. It reads hypothesis H2 (per-voxel
representative conductivity = sigma_iso) with the representative conductivity
G(sigma) taken as the GEOMETRIC MEAN of the three eigenvalues:

    sigma_i = sigma_iso * r ** ((3*lambda_i - 1) / (3*c))

The exponents sum to zero, so cbrt(sigma_1 sigma_2 sigma_3) = sigma_iso exactly
(H2) while sigma_1/sigma_3 = r (H1).

The companion `tractbased_e` targets the effective conductivity instead; the two
differ ONLY by a per-voxel scalar, so they share eigenvectors and eigenvalue
ratios and differ in magnitude alone.
"""

import numpy as np

from core.cond.lib_tractbased_core import TractBasedModelBase


class TractBasedGeoModel(TractBasedModelBase):
    name: str = "tractbased_g"
    short_name: str = "TBg"

    def _eigen_conductivity(self, lam, c, sigma_iso):
        """sigma_i = sigma_iso * s_i, with s_i the shape factor (geometric mean 1).

        Because the shape factors already have geometric mean 1, scaling them by
        sigma_iso is exactly what H2-as-geometric-mean asks for -- no further
        normalisation is needed.
        """
        s, _r = self._shape_factors(lam, c)
        return np.asarray(sigma_iso * s, dtype=np.float64)
