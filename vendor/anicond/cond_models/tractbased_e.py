# -*- coding: utf-8 -*-
"""
Tract-Based Conductivity Model, effective-conductivity variant (TBe)
====================================================================
All the tractbased model base implementation
lives in `TractBasedModelBase` (core/cond/lib_tractbased_core.py).

TBe reads hypothesis H2 with the representative conductivity G(sigma) taken as
the spherical average of the EFFECTIVE conductivity (Kangasmaa et al., 2025, doi:10.1088/1361-6560/adfe31):

K is homogeneous of degree 1 in (sigma_para, sigma_perp), so dividing by it makes
the spherical average of sigma_eff come out at exactly sigma_iso (H2). 
"""

import numpy as np

from core.cond.lib_tractbased_core import (TractBasedModelBase, eff_cond_spherical_mean)


class TractBasedEffModel(TractBasedModelBase):
    name: str = "tractbased_e"
    short_name: str = "TBe"

    def _eigen_conductivity(self, lam, c, sigma_iso):
        """sigma_i = sigma_iso * s_i / K(s_para, s_perp).

        The shape factors s_i fix the anisotropy (H1); K -- the spherical average
        of the effective conductivity they imply -- is the scalar that lands the
        magnitude on sigma_iso (H2).
        """
        s, _r = self._shape_factors(lam, c)

        s_para = float(s[0])
        s_perp = 0.5 * float(s[1] + s[2])
        K = float(eff_cond_spherical_mean(s_para, s_perp, eps=self.eps))

        return np.asarray(sigma_iso * s / max(K, self.eps), dtype=np.float64)
