#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DWI / gradient-table helpers -- VENDORED, TRIMMED. See vendor/VERSION.md.

Kept byte-identical from PROJ_DCONDNET/scripts/core/med/lib_dwi.py:
    load_bvals_bvecs, save_bvals_bvecs, n4_bias_field, get_fa_map

Dropped: build_4d_dwi / find_bval_bvec (they scan a BIDS `dwi_processed` folder
for `<subj>_DTI-NN_processed.nii.gz`; an upload has no such convention),
bvecs_plot (plotly QC page), subsample_bvecs_with_b0 + electrostatic_energy
(cohort preprocessing), apply_bias_field (unused).

CHANGED: `fit_dti_tensor(dwi_fdpath, subj_id, ...)` loaded the 4D series off
disk itself. Its replacement `fit_dti_tensor_from_arrays` takes the array the
caller already holds; everything after that -- Otsu mask with the percentile
fallback, WLS fit, eigenvalue clipping, FA/MD -- is the original, unchanged, so
a served fit and a b5/b6 fit agree.
"""
import os

import numpy as np
import SimpleITK as sitk
from dipy.reconst.dti import TensorModel, fractional_anisotropy, mean_diffusivity

from core import general as gnr
from core.med import lib_mri as lmri


def load_bvals_bvecs(bval_path, bvec_path, return_gtab=False):
    bvals = np.loadtxt(bval_path)
    bvecs = np.loadtxt(bvec_path)
    bvecs = np.asarray(bvecs)

    if bvecs.shape[0] != 3 and bvecs.shape[1] == 3:
        bvecs = bvecs.T

    norms = np.linalg.norm(bvecs, axis=0)
    norms[norms == 0] = 1
    bvecs_norm = bvecs / norms
    
    if return_gtab is True:
        gtab = gradient_table(bvals=bvals, bvecs=bvecs_norm)
        return gtab

    return bvals, bvecs, bvecs_norm

def save_bvals_bvecs(bvals, bvecs, out_fdpath, subj_id):
    bvals = np.asarray(bvals).flatten()   # always (N,)
    bvecs = np.asarray(bvecs)
    
    bval_path = os.path.join(out_fdpath, f"{subj_id}_bval")
    with open(bval_path, "w") as f:
        f.write("\n".join(f"{v:.6g}" for v in bvals))

    np.savetxt(os.path.join(out_fdpath, f"{subj_id}_bvec"), bvecs, fmt="%.6g")
    
    

def n4_bias_field(b0_input):
    """
    Perform N4 bias field correction on b0 image and return:
    - corrected b0 image
    - estimated bias field (same size as b0)
    """        
    if isinstance(b0_input, sitk.Image):
        b0_image = sitk.Cast(b0_input, sitk.sitkFloat32)
    else:
        b0_image = sitk.ReadImage(b0_input, sitk.sitkFloat32)

    mask = sitk.OtsuThreshold(b0_image, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    corrected_b0 = corrector.Execute(b0_image, mask)
    
    # Bias field = original / corrected
    bias_field = sitk.Divide(b0_image, corrected_b0)

    return corrected_b0, bias_field


# apply bias field correction on b0 and reuse the bias field to other DWI images


def fit_dti_tensor_from_arrays(dwi_4d, gtab, b0_arr,
                               fit_method=gnr.dtensor_fit_method,
                               return_dat_for_tracking=False):
    """Fit the diffusion tensor to an already-loaded 4D DWI array.

    Params:
        dwi_4d   : [X, Y, Z, N] DWI series, volume order matching `gtab`
        gtab     : dipy GradientTable built from the same bvals/bvecs
        b0_arr   : [X, Y, Z] b0 volume, used only for the Otsu brain mask
        fit_method : dipy TensorModel fit method (default WLS, as in b5)

    Returns (tenfit, FA, MD, brain_mask), or with the DWI/gtab appended when
    `return_dat_for_tracking` -- the same shape of return the batch b5/b6
    scripts consume, minus the b0 NIfTI object they took from disk.
    """
    if dwi_4d.shape[-1] != gtab.bvals.shape[0]:
        raise RuntimeError(
            f"Mismatch DWI volumes ({dwi_4d.shape[-1]}) vs bvals "
            f"({gtab.bvals.shape[0]})")

    if not np.any(gtab.bvals < 50):
        raise RuntimeError("No valid b0 found (no bval < 50)")

    # Brain mask -- Otsu on the b0, with the same low-count fallback as b5/b6.
    brain_mask, _ = lmri.get_otsu_brain_mask(b0_arr)
    if brain_mask.sum() < 100:
        brain_mask = dwi_4d[..., 0] > np.percentile(dwi_4d[..., 0], 30)

    tenmodel = TensorModel(gtab, fit_method=fit_method)
    tenfit = tenmodel.fit(dwi_4d, mask=brain_mask)
    if np.isnan(tenfit.evals).any():
        print("       Warning: NaN eigenvalues detected")

    evals = np.clip(tenfit.evals, 1e-12, None)
    MD = np.nan_to_num(mean_diffusivity(evals))
    FA = np.nan_to_num(fractional_anisotropy(evals))

    if return_dat_for_tracking is True:
        return tenfit, FA, MD, brain_mask, dwi_4d, gtab
    return tenfit, FA, MD, brain_mask

def get_fa_map(tens_fa, fa_thr=0.2):
    fa_mask = tens_fa.copy()
    fa_mask[fa_mask < fa_thr] = 0
    fa_mask[fa_mask >= fa_thr] = 1
    return fa_mask