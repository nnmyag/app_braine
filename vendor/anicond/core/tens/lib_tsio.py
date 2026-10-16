# -*- coding: utf-8 -*-
"""
Tensor I/O utilities -- VENDORED. See vendor/VERSION.md.

`nib_save_principal_components` is byte-identical to
PROJ_DCONDNET/scripts/core/tens/lib_tsio.py.

ADDED here: the 3x3 <-> 6x1 conversion, lifted verbatim out of the batch script
tools/convert_condtens3x3_to_condtens6x1.py. In the study that conversion is a
post-hoc pass over a whole derivatives tree; in the service the 6x1 tensor IS the
deliverable, so the three pure functions move into the library and the batch
script's cohort walking stays behind. The values are untouched by the conversion
-- no segmentation is read, no diffusion proxy derived, no scale factor applied.
"""
import numpy as np

from core.med import lib_mri as lmri  # avoid circular import

# FSL convention: the upper triangle read row-wise.
UT_IDX = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]


def nib_save_principal_components(evals, affine, out_fdpath, prefix):
    """
    Save principal components (eigenvalues) as separate NIfTI files.

    Params
    ----------
    evals : np.ndarray [X, Y, Z, 3]
    affine : np.ndarray (4, 4)
    out_fdpath : str
    prefix : e.g., 'subj01_cond_evals'
    """

    for i in range(3):
        fname = f"{prefix}_pc{i+1}.nii.gz"
        lmri.nib_save_from_arr(evals[..., i], affine, out_fdpath, fname)


def as_tensor3x3(arr):
    """[X, Y, Z, 3, 3] (or [X, Y, Z, 9]) -> symmetrised [X, Y, Z, 3, 3].

    Returns (C, max_asymmetry). The tensor is symmetrised as 0.5 * (C + C.T),
    so a tiny numerical asymmetry never biases one off-diagonal over its mirror.
    """
    if arr.ndim == 5 and arr.shape[-2:] == (3, 3):
        C = arr
    elif arr.ndim == 4 and arr.shape[-1] == 9:
        C = arr.reshape(arr.shape[:3] + (3, 3))
    else:
        raise ValueError(f"unexpected tensor shape {arr.shape}; "
                         "expected [X, Y, Z, 3, 3] or [X, Y, Z, 9]")

    asym = float(np.max(np.abs(C - np.swapaxes(C, -1, -2)))) if C.size else 0.0
    return 0.5 * (C + np.swapaxes(C, -1, -2)), asym


def to_upper_triangular6(C):
    """symmetric [X, Y, Z, 3, 3] -> [X, Y, Z, 6] float32 in FSL order."""
    return np.stack([C[..., r, c] for r, c in UT_IDX], axis=-1).astype(np.float32)


def from_upper_triangular6(arr6):
    """[X, Y, Z, 6] in FSL order -> full symmetric [X, Y, Z, 3, 3]."""
    C = np.empty(arr6.shape[:3] + (3, 3), dtype=arr6.dtype)
    for n, (r, c) in enumerate(UT_IDX):
        C[..., r, c] = arr6[..., n]
        C[..., c, r] = arr6[..., n]
    return C
