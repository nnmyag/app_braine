#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MRI volume helpers -- VENDORED, TRIMMED. See vendor/VERSION.md.

Kept from PROJ_DCONDNET/scripts/core/med/lib_mri.py: the six functions the
DTI -> conductivity job actually calls. Function bodies are byte-identical to
the source; only the unused ones were removed (label mosaics, generic resample,
RGB fusion, the comparison/histogram figures, tissue boundary), together with
the `core.general` and `core.lib_stats` imports they needed -- nothing left in
here reads either.
"""
import math
import os

import matplotlib
matplotlib.use("Agg")          # server-side: never open a window
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import SimpleITK as sitk
from dipy.segment.mask import median_otsu
from scipy.ndimage import binary_dilation


def generate_axial_mosaic(input_data, save_path, n_shown_slices=20, n_cols=5, title_tag='', cmap='gray'):
    """
    Generate an axial mosaic from either:
        - a NIfTI file path (str), or
        - a 3D numpy array
    """
    # Handle input (path vs array)
    if isinstance(input_data, str):
        img = nib.load(input_data)
        data = img.get_fdata()
        shape = data.shape
        name = os.path.basename(input_data).replace('.nii.gz', '')
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim != 3:
            raise ValueError("Input numpy array must be 3D.")
        data = input_data
        shape = data.shape
        name = "array"
    else:
        raise TypeError("input_data must be a file path (str) or a 3D numpy array.")

    # Slice selection
    n_slices = data.shape[2]
    slice_indices = np.linspace(0, n_slices - 1, n_shown_slices, dtype=int)
    n_rows = math.ceil(n_shown_slices / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes = axes.flatten()

    # Plot slices
    for i, slice_idx in enumerate(slice_indices):
        ax = axes[i]
        ax.imshow(data[:, :, slice_idx].T, cmap=cmap, origin='lower')
        ax.set_title(f"{slice_idx}", fontsize=55, color='gray')
        ax.axis('off')

    # Hide unused subplots
    for i in range(n_shown_slices, len(axes)):
        axes[i].axis('off')

    # Annotations
    dim_text = f"{shape[0]}×{shape[1]}×{shape[2]}"
    fig.text(0.95, 0.97, dim_text, ha='right', va='top', fontsize=50, color='gray')

    plt.suptitle(f'{title_tag}{name}\n  ', fontsize=60)

    plt.tight_layout()

#%% get_otsu_brain_mask
def get_otsu_brain_mask(mri_im, affine=None, output_mask_path=None, visualize=False):
    if hasattr(mri_im, "get_fdata"):  # NIfTI case
        b0_img = mri_im.get_fdata()
    else:  # ndarray case
        b0_img = mri_im

    # ======== Otsu filtering & Mask expansion =======
    b0_mask, mask = median_otsu(b0_img, median_radius=2, numpass=1)
    shape = mask.shape
    radius = int(0.05 * min(shape))  # ~5% of image width
    mask = binary_dilation(mask, iterations=radius)
    b0_mask = binary_dilation(b0_mask, iterations=radius)

    # Save mask (optional)
    if output_mask_path:
        mask_img = nib.Nifti1Image(mask.astype(np.uint8), affine)
        nib.save(mask_img, output_mask_path)

    # Visualization (optional)
    if visualize:
        mid_slice = b0_img.shape[2] // 2
        plt.imshow(b0_img[:, :, mid_slice], cmap='gray')
        plt.imshow(mask[:, :, mid_slice], alpha=0.3, cmap='Greens')
        plt.title('Brain Mask Overlay')
        plt.axis('off')
        plt.show()

    return mask, b0_mask
  

def img_resample_vox_size(img, target_vox_size_mm=(1,1,1), target_dim_vox=None, interpolator=sitk.sitkLinear): 
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_vox_size_mm)
    if target_dim_vox is not None:
        resampler.SetSize(target_dim_vox)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(sitk.sitkLinear)
    return resampler.Execute(img)
    
    
# def normalize_image_minmax(img, out_range=(0, 6000)):
#     arr = sitk.GetArrayViewFromImage(img)
#     old_min, old_max = float(arr.min()), float(arr.max())

#     if old_max == old_min:
#         return img

#     scale = (out_range[1] - out_range[0]) / (old_max - old_min)
#     offset = out_range[0] - old_min * scale

#     return sitk.ShiftScale(img, shift=float(offset), scale=float(scale))



def normalize_image_minmax(img, out_range=(0, 6000), clip_percentile=None ):
    is_sitk = isinstance(img, sitk.Image)

    # Convert to numpy
    if is_sitk:
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
    else:
        arr = np.asarray(img, dtype=np.float32)

    # Optional clipping 
    if clip_percentile is not None:
        p_low, p_high = np.percentile(arr, clip_percentile)
        arr = np.clip(arr, p_low, p_high)

    old_min = arr.min()
    old_max = arr.max()

    # Edge case: constant image
    if old_max == old_min:
        norm_arr = np.full_like(arr, out_range[0], dtype=np.float32)
    else:
        scale = (out_range[1] - out_range[0]) / (old_max - old_min)
        norm_arr = (arr - old_min) * scale + out_range[0]

    # Return same type
    if is_sitk:
        out_img = sitk.GetImageFromArray(norm_arr)
        out_img.CopyInformation(img)  # preserve spacing/origin/direction
        return out_img
    else:
        return norm_arr




def nib_save_from_arr(arr, affine, out_dir, fname):
    # in case this is a boolean mask
    if arr.dtype == np.bool_:
        arr = arr.astype(np.uint8)
        
    # create image
    img = nib.Nifti1Image(arr, affine)

    # optional: enforce header dtype (safer)
    img.set_data_dtype(arr.dtype)

    # save
    nib.save(img, os.path.join(out_dir, fname))
    
    
def nib_load(fpath, dtype=None):
    nii = nib.load(fpath)
    data = nii.get_fdata()
    if dtype is not None:
        data = data.astype(dtype)

    affine = nii.affine

    return data, affine

