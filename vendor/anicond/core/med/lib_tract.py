# -*- coding: utf-8 -*-
"""
Fiber tracts and tractography objects -- VENDORED, TRIMMED. See vendor/VERSION.md.

Kept byte-identical from PROJ_DCONDNET/scripts/core/med/lib_tract.py:
    world_to_voxel, run_tractography, export_streamlines_to_vtk,
    crop_streamlines_to_mask, load_streamlines_from_vtk,
    _write_streamlines_rgb_vtk, export_streamlines_colored_to_vtk

Dropped: export_streamlines_world_vtk and the three tensor-ellipsoid exporters
(export_tensor_ellipsoid / _to_vtp / _to_stl) -- 3D Slicer inspection tools with
no consumer in the service.

Coordinate note, unchanged from the source and load-bearing here: tracking runs
with the b0 affine, so run_tractography returns RASMM (world) coordinates, while
the conductivity models index voxels and need VOX. The service therefore writes
the .vtk and reads it back with space_str='vox' -- the same round-trip b6 -> c1
does, rather than a second conversion that could drift from it.
"""
import os

import numpy as np
import nibabel as nib

from dipy.data import get_sphere
from dipy.direction import peaks_from_model
from dipy.reconst.shm import CsaOdfModel
from dipy.tracking.stopping_criterion import BinaryStoppingCriterion
from dipy.tracking import utils
from dipy.tracking.local_tracking import LocalTracking
from dipy.tracking.streamline import Streamlines, length
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram, load_tractogram

from core import general as gnr
from core.med import lib_mri as lmri

# segment length helper used by the cropping routine below
tract_len = length

streamlines_min_length = 5  # in mm
streamlines_min_length_for_whole_brain_tractography = 20  # in mm


# Convert world (RASMM) streamline coordinates to voxel coordinates
def world_to_voxel(streamline, affine):
    # Use np.linalg.inv(affine) to map world to voxel
    return nib.affines.apply_affine(np.linalg.inv(affine), streamline)


# =================================================================
# Perform CSA-based tractography and return filtered streamlines
# =================================================================
def run_tractography(
    dwi_4d, gtab, affine, 
    tract_len_thr=None, mask=None, sphere=None,      
    peak_threshold=None, min_separation_angle=None,
    step_size=None, seed_density=None, sh_order_max=None,
    model='csa'
):
    
    # Safe param loading
    def get_param(val, gnr_name, default):
        if val is not None:
            return val
        return getattr(gnr, gnr_name, default)
    
    tract_len_thr = get_param(tract_len_thr, "tract_min_len_for_wholebrain_tracking", 20)   # mm
    sh_order_max = get_param(sh_order_max, "sh_order_max", 6)
    peak_threshold = get_param(peak_threshold, "peak_threshold", 0.5)
    min_separation_angle = get_param(min_separation_angle, "min_separation_angle", 45)  # degrees
    step_size = get_param(step_size, "step_size", .875)  # mm
    seed_density = get_param(seed_density, "seed_density", 1)    
    

    # Prepare mask and sphere
    if mask is None:
        b0_idx = np.where(gtab.b0s_mask)[0][0]
        mask, _ = lmri.get_otsu_brain_mask(dwi_4d[..., b0_idx])    
    if sphere is None:
        sphere = get_sphere(name='repulsion724')
        

    # Estimate peaks (CSA-ODF)
    """
    The CSA-ODF (2012) is the distribution of water diffusion as a function of direction.
    The peaks of an CSA-ODF are good estimates for the orientation of tract segments at each voxel. 

    'csa_peaks' (PeaksAndMetrics class) encapsulates following properties:
        + peak_dirs: principal diffusion directions. Its dimension is (X, Y, Z, N, 3), with N is number of directions per voxel (e.g., 1-5) and 3 is the vector components (x, y, z).
        + peak_values: magnitude of each direction. Its dimension is (X, Y, Z, N).
        + peak_indices
        
    Read more: https://workshop.dipy.org/documentation/1.7.0/reference/dipy.direction/    
    """
    if model == 'csa':
        csa_model = CsaOdfModel(gtab, sh_order_max=sh_order_max)
        peaks = peaks_from_model(
            model=csa_model,
            data=dwi_4d,
            mask=mask, sphere=sphere,
            relative_peak_threshold=peak_threshold,
            min_separation_angle=min_separation_angle,        
            return_sh=False,
            return_odf=False,
            normalize_peaks=True
        )

    # Tracking
    stopping_criterion = BinaryStoppingCriterion(mask)
    seeds = utils.seeds_from_mask(mask, affine=affine, density=seed_density)
    streamlines_generator = LocalTracking(
        peaks,
        stopping_criterion,
        seeds,
        affine=affine,
        step_size=step_size
    )
    del peaks
    
    streamlines = Streamlines(streamlines_generator)

    # Filter streamlines    
    line_lengths = list(map(length, streamlines))
    streamlines = [s for s, l in zip(streamlines, line_lengths) if l >= tract_len_thr]
    # streamlines = [s for s in streamlines if length(s) >= tract_len_thr]

    return streamlines
    
    
def export_streamlines_to_vtk(streamlines, header, save_fpath, space_str='vox'):
    """
    Export streamlines to file, so that it can be loaded in 3D Slicer.

    Params:
        header     : nibabel header/image used as the spatial reference.
        save_fpath : full output path (trk/tck/vtk/vtp/fib/dpy).
        space_str  : current space of the streamlines (vox | voxmm | rasmm | auto).
                     After tracking the space is VOX (voxel indices); after loading
                     with nibabel the space is RASMM. 'auto' guesses from coordinate
                     sign (any negative -> rasmm, else vox).
    """
    s = space_str.lower().strip()
    if s == 'auto':
        s = 'rasmm' if np.min(np.vstack(streamlines)) < 0 else 'vox'
    if s == 'vox':
        spc = Space.VOX
    elif s == 'voxmm':
        spc = Space.VOXMM
    elif s == 'rasmm':
        spc = Space.RASMM
    else:
        raise ValueError(f"Unsupported space_str: {space_str}")

    sft = StatefulTractogram(
        streamlines,
        reference=header,
        space=spc,
        origin=Origin.NIFTI,   # explicit is better than implicit
    )
    sft.remove_invalid_streamlines(epsilon=0.001)
    # Save (supports multiple formats: trk/tck/vtk/vtp/fib/dpy)
    save_tractogram(sft, save_fpath, bbox_valid_check=False)


#%% Crop streamlines to keep only segments that pass through the mask.
def crop_streamlines_to_mask(streamlines, mask, affine, tract_len_thr = streamlines_min_length): 
    cropped_streamlines = []

    for sl in streamlines:
        # Convert streamline points to voxel space
        vox_coords = world_to_voxel(sl, affine)
        vox_int = np.round(vox_coords).astype(int)
        inside_mask = []
        for idx, (x, y, z) in enumerate(vox_int):
            if (0 <= x < mask.shape[0] and 0 <= y < mask.shape[1] and 0 <= z < mask.shape[2] and mask[x, y, z]):
                inside_mask.append(True)
            else:
                inside_mask.append(False)

        # Extract continuous inside segments
        segment = []
        for p, inside in zip(sl, inside_mask):
            if inside:
                segment.append(p)
            else:
                if len(segment) >= 2:
                    cropped_streamlines.append(np.array(segment))
                segment = []
        # last segment
        if len(segment) >= 2:
            cropped_streamlines.append(np.array(segment))
            
    # [OPTIONAL] Filter long/major streamlines
    line_lengths = list(map(tract_len, cropped_streamlines))
    cropped_streamlines = [s for s, l in zip(cropped_streamlines, line_lengths) if l >= tract_len_thr]

    return cropped_streamlines


#%% Load whole-brain tracts from a VTK/TRK/TCK file into VOX space
def load_streamlines_from_vtk(tract_fpath, ref_img, space_str='vox'):
    """
    Load a saved tractogram (.vtk/.vtp/.trk/.tck/.fib/.dpy) and return its
    streamlines as a list of (P, 3) float arrays.

    Params:
        tract_fpath : path to the tractogram file
        ref_img     : reference Nifti image (provides header/affine)
        space_str   : 'vox' | 'voxmm' | 'rasmm' coordinate space of the output

    Returns:
        list of np.ndarray, each (P, 3), expressed in the requested space.
        Default 'vox' makes points directly indexable as voxel coordinates.
    """
    space_map = {
        'vox': Space.VOX,
        'voxmm': Space.VOXMM,
        'rasmm': Space.RASMM,
    }
    space_key = space_str.lower().strip()
    if space_key not in space_map:
        raise ValueError(f"Unsupported space: {space_str}")

    sft = load_tractogram(
        tract_fpath,
        reference=ref_img,
        to_space=space_map[space_key],
        bbox_valid_check=False,
    )
    return [np.asarray(s, dtype=np.float64) for s in sft.streamlines]


#%% Shared writer: streamlines + per-point RGB -> .vtk PolyData
def _write_streamlines_rgb_vtk(streamlines, rgb_list, out_fdpath, out_fname,
                               scalar_name="colors", affine=None):
    """
    Low-level writer. `rgb_list[k]` is an (P_k, 3) uint8 array of per-point colors
    matching `streamlines[k]`. Streamlines with < 2 points are skipped.

    `affine` (4x4, optional): if given, point coordinates are mapped through it
    before writing. Streamlines are usually in VOX space, while 3D Slicer expects
    world/RAS geometry, so pass the reference image affine to make the saved VTK
    overlay the NIfTI correctly. Defaults to identity (coordinates written as-is).
    """
    import vtk

    points = vtk.vtkPoints()
    lines = vtk.vtkCellArray()
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName(scalar_name)

    pid = 0
    for sl, rgb in zip(streamlines, rgb_list):
        sl = np.asarray(sl, dtype=np.float64)
        if affine is not None:
            sl = nib.affines.apply_affine(np.asarray(affine, dtype=np.float64), sl)
        rgb = np.asarray(rgb)
        n = len(sl)
        if n < 2:
            continue
        line = vtk.vtkPolyLine()
        line.GetPointIds().SetNumberOfIds(n)
        for i in range(n):
            points.InsertNextPoint(float(sl[i, 0]), float(sl[i, 1]), float(sl[i, 2]))
            r, g, b = (int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2]))
            colors.InsertNextTuple3(r, g, b)
            line.GetPointIds().SetId(i, pid)
            pid += 1
        lines.InsertNextCell(line)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetLines(lines)
    polydata.GetPointData().SetScalars(colors)

    os.makedirs(out_fdpath, exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileName(os.path.join(out_fdpath, out_fname))
    writer.SetInputData(polydata)
    writer.Write()


#%% Export streamlines colored per-point by a scalar (jet, centered at iso_val)
def export_streamlines_colored_to_vtk(
    streamlines, point_scalars, out_fdpath, out_fname,
    iso_val=None, cmap='jet', vmin=None, vmax=None, scalar_name="sigma1",
    affine=None,
):
    """
    Write a .vtk PolyData of streamlines whose points carry RGB colors derived
    from a per-point scalar (e.g. the first conductivity eigenvalue).

    The scalar is mapped through `cmap` (default 'jet': blue->green->red). When
    `iso_val` is given, a two-slope normalization centers it on the colormap
    midpoint, so points equal to iso_val render green, lower values bluish,
    higher values reddish.

    Params:
        streamlines   : list of (P, 3) arrays (coordinates used as written)
        point_scalars : list of (P,) arrays, one scalar per streamline point
        out_fdpath    : output folder
        out_fname     : output filename (should end with .vtk)
        iso_val       : scalar mapped to the colormap center (green for 'jet')
        cmap          : matplotlib colormap name
        vmin, vmax    : scalar range; defaults to 1st/99th percentile of data
        affine        : 4x4 vox->world transform applied to geometry (for Slicer
                        alignment); None writes coordinates as-is
    """
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize, TwoSlopeNorm

    if len(streamlines) != len(point_scalars):
        raise ValueError("streamlines and point_scalars must have equal length")

    all_vals = np.concatenate([np.asarray(s).ravel() for s in point_scalars]) \
        if len(point_scalars) else np.array([0.0, 1.0])
    if vmin is None:
        vmin = float(np.percentile(all_vals, 1))
    if vmax is None:
        vmax = float(np.percentile(all_vals, 99))
    if vmax <= vmin:
        vmax = vmin + 1e-6

    if iso_val is not None and vmin < iso_val < vmax:
        norm = TwoSlopeNorm(vmin=vmin, vcenter=float(iso_val), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)
    colormap = cm.get_cmap(cmap)

    rgb_list = []
    for sc in point_scalars:
        rgba = colormap(norm(np.asarray(sc)))           # (n, 4) in [0, 1]
        rgb_list.append((rgba[:, :3] * 255).astype(np.uint8))

    _write_streamlines_rgb_vtk(streamlines, rgb_list, out_fdpath, out_fname,
                               scalar_name, affine=affine)


