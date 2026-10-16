# ============================================================================
# tasks/anicond/tractio.py — tractogram files without fury
# ----------------------------------------------------------------------------
# dipy hands every .vtk/.vtp tractogram to fury (dipy.io.vtk), and dipy 1.11
# refuses any fury outside 0.10-1.0 -- a pin that broke the moment fury 2.0
# was installed. This module does the same two things on the `vtk` package
# alone, which the app already requires:
#
#   save_tractogram()   streamlines in RAS mm  -> .vtk polydata
#   load_tractogram()   .vtk/.vtp/.trk/.tck    -> streamlines in VOX space
#
# CONVENTIONS, matched to what dipy + fury wrote and read, so the .vtk a run
# offers still overlays the NIfTI in 3D Slicer and every tractogram the study
# already produced (PROJ_DCONDNET's *_wb_tracts.vtk) still loads unchanged:
#
#   - on disk the points are LPS mm (x and y negated), the VTK file convention
#     dipy.io.vtk.save_vtk_streamlines(to_lps=True) follows; legacy binary
#     .vtk, the format save_tractogram picks for that extension
#   - before writing, a streamline with any point outside the reference grid
#     is dropped, as StatefulTractogram.remove_invalid_streamlines(1e-3) did
#   - loading returns VOX coordinates with the NIfTI (voxel-centre) origin,
#     i.e. inv(affine) applied to the RAS mm points -- exactly the space the
#     conductivity models index voxels in
#
# .trk/.tck never needed fury (nibabel reads them, already in RAS mm) and go
# through nibabel here too.
# ============================================================================
from pathlib import Path
from typing import List, Sequence, Tuple

import nibabel as nib
import numpy as np
import vtk
from vtk.util import numpy_support as vtknp

# LPS <-> RAS: negate x and y. Its own inverse.
_LPS = np.diag([-1.0, -1.0, 1.0, 1.0])

# remove_invalid_streamlines' tolerance, in voxels.
_BBOX_EPS = 1e-3


# ============================================================================
# Coordinate helpers
# ============================================================================
def _apply(affine: np.ndarray, streamlines: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [nib.affines.apply_affine(affine, np.asarray(s, dtype=np.float64))
            for s in streamlines]


def rasmm_to_vox(streamlines, ref_img) -> List[np.ndarray]:
    """RAS mm -> voxel coordinates of `ref_img`, voxel-centre origin."""
    return _apply(np.linalg.inv(ref_img.affine), streamlines)


def drop_outside_grid(streamlines_vox: Sequence[np.ndarray], shape
                      ) -> Tuple[List[np.ndarray], int]:
    """Drop every streamline with a point outside the grid.

    dipy's rule, kept so a tractogram written here has the same streamline
    set as one written through dipy: judged in corner-origin voxel space
    (vox + 0.5), a point below eps or above dim - eps invalidates its whole
    streamline. Returns (kept, number dropped).
    """
    dims = np.asarray(shape[:3], dtype=np.float64)
    kept = []
    for s in streamlines_vox:
        corner = np.asarray(s, dtype=np.float64) + 0.5
        if corner.min() < _BBOX_EPS or np.any(corner > dims - _BBOX_EPS):
            continue
        kept.append(s)
    return kept, len(streamlines_vox) - len(kept)


# ============================================================================
# Writing
# ============================================================================
def _lines_to_polydata(streamlines: Sequence[np.ndarray]) -> vtk.vtkPolyData:
    """One polyline per streamline. Built from two flat arrays (offsets +
    connectivity) rather than one InsertNextCell per line: a whole-brain
    tractogram is ~10^5 lines and ~10^7 points."""
    lengths = np.array([len(s) for s in streamlines], dtype=np.int64)
    # float32 on disk, as fury wrote: sub-micrometre at brain scale, and half
    # the file (a whole-brain tractogram is a ~120 MB download either way).
    points = np.vstack(streamlines).astype(np.float32)

    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtknp.numpy_to_vtk(points, deep=True))

    offsets = np.concatenate([[0], np.cumsum(lengths)])
    connectivity = np.arange(len(points), dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(vtknp.numpy_to_vtkIdTypeArray(offsets, deep=True),
                  vtknp.numpy_to_vtkIdTypeArray(connectivity, deep=True))

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)
    return polydata


def save_tractogram(streamlines_rasmm: Sequence[np.ndarray], ref_img,
                    path) -> int:
    """Write streamlines given in RAS mm as a legacy binary .vtk.

    Streamlines leaving the reference grid are dropped first (see
    drop_outside_grid). Returns the number of streamlines written.
    """
    path = Path(path)
    if path.suffix.lower() not in (".vtk", ".vtp"):
        raise ValueError(f"save_tractogram writes .vtk/.vtp, not {path.suffix}")

    vox = rasmm_to_vox(streamlines_rasmm, ref_img)
    kept_vox, _dropped = drop_outside_grid(vox, ref_img.shape)
    if not kept_vox:
        raise ValueError("No streamline lies inside the reference grid")
    kept_lps = _apply(_LPS @ ref_img.affine, kept_vox)   # back to mm, flipped

    polydata = _lines_to_polydata(kept_lps)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    else:
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileTypeToBinary()
    writer.SetFileName(str(path))
    writer.SetInputData(polydata)
    if not writer.Write():
        raise OSError(f"Could not write {path}")
    return len(kept_lps)


# ============================================================================
# Reading
# ============================================================================
def _polydata_lines(polydata: vtk.vtkPolyData) -> List[np.ndarray]:
    points = vtknp.vtk_to_numpy(polydata.GetPoints().GetData()).astype(np.float64)
    lines = polydata.GetLines()
    if lines is None or lines.GetNumberOfCells() == 0:
        return []
    offsets = vtknp.vtk_to_numpy(lines.GetOffsetsArray())
    connectivity = vtknp.vtk_to_numpy(lines.GetConnectivityArray())
    return [points[connectivity[a:b]] for a, b in zip(offsets[:-1], offsets[1:])]


def load_tractogram(path, ref_img) -> List[np.ndarray]:
    """Streamlines of a .vtk/.vtp/.trk/.tck file, in VOX space of `ref_img`."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".vtk", ".fib", ".vtp"):
        reader = (vtk.vtkXMLPolyDataReader() if ext == ".vtp"
                  else vtk.vtkPolyDataReader())
        reader.SetFileName(str(path))
        reader.Update()
        polydata = reader.GetOutput()
        if polydata is None or polydata.GetNumberOfPoints() == 0:
            raise ValueError(f"{path.name}: no polydata could be read")
        rasmm = _apply(_LPS, _polydata_lines(polydata))
    elif ext in (".trk", ".tck"):
        # nibabel already applies affine_to_rasmm: these are RAS mm.
        rasmm = [np.asarray(s, dtype=np.float64)
                 for s in nib.streamlines.load(str(path)).tractogram.streamlines]
    else:
        raise ValueError(f"{path.name}: unsupported tractogram format {ext}; "
                         f"use .vtk, .vtp, .trk or .tck")
    return rasmm_to_vox(rasmm, ref_img)
