# ============================================================================
# vendor/gii2vtk — GIFTI cortical surfaces -> legacy VTK PolyData
# ----------------------------------------------------------------------------
# The conversion core of PROJ_DCONDNET/scripts/tools/
# gui___convert_surfaces_gii_to_vtk.py, without its PyQt front-end: reading
# a SimNIBS / CAT12 surface (POINTSET + TRIANGLE arrays), the matching
# FreeSurfer-curv ".sigma" file charm writes, merging the two hemispheres
# into one model, and writing a binary legacy .vtk whose title line carries
# the SPACE=RAS / SPACE=LPS tag 3D Slicer reads to place it correctly.
#
# Kept free of Qt and of anything task-specific so a web task, a cli and the
# GUI tool can all call the same functions.
# ============================================================================
from .convert import (INTENT_POINTSET, INTENT_TRIANGLE, SURFACE_NAMES,
                      convert_m2m_surfaces, merge_surfaces, parse_surface_name,
                      ras_to_lps, read_gii_surface, read_sigma, scan_folder,
                      sigma_path_for, write_vtk_polydata)

__all__ = [
    "INTENT_POINTSET", "INTENT_TRIANGLE", "SURFACE_NAMES",
    "convert_m2m_surfaces", "merge_surfaces", "parse_surface_name",
    "ras_to_lps", "read_gii_surface", "read_sigma", "scan_folder",
    "sigma_path_for", "write_vtk_polydata",
]
