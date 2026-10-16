# ============================================================================
# vendor/gii2vtk/convert.py — the conversion itself
# ----------------------------------------------------------------------------
# Source: PROJ_DCONDNET/scripts/tools/gui___convert_surfaces_gii_to_vtk.py
# (its non-GUI functions, reorganised; behaviour preserved except where noted).
#
# Coordinates: SimNIBS GIFTI surfaces are in scanner RAS (mm) of the subject
# T1. 3D Slicer works in RAS but reads an untagged legacy .vtk as LPS, so an
# untagged RAS file lands rotated 180 deg about the S axis. Every file written
# here is tagged "SPACE=RAS" or "SPACE=LPS" in its title line, which Slicer
# honours (>= 4.11); the LPS option negates X and Y for ITK-based tools.
#
# Merging: lh + rh become one mesh by stacking the vertices and offsetting the
# right hemisphere's face indices. The GUI also attaches a 0/1 "hemisphere"
# scalar; here that is opt-in (hemisphere_scalar=False), so a merged model
# shows in ONE colour unless the caller asks otherwise.
# ============================================================================
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

INTENT_POINTSET = 1008          # NIFTI_INTENT_POINTSET
INTENT_TRIANGLE = 1009          # NIFTI_INTENT_TRIANGLE

# The anatomical surfaces charm writes; the spheres are registrations, not
# anatomy, and are left out by default just as the GUI leaves them unticked.
SURFACE_NAMES = ("central", "pial", "white")


# ============================================================================
# GIFTI / sigma reading
# ============================================================================
def read_gii_surface(path) -> Tuple[np.ndarray, np.ndarray]:
    """(points Nx3 float32, faces Mx3 int32) from a GIFTI surface file."""
    g = nib.load(str(path))
    points = faces = None
    for d in g.darrays:
        if d.intent == INTENT_POINTSET and points is None:
            points = np.asarray(d.data)
        elif d.intent == INTENT_TRIANGLE and faces is None:
            faces = np.asarray(d.data)
    # fall back on shape/dtype if the intent codes are missing
    if points is None or faces is None:
        for d in g.darrays:
            a = np.asarray(d.data)
            if a.ndim == 2 and a.shape[1] == 3:
                if np.issubdtype(a.dtype, np.floating) and points is None:
                    points = a
                elif np.issubdtype(a.dtype, np.integer) and faces is None:
                    faces = a
    if points is None:
        raise ValueError(f"{path}: no POINTSET (vertex) array found")
    if faces is None:
        raise ValueError(f"{path}: no TRIANGLE (face) array found")
    return (np.ascontiguousarray(points, dtype=np.float32),
            np.ascontiguousarray(faces, dtype=np.int32))


def sigma_path_for(gii_path) -> Optional[str]:
    """'.../lh.white.gii' -> '.../lh.white.sigma' if it exists, else None."""
    gii_path = str(gii_path)
    cand = gii_path[:-4] + ".sigma" if gii_path.endswith(".gii") else None
    return cand if cand and os.path.isfile(cand) else None


def read_sigma(path, n_expected: int) -> np.ndarray:
    """A FreeSurfer curv file (charm's per-vertex uncertainty, mm), checked
    against the surface's vertex count."""
    data = np.asarray(nib.freesurfer.io.read_morph_data(str(path)),
                      dtype=np.float32)
    if data.shape[0] != n_expected:
        raise ValueError(f"{os.path.basename(str(path))} has {data.shape[0]} "
                         f"values but the surface has {n_expected} vertices")
    return data


# ============================================================================
# Naming / scanning
# ============================================================================
def parse_surface_name(fname: str) -> Tuple[str, str]:
    """'lh.sphere.reg.gii' -> ('lh', 'sphere.reg'); no hemisphere -> ('', base)."""
    base = fname[:-4] if fname.endswith(".gii") else fname
    parts = base.split(".")
    if parts and parts[0] in ("lh", "rh"):
        return parts[0], ".".join(parts[1:]) or base
    return "", base


def scan_folder(folder) -> List[Dict]:
    """Every *.gii in `folder`, as dicts (path, name, hemi, surf, sigma),
    sorted by surface type then hemisphere."""
    items = []
    for path in sorted(glob.glob(os.path.join(str(folder), "*.gii"))):
        fname = os.path.basename(path)
        hemi, surf = parse_surface_name(fname)
        items.append(dict(path=path, name=fname, hemi=hemi, surf=surf,
                          sigma=sigma_path_for(path)))
    items.sort(key=lambda d: (d["surf"], d["hemi"]))
    return items


# ============================================================================
# Coordinates
# ============================================================================
def ras_to_lps(points: np.ndarray) -> np.ndarray:
    out = np.array(points, copy=True)
    out[:, 0] *= -1.0
    out[:, 1] *= -1.0
    return out


# ============================================================================
# Writing
# ============================================================================
def write_vtk_polydata(path, points: np.ndarray, faces: np.ndarray,
                       arrays: Optional[Sequence[Tuple[str, np.ndarray]]] = None,
                       binary: bool = True, title: Optional[str] = None,
                       space: str = "RAS") -> Path:
    """Write a legacy VTK PolyData file and return its path.

    arrays : (name, 1-D array) pairs attached as point data. The first
             becomes the active SCALARS, the rest go into a FIELD block.
    space  : "RAS" or "LPS" -- what `points` are in. Written into the title
             line as "SPACE=RAS" / "SPACE=LPS" for 3D Slicer (which assumes
             LPS for an untagged file). The points are written AS GIVEN; use
             ras_to_lps() first when writing LPS.
    """
    arrays = list(arrays or [])
    path = Path(path)
    space = space.upper()
    if space not in ("RAS", "LPS"):
        raise ValueError(f"space must be 'RAS' or 'LPS', got {space!r}")
    tag = f" SPACE={space}"
    # VTK caps the title line at 256 characters; keep the tag whatever the name.
    title = (title or path.name)[:250 - len(tag)] + tag

    n_pts, n_cells = points.shape[0], faces.shape[0]
    cells = np.empty((n_cells, 4), dtype=np.int32)       # [3, i, j, k] each
    cells[:, 0] = 3
    cells[:, 1:] = faces

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        def w(text):
            fh.write(text.encode("ascii"))

        w("# vtk DataFile Version 3.0\n")
        w(title + "\n")
        w("BINARY\n" if binary else "ASCII\n")
        w("DATASET POLYDATA\n")

        w(f"POINTS {n_pts} float\n")
        if binary:
            fh.write(np.ascontiguousarray(points).astype(">f4", copy=False).tobytes())
            w("\n")
        else:
            np.savetxt(fh, points, fmt="%.6f")

        w(f"POLYGONS {n_cells} {cells.size}\n")
        if binary:
            fh.write(cells.astype(">i4", copy=False).tobytes())
            w("\n")
        else:
            np.savetxt(fh, cells, fmt="%d")

        if not arrays:
            return path

        w(f"POINT_DATA {n_pts}\n")
        name, values = arrays[0]
        w(f"SCALARS {name} float 1\nLOOKUP_TABLE default\n")
        if binary:
            fh.write(np.ascontiguousarray(values).astype(">f4", copy=False).tobytes())
            w("\n")
        else:
            np.savetxt(fh, np.asarray(values).reshape(-1, 1), fmt="%.6f")

        if len(arrays) > 1:
            w(f"FIELD FieldData {len(arrays) - 1}\n")
            for name, values in arrays[1:]:
                w(f"{name} 1 {n_pts} float\n")
                if binary:
                    fh.write(np.ascontiguousarray(values).astype(">f4", copy=False).tobytes())
                    w("\n")
                else:
                    np.savetxt(fh, np.asarray(values).reshape(-1, 1), fmt="%.6f")
    return path


# ============================================================================
# Merging + the one-call conversion
# ============================================================================
def merge_surfaces(gii_paths: Sequence, with_sigma: bool = True,
                   hemisphere_scalar: bool = False
                   ) -> Tuple[np.ndarray, np.ndarray, List[Tuple[str, np.ndarray]]]:
    """Stack the given surfaces into one mesh: (points, faces, arrays).

    Faces of each later surface are offset by the vertices before it.
    `with_sigma` attaches the ".sigma" values as the active scalar when at
    least one surface has them (zeros for the others). `hemisphere_scalar`
    adds the GUI's 0/1 (lh/rh) array; off by default so the merged model
    shows in one colour.
    """
    pts_l, faces_l, sigmas, hemis = [], [], [], []
    offset, any_sigma = 0, False
    for p in gii_paths:
        pts, faces = read_gii_surface(p)
        pts_l.append(pts)
        faces_l.append(faces + offset)
        sigma_file = sigma_path_for(p) if with_sigma else None
        if sigma_file:
            sigmas.append(read_sigma(sigma_file, pts.shape[0]))
            any_sigma = True
        else:
            sigmas.append(np.zeros(pts.shape[0], np.float32))
        hemi, _surf = parse_surface_name(os.path.basename(str(p)))
        hemis.append(np.full(pts.shape[0], 1.0 if hemi == "rh" else 0.0,
                             np.float32))
        offset += pts.shape[0]
    arrays: List[Tuple[str, np.ndarray]] = []
    if any_sigma:
        arrays.append(("sigma", np.concatenate(sigmas)))
    if hemisphere_scalar and len(gii_paths) > 1:
        arrays.append(("hemisphere", np.concatenate(hemis)))
    return np.vstack(pts_l), np.vstack(faces_l), arrays


def convert_m2m_surfaces(surfaces_dir, out_dir,
                         surfaces: Sequence[str] = SURFACE_NAMES,
                         merge: bool = True, with_sigma: bool = True,
                         hemisphere_scalar: bool = False,
                         space: str = "RAS", binary: bool = True,
                         name_prefix: str = "") -> Dict[str, Path]:
    """Convert charm's surfaces/ folder. Returns {output stem: path}.

    With `merge`, each surface type becomes one "both.<surf>.vtk" holding
    both hemispheres (the GUI's naming); without it, one file per .gii.
    `name_prefix` is put in front of every output name ("IXI034_" ->
    "IXI034_both.central.vtk").
    """
    items = [i for i in scan_folder(surfaces_dir) if i["surf"] in surfaces]
    if not items:
        raise FileNotFoundError(f"no {'/'.join(surfaces)} .gii files in "
                                f"{surfaces_dir}")
    groups: List[Tuple[str, List[str]]] = []
    if merge:
        by_surf: Dict[str, List[Dict]] = {}
        for it in items:
            by_surf.setdefault(it["surf"] if it["hemi"] else it["name"], []).append(it)
        for key, its in by_surf.items():
            its = sorted(its, key=lambda d: d["hemi"])
            stem = f"both.{key}" if len(its) > 1 else os.path.splitext(its[0]["name"])[0]
            groups.append((stem, [i["path"] for i in its]))
    else:
        groups = [(os.path.splitext(i["name"])[0], [i["path"]]) for i in items]

    written: Dict[str, Path] = {}
    for stem, paths in groups:
        pts, faces, arrays = merge_surfaces(paths, with_sigma, hemisphere_scalar)
        if space.upper() == "LPS":
            pts = ras_to_lps(pts)
        out = Path(out_dir) / f"{name_prefix}{stem}.vtk"
        write_vtk_polydata(out, pts, faces, arrays, binary=binary,
                           title=out.name, space=space)
        written[stem] = out
    return written
