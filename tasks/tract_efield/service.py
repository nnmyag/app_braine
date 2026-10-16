# ============================================================================
# tasks/tract_efield/service.py — TMS field on the white surface -> tracts
# ----------------------------------------------------------------------------
# THE seam. UI-agnostic: run_job() takes paths and returns paths, so the web
# page and a later cli.py produce byte-identical run folders.
#
# What one job does
#   1. unpack     the archive mesh_sim produced (<id>_mesh_sim_<type>___output.zip): the
#                 white surface carrying |E| (vtk/), the field NIfTI and the
#                 T1 (subject_volumes/), the run's percentile CSV
#   2. tracts     the whole-brain tractogram (task 2's <id>_wb_tracts.vtk,
#                 or any .vtk/.vtp/.trk/.tck in the same space), with two
#                 checks that it IS in the same space
#   3. hot spot   the white-surface vertices at or above the chosen
#                 percentile of |E| (area-weighted), or an absolute V/m
#   4. select     the streamlines with any point (or an endpoint) within
#                 the contact distance of a hot-spot vertex -- optionally
#                 followed across the breaks tractography leaves in a long
#                 tract (follow_breaks)
#   5. activation the activating function f = -dE_s/ds along every selected
#                 streamline and the terminal field at its ends, from the E
#                 VECTOR volume (activation_along)
#   6. outputs    the selected tracts as three VTKs -- coloured by |E|
#                 along them, by direction, by f -- a one-row CSV, a
#                 per-streamline CSV, three mosaics of identical geometry,
#                 and the archive. No white-surface VTK: mesh_sim's archive
#                 already carries the white surface with |E|, and the hot
#                 spot is only a threshold on that scalar.
#
# Why a separate task from mesh_sim: the tractogram is task 2's product and
# mesh_sim never sees it; the selection is tuned (percentile, distance) far
# more often than a simulation is re-run; and nothing here needs SimNIBS --
# it runs in the app's own environment on files the archive already holds.
#
# Space: the tractogram is in the DWI's RAS mm (task 2), the simulation in
# the head model's, which task 3 rewrote into the DWI's b0 space -- so the
# two overlay directly, and the run says so with numbers rather than
# assuming it (see space_check).
# ============================================================================
import csv
import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

from app import jobs
from tasks.anicond import tractio
from tasks.mesh_sim import service as mesh_sim_service
from tasks.tract_efield import preview
from tasks.tract_efield import settings as task_settings
from vendor import tractviz

logger = logging.getLogger(__name__)

TASK_SLUG = "tract_efield"

TRACT_TYPES = (".vtk", ".vtp", ".trk", ".tck")


# ============================================================================
# What a run produces
# ============================================================================
FILE_MEANING: Dict[str, str] = {
    "tracts_orientation_vtk": "every selected streamline AS TRACKED, coloured "
                              "by direction (red left–right, green "
                              "anterior–posterior, blue superior–inferior; "
                              "the RGB is written into the file), with |E| "
                              "per point and the hop per streamline as plain "
                              "arrays — the file the numbers and the CSVs "
                              "refer to, line for line (LPS on disk, as "
                              "every tractogram this app writes)",
    "tracts_orientation_simplified_vtk":
        f"the selection simplified for viewing (vendor/tractviz: broken "
        f"pieces healed at gap ≤ {task_settings.SIMPLIFY_HEAL_GAP_MM:g} mm, "
        f"≤ {task_settings.SIMPLIFY_HEAL_ANGLE_DEG:g}°; fragments under "
        f"{task_settings.SIMPLIFY_MIN_LENGTH_MM:g} mm dropped; "
        f"{task_settings.SIMPLIFY_KEEP_PERCENT:g} % of every bundle kept; "
        f"smoothed σ {task_settings.SIMPLIFY_SMOOTH_MM:g} mm), coloured by "
        f"direction, with the pieces each streamline was healed from — "
        f"lighter to open in a viewer",
    "tracts_efield_simplified_vtk":
        "the simplified selection coloured by |E| along it (jet, red "
        "strongest, the same window as the pictures), |E| re-sampled on "
        "the simplified points and written as a plain array too",
    "tracts_activation_simplified_vtk":
        "the simplified selection coloured by the activating function "
        "f = -dE_s/ds along it (blue hyperpolarising, red depolarising, "
        "symmetric window), f, E_s and |E| per point re-computed on the "
        "simplified points, and per streamline the terminal field E·t_out "
        "at each end",
    "stats_csv": "this run as one row: threshold, contact distance, "
                 "streamlines selected — for cohort tables",
    "tracts_png": "the selected streamlines as tracked over a darkened "
                  "T1, coloured by direction (red left–right, green "
                  "anterior–posterior, blue superior–inferior) and fading "
                  "with depth from each slice; the gray/white boundary as "
                  "a thin grey line, the hot spot in white; healed pieces "
                  "drawn thinner",
    "tracts_efield_png": "the same mosaic, same geometry, with the "
                         "streamlines coloured by |E| along them instead "
                         "(jet, red strongest; the window is in the legend)",
    "tracts_activation_png": "the same mosaic, same geometry, with the "
                             "streamlines coloured by the activating "
                             "function and a dot at each fibre end coloured "
                             "by its terminal field",
    "activation_csv": "one row per selected streamline AS TRACKED: hop, "
                      "length, peak activating function and where along the "
                      "fibre it is, the terminal field at both ends",
}
FILE_ORDER = ("stats_csv", "activation_csv", "tracts_orientation_vtk",
              "tracts_orientation_simplified_vtk",
              "tracts_efield_simplified_vtk",
              "tracts_activation_simplified_vtk",
              "tracts_png", "tracts_efield_png", "tracts_activation_png")

@dataclass
class Simulation:
    """What was found inside the mesh_sim archive."""
    root: Path
    subject_id: str
    conductivity_type: str
    white_vtk: Path
    field_nii: Path
    vector_nii: Optional[Path] = None      # E as 3 components per voxel
    t1_nii: Optional[Path] = None
    archive_name: str = ""
    # the coil placement, from the simulation's own job spec (if present)
    coil_center: str = ""
    coil_ydir: str = ""


@dataclass
class JobResult:
    """What one completed job produced. Paths and numbers, no arrays."""
    run: jobs.RunDir
    subject_id: str
    conductivity_type: str
    simulation: Simulation
    files: Dict[str, Path] = field(default_factory=dict)
    tractogram_name: str = ""
    # settings the selection ran with
    percentile: float = task_settings.DEFAULT_PERCENTILE
    absolute_v_per_m: float = 0.0
    contact_mm: float = task_settings.DEFAULT_CONTACT_MM
    contact_mode: str = task_settings.DEFAULT_CONTACT_MODE
    follow_breaks: bool = task_settings.DEFAULT_FOLLOW_BREAKS
    join_mm: float = task_settings.DEFAULT_JOIN_MM
    join_max_angle_deg: float = task_settings.JOIN_MAX_ANGLE_DEG
    max_hops: int = task_settings.DEFAULT_MAX_HOPS
    # numbers
    threshold_v_per_m: float = 0.0
    threshold_source: str = ""               # "surface percentile" / "absolute"
    n_vertices: int = 0
    n_hotspot_vertices: int = 0
    hotspot_area_mm2: float = 0.0
    surface_max_v_per_m: float = 0.0
    n_streamlines: int = 0
    n_selected: int = 0                      # touching the hot spot + followed
    n_direct: int = 0                        # touching the hot spot
    n_followed: int = 0                      # added by following breaks
    tract_field: Dict[str, float] = field(default_factory=dict)   # |E| along
    space_check: Dict[str, float] = field(default_factory=dict)
    space_warning: str = ""
    # for the canvas, kept in memory for the session only
    selected_mm: List[np.ndarray] = field(default_factory=list, repr=False)
    selected_hops: Optional[np.ndarray] = field(default=None, repr=False)
    # |E| at every point of every selected streamline, one array each
    selected_e: List[np.ndarray] = field(default_factory=list, repr=False)
    e_color_top: float = 0.0            # the |E| colour window's top (V/m)
    # the activating function per point, one array per streamline, and
    # the terminal field at each streamline's two ends [n, 2]
    selected_f: List[np.ndarray] = field(default_factory=list, repr=False)
    terminal_e: Optional[np.ndarray] = field(default=None, repr=False)
    f_color_top: float = 0.0            # the |f| colour window (V/m^2)
    # The selection SIMPLIFIED (vendor/tractviz, its defaults): what the
    # *___simplified.vtk files carry, with |E|, f and the terminal field
    # re-computed on its points. The numbers, the CSVs, the pictures and
    # tracts_coloredby_orientation.vtk stay with the selection as tracked
    # (above). Empty when nothing survives the simplification.
    simplified_mm: List[np.ndarray] = field(default_factory=list, repr=False)
    simplified_pieces: Optional[np.ndarray] = field(default=None, repr=False)
    simplified_e: List[np.ndarray] = field(default_factory=list, repr=False)
    simplified_f: List[np.ndarray] = field(default_factory=list, repr=False)
    simplified_terminal: Optional[np.ndarray] = field(default=None, repr=False)
    n_simplified: int = 0
    activation: Dict[str, float] = field(default_factory=dict)
    # the white surface (vertices mm, faces, |E|), both hemispheres merged
    # as the VTK holds them
    surface: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = field(
        default=None, repr=False)
    zip_path: Optional[Path] = None
    elapsed_s: float = 0.0
    completed_at: Optional[datetime] = None
    summary: str = ""

    @property
    def selection_line(self) -> str:
        mode = ("either endpoint" if self.contact_mode == "endpoint"
                else "any point")
        follow = (f", broken tracts healed (gap ≤ {self.join_mm:g} mm, "
                  f"≤ {self.join_max_angle_deg:g}°, ≤ {self.max_hops} hops)"
                  if self.follow_breaks else "")
        return (f"|E| ≥ {self.threshold_v_per_m:.3g} V/m "
                f"({self.threshold_source}), {mode} within "
                f"{self.contact_mm:g} mm{follow}")

    def guide(self, roles: Sequence[str] = FILE_ORDER
              ) -> List[Tuple[str, Path, str]]:
        return [(role, self.files[role], FILE_MEANING[role])
                for role in roles
                if role in self.files and Path(self.files[role]).exists()]


# ============================================================================
# Small helpers
# ============================================================================
def say(report: jobs.Reporter, message: str) -> None:
    logger.info(message)
    if report is not None:
        report(message)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray,
                       q: float) -> float:
    """Weighted quantile by the cumulative-weight rule SimNIBS uses and
    mesh_sim's worker copies, so a surface percentile here and a volume
    percentile there are the same kind of number."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    cumulative /= cumulative[-1]
    return float(values[order][np.argmin(np.abs(cumulative - q))])


def vertex_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """A third of each adjacent triangle's area, per vertex."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    tri = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    out = np.zeros(len(vertices))
    for i in range(3):
        np.add.at(out, faces[:, i], tri / 3.0)
    return out


# ============================================================================
# 1. The simulation archive
# ============================================================================
_ARCHIVE_RE = re.compile(r"^(?P<id>.+)_tms_(?P<type>[a-z0-9]+)$")


def identify_zip(zip_path) -> Tuple[str, str]:
    """(subject id, conductivity type) from a mesh_sim archive's index,
    without unpacking: the white VTK inside is named
    <id>_<type>_both.white.magn.vtk. An older archive (before 2026-09-17)
    nested everything under a <id>_tms_<type> folder, which says the same
    and is still honoured."""
    try:
        with zipfile.ZipFile(str(zip_path)) as archive:
            names = archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return "", ""
    for name in names:
        parts = Path(name).parts
        if parts:
            match = _ARCHIVE_RE.match(parts[0])
            if match:
                return match.group("id"), match.group("type")
    for name in names:
        base = Path(name).name
        match = re.match(r"^(?P<id>.+?)_(?P<type>[a-z0-9]+)_both\.white\.magn"
                         r"\.vtk$", base)
        if match:
            return match.group("id"), match.group("type")
    return "", ""


def unpack_simulation(zip_path, dest: Path,
                      report: jobs.Reporter = None) -> Simulation:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as archive:
        members = archive.namelist()
        for name in members:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe path in archive: {name}")
        total = sum(info.file_size for info in archive.infolist())
        say(report, f"Simulation: unpacking {len(members)} files "
                    f"({total / 1e6:.0f} MB) …")
        archive.extractall(dest)

    def find(pattern: str) -> Optional[Path]:
        hits = sorted(p for p in dest.rglob(pattern) if p.is_file())
        return hits[0] if hits else None

    white = find("*both.white.magn.vtk")
    if white is None:
        raise FileNotFoundError(
            "No white-surface field (vtk/<id>_<type>_both.white.magn.vtk) in "
            "the archive. Upload the archive the Head Mesh → TMS Simulation "
            "page produces; runs older than 2026-09-14 did not write it.")
    field_nii = find("*_magnE.nii.gz")
    if field_nii is None:
        raise FileNotFoundError("No E-field magnitude NIfTI (subject_volumes/"
                                "*_magnE.nii.gz) in the archive.")
    subject, cond_type = identify_zip(zip_path)
    if not subject:
        match = re.match(r"^(.+?)_([a-z0-9]+)_both\.white", white.name)
        subject, cond_type = (match.group(1), match.group(2)) if match \
            else (white.name.split("_")[0], "")
    sim = Simulation(root=dest, subject_id=subject,
                     conductivity_type=cond_type, white_vtk=white,
                     field_nii=field_nii, vector_nii=find("*_E.nii.gz"),
                     t1_nii=find("*_T1.nii.gz"),
                     archive_name=Path(str(zip_path)).name)
    spec_path = find("*_job_spec.json")
    if spec_path is not None:
        try:
            spec = json.loads(spec_path.read_text())
            sim.coil_center = str(spec.get("coil_center", "") or "")
            sim.coil_ydir = str(spec.get("coil_ydir", "") or "")
        except (OSError, json.JSONDecodeError):
            pass
    say(report, f"Simulation: subject {subject}, conductivity type "
                f"{cond_type or '?'}" + ("" if sim.t1_nii else
                " — no T1 in the archive, the field volume is the "
                "background"))
    return sim


def read_vtk_surface(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(vertices, faces, |E| per vertex) of a legacy polydata surface --
    what mesh_sim writes. Points are read as stored; those files are RAS
    (their header says SPACE=RAS)."""
    import vtk
    from vtk.util import numpy_support as vtknp
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetNumberOfPoints() == 0:
        raise ValueError(f"{Path(path).name}: no polydata could be read")
    points = vtknp.vtk_to_numpy(poly.GetPoints().GetData()).astype(np.float64)
    polys = poly.GetPolys()
    connectivity = vtknp.vtk_to_numpy(polys.GetConnectivityArray())
    offsets = vtknp.vtk_to_numpy(polys.GetOffsetsArray())
    if not np.all(np.diff(offsets) == 3):
        raise ValueError(f"{Path(path).name}: not a triangle surface")
    faces = connectivity.reshape(-1, 3).astype(np.int64)
    # The field: mesh_sim writes it as a named float array beside the RGB
    # 'Colors' it sets as the active scalars (since 2026-09-14); an older
    # file has the field itself as the active scalars.
    point_data = poly.GetPointData()
    array = point_data.GetArray("E_magnitude_V_per_m")
    if array is None:
        array = point_data.GetScalars()
    if array is None or array.GetNumberOfComponents() != 1:
        raise ValueError(f"{Path(path).name}: carries no one-component field "
                         f"array")
    values = vtknp.vtk_to_numpy(array).astype(np.float64).reshape(-1)
    return points, faces, values


# ============================================================================
# 2. The tractogram, and whether it is in the simulation's space
# ============================================================================
def load_tracts(path, ref_img) -> List[np.ndarray]:
    """Streamlines in RAS mm. tractio returns the reference grid's voxel
    space (what the conductivity models want); here mm is the unit
    everything else is in, so they are put straight back."""
    vox = tractio.load_tractogram(path, ref_img)
    return [nib.affines.apply_affine(ref_img.affine, s) for s in vox]


def space_check(streamlines_mm: Sequence[np.ndarray], ref_img,
                white_vertices: np.ndarray) -> Dict[str, float]:
    """Two fractions: streamline points inside the field grid, and
    streamline endpoints within ENDPOINT_NEAR_MM of the white surface."""
    from scipy.spatial import cKDTree
    points = np.vstack(streamlines_mm)
    vox = preview.to_voxels(ref_img.affine, points)
    dims = np.asarray(ref_img.shape[:3], dtype=float)
    inside = np.all((vox + 0.5 >= 0) & (vox + 0.5 <= dims), axis=1)
    ends = np.vstack([np.stack([s[0], s[-1]]) for s in streamlines_mm])
    distance, _ = cKDTree(white_vertices).query(ends, k=1)
    return {
        "grid_overlap": float(inside.mean()),
        "endpoints_near_white": float(
            (distance <= task_settings.ENDPOINT_NEAR_MM).mean()),
        "endpoint_median_distance_mm": float(np.median(distance)),
    }


# ============================================================================
# 3 + 4. The hot spot and the streamlines that reach it
# ============================================================================
def hotspot_mask(values: np.ndarray, areas: np.ndarray, percentile: float,
                 absolute: float = 0.0) -> Tuple[np.ndarray, float, str]:
    """(mask, threshold, how the threshold was set)."""
    if absolute and absolute > 0:
        threshold, source = float(absolute), "absolute"
    else:
        threshold = _weighted_quantile(values, areas, percentile / 100.0)
        source = f"p{percentile:g} of |E| on the white surface, area-weighted"
    return values >= threshold, threshold, source


def select_streamlines(streamlines_mm: Sequence[np.ndarray],
                       hotspot_mm: np.ndarray, contact_mm: float,
                       mode: str) -> np.ndarray:
    """Boolean mask over the streamlines: which reach the hot spot."""
    from scipy.spatial import cKDTree
    if len(hotspot_mm) == 0 or not streamlines_mm:
        return np.zeros(len(streamlines_mm), dtype=bool)
    tree = cKDTree(hotspot_mm)
    if mode == "endpoint":
        ends = np.vstack([np.stack([s[0], s[-1]]) for s in streamlines_mm])
        distance, _ = tree.query(ends, k=1, distance_upper_bound=contact_mm)
        return np.isfinite(distance).reshape(-1, 2).any(axis=1)
    points = np.vstack(streamlines_mm)
    distance, _ = tree.query(points, k=1, distance_upper_bound=contact_mm)
    near = np.isfinite(distance)
    lengths = np.array([len(s) for s in streamlines_mm])
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    return np.array([near[s:s + n].any() for s, n in zip(starts, lengths)])


def _exit_tangents(streamlines_mm: Sequence[np.ndarray],
                   n_points: int = task_settings.TANGENT_POINTS
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """(endpoints [2N, 3], exit tangents [2N, 3]): for each streamline its
    first and last point, and at each the unit direction pointing OUT of
    the streamline, measured over `n_points` steps in from the end."""
    ends, tangents = [], []
    for s in streamlines_mm:
        k = min(n_points, len(s) - 1)
        for end, inner in ((s[0], s[k]), (s[-1], s[-1 - k])):
            d = np.asarray(end, float) - np.asarray(inner, float)
            n = np.linalg.norm(d)
            ends.append(end)
            tangents.append(d / n if n else np.zeros(3))
    return np.asarray(ends, float), np.asarray(tangents, float)


def follow_breaks(streamlines_mm: Sequence[np.ndarray], selected: np.ndarray,
                  join_mm: float, max_hops: int,
                  max_angle_deg: float = task_settings.JOIN_MAX_ANGLE_DEG
                  ) -> np.ndarray:
    """Hop number per streamline: 0 for the selection given, 1.. for the
    pieces reached by following breaks, -1 for the rest.

    A piece B continues a selected piece A when an endpoint of B lies within
    `join_mm` of an endpoint of A and the two exit tangents are within
    `max_angle_deg` of head-to-tail (A leaves toward B, B leaves toward A:
    the dot product of the two OUTWARD tangents is at most -cos(angle)).
    Two streamlines that merely terminate side by side at the cortex leave
    in the same direction and are not joined -- the test that keeps this
    from spreading over the whole gray/white boundary.
    """
    from scipy.spatial import cKDTree
    hop = np.where(selected, 0, -1).astype(np.int64)
    if len(streamlines_mm) == 0 or not selected.any() or max_hops < 1:
        return hop
    ends, tangents = _exit_tangents(streamlines_mm)
    tree = cKDTree(ends)
    limit = -np.cos(np.radians(max_angle_deg))
    frontier = np.flatnonzero(np.repeat(selected, 2))     # endpoint indices
    for level in range(1, int(max_hops) + 1):
        added = []
        for e in frontier:
            for j in tree.query_ball_point(ends[e], join_mm):
                line = j // 2
                if hop[line] >= 0 or line == e // 2:
                    continue
                if float(tangents[e] @ tangents[j]) <= limit:
                    hop[line] = level
                    added.append(j ^ 1)         # the piece's OTHER endpoint
        if not added:
            break
        frontier = np.asarray(added)
    return hop


# ============================================================================
# 5. The activating function along the selected streamlines
# ----------------------------------------------------------------------------
# See settings.py ("Activating function") for what is computed and why.
# ============================================================================
def sample_vector_field(vector_img, points_mm: np.ndarray,
                        min_weight: float = task_settings.ACTIVATION_MIN_MASK_WEIGHT
                        ) -> np.ndarray:
    """E [n, 3] at world points, mask-aware: the NIfTI is zero outside the
    tissues it was written for, so a plain trilinear sample within a voxel
    of the boundary is blended with zero. Interpolating E*mask and mask
    separately and dividing gives the field of the tissue side alone; a
    point whose mask weight is below `min_weight` lies outside and reads
    zero."""
    from scipy.ndimage import map_coordinates
    data = np.asarray(vector_img.dataobj, dtype=np.float64)
    mask = np.any(data != 0, axis=-1).astype(np.float64)
    voxel = preview.to_voxels(vector_img.affine, points_mm).T
    weight = map_coordinates(mask, voxel, order=1, mode="constant", cval=0.0)
    out = np.zeros((len(points_mm), 3))
    inside = weight >= min_weight
    for i in range(3):
        component = map_coordinates(data[..., i] * mask, voxel, order=1,
                                    mode="constant", cval=0.0)
        out[inside, i] = component[inside] / weight[inside]
    return out


def activation_along(streamlines_mm: Sequence[np.ndarray], vector_img,
                     smooth_mm: float = task_settings.ACTIVATION_SMOOTH_MM
                     ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """(E_s per streamline, f per streamline, terminal field [n, 2]).

    E_s = E · t along arc length s, smoothed with a Gaussian of `smooth_mm`
    along the fibre; f = -dE_s/ds in V/m^2 (s converted to metres). The
    terminal field is E · t_out at the first and last point, t_out the
    tangent pointing out of the fibre through that end.
    """
    from scipy.ndimage import gaussian_filter1d
    points = np.vstack(streamlines_mm)
    e_all = sample_vector_field(vector_img, points)
    splits = np.cumsum([len(s) for s in streamlines_mm])[:-1]
    e_s_list, f_list, terminal = [], [], []
    for s, e in zip(streamlines_mm, np.split(e_all, splits)):
        s = np.asarray(s, dtype=np.float64)
        n = len(s)
        if n < 3:
            e_s_list.append(np.zeros(n))
            f_list.append(np.zeros(n))
            terminal.append((0.0, 0.0))
            continue
        d = np.empty_like(s)
        d[0], d[-1], d[1:-1] = s[1] - s[0], s[-1] - s[-2], s[2:] - s[:-2]
        norm = np.linalg.norm(d, axis=1, keepdims=True)
        t = d / np.where(norm == 0, 1.0, norm)
        e_s = np.einsum("ij,ij->i", e, t)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
            np.diff(s, axis=0), axis=1))])
        step = arc[-1] / max(n - 1, 1)
        sigma = smooth_mm / step if step > 0 else 0.0
        smooth = (gaussian_filter1d(e_s, sigma, mode="nearest") if sigma > 0
                  else e_s)
        f = -np.gradient(smooth, arc / 1000.0)      # s in metres -> V/m^2
        e_s_list.append(e_s)
        f_list.append(f)
        # out of the fibre: at the start, back along -t; at the end, along t.
        terminal.append((float(e[0] @ -t[0]), float(e[-1] @ t[-1])))
    return e_s_list, f_list, np.asarray(terminal, dtype=np.float64)


# ============================================================================
# 6. Outputs
# ============================================================================
def point_directions(streamlines_mm: Sequence[np.ndarray]) -> np.ndarray:
    """[N, 3] unsigned unit direction at every point, the local tangent by
    central difference (one-sided at the ends) -- what a tractography
    viewer colours by."""
    out = []
    for s in streamlines_mm:
        s = np.asarray(s, dtype=np.float64)
        if len(s) < 2:
            out.append(np.zeros((len(s), 3)))
            continue
        d = np.empty_like(s)
        d[0] = s[1] - s[0]
        d[-1] = s[-1] - s[-2]
        d[1:-1] = s[2:] - s[:-2]
        n = np.linalg.norm(d, axis=1, keepdims=True)
        out.append(np.abs(d / np.where(n == 0, 1.0, n)))
    return np.vstack(out) if out else np.empty((0, 3))


def _export_vtk(files: Dict[str, Path], role: str, path: Path,
                streamlines_mm: Sequence[np.ndarray], colors: np.ndarray,
                report: jobs.Reporter, **arrays) -> None:
    """write_tracts_vtk into `files[role]`, reported; a failure is logged
    and said, never raised -- an export, not the result."""
    try:
        files[role] = write_tracts_vtk(path, streamlines_mm, colors, **arrays)
        say(report, f"VTK: {files[role].name} "
                    f"({files[role].stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:              # noqa: BLE001
        logger.exception("tract_efield %s failed", role)
        say(report, f"VTK: {path.name} skipped ({type(exc).__name__}: {exc})")


def jet_colors(values: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """[N, 3] RGB in 0..1, jet, clipped to [vmin, vmax]."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    return colormaps["jet"](Normalize(vmin, vmax, clip=True)(values))[:, :3]


def write_tracts_vtk(path: Path, streamlines_mm: Sequence[np.ndarray],
                     colors: np.ndarray,
                     point_arrays: Sequence[Tuple[str, np.ndarray]] = (),
                     cell_arrays: Sequence[Tuple[str, np.ndarray]] = (),
                     cell_float_arrays: Sequence[Tuple[str, np.ndarray]] = ()
                     ) -> Path:
    """The selection as legacy binary polydata: polylines whose points carry
    an explicit RGB ('Colors', unsigned char x3, the active scalars, so a
    viewer shows the colours as they are without a colour map), plus any
    numeric per-point and per-line arrays. LPS on disk, tractio's
    convention, so it lands where task 2's tractogram lands in 3D Slicer;
    the header title says so."""
    import vtk
    from vtk.util import numpy_support as vtknp
    lengths = np.array([len(s) for s in streamlines_mm], dtype=np.int64)
    points = np.vstack(streamlines_mm).astype(np.float64)
    points[:, :2] *= -1.0                                     # RAS -> LPS
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtknp.numpy_to_vtk(points.astype(np.float32), deep=True))
    offsets = np.concatenate([[0], np.cumsum(lengths)])
    cells = vtk.vtkCellArray()
    cells.SetData(vtknp.numpy_to_vtkIdTypeArray(offsets, deep=True),
                  vtknp.numpy_to_vtkIdTypeArray(
                      np.arange(len(points), dtype=np.int64), deep=True))
    poly = vtk.vtkPolyData()
    poly.SetPoints(vtk_points)
    poly.SetLines(cells)

    rgb = np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0)
    rgb = np.rint(rgb * 255).astype(np.uint8)
    color_array = vtknp.numpy_to_vtk(np.ascontiguousarray(rgb), deep=True,
                                     array_type=vtk.VTK_UNSIGNED_CHAR)
    color_array.SetName("Colors")
    poly.GetPointData().SetScalars(color_array)
    for name, values in point_arrays:
        array = vtknp.numpy_to_vtk(np.asarray(values, dtype=np.float32),
                                   deep=True)
        array.SetName(name)
        poly.GetPointData().AddArray(array)
    for name, values in cell_arrays:
        array = vtknp.numpy_to_vtk(np.asarray(values, dtype=np.int32),
                                   deep=True)
        array.SetName(name)
        poly.GetCellData().AddArray(array)
    for name, values in cell_float_arrays:
        array = vtknp.numpy_to_vtk(np.asarray(values, dtype=np.float32),
                                   deep=True)
        array.SetName(name)
        poly.GetCellData().AddArray(array)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileTypeToBinary()
    writer.SetHeader(f"{path.name} SPACE=LPS")
    # The 4.2 legacy layout (32-bit cell lists), not the writer's default
    # 5.1 (64-bit offsets + connectivity): every reader opens it, and a
    # half-million-vertex surface is 40% smaller.
    writer.SetFileVersion(42)
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    if not writer.Write():
        raise OSError(f"Could not write {path}")
    return path


STATS_COLUMNS = ("subject_id", "conductivity_type", "percentile",
                 "threshold_v_per_m", "threshold_source", "contact_mm",
                 "n_streamlines", "n_selected", "fraction_selected",
                 "activation_max_v_per_m2", "activation_p95_abs_v_per_m2",
                 "e_terminal_max_v_per_m")


ACTIVATION_COLUMNS = ("streamline", "hop", "length_mm",
                      "f_peak_v_per_m2", "f_peak_at_mm", "f_peak_at_fraction",
                      "e_terminal_start_v_per_m", "e_terminal_end_v_per_m",
                      "e_magnitude_max_v_per_m")


def write_activation_csv(result: JobResult, path: Path) -> Path:
    """One row per selected streamline: where along it the activating
    function peaks, and the terminal field at both ends."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ACTIVATION_COLUMNS)
        writer.writeheader()
        for i, (s, f, e) in enumerate(zip(result.selected_mm,
                                          result.selected_f,
                                          result.selected_e)):
            arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(
                np.diff(np.asarray(s), axis=0), axis=1))])
            k = int(np.argmax(np.abs(f))) if len(f) else 0
            length = float(arc[-1]) if len(arc) else 0.0
            term = (result.terminal_e[i] if result.terminal_e is not None
                    else (0.0, 0.0))
            writer.writerow({
                "streamline": i,
                "hop": int(result.selected_hops[i])
                       if result.selected_hops is not None else 0,
                "length_mm": f"{length:.1f}",
                "f_peak_v_per_m2": f"{f[k]:.6g}" if len(f) else "",
                "f_peak_at_mm": f"{arc[k]:.1f}" if len(f) else "",
                "f_peak_at_fraction": (f"{arc[k] / length:.3f}"
                                       if length > 0 and len(f) else ""),
                "e_terminal_start_v_per_m": f"{term[0]:.6g}",
                "e_terminal_end_v_per_m": f"{term[1]:.6g}",
                "e_magnitude_max_v_per_m": f"{np.max(e):.6g}" if len(e) else "",
            })
    return path


def write_stats_csv(result: JobResult, path: Path) -> Path:
    """One row per run, for concatenation across a cohort."""
    row = {
        "subject_id": result.subject_id,
        "conductivity_type": result.conductivity_type,
        "percentile": f"{result.percentile:g}",
        "threshold_v_per_m": f"{result.threshold_v_per_m:.6g}",
        "threshold_source": result.threshold_source,
        "contact_mm": f"{result.contact_mm:g}",
        "n_streamlines": result.n_streamlines,
        "n_selected": result.n_selected,
        "fraction_selected": (f"{result.n_selected / result.n_streamlines:.6g}"
                              if result.n_streamlines else "0"),
        "activation_max_v_per_m2": (f"{result.activation['max']:.6g}"
                                    if "max" in result.activation else ""),
        "activation_p95_abs_v_per_m2": (f"{result.activation['p95_abs']:.6g}"
                                        if "p95_abs" in result.activation
                                        else ""),
        "e_terminal_max_v_per_m": (f"{result.activation['terminal_max']:.6g}"
                                   if "terminal_max" in result.activation
                                   else ""),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_COLUMNS)
        writer.writeheader()
        writer.writerow(row)
    return path


# ============================================================================
# The job
# ============================================================================
def run_job(simulation_zip, tractogram,
            percentile: float = task_settings.DEFAULT_PERCENTILE,
            absolute_v_per_m: float = task_settings.DEFAULT_ABSOLUTE_V_PER_M,
            contact_mm: float = task_settings.DEFAULT_CONTACT_MM,
            contact_mode: str = task_settings.DEFAULT_CONTACT_MODE,
            follow_breaks_on: bool = task_settings.DEFAULT_FOLLOW_BREAKS,
            join_mm: float = task_settings.DEFAULT_JOIN_MM,
            join_max_angle_deg: float = task_settings.JOIN_MAX_ANGLE_DEG,
            max_hops: int = task_settings.DEFAULT_MAX_HOPS,
            run: Optional[jobs.RunDir] = None,
            report: jobs.Reporter = None) -> JobResult:
    """Select the streamlines reaching the white-surface hot spot of one
    TMS simulation."""
    t_start = time.time()
    if not simulation_zip:
        raise ValueError("The simulation archive (.zip) is required.")
    if not tractogram:
        raise ValueError("A tractogram (.vtk / .vtp / .trk / .tck) is "
                         "required.")
    if Path(str(tractogram)).suffix.lower() not in TRACT_TYPES:
        raise ValueError(f"Tractogram: {Path(str(tractogram)).name} is not "
                         f"one of {', '.join(TRACT_TYPES)}")
    percentile = float(np.clip(percentile, task_settings.PERCENTILE_MIN,
                               task_settings.PERCENTILE_MAX))
    contact_mm = float(np.clip(contact_mm, task_settings.CONTACT_MM_MIN,
                               task_settings.CONTACT_MM_MAX))
    if contact_mode not in {v for _l, v in task_settings.CONTACT_MODES}:
        contact_mode = task_settings.DEFAULT_CONTACT_MODE
    absolute_v_per_m = max(float(absolute_v_per_m or 0.0), 0.0)
    join_mm = float(np.clip(join_mm, task_settings.JOIN_MM_MIN,
                            task_settings.JOIN_MM_MAX))
    join_max_angle_deg = float(np.clip(join_max_angle_deg,
                                       task_settings.JOIN_ANGLE_MIN,
                                       task_settings.JOIN_ANGLE_MAX))
    max_hops = int(np.clip(int(max_hops), task_settings.MAX_HOPS_MIN,
                           task_settings.MAX_HOPS_MAX))

    run = run or jobs.new_run(TASK_SLUG)
    say(report, f"Job {run.job_id}")

    # -- 1. the simulation --------------------------------------------------
    sim = unpack_simulation(simulation_zip, run.inputs / "simulation", report)
    if sim.conductivity_type == "iso":
        say(report, "Note: the simulation is isotropic ('iso'). The selection "
                    "works the same; the study's question is about "
                    "anisotropic runs.")
    vertices, faces, values = read_vtk_surface(sim.white_vtk)
    areas = vertex_areas(vertices, faces)
    say(report, f"White surface: {len(vertices):,} vertices, |E| up to "
                f"{values.max():.3g} V/m")
    ref_img = nib.load(str(sim.field_nii))

    # -- 2. the tractogram --------------------------------------------------
    staged = jobs.stage_upload(tractogram, run,
                               f"{sim.subject_id}_tracts"
                               f"{Path(str(tractogram)).suffix.lower()}")
    say(report, f"Tracts: loading {Path(str(tractogram)).name} …")
    streamlines = load_tracts(staged, ref_img)
    if not streamlines:
        raise ValueError(f"{Path(str(tractogram)).name} contains no "
                         f"streamline")
    check = space_check(streamlines, ref_img, vertices)
    say(report, f"Tracts: {len(streamlines):,} streamlines; "
                f"{check['grid_overlap']:.1%} of their points inside the "
                f"field grid, {check['endpoints_near_white']:.1%} of "
                f"endpoints within {task_settings.ENDPOINT_NEAR_MM:g} mm of "
                f"the white surface (median "
                f"{check['endpoint_median_distance_mm']:.1f} mm)")
    warning = ""
    if check["grid_overlap"] < task_settings.MIN_GRID_OVERLAP:
        warning = (f"only {check['grid_overlap']:.0%} of the streamline "
                   f"points fall inside the field volume: the tractogram is "
                   f"probably not in this simulation's space (was the head "
                   f"model registered to this DWI's b0?).")
    elif check["endpoints_near_white"] < task_settings.MIN_ENDPOINT_NEAR_FRACTION:
        warning = (f"only {check['endpoints_near_white']:.0%} of streamline "
                   f"endpoints lie within {task_settings.ENDPOINT_NEAR_MM:g} "
                   f"mm of the white surface: the tractogram and the "
                   f"simulation may not share a space, or the tracking "
                   f"stopped far from the cortex.")
    if warning:
        say(report, f"Tracts: WARNING — {warning}")

    # -- 3. the hot spot ----------------------------------------------------
    mask, threshold, source = hotspot_mask(values, areas, percentile,
                                           absolute_v_per_m)
    hotspot = vertices[mask]
    say(report, f"Hot spot: |E| ≥ {threshold:.3g} V/m ({source}) — "
                f"{int(mask.sum()):,} vertices, {areas[mask].sum():.0f} mm²")

    # -- 4. the streamlines that reach it -----------------------------------
    selected = select_streamlines(streamlines, hotspot, contact_mm,
                                  contact_mode)
    n_direct = int(selected.sum())
    say(report, f"Selection: {n_direct:,} of {len(streamlines):,} "
                f"streamlines have "
                + ("an endpoint" if contact_mode == "endpoint" else "a point")
                + f" within {contact_mm:g} mm of the hot spot")
    hops = np.where(selected, 0, -1)
    if follow_breaks_on:
        hops = follow_breaks(streamlines, selected, join_mm, max_hops,
                             max_angle_deg=join_max_angle_deg)
        per_hop = [int((hops == h).sum()) for h in range(1, max_hops + 1)]
        say(report, f"Heal: +{sum(per_hop):,} streamlines continue a "
                    f"selected one head-to-tail (endpoint gap ≤ {join_mm:g} "
                    f"mm, ≤ {join_max_angle_deg:g}°) — per hop "
                    f"{', '.join(f'{n:,}' for n in per_hop)}")
    keep = hops >= 0
    chosen = [s for s, k in zip(streamlines, keep) if k]
    chosen_hops = hops[keep]

    result = JobResult(run=run, subject_id=sim.subject_id,
                       conductivity_type=sim.conductivity_type, simulation=sim,
                       tractogram_name=Path(str(tractogram)).name,
                       percentile=percentile, absolute_v_per_m=absolute_v_per_m,
                       contact_mm=contact_mm, contact_mode=contact_mode,
                       follow_breaks=bool(follow_breaks_on), join_mm=join_mm,
                       join_max_angle_deg=join_max_angle_deg,
                       max_hops=max_hops,
                       threshold_v_per_m=threshold, threshold_source=source,
                       n_vertices=len(vertices),
                       n_hotspot_vertices=int(mask.sum()),
                       hotspot_area_mm2=float(areas[mask].sum()),
                       surface_max_v_per_m=float(values.max()),
                       n_streamlines=len(streamlines), n_selected=len(chosen),
                       n_direct=n_direct, n_followed=len(chosen) - n_direct,
                       space_check=check, space_warning=warning,
                       selected_mm=chosen, selected_hops=chosen_hops,
                       surface=(vertices, faces, values))

    # -- 5. outputs ---------------------------------------------------------
    files = result.files
    tag = f"{sim.subject_id}_{sim.conductivity_type}"

    field_volume = np.asarray(ref_img.dataobj, dtype=np.float64)
    vtk_dir = run.outputs / "vtk"
    if chosen:
        # -- 5a. the numbers: |E| along the selection AS TRACKED ------------
        points = np.vstack(chosen)
        along = preview.sample_volume_at(field_volume, ref_img.affine, points)
        result.tract_field = {"mean": float(along.mean()),
                              "median": float(np.median(along)),
                              "p95": float(np.percentile(along, 95)),
                              "max": float(along.max())}
        say(report, f"Field along the selection: mean "
                    f"{result.tract_field['mean']:.3g}, p95 "
                    f"{result.tract_field['p95']:.3g} V/m")
        top = float(np.percentile(along, task_settings.TRACT_COLOR_PCTL))
        result.selected_e = list(np.split(
            along, np.cumsum([len(s) for s in chosen])[:-1]))
        result.e_color_top = top
        # The selection as tracked, coloured by direction: the one file the
        # numbers and the CSVs refer to line for line.
        _export_vtk(files, "tracts_orientation_vtk",
                    vtk_dir / f"{tag}_tracts_coloredby_orientation.vtk",
                    chosen, point_directions(chosen), report,
                    point_arrays=[("E_magnitude_V_per_m", along)],
                    cell_arrays=[("hop", chosen_hops)])

        # -- 5b. the selection simplified for viewing -----------------------
        # vendor/tractviz at this task's own settings (SIMPLIFY_*, gentler
        # than tract_simplify's page defaults). The coloured *___simplified
        # .vtk files carry this copy -- lighter to open in a viewer -- with
        # |E| re-sampled on its points; the numbers and the pictures never
        # see it.
        simplified = None
        try:
            simplified = tractviz.simplify_tracts(
                chosen,
                heal=task_settings.SIMPLIFY_HEAL,
                heal_gap_mm=task_settings.SIMPLIFY_HEAL_GAP_MM,
                heal_angle_deg=task_settings.SIMPLIFY_HEAL_ANGLE_DEG,
                min_length_mm=task_settings.SIMPLIFY_MIN_LENGTH_MM,
                keep_percent=task_settings.SIMPLIFY_KEEP_PERCENT,
                smooth_mm=task_settings.SIMPLIFY_SMOOTH_MM,
                report=lambda line: say(report, f"Simplify: {line}"))
        except Exception as exc:      # noqa: BLE001 - a display copy, not the result
            logger.exception("tract_efield simplification failed")
            say(report, f"Simplify: failed ({type(exc).__name__}: {exc}); "
                        f"no *___simplified.vtk this run")
        if simplified is not None and simplified.streamlines:
            simp = simplified.streamlines
            result.simplified_mm = simp
            result.simplified_pieces = simplified.prepared.pieces[
                simplified.kept]
            result.n_simplified = len(simp)
            along_s = preview.sample_volume_at(field_volume, ref_img.affine,
                                               np.vstack(simp))
            result.simplified_e = list(np.split(
                along_s, np.cumsum([len(s) for s in simp])[:-1]))
            simp_arrays = dict(
                point_arrays=[("E_magnitude_V_per_m", along_s)],
                cell_arrays=[("pieces", result.simplified_pieces)])
            _export_vtk(files, "tracts_orientation_simplified_vtk",
                        vtk_dir / f"{tag}_tracts_coloredby_orientation"
                                  f"___simplified.vtk",
                        simp, point_directions(simp), report, **simp_arrays)
            _export_vtk(files, "tracts_efield_simplified_vtk",
                        vtk_dir / f"{tag}_tracts_coloredby_efield"
                                  f"___simplified.vtk",
                        simp, jet_colors(along_s, 0.0, top), report,
                        **simp_arrays)
            say(report, f"VTK: |E| colours span 0 to {top:.3g} V/m "
                        f"(p{task_settings.TRACT_COLOR_PCTL:g} along the "
                        f"selection as tracked)")
        elif simplified is not None:
            say(report, "Simplify: no streamline survives (all shorter than "
                        "the length cut); no *___simplified.vtk this run")

        # -- 5c. the activating function ------------------------------------
        # The numbers, the CSV and the picture on the selection as tracked;
        # the coloured *___simplified.vtk on the simplified copy, f
        # re-computed there.
        if sim.vector_nii is None:
            say(report, "Activation: no E vector volume (subject_volumes/"
                        "*_E.nii.gz) in the archive, skipped")
        else:
            try:
                say(report, "Activation: E_s and f = -dE_s/ds along the "
                            "selected streamlines …")
                vector_img = nib.load(str(sim.vector_nii))
                e_s, f_list, terminal = activation_along(chosen, vector_img)
                f_all = np.concatenate(f_list)
                f_top = float(np.percentile(np.abs(f_all),
                                            task_settings.ACTIVATION_COLOR_PCTL))
                result.selected_f, result.terminal_e = f_list, terminal
                result.f_color_top = f_top
                result.activation = {
                    "max": float(f_all.max()),
                    "min": float(f_all.min()),
                    "p95_abs": float(np.percentile(np.abs(f_all), 95)),
                    "terminal_max": float(terminal.max()),
                }
                say(report, f"Activation: f from {result.activation['min']:.3g} "
                            f"to {result.activation['max']:.3g} V/m² "
                            f"(p95 |f| {result.activation['p95_abs']:.3g}); "
                            f"terminal field up to "
                            f"{result.activation['terminal_max']:.3g} V/m; "
                            f"colours ±{f_top:.3g} V/m²")
                files["activation_csv"] = write_activation_csv(
                    result, run.output(f"{sim.subject_id}_tracts_activation"
                                       f".csv"))
                if result.simplified_mm:
                    e_s_s, f_s, term_s = activation_along(result.simplified_mm,
                                                          vector_img)
                    result.simplified_f = f_s
                    result.simplified_terminal = term_s
                    f_all_s = np.concatenate(f_s)
                    colors = preview.ACTIVATION_CMAP(
                        np.clip(f_all_s / (2 * f_top) + 0.5, 0, 1))[:, :3] \
                        if f_top > 0 else np.full((len(f_all_s), 3), 0.42)
                    _export_vtk(
                        files, "tracts_activation_simplified_vtk",
                        vtk_dir / f"{tag}_tracts_coloredby_activation"
                                  f"___simplified.vtk",
                        result.simplified_mm, colors, report,
                        point_arrays=[("activation_V_per_m2", f_all_s),
                                      ("E_tangential_V_per_m",
                                       np.concatenate(e_s_s)),
                                      ("E_magnitude_V_per_m",
                                       np.concatenate(result.simplified_e))],
                        cell_arrays=[("pieces", result.simplified_pieces)],
                        cell_float_arrays=[("E_terminal_start_V_per_m",
                                            term_s[:, 0]),
                                           ("E_terminal_end_V_per_m",
                                            term_s[:, 1])])
            except Exception as exc:      # noqa: BLE001 - an analysis, not the selection
                logger.exception("tract_efield activation failed")
                say(report, f"Activation: skipped ({type(exc).__name__}: "
                            f"{exc})")
    else:
        say(report, "Selection: no streamline reaches the hot spot — try a "
                    "larger contact distance or a lower percentile")

    say(report, "Previews: the connected tracts, by direction, by |E| and "
                "by activation …")
    for role, name, by in (
            ("tracts_png", f"{sim.subject_id}_field_with_tracts_"
                                 f"{sim.conductivity_type}.png", "orientation"),
            ("tracts_efield_png", f"{sim.subject_id}_field_with_tracts_"
                                        f"by_efield_{sim.conductivity_type}"
                                        f".png", "efield"),
            ("tracts_activation_png",
             f"{sim.subject_id}_field_with_tracts_by_activation_"
             f"{sim.conductivity_type}.png", "activation")):
        try:
            image = _tracts_view(result, saved=True, by=by)
            if image is not None:
                files[role] = run.output(name)
                preview.save_png(image, files[role])
        except Exception as exc:          # noqa: BLE001 - a picture, not the result
            logger.exception("tract_efield preview %s failed", role)
            say(report, f"Previews: {role} skipped ({type(exc).__name__}: "
                        f"{exc})")

    files["stats_csv"] = write_stats_csv(
        result, run.output(f"{sim.subject_id}_tracts_stats.csv"))
    result.completed_at = datetime.now()
    say(report, "Archiving outputs/ …")
    result.zip_path = run.zip_outputs(subject_id=sim.subject_id,
                                      model=sim.conductivity_type)
    result.elapsed_s = time.time() - t_start
    result.summary = _summarize(result)
    return result


def _summarize(result: JobResult) -> str:
    lines = [f"Done in {result.elapsed_s / 60:.1f} min — "
             f"{result.selection_line}",
             f"{result.n_selected:,} of {result.n_streamlines:,} streamlines "
             f"selected ({result.n_direct:,} touching the hot spot"
             + (f" + {result.n_followed:,} followed" if result.n_followed
                else "")
             + f"; hot spot {result.n_hotspot_vertices:,} vertices, "
             f"{result.hotspot_area_mm2:.0f} mm²)"]
    if result.space_warning:
        lines.append(f"WARNING: {result.space_warning}")
    if result.zip_path:
        lines.append(f"Archive: {Path(result.zip_path).name}, "
                     f"{Path(result.zip_path).stat().st_size / 1e6:.0f} MB")
    return "\n".join(lines)


# ============================================================================
# Views for the canvas
# ============================================================================
VIEW_T1 = "T1 (from the simulation archive)"
VIEW_TRACTS = "Connected tracts — coloured by direction"
VIEW_TRACTS_EFIELD = "Connected tracts — coloured by |E|"
VIEW_TRACTS_ACTIVATION = "Connected tracts — coloured by activating function"
VIEW_LABELS = (VIEW_T1, VIEW_TRACTS, VIEW_TRACTS_EFIELD,
               VIEW_TRACTS_ACTIVATION)
DEFAULT_VIEW = VIEW_T1
RESULT_VIEW = VIEW_TRACTS


def view_labels() -> List[str]:
    return list(VIEW_LABELS)


def _figure_kwargs(saved: bool) -> Dict:
    """The one difference between the canvas picture and the saved one:
    resolution and line weight. Both are above the app's defaults here --
    this mosaic is read one slice at a time."""
    if saved:
        return {"dpi": task_settings.FIGURE_DPI,
                "slice_fontsize": task_settings.SLICE_FONTSIZE,
                "linewidth": task_settings.TRACT_LINEWIDTH_SAVED,
                "surface_linewidth": task_settings.SURFACE_LINEWIDTH_SAVED}
    return {"dpi": task_settings.CANVAS_DPI,
            "slice_fontsize": task_settings.SLICE_FONTSIZE,
            "linewidth": task_settings.TRACT_LINEWIDTH_CANVAS,
            "surface_linewidth": task_settings.SURFACE_LINEWIDTH_CANVAS}


def _background(result: JobResult):
    """(T1, affine) on the FIELD's grid -- the grid the streamlines and the
    surface are drawn against. The archive's T1 is on that grid already
    (SimNIBS writes the field on the T1's grid); resampled if ever not.
    Without a T1 in the archive, the field volume itself is the anatomy."""
    field, affine = preview.load_canonical(result.simulation.field_nii)
    if result.simulation.t1_nii:
        t1, t1_affine = preview.load_canonical(result.simulation.t1_nii)
        if t1.shape != field.shape or not np.allclose(t1_affine, affine,
                                                       atol=1e-3):
            t1 = preview.resample_to_grid(t1, t1_affine, field.shape, affine)
        return t1, affine
    return field, affine


def _drawn(result: JobResult):
    """The drawn subset of the selection AS TRACKED (subsampled when large
    -- the same subset for every picture), as a dict of aligned pieces:
    streamlines, hops, |E| per point, f per point, the terminal field
    [n, 2]. The simplified copy is for the *___simplified.vtk files only:
    at its defaults it keeps a tenth of every bundle, too few to make a
    picture of a small selection."""
    chosen = result.selected_mm
    limit = task_settings.MAX_DRAWN_STREAMLINES
    keep = np.arange(len(chosen))
    if len(chosen) > limit:
        rng = np.random.default_rng(task_settings.DRAW_SEED)
        keep = np.sort(rng.choice(len(chosen), size=limit, replace=False))

    def pick(items):
        return [items[i] for i in keep] if items else None

    return {"streamlines": pick(chosen),
            "hops": (result.selected_hops[keep]
                     if result.selected_hops is not None else None),
            "e": pick(result.selected_e), "f": pick(result.selected_f),
            "terminal": (result.terminal_e[keep]
                         if result.terminal_e is not None else None)}


def _legend(result: JobResult, by: str = "orientation") -> str:
    """Two short lines for the strip under the tiles."""
    drawn = min(len(result.selected_mm), task_settings.MAX_DRAWN_STREAMLINES)
    which = ("absolute" if result.threshold_source == "absolute"
             else f"p{result.percentile:g} of the white surface")
    first = (f"hot spot: |E| ≥ {result.threshold_v_per_m:.3g} V/m ({which}), "
             f"{result.n_hotspot_vertices:,} vertices, "
             f"{result.hotspot_area_mm2:.0f} mm²")
    second = (f"tracts: {result.n_selected:,} of {result.n_streamlines:,}"
              + (f" ({result.n_direct:,} touching + {result.n_followed:,} "
                 f"followed)" if result.n_followed else "")
              + (f", {drawn:,} drawn" if drawn < result.n_selected else "")
              + (" · red L–R, green A–P, blue S–I" if by == "orientation"
                 else f" · colour: |E| along the tract, jet 0 (blue) → "
                      f"{result.e_color_top:.3g} V/m (red)" if by == "efield"
                 else f" · colour: −dE_s/ds ±{result.f_color_top:.3g} V/m² "
                      f"(blue hyper-, red depolarising) · dots: terminal "
                      f"field at fibre ends")
              + " · grey line: gray/white boundary · white: hot spot")
    return f"{first}\n{second}"


def _region_label(result: JobResult) -> str:
    """'C3 (left M1, hand area)' from the simulation's coil centre, the
    way mesh_sim's own dropdown names it; '' when the archive has no spec."""
    centre = result.simulation.coil_center
    if not centre:
        return ""
    known = dict(mesh_sim_service.ELECTRODE_REGIONS)
    return f"{centre} ({known[centre]})" if centre in known else centre


def _tracts_view(result: JobResult, saved: bool = False,
                       by: str = "orientation"):
    """The connected tracts, coloured `by` "orientation" or "efield"."""
    if result.surface is None:
        return None
    background, affine = _background(result)
    drawn = _drawn(result)
    region = _region_label(result)
    tag = {"efield": " coloured by |E|",
           "activation": " coloured by activating function"}.get(by, "")
    title = " · ".join(part for part in (
        result.subject_id, region,
        f"|E| ≥ {result.threshold_v_per_m:.3g} V/m",
        "connected fiber tracts" + tag) if part)
    by_field = {}
    if by == "efield" and drawn["e"]:
        by_field = dict(point_values=drawn["e"],
                        value_range=(0.0, result.e_color_top))
    elif by == "activation":
        if not drawn["f"]:
            return None
        top = result.f_color_top or 1.0
        ends = np.vstack([np.stack([s[0], s[-1]]) for s in drawn["streamlines"]])
        by_field = dict(point_values=drawn["f"], value_range=(-top, top),
                        value_cmap=preview.ACTIVATION_CMAP,
                        endpoints_mm=ends,
                        endpoint_values=drawn["terminal"].reshape(-1)
                        if drawn["terminal"] is not None else None)
    mode = "any point" if result.contact_mode == "any" else "an endpoint"
    subtitle = (f"conductivity {result.conductivity_type} · {mode} within "
                f"{result.contact_mm:g} mm of the hot spot"
                + (f" · broken tracts healed (gap ≤ "
                   f"{result.join_mm:g} mm, ≤ {result.join_max_angle_deg:g}°, "
                   f"≤ {result.max_hops} hops, thinner)"
                   if result.follow_breaks else ""))
    return preview.tract_mosaic(
        background, affine, drawn["streamlines"], [result.surface], title,
        threshold=result.threshold_v_per_m, hops=drawn["hops"],
        legend=_legend(result, by), subtitle=subtitle, **by_field,
        **_figure_kwargs(saved))


def render_view(view_label: str, result: Optional[JobResult]):
    if result is None:
        return None
    try:
        if view_label == VIEW_T1:
            background, _affine = _background(result)
            return preview.anatomy_mosaic(
                background, f"{result.subject_id}: T1 (from the simulation "
                            f"archive)  [{result.conductivity_type}]",
                "T1 raw intensity (a.u.)")
        if view_label == VIEW_TRACTS:
            return _tracts_view(result)
        if view_label == VIEW_TRACTS_EFIELD:
            return _tracts_view(result, by="efield")
        if view_label == VIEW_TRACTS_ACTIVATION:
            return _tracts_view(result, by="activation")
    except Exception:
        logger.exception("Could not render %s", view_label)
        return None
    return None
