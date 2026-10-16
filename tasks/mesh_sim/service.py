# ============================================================================
# tasks/mesh_sim/service.py — head model -> TMS induced E-field
# ----------------------------------------------------------------------------
# THE seam. UI-agnostic: run_job() takes paths and returns paths, so the web
# page and a later cli.py produce byte-identical run folders.
#
# What one job does
#   1. probe        SimNIBS present and usable -- mesh_head's own check,
#                   reused rather than copied, since both tasks ask the same
#                   question of the same install
#   2. unpack       the .zip mesh_head produced, into inputs/; the m2m folder
#                   inside it is what SimNIBS works from
#   3. spec         a plain-JSON job description (paths, names, numbers)
#   4. simulate     `simnibs_python worker.py spec.json result.json`, stdout
#                   streamed into the transcript. The FEM solve itself is
#                   seconds; the post-processing (field onto the cortical
#                   surface, field onto a NIfTI grid) is the longer half.
#   5. vtk          the cortical surfaces (central, pial, white) carrying
#                   the field, one Slicer-ready polydata each (vendor/gii2vtk)
#   6. previews     field-on-cortex, volume, white-surface and coil pictures
#   7. report + zip a one-page .docx run report (SimNIBS's own field summary
#                   folded in), then everything under outputs/
#
# Why a subprocess: `simnibs` cannot be imported from this app's environment
# (SimNIBS ships its own Python 3.11) and the two never share a dependency
# tree. worker.py therefore imports nothing from this repository, and the
# only things crossing between them are two JSON files and stdout.
#
# The catalogue helpers below (coils, caps, electrodes, handle angles) are
# the live scans settings.py was written to expect: its static lists are now
# only the fallback for when nothing can be scanned yet -- at page build, for
# instance, when no head model has been uploaded.
# ============================================================================
import json
import logging
import math
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np

import settings as global_settings
from app import jobs
from app import report as report_doc
from tasks.anicond import runctx as anicond_runctx
from tasks.mesh_head import service as mesh_head_service
from tasks.mesh_sim import preview
from tasks.mesh_sim import settings as task_settings
from vendor.gii2vtk import read_gii_surface

logger = logging.getLogger(__name__)

TASK_SLUG = "mesh_sim"


# ============================================================================
# SimNIBS availability
# ----------------------------------------------------------------------------
# Deliberately not a second implementation: mesh_head already probes this
# install (SIMNIBS_ROOT, charm, `import simnibs`, version >= 4) and caches the
# answer for the process. A simulation needs exactly the same things.
# ============================================================================
def simnibs_status():
    return mesh_head_service.simnibs_status()


# ============================================================================
# Catalogues
# ============================================================================
def coil_choices() -> List[Tuple[str, str]]:
    """(label, filename) for every figure-eight coil the install ships.

    Scanned, not listed: a SimNIBS upgrade that adds or renames a coil is
    picked up without an edit here. Scoped to the Drakaki_BrainStim_2022
    folder -- see settings.py for why that folder is the figure-eight one.
    Falls back to settings.COIL_CHOICES if the folder cannot be read.
    """
    folder = Path(task_settings.FIGURE8_COIL_DIR)
    try:
        files = sorted(p.name for p in folder.iterdir()
                       if p.suffix.lower() in (".ccd", ".tcd"))
    except OSError:
        files = []
    if not files:
        return list(task_settings.COIL_CHOICES)
    return [(_coil_label(name), name) for name in files]


def _coil_label(filename: str) -> str:
    """'MagVenture_MCF-B65.ccd' -> 'MagVenture MCF-B65'."""
    stem = Path(filename).stem
    stem = re.sub(r"^(MagVenture|MagStim|Magstim|Deymed|MagMore)_",
                  lambda m: m.group(1).replace("MagStim", "Magstim") + " ", stem)
    return stem.replace("_", " ")


_DIDTMAX_RE = re.compile(r"dIdtmax\s*=\s*([0-9.]+)", re.IGNORECASE)


def coil_didtmax(coil: str) -> Optional[float]:
    """The coil's measured 100% MSO dI/dt in A/s, from the `dIdtmax=<A/us>`
    field every .ccd in the Drakaki 2022 folder carries in its header
    line; None for a coil file without one (a .tcd, or a file from
    elsewhere). This is the physical ceiling the page caps dI/dt at."""
    if not coil:
        return None
    path = Path(task_settings.FIGURE8_COIL_DIR) / coil
    try:
        with open(path, errors="replace") as fh:
            for _ in range(5):                    # the header is line 1
                match = _DIDTMAX_RE.search(fh.readline())
                if match:
                    return float(match.group(1)) * 1e6
    except OSError:
        return None
    return None


def default_coil(choices: Optional[Sequence[Tuple[str, str]]] = None) -> str:
    """settings.DEFAULT_COIL if the scan still offers it, else the first."""
    choices = choices if choices is not None else coil_choices()
    names = [value for _label, value in choices]
    if task_settings.DEFAULT_COIL in names:
        return task_settings.DEFAULT_COIL
    return names[0] if names else task_settings.DEFAULT_COIL


def cap_choices(zip_path=None) -> List[str]:
    """The cap files a head model ships, read straight out of the .zip.

    Reading the archive's index (and two small CSVs from it) is fast even for
    a several-hundred-megabyte head model, so the page can fill its dropdowns
    the moment a file is chosen, long before a run unpacks anything.
    """
    if zip_path:
        try:
            with zipfile.ZipFile(str(zip_path)) as archive:
                caps = sorted({Path(n).name for n in archive.namelist()
                               if "/eeg_positions/" in n and n.endswith(".csv")
                               and not Path(n).name.startswith("Fiducials")})
            if caps:
                return caps
        except (OSError, zipfile.BadZipFile) as exc:
            logger.info("mesh_sim: could not read caps from %s: %s",
                        zip_path, exc)
    return list(task_settings.EEG_CAP_CHOICES)


def default_cap(choices: Optional[Sequence[str]] = None) -> str:
    choices = choices if choices is not None else cap_choices()
    if task_settings.DEFAULT_EEG_CAP in choices:
        return task_settings.DEFAULT_EEG_CAP
    return choices[0] if choices else task_settings.DEFAULT_EEG_CAP


# ---------------------------------------------------------------- positions --
def _read_cap_csv(text: str) -> Dict[str, np.ndarray]:
    """{electrode name: xyz} from a SimNIBS eeg_positions csv.

    Rows are `Type,x,y,z,name`; Fiducials.csv uses the same shape with type
    `Fiducial`, which is how Nz/Iz are read for the anterior-posterior axis.
    """
    points: Dict[str, np.ndarray] = {}
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        except ValueError:
            continue
        points[parts[4]] = xyz
    return points


def read_cap_from_zip(zip_path, cap_name: str) -> Tuple[Dict, Dict]:
    """(cap points, fiducial points) read out of a head-model .zip."""
    caps: Dict[str, np.ndarray] = {}
    fiducials: Dict[str, np.ndarray] = {}
    with zipfile.ZipFile(str(zip_path)) as archive:
        for name in archive.namelist():
            if "/eeg_positions/" not in name:
                continue
            base = Path(name).name
            if base == cap_name:
                caps = _read_cap_csv(archive.read(name).decode("utf-8", "replace"))
            elif base == "Fiducials.csv":
                fiducials = _read_cap_csv(
                    archive.read(name).decode("utf-8", "replace"))
    return caps, fiducials


def read_cap_from_m2m(m2m_dir, cap_name: str) -> Tuple[Dict, Dict]:
    """The same, from an unpacked m2m folder."""
    folder = Path(m2m_dir) / "eeg_positions"
    caps, fiducials = {}, {}
    cap_file = folder / cap_name
    if cap_file.is_file():
        caps = _read_cap_csv(cap_file.read_text(errors="replace"))
    fid_file = folder / "Fiducials.csv"
    if fid_file.is_file():
        fiducials = _read_cap_csv(fid_file.read_text(errors="replace"))
    return caps, fiducials


def electrode_choices(caps: Dict[str, np.ndarray]) -> List[Tuple[str, str]]:
    """(label, name) for the coil centre, annotated where the site is known.

    The regions are the well-established ones only: the motor strip and the
    prefrontal sites people actually target. Anything else keeps its bare
    name rather than being given a label this code cannot stand behind.
    """
    if not caps:
        return list(task_settings.COIL_CENTER_CHOICES)
    known = dict(ELECTRODE_REGIONS)
    named = [(f"{n} ({known[n]})" if n in known else n, n)
             for n in sorted(caps, key=_electrode_sort_key)]
    return named


ELECTRODE_REGIONS = (
    ("C3", "left M1, hand area"), ("C4", "right M1, hand area"),
    ("Cz", "vertex: M1 leg area / SMA"),
    ("F3", "left DLPFC"), ("F4", "right DLPFC"),
    ("P3", "left parietal"), ("P4", "right parietal"),
    ("T7", "left temporal"), ("T8", "right temporal"),
    ("Oz", "occipital / visual"),
)


def _electrode_sort_key(name: str):
    """Front-to-back by row letter, then midline-outward by number, so the
    dropdown reads like the cap rather than like an ASCII sort."""
    rows = ("Fp", "AF", "F", "FT", "FC", "T", "C", "TP", "CP", "P", "PO", "O", "I")
    match = re.match(r"^([A-Za-z]+)(\d*)", name)
    if not match:
        return (99, 99, name)
    letters, digits = match.group(1), match.group(2)
    try:
        row = rows.index(letters)
    except ValueError:
        row = len(rows)
    number = int(digits) if digits else 0
    return (row, number, name)


def handle_choices(caps: Dict[str, np.ndarray], fiducials: Dict[str, np.ndarray],
                   centre: str, max_options: int = 10
                   ) -> List[Tuple[str, str]]:
    """(label, name) for the handle-direction electrode, angles computed live.

    This is what settings.COIL_YDIR_CHOICES was a stand-in for. For each
    electrode near `centre`, the direction from the centre to it is projected
    onto the scalp's tangent plane there and measured against the tangent
    projection of the nasion->inion (straight back) axis. The label states the
    angle and the direction in words.

    Medial vs lateral is decided by comparing distance from the midline, NOT
    by the sign of the angle: the sign flips between the left and the right
    hemisphere, and reading it off the coordinates is right on both.
    """
    if not caps or centre not in caps:
        return list(task_settings.COIL_YDIR_CHOICES)

    origin = np.asarray(caps[centre], dtype=float)
    normal = origin / (np.linalg.norm(origin) or 1.0)

    # Straight-back reference. The fiducials give it exactly; without them,
    # fall back on the cap's own front-to-back spread (-y in SimNIBS's RAS).
    if "Nz" in fiducials and "Iz" in fiducials:
        posterior = np.asarray(fiducials["Iz"]) - np.asarray(fiducials["Nz"])
    else:
        posterior = np.array([0.0, -1.0, 0.0])
    posterior = posterior / (np.linalg.norm(posterior) or 1.0)

    def tangent(vector):
        flat = vector - normal * float(np.dot(vector, normal))
        length = float(np.linalg.norm(flat))
        return flat / length if length else None

    reference = tangent(posterior)
    if reference is None:
        return list(task_settings.COIL_YDIR_CHOICES)

    scored = []
    for name, point in caps.items():
        if name == centre:
            continue
        offset = np.asarray(point, dtype=float) - origin
        distance = float(np.linalg.norm(offset))
        # A neighbour, not the far side of the head: the tangent-plane
        # approximation only holds close by, and a handle reference far away
        # is no more informative than one next door.
        if not (task_settings.HANDLE_MIN_MM <= distance
                <= task_settings.HANDLE_MAX_MM):
            continue
        direction = tangent(offset)
        if direction is None:
            continue
        angle = math.degrees(math.acos(
            float(np.clip(np.dot(reference, direction), -1.0, 1.0))))
        scored.append((distance, angle, name, np.asarray(point, dtype=float)))

    if not scored:
        return list(task_settings.COIL_YDIR_CHOICES)

    scored.sort(key=lambda row: row[0])            # nearest first
    scored = scored[:max_options]
    scored.sort(key=lambda row: row[1])            # then present by angle

    # "Midline" has to be relative, not an absolute few millimetres: Cz sits
    # ~9 mm off x=0 on this subject, which is anatomy, not a midline miss.
    half_width = max((abs(float(p[0])) for p in caps.values()), default=1.0)

    out = []
    for _distance, angle, name, point in scored:
        out.append((f"{name} (≈{angle:.0f}° "
                    f"{_direction_word(angle, origin, point, half_width)})",
                    name))
    return out


def _direction_word(angle: float, origin: np.ndarray, point: np.ndarray,
                    half_width: float = 1.0) -> str:
    """Plain English for an angle measured from 'straight posterior'.

    The sideways half comes from the coordinates rather than the angle's
    sign, so it stays right whichever hemisphere the centre is on: the sign
    mirrors between left and right, "further from the midline" does not.
    From a midline site there is no medial or lateral, so it says which side
    instead.
    """
    if angle < 22.5:
        return "straight posterior"
    if angle > 157.5:
        return "straight anterior"
    if abs(origin[0]) < 0.2 * half_width:          # a midline site: Cz, Fz, …
        sideways = "left" if point[0] < origin[0] else "right"
        joiner = "-"
    else:
        sideways = ("medial" if abs(point[0]) < abs(origin[0]) else "lateral")
        joiner = ""
    if angle < 67.5:
        return f"postero{joiner}{sideways}"
    if angle < 112.5:
        return sideways
    return f"antero{joiner}{sideways}"


def default_handle(choices: Sequence[Tuple[str, str]], centre: str) -> str:
    """The option closest to 45 degrees on the postero-medial side (CP1's
    own family for a C3-like centre), matching settings.DEFAULT_COIL_YDIR.

    Falls back to whatever is nearest 45 degrees in any direction, and then
    to the first option, so a cap or a centre with no medial neighbour still
    produces a usable default rather than nothing. "medial" as a substring
    also matches "anteromedial"; the closest-to-45 step below still picks
    the postero-medial one over it.
    """
    if not choices:
        return task_settings.DEFAULT_COIL_YDIR
    def angle_of(label):
        match = re.search(r"≈(\d+)°", label)
        return float(match.group(1)) if match else 999.0
    medial = [c for c in choices if "medial" in c[0]]
    pool = medial or list(choices)
    best = min(pool, key=lambda c: abs(angle_of(c[0]) - 45.0))
    return best[1]


# ============================================================================
# The head model
# ============================================================================
def m2m_name_in_zip(zip_path) -> Optional[str]:
    """The m2m_<subject> folder inside a head-model .zip, without unpacking."""
    try:
        with zipfile.ZipFile(str(zip_path)) as archive:
            for name in archive.namelist():
                for part in Path(name).parts:
                    if part.startswith("m2m_"):
                        return part
    except (OSError, zipfile.BadZipFile) as exc:
        logger.info("mesh_sim: cannot read %s: %s", zip_path, exc)
    return None


def subject_id_from_zip(zip_path) -> str:
    """'m2m_IXI034' -> 'IXI034'; '' when the archive has no m2m folder."""
    name = m2m_name_in_zip(zip_path) if zip_path else None
    return name[len("m2m_"):] if name else ""


# ============================================================================
# The isotropic conductivity table
# ----------------------------------------------------------------------------
# anicond's target table (settings.CONDUCTIVITY_CSV), read through anicond's
# own parser so the two tasks cannot disagree about a value, joined to
# SimNIBS's element tags through the table's `simnibs_tissue` column.
# ============================================================================
def conductivity_table(csv_path=None, freq: Optional[str] = None
                       ) -> Tuple[List[Dict], str]:
    """([{tag, simnibs_name, table_id, table_tissue, value}], source text).

    One entry per table row that names a SimNIBS tissue, sorted by tag. A
    row whose `simnibs_tissue` is not a SimNIBS tissue name is an error (a
    typo in the table should stop the run, not silently hand SimNIBS its
    built-in value); a row with no name is simply not a tissue charm labels.
    """
    csv_path = Path(str(csv_path or task_settings.CONDUCTIVITY_CSV))
    freq = (freq or task_settings.CONDUCTIVITY_FREQ or "").strip().lower() \
        or anicond_runctx.default_freq(csv_path)
    values = anicond_runctx.parse_conductivity_csv(csv_path, freq)
    names = anicond_runctx.parse_simnibs_tissues(csv_path)
    if not names:
        raise ValueError(f"{csv_path.name} has no `simnibs_tissue` column, so "
                         f"its rows cannot be matched to SimNIBS's tissues")
    tags = {k.lower(): (k, v) for k, v in task_settings.SIMNIBS_TISSUE_TAGS.items()}
    tissue_names = {}
    try:
        from core import general as gnr          # anicond's tissue list
        tissue_names = dict(gnr.dict_tissue)
    except Exception:                             # noqa: BLE001 - names only
        pass
    entries, seen = [], {}
    for table_id, simnibs_name in names.items():
        key = simnibs_name.lower()
        if key not in tags:
            raise ValueError(
                f"{csv_path.name}: tissue id {table_id} names SimNIBS tissue "
                f"{simnibs_name!r}, which is not one of "
                f"{', '.join(task_settings.SIMNIBS_TISSUE_TAGS)}")
        if table_id not in values:
            continue
        canonical, tag = tags[key]
        if tag in seen:
            raise ValueError(f"{csv_path.name}: tissue ids {seen[tag]} and "
                             f"{table_id} both map to SimNIBS {canonical}")
        seen[tag] = table_id
        entries.append({"tag": tag, "simnibs_name": canonical,
                        "table_id": table_id,
                        "table_tissue": tissue_names.get(table_id, ""),
                        "value": float(values[table_id])})
    entries.sort(key=lambda e: e["tag"])
    source = f"{csv_path.name}, column {anicond_runctx._COND_PREFIX}{freq}"
    return entries, source


def conductivity_frequency(csv_path=None, freq: Optional[str] = None) -> str:
    """The frequency the isotropic conductivities are tabulated at, as the
    table's own column label ('10khz'), rendered for a page ('10 kHz').

    This is the one frequency the whole simulation stands on: SimNIBS's
    quasi-static solve has no frequency of its own, the tissue
    conductivities carry it. Shown on the page so a run's report and its
    figures can be read against the right exposure limit."""
    csv_path = Path(str(csv_path or task_settings.CONDUCTIVITY_CSV))
    label = (freq or task_settings.CONDUCTIVITY_FREQ or "").strip().lower() \
        or anicond_runctx.default_freq(csv_path)
    match = re.match(r"^([0-9.]+)\s*([kmg]?)hz$", label)
    if not match:
        return label
    # SI prefixes keep their case: kilo is 'k', mega 'M', giga 'G'.
    number, prefix = match.group(1), {"k": "k", "m": "M", "g": "G"}.get(
        match.group(2), "")
    return f"{number} {prefix}Hz"


def conductivity_frequency_hz(csv_path=None, freq: Optional[str] = None
                              ) -> Optional[float]:
    """The same frequency as a number in Hz, or None if the label is not of
    the form <number><k|M|G>Hz."""
    label = conductivity_frequency(csv_path, freq)
    match = re.match(r"^([0-9.]+)\s*([kMG]?)Hz$", label)
    if not match:
        return None
    return float(match.group(1)) * {"": 1.0, "k": 1e3, "M": 1e6,
                                    "G": 1e9}[match.group(2)]


def icnirp_basic_restriction(freq_hz: float) -> Dict[str, float]:
    """{'occupational': V/m rms, 'general_public': V/m rms}: ICNIRP 2010's
    basic restriction on the internal E-field in CNS tissue of the head at
    `freq_hz` (settings.ICNIRP_CNS_HEAD_RMS). Empty outside 1 Hz-10 MHz."""
    out: Dict[str, float] = {}
    for group, bands in task_settings.ICNIRP_CNS_HEAD_RMS.items():
        for lo, hi, rule in bands:
            if lo <= freq_hz <= hi:
                out[group] = (float(rule) if not isinstance(rule, str)
                              else float(eval(rule, {"f": freq_hz})))  # noqa: S307
                break
    return out


def guideline_check(result: "JobResult") -> Dict:
    """This run against ICNIRP 2010, at the conductivity table's frequency.

    The run's 99th percentile of |E| in gray and white matter (the CNS
    tissues in the head model; SimNIBS's percentiles are volume-weighted,
    the guideline's are over 2 mm cubes -- close enough for a scaling) is
    compared with the weighted-peak restriction, sqrt(2) x the rms value.
    Since |E| is linear in dI/dt, the dI/dt at which the larger of the two
    p99s would exactly meet each restriction follows by one ratio.
    """
    freq = conductivity_frequency_hz()
    if freq is None:
        return {}
    limits = icnirp_basic_restriction(freq)
    p99 = {}
    for tissue, entry in result.tissue_stats.items():
        tail = entry.get("percentiles", {}).get("99")
        if tail:
            p99[tissue] = float(tail["threshold"])
    if not limits or not p99 or result.didt <= 0:
        return {"frequency_hz": freq, "limits_rms": limits, "p99": p99}
    worst = max(p99.values())
    out = {"frequency_hz": freq, "limits_rms": limits, "p99": p99,
           "p99_max": worst, "didt": result.didt, "at_limit": {}}
    for group, rms in limits.items():
        peak = rms * task_settings.ICNIRP_PEAK_FACTOR
        out["at_limit"][group] = {"rms": rms, "peak": peak,
                                  "didt": result.didt * peak / worst,
                                  "ratio": worst / peak}
    return out


def unpack_head_model(zip_path, dest: Path, report: jobs.Reporter = None) -> Path:
    """Extract the archive into `dest` and return the m2m folder inside it."""
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path)) as archive:
        members = archive.namelist()
        total = sum(info.file_size for info in archive.infolist())
        say(report, f"Head model: unpacking {len(members)} files "
                    f"({total / 1e6:.0f} MB) …")
        # Refuse absolute or parent-escaping members rather than trusting an
        # uploaded archive to stay inside the folder it is given.
        for name in members:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"unsafe path in archive: {name}")
        archive.extractall(dest)

    candidates = sorted(p for p in dest.rglob("m2m_*") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(
            "No m2m_<subject> folder in the uploaded .zip. Upload the archive "
            "the T1/T2 → Head Mesh page produces, not a folder of your own.")
    m2m = candidates[0]
    if not (m2m / f"{m2m.name[len('m2m_'):]}.msh").is_file():
        # The head mesh is the one file a simulation cannot do without.
        raise FileNotFoundError(
            f"{m2m.name} has no head mesh ({m2m.name[len('m2m_'):]}.msh). The "
            f"archive looks incomplete -- was the head-model run finished?")
    say(report, f"Head model: {m2m.name}")
    return m2m


# ============================================================================
# What a run produces
# ----------------------------------------------------------------------------
# Roles rather than filenames: SimNIBS names its own outputs after the mesh,
# the coil and the position ("IXI034_TMS_1-0001_MagVenture_MCF-B65_scalar"),
# so the paths cannot be written down in advance the way mesh_head's could.
# The job discovers the files and files them under these roles instead.
# ============================================================================
FILE_MEANING: Dict[str, str] = {
    "field_mesh": "the result: induced E-field on the whole head mesh (gmsh "
                  ".msh, opens in SimNIBS's own viewer)",
    "field_magn_nii": "E-field magnitude in gray AND white matter, on a "
                      "NIfTI grid (V/m) — for viewing over the T1 in any "
                      "volume viewer",
    "field_vec_nii": "E-field as a vector volume (3 components per voxel)",
    "t1_nii": "the head model's T1, on the same grid as the field volumes — "
              "the anatomy to overlay them on (and what the TMS Field → "
              "Tracts task draws over)",
    "cortex_mesh": "field on the central cortical surface, both hemispheres "
                   "(gmsh .msh, all four quantities)",
    "vtk_magn": "central cortical surface coloured by E-field magnitude — "
                "jet, the mosaics' own scale, written into the file as RGB "
                "with the |E| values alongside; load in 3D Slicer as a "
                "Model (RAS-tagged, no transform needed)",
    "vtk_normal": "the same surface carrying the field's normal component "
                  "(into/out of the cortex; signed, coolwarm about zero)",
    "vtk_pial_magn": "pial surface (gray matter / CSF boundary) coloured by "
                     "E-field magnitude (jet), read from the gray-matter "
                     "elements",
    "vtk_white_magn": "white surface (gray / white matter boundary) coloured "
                      "by E-field magnitude (jet), read from the "
                      "white-matter elements",
    "coil_geo": "the coil's position and windings as a gmsh overlay, to view "
                "beside the field mesh",
    "cortex_coil_png": "field-on-cortex mosaic with the per-tissue percentile "
                       "table, and the coil drawn on top — where it sat and "
                       "which way it faced",
    "volume_png": "E-field magnitude in gray and white matter as a volume "
                  "mosaic, the gray/white boundary drawn in white",
    "wm_surface_png": "E-field on the white surface (the gray/white boundary)",
    "tensor_orientation_png": "the uploaded conductivity tensor's principal "
                              "axis, direction-encoded, with its anisotropy "
                              "index per tissue — a QC check on the tensor "
                              "itself, not the simulation (anisotropic runs "
                              "only)",
    "coil_png": "the axial slice through the coil centre, with the coil "
                "marked",
    "exceedance_png": "ICNIRP exceedance: |E| in gray and white matter "
                      "divided by the ICNIRP 2010 occupational peak "
                      "restriction at the table's frequency, drawn only "
                      "where the ratio is above 1 (the gray/white boundary "
                      "in white)",
    "coil_vtk": "the coil's own casing placed where it sat, plus its axis — "
                "load in 3D Slicer beside the T1 and the field (RAS-tagged)",
    "coil_axis_vtk": "the coil's position without its casing: a sphere at "
                     "the coil centre, a thick rod along the coil axis into "
                     "the head and a thinner one along the handle — solid "
                     "geometry, so it reads at any zoom in 3D Slicer "
                     "(RAS-tagged; 'part' scalar 0/1/2 = centre/axis/handle)",
    "exceedance_nii": "ICNIRP exceedance as a volume: |E| / the occupational "
                      "peak restriction on the field grid, 0 where the ratio "
                      "is at or below 1",
    "gm_wm_labels_nii": "gray and white matter as a label map on the field "
                        "grid (1 = white matter, 2 = gray matter, SimNIBS's "
                        "own tags; anisotropic runs)",
    "tensor_rgb_nii": "the conductivity tensor's principal axis as an RGB "
                      "colour volume on the tensor's own grid (red "
                      "left–right, green anterior–posterior, blue "
                      "superior–inferior, brightness = anisotropy index; "
                      "anisotropic runs) — opens in 3D Slicer as a colour "
                      "volume",
    "stats_csv": "one row per tissue and percentile: the |E| threshold at "
                 "p99 / p99.5 / p99.9, for cohort plots",
    "run_report": "this run in one page (.docx): settings, conductivities, "
                  "the field percentiles and focality, and what every file "
                  "is",
    "spec": "the exact job description handed to SimNIBS",
    "sim_log": "SimNIBS's own log for the run",
}

# Display order for the downloads table; roles the run did not produce are
# skipped rather than shown empty.
FILE_ORDER = ("run_report", "stats_csv", "field_mesh", "vtk_magn",
              "vtk_normal", "vtk_pial_magn", "vtk_white_magn", "coil_vtk",
              "field_magn_nii", "field_vec_nii", "t1_nii", "cortex_mesh",
              "coil_geo", "coil_axis_vtk", "exceedance_nii",
              "gm_wm_labels_nii", "tensor_rgb_nii", "cortex_coil_png",
              "volume_png", "wm_surface_png", "tensor_orientation_png",
              "coil_png", "exceedance_png", "spec", "sim_log")

# The subset the run report's file table lists: what someone reads, views or
# analyses. The rest (the report itself, gmsh side files, the job spec,
# SimNIBS's log) stays in the folder and in the page's downloads table, but
# is bookkeeping, not a result.
REPORT_ROLES = ("stats_csv", "field_mesh", "vtk_magn", "vtk_normal",
                "vtk_pial_magn", "vtk_white_magn", "coil_vtk",
                "coil_axis_vtk", "field_magn_nii", "field_vec_nii", "t1_nii",
                "exceedance_nii", "gm_wm_labels_nii", "tensor_rgb_nii",
                "cortex_coil_png", "volume_png", "wm_surface_png",
                "tensor_orientation_png", "coil_png", "exceedance_png")

FOLDER_MEANING = (
    ("vtk", "cortical surfaces with the field attached and the coil's own "
            "casing, ready for 3D Slicer"),
    ("subject_overlays", "the field on the central, pial and white surfaces, "
                         "per hemisphere, as FreeSurfer curv files (central: "
                         "magnitude, normal, tangential, angle; pial and "
                         "white: magnitude, normal)"),
    ("subject_volumes", "the field resampled onto a NIfTI grid"),
)


@dataclass
class JobResult:
    """What one completed job produced. Paths only -- no widgets, no arrays."""
    run: jobs.RunDir
    subject_id: str
    m2m_dir: Path
    files: Dict[str, Path] = field(default_factory=dict)
    # What was asked for, kept so the report and the page can state it back.
    coil: str = ""
    coil_center: str = ""
    coil_ydir: str = ""
    didt: float = 0.0
    distance: float = 0.0
    eeg_cap: str = ""
    anisotropy: str = "scalar"
    conductivity_type: str = "iso"          # iso / dm / vn / tbe -- see settings
    tensor_name: str = ""
    # worker.tissue_field_stats's output, keyed by tissue then percentile.
    tissue_stats: Dict = field(default_factory=dict)
    # SimNIBS's own fields_summary.txt, parsed into tables
    # (_parse_fields_summary) -- the file itself is not kept.
    field_summary: List[Dict] = field(default_factory=list)
    percentiles: Sequence[float] = ()
    # {tissue: preview.tensor_tissue_stats(...)} of the uploaded conductivity
    # tensor's own anisotropy index -- empty for an isotropic run.
    tensor_stats: Dict = field(default_factory=dict)
    # Per-tissue isotropic values the solver ran with, as the worker read
    # them back: [{tag, simnibs_name, value, source, table_tissue}], where
    # source is "table" or "simnibs_default".
    conductivities: List[Dict] = field(default_factory=list)
    conductivity_source: str = ""           # which table and column
    tensor_as_is: bool = False              # 'dir' without SimNIBS's rescaling
    # Eigenvalue range of the conductivity the solver used, per anisotropic
    # tissue tag (worker.tensor_qc) -- anisotropic runs only.
    tensor_qc: Dict = field(default_factory=dict)
    simnibs: str = ""
    guideline: Dict = field(default_factory=dict)   # guideline_check()
    coil_centre_xyz: Optional[List[float]] = None
    coil_normal_xyz: Optional[List[float]] = None
    matsimnibs: Optional[List[List[float]]] = None
    peak_field: Optional[float] = None          # 99.9th percentile, V/m
    inputs: Dict[str, Tuple[str, Path]] = field(default_factory=dict)
    zip_path: Optional[Path] = None
    simulate_s: float = 0.0
    elapsed_s: float = 0.0
    completed_at: Optional[datetime] = None
    summary: str = ""

    @property
    def placement(self) -> str:
        return (f"{self.coil_center} → {self.coil_ydir}, "
                f"{self.distance:g} mm off the scalp")

    def guide(self, roles: Sequence[str] = FILE_ORDER
              ) -> List[Tuple[str, Path, str]]:
        """(role, path, meaning) for every guided file this run wrote."""
        return [(role, self.files[role], FILE_MEANING[role])
                for role in roles
                if role in self.files and Path(self.files[role]).exists()]

    def folders(self) -> List[Tuple[Path, str]]:
        out = []
        for name, meaning in FOLDER_MEANING:
            path = self.run.outputs / name
            if path.is_dir():
                out.append((path, meaning))
        return out


# ============================================================================
# Small helpers
# ============================================================================
def say(report: jobs.Reporter, message: str) -> None:
    """Send one progress line, if anybody is listening, and always log it."""
    logger.info(message)
    if report is not None:
        report(message)


def _clean_env() -> Dict[str, str]:
    """The environment the worker runs in: ours, minus anything that would
    point SimNIBS's interpreter at this app's packages."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env["PATH"] = f"{global_settings.SIMNIBS_ROOT / 'bin'}:{env.get('PATH', '')}"
    return env


# SimNIBS's own lines are worth showing; the solver's inner chatter is not.
CONSOLE_MARKERS = ("[ simnibs ]", "summary |", "Coil:", "Conductivity:",
                   "SimNIBS:", "Solving", "Surfaces:", "Statistics:", "Worker",
                   "VTK:", "Error", "ERROR", "Traceback", "Exception",
                   "WARNING")
RAW_TAIL_ON_ERROR = 40


def _console_worthy(line: str) -> bool:
    return any(marker in line for marker in CONSOLE_MARKERS)


def run_worker(spec_path: Path, result_path: Path, raw_log: Path,
               report: jobs.Reporter = None) -> Dict:
    """Run worker.py under SimNIBS's interpreter, streaming its output.

    Returns the parsed result file. A worker that fails writes the reason
    into that file, so the error the page shows is SimNIBS's own rather than
    an exit code.
    """
    cmd = [str(global_settings.SIMNIBS_PYTHON),
           str(task_settings.WORKER), str(spec_path), str(result_path)]
    say(report, f"Worker: {Path(cmd[0]).name} {Path(cmd[1]).name} "
                f"(full output in {raw_log.name})")

    proc = subprocess.Popen(cmd, cwd=str(spec_path.parent), env=_clean_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1)
    timed_out = threading.Event()

    def _kill():
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(task_settings.SIM_TIMEOUT_S, _kill)
    watchdog.daemon = True
    watchdog.start()
    tail: List[str] = []
    try:
        assert proc.stdout is not None
        with open(raw_log, "a") as raw:
            for line in proc.stdout:
                raw.write(line)
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                del tail[:-RAW_TAIL_ON_ERROR]
                if _console_worthy(line):
                    say(report, f"simnibs | {line}")
        proc.wait()
    finally:
        watchdog.cancel()

    if timed_out.is_set():
        raise TimeoutError(f"the simulation exceeded "
                           f"{task_settings.SIM_TIMEOUT_S}s and was killed")

    result: Dict = {}
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text())
        except json.JSONDecodeError:
            result = {}

    if proc.returncode != 0 or not result.get("ok"):
        detail = result.get("error") or f"exit code {proc.returncode}"
        for line in tail:
            say(report, f"simnibs | {line}")
        raise RuntimeError(f"SimNIBS failed: {detail}. See {raw_log.name}.")
    return result


# ============================================================================
# The cortical surfaces, as something 3D Slicer opens
# ============================================================================
# The overlays are per-hemisphere FreeSurfer curv files: one value per vertex,
# in the same order as the m2m folder's own gifti surfaces. SimNIBS writes
# them for the central surface (magnitude, normal, tangential, angle) and the
# worker adds the pial and white surfaces (magnitude, normal), all in one
# naming scheme.
#
# Each VTK carries the field TWICE: as the plain values (a float point array
# a viewer can remap) and as explicit RGB in an unsigned-char 'Colors' array
# set as the active scalars -- 3D Slicer maps a 3-component unsigned-char
# scalar directly, so the file opens in the colour scale the mosaics use
# (jet for a magnitude, coolwarm about zero for the signed normal
# component) instead of the viewer's default grey. The window is the
# mosaics' own (FIELD_PCTL of the values; symmetric for a signed one) and
# is written into the header line, so the file states its own scale.
#
# (surface, quantity, file role, scalar name in the VTK, signed?)
SURFACE_VTKS = (("central", "magn", "vtk_magn", "E_magnitude_V_per_m", False),
                ("central", "normal", "vtk_normal", "E_normal_V_per_m", True),
                ("pial", "magn", "vtk_pial_magn", "E_magnitude_V_per_m", False),
                ("white", "magn", "vtk_white_magn", "E_magnitude_V_per_m",
                 False))
SIGNED_CMAP = "coolwarm"


def _overlay_for(overlays: Sequence[str], hemi: str, quantity: str,
                 surface: str = "central") -> Optional[Path]:
    suffix = f".{surface}.E.{quantity}"
    for path in overlays:
        name = Path(path).name
        if name.startswith(f"{hemi}.") and name.endswith(suffix):
            return Path(path)
    return None


def field_window(values: np.ndarray, signed: bool) -> Tuple[float, float]:
    """The colour window the mosaics use for these values."""
    finite = values[np.isfinite(values)]
    if not finite.size:
        return 0.0, 1.0
    if signed:
        top = float(np.percentile(np.abs(finite), preview.FIELD_PCTL[1]))
        return (-top, top) if top > 0 else (-1.0, 1.0)
    lo, hi = (float(v) for v in np.percentile(finite, preview.FIELD_PCTL))
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def write_surface_vtk(path: Path, points: np.ndarray, faces: np.ndarray,
                      values: np.ndarray, scalar_name: str, cmap: str,
                      vmin: float, vmax: float, title: str) -> Path:
    """Legacy binary polydata: triangles, explicit RGB as the active scalars,
    the values as a named float array. Points written as given (RAS)."""
    import vtk
    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from vtk.util import numpy_support as vtknp

    poly = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_points.SetData(vtknp.numpy_to_vtk(
        np.ascontiguousarray(points, dtype=np.float32), deep=True))
    poly.SetPoints(vtk_points)
    n = len(faces)
    offsets = np.arange(0, 3 * n + 1, 3, dtype=np.int64)
    cells = vtk.vtkCellArray()
    cells.SetData(vtknp.numpy_to_vtkIdTypeArray(offsets, deep=True),
                  vtknp.numpy_to_vtkIdTypeArray(
                      np.ascontiguousarray(faces, dtype=np.int64).reshape(-1),
                      deep=True))
    poly.SetPolys(cells)

    rgb = colormaps[cmap](Normalize(vmin, vmax, clip=True)(
        np.nan_to_num(values, nan=vmin)))[:, :3]
    colors = vtknp.numpy_to_vtk(
        np.ascontiguousarray(np.rint(rgb * 255).astype(np.uint8)), deep=True,
        array_type=vtk.VTK_UNSIGNED_CHAR)
    colors.SetName("Colors")
    poly.GetPointData().SetScalars(colors)
    field = vtknp.numpy_to_vtk(np.asarray(values, dtype=np.float32), deep=True)
    field.SetName(scalar_name)
    poly.GetPointData().AddArray(field)

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileTypeToBinary()
    writer.SetHeader(title[:250])
    # The 4.2 legacy layout (32-bit cell lists), not the writer's default
    # 5.1 (64-bit offsets + connectivity): every reader opens it, and a
    # half-million-vertex surface is 40% smaller.
    writer.SetFileVersion(42)
    writer.SetFileName(str(path))
    writer.SetInputData(poly)
    if not writer.Write():
        raise OSError(f"Could not write {path}")
    return path


def cortex_surface_vtk(m2m_dir: Path, overlays: Sequence[str], out_dir: Path,
                       hemispheres: Sequence[str], prefix: str,
                       report: jobs.Reporter = None) -> Dict[str, Path]:
    """One .vtk per (surface, quantity): both hemispheres, the field as
    colours and as values."""
    written: Dict[str, Path] = {}
    for surface, quantity, role, scalar_name, signed in SURFACE_VTKS:
        points, faces, values, offset = [], [], [], 0
        for hemi in hemispheres:
            gii = m2m_dir / "surfaces" / f"{hemi}.{surface}.gii"
            overlay = _overlay_for(overlays, hemi, quantity, surface)
            if not gii.is_file() or overlay is None:
                points = []
                break
            vertices, triangles = read_gii_surface(gii)
            per_vertex = preview.read_curv_values(overlay, len(vertices))
            points.append(vertices)
            faces.append(triangles + offset)
            values.append(per_vertex)
            offset += len(vertices)
        if not points:
            say(report, f"VTK: no {surface} {quantity} overlay to attach, "
                        f"skipped")
            continue
        all_values = np.concatenate(values)
        vmin, vmax = field_window(all_values, signed)
        cmap = SIGNED_CMAP if signed else preview.FIELD_CMAP
        out = out_dir / f"{prefix}both.{surface}.{quantity}.vtk"
        write_surface_vtk(out, np.vstack(points), np.vstack(faces), all_values,
                          scalar_name, cmap, vmin, vmax,
                          title=f"{out.name} SPACE=RAS {cmap} "
                                f"{vmin:.4g}..{vmax:.4g} V/m")
        written[role] = out
        say(report, f"VTK: {out.name} ({out.stat().st_size / 1e6:.0f} MB, "
                    f"{offset:,} vertices, {cmap} {vmin:.3g}..{vmax:.3g} V/m)")
    return written


# ============================================================================
# The coil itself, as something 3D Slicer opens
# ----------------------------------------------------------------------------
# Every coil in the catalogue ships an .stl of its casing beside the .ccd, in
# the coil's OWN millimetre frame: z = 0 is the face that meets the scalp and
# the body extends to negative z. matsimnibs is exactly the transform that
# puts that frame into the subject's, so the casing lands where the coil
# actually sat, distance and all -- no scaling, both are millimetres.
#
# Written as ONE legacy VTK carrying both a POLYGONS section (the casing) and
# a LINES section (the coil axis, from the centre into the head). Legacy
# polydata allows both in one file; vendor/gii2vtk's writer only emits
# triangles, so this one is written here rather than bending that.
# ============================================================================
def read_binary_stl(path) -> Tuple[np.ndarray, np.ndarray]:
    """(vertices [3m, 3], faces [m, 3]) from a binary STL.

    Vertices are left duplicated per triangle, as the format stores them:
    de-duplicating 6k triangles buys nothing a viewer cares about.
    """
    import struct
    with open(path, "rb") as fh:
        fh.read(80)                                    # header, unused
        count = struct.unpack("<I", fh.read(4))[0]
        raw = np.frombuffer(fh.read(count * 50), dtype=np.uint8)
    if raw.size != count * 50:
        raise ValueError(f"{Path(path).name}: truncated STL "
                         f"({raw.size} bytes for {count} triangles)")
    records = raw.reshape(count, 50)
    # bytes 0-11 are the facet normal, 12-47 the three vertices, 48-49 an
    # attribute count nothing uses.
    vertices = records[:, 12:48].copy().view("<f4").reshape(count * 3, 3)
    faces = np.arange(count * 3, dtype=np.int64).reshape(count, 3)
    return np.asarray(vertices, dtype=np.float64), faces


def _write_coil_vtk(path: Path, casing: np.ndarray, faces: np.ndarray,
                    axis: Optional[np.ndarray] = None,
                    title: Optional[str] = None) -> Path:
    """Legacy VTK polydata: casing triangles plus an optional axis polyline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    points = casing if axis is None else np.vstack([casing, axis])
    tag = " SPACE=RAS"
    title = (title or path.name)[:250 - len(tag)] + tag
    with open(path, "wb") as fh:
        def w(text):
            fh.write(text.encode("ascii"))

        w("# vtk DataFile Version 3.0\n")
        w(title + "\n")
        w("BINARY\nDATASET POLYDATA\n")
        w(f"POINTS {len(points)} float\n")
        fh.write(np.ascontiguousarray(points).astype(">f4", copy=False).tobytes())
        w("\n")

        cells = np.empty((len(faces), 4), dtype=np.int32)
        cells[:, 0] = 3
        cells[:, 1:] = faces
        w(f"POLYGONS {len(faces)} {cells.size}\n")
        fh.write(cells.astype(">i4", copy=False).tobytes())
        w("\n")

        if axis is not None:
            first = len(casing)
            line = np.array([len(axis)] + list(range(first, first + len(axis))),
                            dtype=np.int32)
            w(f"LINES 1 {line.size}\n")
            fh.write(line.astype(">i4", copy=False).tobytes())
            w("\n")
    return path


def coil_vtk(coil_file: Path, matsimnibs: Sequence[Sequence[float]],
             out_path: Path, axis_mm: float = 60.0,
             report: jobs.Reporter = None) -> Optional[Path]:
    """The coil's casing and axis, placed in subject space, as one .vtk."""
    stl = Path(coil_file).with_suffix(".stl")
    if not stl.is_file():
        say(report, f"Coil VTK: no casing beside {Path(coil_file).name}, "
                    f"skipped")
        return None
    vertices, faces = read_binary_stl(stl)
    matrix = np.asarray(matsimnibs, dtype=float)
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    casing = vertices @ rotation.T + translation
    # The axis runs from the coil centre INTO the head: matsimnibs's third
    # column is that direction by SimNIBS's own convention.
    axis = np.vstack([translation, translation + rotation[:, 2] * axis_mm])
    _write_coil_vtk(out_path, casing, faces, axis, title=out_path.name)
    say(report, f"Coil VTK: {out_path.name} "
                f"({out_path.stat().st_size / 1e6:.1f} MB, "
                f"{len(faces):,} triangles + axis)")
    return out_path


# ============================================================================
# The coil's position as solid geometry
# ----------------------------------------------------------------------------
# A VTK line has no thickness: 3D Slicer draws it one pixel wide whatever
# the zoom, so a bare axis line is invisible next to a head model. This
# file is the same information as tubes and a sphere -- polygons, which
# render as solid objects at any scale with no display settings.
# ============================================================================
COIL_AXIS_MM = 60.0            # into the head, matsimnibs's 3rd column
COIL_HANDLE_MM = 40.0          # along the handle, -1 x matsimnibs's 2nd column
COIL_AXIS_RADIUS_MM = 1.5
COIL_HANDLE_RADIUS_MM = 1.0
COIL_CENTRE_RADIUS_MM = 4.0


def coil_axis_vtk(matsimnibs: Sequence[Sequence[float]], out_path: Path,
                  report: jobs.Reporter = None) -> Path:
    """Sphere at the coil centre + tube along the axis into the head + a
    thinner tube along the handle, one polydata, 'part' point scalar
    (0 centre, 1 axis, 2 handle) so Slicer can colour them apart."""
    import vtk
    from vtk.util import numpy_support as vtknp
    matrix = np.asarray(matsimnibs, dtype=float)
    # matsimnibs's 2nd column is the coil's local +y axis in world space, and
    # SimNIBS's own coil convention (coil_axesorientation.pdf, calc_matsimnibs
    # in mesh_io.py: y = pos_ydir - centre) has +y pointing AWAY from the
    # physical handle, not along it -- the handle is a "prolongation" of the
    # y-axis LINE, but the wrong way along it. Drawn as-is this tube pointed
    # into the coil casing instead of out along the handle a reader holds;
    # negating it draws the true handle direction.
    centre, handle, axis = matrix[:3, 3], -matrix[:3, 1], matrix[:3, 2]

    def tube(start, end, radius):
        line = vtk.vtkLineSource()
        line.SetPoint1(*start)
        line.SetPoint2(*end)
        line.SetResolution(1)
        filt = vtk.vtkTubeFilter()
        filt.SetInputConnection(line.GetOutputPort())
        filt.SetRadius(radius)
        filt.SetNumberOfSides(24)
        filt.CappingOn()
        filt.Update()
        return filt.GetOutput()

    sphere = vtk.vtkSphereSource()
    sphere.SetCenter(*centre)
    sphere.SetRadius(COIL_CENTRE_RADIUS_MM)
    sphere.SetThetaResolution(32)
    sphere.SetPhiResolution(32)
    sphere.Update()
    parts = ((sphere.GetOutput(), 0),
             (tube(centre, centre + axis * COIL_AXIS_MM, COIL_AXIS_RADIUS_MM),
              1),
             (tube(centre, centre + handle * COIL_HANDLE_MM,
                   COIL_HANDLE_RADIUS_MM), 2))
    append = vtk.vtkAppendPolyData()
    for poly, label in parts:
        tag = vtknp.numpy_to_vtk(
            np.full(poly.GetNumberOfPoints(), label, dtype=np.int32), deep=True)
        tag.SetName("part")
        poly.GetPointData().SetScalars(tag)
        append.AddInputData(poly)
    append.Update()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtk.vtkPolyDataWriter()
    writer.SetFileTypeToBinary()
    writer.SetFileVersion(42)
    writer.SetHeader(f"{out_path.name} SPACE=RAS")
    writer.SetFileName(str(out_path))
    writer.SetInputData(append.GetOutput())
    if not writer.Write():
        raise OSError(f"Could not write {out_path}")
    say(report, f"Coil VTK: {out_path.name} (centre sphere, axis and handle "
                f"tubes)")
    return out_path


# ============================================================================
# Extra volumes
# ============================================================================
def write_exceedance_nifti(result: "JobResult", path: Path) -> Optional[Path]:
    """|E| / the ICNIRP occupational peak restriction, 0 where <= 1, on the
    field volume's own grid (its affine is kept as is)."""
    src = result.files.get("field_magn_nii")
    peak = occupational_peak()
    if not src or not Path(src).is_file() or not peak:
        return None
    img = nib.load(str(src))
    field = np.nan_to_num(np.asarray(img.dataobj, dtype=np.float32))
    ratio = field / np.float32(peak)
    ratio[ratio <= 1.0] = 0.0
    out = nib.Nifti1Image(ratio.astype(np.float32), img.affine)
    out.header.set_xyzt_units("mm")
    out.header["descrip"] = (f"|E| / {peak:.4g} V/m ICNIRP2010 occ peak, "
                             f"0 where <=1")[:79].encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return path


def write_gm_wm_labels(m2m_dir: Path, path: Path) -> Optional[Path]:
    """final_tissues with only WM (1) and GM (2) kept, as a uint8 label map
    on the head model's grid -- the same grid the field volumes are on."""
    src = m2m_dir / "final_tissues.nii.gz"
    if not src.is_file():
        return None
    img = nib.load(str(src))
    # charm writes final_tissues with a trailing singleton (x, y, z, 1);
    # a label map is 3-D.
    labels = np.rint(np.asarray(img.dataobj, dtype=np.float32)).astype(np.int32)
    labels = labels.reshape(labels.shape[:3])
    keep = np.isin(labels, [tag for _n, tag in task_settings.STAT_TISSUES])
    out = nib.Nifti1Image(np.where(keep, labels, 0).astype(np.uint8),
                          img.affine)
    out.header.set_xyzt_units("mm")
    out.header["descrip"] = b"SimNIBS tags: 1 white matter, 2 gray matter"
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return path


def write_tensor_rgb(result: "JobResult", path: Path) -> Optional[Path]:
    """The conductivity tensor's principal eigenvector, direction-encoded
    and scaled by the anisotropy index, as an RGB24 NIfTI on the tensor's
    own grid -- the QC mosaic's picture, as a volume 3D Slicer opens as a
    colour ('vector') volume. Only voxels the tensor covers and the
    tissue masks include are coloured; the rest are black."""
    arrays = _tensor_grid_arrays(result)
    if arrays is None:
        return None
    principal, anisotropy, _background, combined, _masks = arrays
    tensor_file = _tensor_path(result)
    canonical = nib.as_closest_canonical(nib.load(str(tensor_file)))
    rgb = np.abs(np.nan_to_num(principal, nan=0.0))
    weight = np.clip(np.nan_to_num(anisotropy, nan=0.0), 0.0, 1.0)
    if combined is not None:
        weight = np.where(combined, weight, 0.0)
    scaled = np.rint(np.clip(rgb * weight[..., None], 0.0, 1.0) * 255)
    scaled = scaled.astype(np.uint8)
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    packed = np.zeros(scaled.shape[:3], dtype=rgb_dtype)
    packed["R"], packed["G"], packed["B"] = (scaled[..., 0], scaled[..., 1],
                                            scaled[..., 2])
    out = nib.Nifti1Image(packed, canonical.affine)
    out.header.set_data_dtype(rgb_dtype)
    out.header.set_xyzt_units("mm")
    out.header["descrip"] = b"principal cond. eigvec RGB x anisotropy index"
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out, str(path))
    return path


# ============================================================================
# The job
# ============================================================================
def run_job(head_model_zip, coil: str, eeg_cap: str, coil_center: str,
            coil_ydir: str, didt: float, distance: float,
            anisotropic: bool = False, tensor_path=None,
            anisotropy_type: str = task_settings.DEFAULT_ANISOTROPY_TYPE,
            conductivity_type: str = task_settings.DEFAULT_CONDUCTIVITY_TYPE,
            run: Optional[jobs.RunDir] = None,
            report: jobs.Reporter = None) -> JobResult:
    """Simulate one TMS coil position on an uploaded head model."""
    t_start = time.time()
    status = simnibs_status()
    if not status.available:
        raise RuntimeError(f"This function is not supported on this server: "
                           f"{status.message}")
    if not head_model_zip:
        raise ValueError("A head model (.zip) is required.")
    if anisotropic and not tensor_path:
        raise ValueError("Anisotropic conductivity is on, so a conductivity "
                         "tensor (.nii.gz) is required. Generate one on the "
                         "DTI → Cond. Tensor page, or turn the option off.")
    if coil_center == coil_ydir:
        raise ValueError("The coil centre and the handle direction cannot be "
                         "the same electrode: there would be no direction to "
                         "point the handle in.")

    # It ends up in a file name and in every CSV row, so pin it to one of the
    # known tags rather than trusting whatever the caller passed.
    conductivity_type = str(conductivity_type or "").strip().lower()
    if conductivity_type not in {v for _, v in task_settings.CONDUCTIVITY_TYPES}:
        conductivity_type = task_settings.DEFAULT_CONDUCTIVITY_TYPE

    run = run or jobs.new_run(TASK_SLUG)
    say(report, f"Job {run.job_id}: {status.message}")

    # The conductivity label is what a cohort's box plots get grouped by, and
    # nothing downstream can recover it if it is wrong -- so say plainly when
    # it contradicts the run itself rather than silently mislabelling a row.
    if anisotropic and conductivity_type == "iso":
        say(report, "Conductivity type: WARNING — labelled 'iso' but a tensor "
                    "was supplied. The label is what the results CSV records; "
                    "set it to the model that produced the tensor.")
    elif not anisotropic and conductivity_type != "iso":
        say(report, f"Conductivity type: WARNING — labelled "
                    f"'{conductivity_type}' but this run is isotropic (no "
                    f"tensor). The results CSV will say "
                    f"'{conductivity_type}' regardless.")

    # -- 2. unpack ----------------------------------------------------------
    # Extracted straight from the upload rather than staged first, which is
    # the one place this task departs from the usual jobs.stage_upload()
    # convention: a head model is the better part of a gigabyte, and keeping
    # the archive AND its extraction would double every run's footprint for
    # no gain -- the unpacked m2m folder IS the complete record of what was
    # computed from, and the archive's own name is kept in the report.
    m2m_dir = unpack_head_model(head_model_zip, run.inputs / "head_model",
                                report)
    subject = m2m_dir.name[len("m2m_"):]
    staged = {"head model": (Path(str(head_model_zip)).name, m2m_dir)}

    tensor_staged = None
    if anisotropic:
        tensor_staged = jobs.stage_upload(tensor_path, run,
                                          f"{subject}_cond_tensor.nii.gz")
        staged["tensor"] = (Path(str(tensor_path)).name, tensor_staged)

    cap_path = m2m_dir / "eeg_positions" / eeg_cap
    if not cap_path.is_file():
        available = sorted(p.name for p in
                           (m2m_dir / "eeg_positions").glob("*.csv"))
        raise FileNotFoundError(
            f"This head model has no {eeg_cap}. It ships: "
            f"{', '.join(available) or '(none)'}")
    caps, _fiducials = read_cap_from_m2m(m2m_dir, eeg_cap)
    for name, label in ((coil_center, "coil centre"),
                        (coil_ydir, "handle direction")):
        if caps and name not in caps:
            raise ValueError(f"{eeg_cap} has no electrode {name!r} for the "
                             f"{label}.")

    coil_file = Path(task_settings.FIGURE8_COIL_DIR) / coil
    if not coil_file.is_file():
        raise FileNotFoundError(f"Coil model not found: {coil_file}")
    # The coil's measured 100% MSO ceiling: a dI/dt above it is not a
    # setting any stimulator reaches with this coil, so it is capped here
    # too, not only on the page (a CLI caller gets the same treatment).
    ceiling = coil_didtmax(coil)
    if ceiling is not None and float(didt) > ceiling:
        say(report, f"dI/dt: WARNING — {float(didt):.4g} A/s exceeds "
                    f"{coil}'s 100% MSO ({ceiling:.4g} A/s); capped there")
        didt = ceiling

    conductivities, conductivity_source = conductivity_table()
    say(report, f"Conductivity table: {conductivity_source} -- " +
                ", ".join(f"{e['simnibs_name']} {e['value']:g}"
                          for e in conductivities) + " S/m")
    tensor_as_is = bool(anisotropic and anisotropy_type == "dir"
                        and task_settings.TENSOR_AS_IS)

    # -- 3. spec ------------------------------------------------------------
    spec = {
        "m2m_dir": str(m2m_dir),
        "out_dir": str(run.outputs),
        "coil_file": str(coil_file),
        "eeg_cap": str(cap_path),
        "coil_center": coil_center,
        "coil_ydir": coil_ydir,
        "distance": float(distance),
        "didt": float(didt),
        "anisotropy_type": anisotropy_type if anisotropic else "scalar",
        "tensor_file": str(tensor_staged) if tensor_staged else None,
        # The table's isotropic values, by SimNIBS tag (worker: apply_...);
        # and, for 'dir', whether to hand the tensor through untouched.
        "conductivities": [{"tag": e["tag"], "value": e["value"],
                            "table_tissue": e["table_tissue"]}
                           for e in conductivities],
        "tensor_as_is": tensor_as_is,
        "tensor_max_cond": task_settings.TENSOR_MAX_COND,
        "tensor_max_ratio": task_settings.TENSOR_MAX_RATIO,
        "cpus": int(task_settings.SIM_CPUS),
        # The white-matter views and the white-matter percentiles both need
        # tag 1 in the NIfTI, which SimNIBS does not write by default.
        "tissues_in_niftis": [tag for _name, tag in task_settings.STAT_TISSUES],
        "percentiles": list(task_settings.FIELD_PERCENTILES),
    }
    spec_path = run.output(f"{subject}_job_spec.json")
    spec_path.write_text(json.dumps(spec, indent=2))
    say(report, f"Coil: {coil} at {coil_center}, handle toward {coil_ydir}, "
                f"{distance:g} mm off the scalp, dI/dt {didt:g} A/s")

    # -- 4. simulate --------------------------------------------------------
    say(report, "Simulation: the FEM solve is seconds; mapping the field onto "
                "the cortex and onto a NIfTI grid is the longer half.")
    t_sim = time.time()
    worker_result = run_worker(spec_path, run.root / "worker_result.json",
                               run.root / "simnibs_stdout.log", report)
    simulate_s = time.time() - t_sim
    say(report, f"Simulation: finished in {simulate_s / 60:.1f} min")

    files: Dict[str, Path] = {"spec": spec_path}
    result = JobResult(run=run, subject_id=subject, m2m_dir=m2m_dir,
                       files=files, coil=coil, coil_center=coil_center,
                       coil_ydir=coil_ydir, didt=float(didt),
                       distance=float(distance), eeg_cap=eeg_cap,
                       anisotropy=(anisotropy_type if anisotropic else "scalar"),
                       conductivity_type=conductivity_type,
                       percentiles=tuple(task_settings.FIELD_PERCENTILES),
                       tensor_name=(Path(str(tensor_path)).name
                                    if anisotropic else ""),
                       simnibs=status.message, inputs=staged,
                       simulate_s=simulate_s)

    # -- what the worker produced -------------------------------------------
    for path in worker_result.get("field_meshes", []):
        files["field_mesh"] = Path(path)
    for path in worker_result.get("volumes", []):
        name = Path(path).name
        if name.endswith("_magnE.nii.gz"):
            files["field_magn_nii"] = Path(path)
        elif name.endswith("_E.nii.gz"):
            files["field_vec_nii"] = Path(path)
    overlays = worker_result.get("overlays", [])
    for path in overlays:
        if Path(path).name.endswith("_central.msh"):
            files["cortex_mesh"] = Path(path)
    # SimNIBS's fields_summary.txt is read into the report and then removed:
    # its two small tables are the report's "Result" section, and a loose
    # copy beside it would only be a second place to look for one number.
    summary_path = worker_result.get("fields_summary")
    if summary_path and Path(summary_path).is_file():
        result_summary = _parse_fields_summary(Path(summary_path))
        try:
            Path(summary_path).unlink()
        except OSError:
            pass
    else:
        result_summary = []
    for path in worker_result.get("coil_geo", []):
        files["coil_geo"] = Path(path)
    for log in sorted(run.outputs.glob("simnibs_simulation*.log")):
        files["sim_log"] = log
    # The head model's T1 travels with the field volumes: the archive is then
    # enough on its own for anything drawn over the anatomy downstream (the
    # TMS Field -> Tracts task), without re-uploading the head model.
    t1_src = m2m_dir / "T1.nii.gz"
    if t1_src.is_file():
        files["t1_nii"] = run.output("subject_volumes", f"{subject}_T1.nii.gz")
        shutil.copyfile(t1_src, files["t1_nii"])

    result.coil_centre_xyz = worker_result.get("coil_centre")
    result.coil_normal_xyz = worker_result.get("coil_normal")
    result.matsimnibs = worker_result.get("matsimnibs")
    result.tissue_stats = worker_result.get("tissue_stats", {})
    if worker_result.get("percentiles"):
        result.percentiles = tuple(worker_result["percentiles"])
    result.field_summary = result_summary
    result.peak_field = _peak_field(result_summary)
    hemispheres = worker_result.get("hemispheres", ["lh", "rh"])
    result.conductivities = worker_result.get("conductivities", [])
    result.conductivity_source = conductivity_source
    result.tensor_as_is = bool(worker_result.get("tensor_as_is", False))
    result.tensor_qc = worker_result.get("tensor_qc", {}) or {}
    # Slots 100+ are tDCS electrode rubber and saline, never in a TMS mesh:
    # not worth a line, in the console or the report.
    result.conductivities = [c for c in result.conductivities
                             if int(c.get("tag", 0)) < 100]
    defaults = [c for c in result.conductivities
                if c.get("source") == "simnibs_default"]
    if defaults:
        say(report, "Conductivity: SimNIBS's built-in value kept for " +
                    ", ".join(f"{c['simnibs_name']} (tag {c['tag']}, "
                              f"{c['value']:g} S/m)" for c in defaults)
                    + " -- the table has no row for these")

    if "field_mesh" not in files:
        raise RuntimeError("SimNIBS reported success but wrote no field mesh; "
                           f"see {(run.root / 'simnibs_stdout.log').name}")

    # -- 5. vtk -------------------------------------------------------------
    say(report, "VTK: attaching the field to the cortical surface …")
    try:
        files.update(cortex_surface_vtk(
            m2m_dir, overlays, run.outputs / "vtk", hemispheres,
            prefix=f"{subject}_{conductivity_type}_", report=report))
    except Exception as exc:              # noqa: BLE001 - an export, not the result
        logger.exception("mesh_sim vtk export failed")
        say(report, f"VTK: skipped ({type(exc).__name__}: {exc})")

    if result.matsimnibs:
        try:
            placed = coil_vtk(coil_file, result.matsimnibs,
                              run.outputs / "vtk" /
                              f"{subject}_{conductivity_type}_coil.vtk",
                              report=report)
            if placed is not None:
                files["coil_vtk"] = placed
        except Exception as exc:          # noqa: BLE001 - an export, not the result
            logger.exception("mesh_sim coil vtk failed")
            say(report, f"Coil VTK: skipped ({type(exc).__name__}: {exc})")
        try:
            files["coil_axis_vtk"] = coil_axis_vtk(
                result.matsimnibs,
                run.outputs / "vtk" / f"{subject}_{conductivity_type}_coil_axis.vtk",
                report=report)
        except Exception as exc:          # noqa: BLE001 - an export, not the result
            logger.exception("mesh_sim coil axis vtk failed")
            say(report, f"Coil VTK: axis skipped ({type(exc).__name__}: {exc})")

    # -- 5b. tensor QC (anisotropic only) ------------------------------------
    # Voxel-based, unweighted, off the tensor's own grid -- unlike
    # tissue_field_stats above, which is volume-weighted over a FEM mesh.
    # There is nothing here for a weight to correct for: every voxel of a
    # NIfTI is the same size.
    if result.anisotropy != "scalar":
        try:
            arrays = _tensor_grid_arrays(result)
            if arrays is not None:
                _principal, anisotropy, _bg, _mask, tissue_masks = arrays
                result.tensor_stats = {
                    name: preview.tensor_tissue_stats(anisotropy, tissue_mask)
                    for name, tissue_mask in tissue_masks.items()}
                say(report, f"Tensor QC: anisotropy index over "
                            f"{len(result.tensor_stats)} tissue(s)")
        except Exception as exc:          # noqa: BLE001 - a QC stat, not the result
            logger.exception("mesh_sim tensor stats failed")
            say(report, f"Tensor QC: skipped ({type(exc).__name__}: {exc})")

    # -- 5c. the exposure guideline -------------------------------------------
    try:
        result.guideline = guideline_check(result)
        for line in _guideline_lines(result):
            say(report, f"Guideline: {line}")
    except Exception as exc:              # noqa: BLE001 - a comparison, not the result
        logger.exception("mesh_sim guideline check failed")
        say(report, f"Guideline: skipped ({type(exc).__name__}: {exc})")

    # -- 6. previews --------------------------------------------------------
    # Each saved figure is the SAME call the canvas makes, with the file
    # settings (higher DPI, thicker lines) instead of the canvas ones -- so
    # what is downloaded and what was looked at cannot drift apart.
    #
    # Every filename below carries the conductivity type, same reason as the
    # archive's own name (run_job step 7): these are also offered as loose,
    # individually-downloaded files, and two runs of the same subject under
    # a different model would otherwise both be "IXI034_field_cortex.png"
    # and overwrite each other in a downloads folder.
    say(report, "Previews: field on the cortex, in the volume, on the white "
                "surface, and the coil …")
    tag = f"_{conductivity_type}"
    saved = {
        "cortex_coil_png": (f"{subject}_field_cortex_with_coil_position{tag}.png",
                            lambda: _cortex_view(result, "magn", saved=True,
                                                 with_coil=True)),
        "volume_png": (f"{subject}_field_gray_white_matter_volume{tag}.png",
                       lambda: _volume_view(result, saved=True)),
        "exceedance_png": (f"{subject}_icnirp_exceedance{tag}.png",
                           lambda: _exceedance_view(result, saved=True)),
        "wm_surface_png": (f"{subject}_field_white_matter_surface{tag}.png",
                           lambda: _wm_surface_view(result, saved=True)),
        "tensor_orientation_png": (f"{subject}_conductivity_tensor_orientation{tag}.png",
                                   lambda: _tensor_orientation_view(result, saved=True)),
        "coil_png": (f"{subject}_coil_position{tag}.png",
                     lambda: _coil_view(result, saved=True)),
    }
    for role, (name, build) in saved.items():
        try:
            image = build()
            if image is None:
                say(report, f"Previews: {role} has nothing to draw, skipped")
                continue
            path = run.output(name)
            preview.save_png(image, path)
            files[role] = path
        except Exception as exc:          # noqa: BLE001 - a picture, not the result
            logger.exception("mesh_sim preview %s failed", role)
            say(report, f"Previews: {role} skipped "
                        f"({type(exc).__name__}: {exc})")

    # -- 6a. extra volumes ---------------------------------------------------
    # The exceedance map as data (the mosaic above is its picture), and for
    # an anisotropic run the two tissues the tensor applies to and the
    # tensor's own principal axis as a colour volume -- what Slicer needs
    # to read the tensor QC mosaic in 3D.
    extra = [("exceedance_nii",
              run.outputs / "subject_volumes" / f"{subject}_icnirp_exceedance{tag}.nii.gz",
              lambda p: write_exceedance_nifti(result, p))]
    if result.anisotropy != "scalar":
        extra += [("gm_wm_labels_nii",
                   run.outputs / "subject_volumes" / f"{subject}_gm_wm_labels.nii.gz",
                   lambda p: write_gm_wm_labels(m2m_dir, p)),
                  ("tensor_rgb_nii",
                   run.outputs / "subject_volumes"
                   / f"{subject}_conductivity_tensor_orientation_rgb{tag}.nii.gz",
                   lambda p: write_tensor_rgb(result, p))]
    for role, path, build in extra:
        try:
            written = build(path)
            if written is None:
                say(report, f"Volumes: {role} has nothing to write, skipped")
                continue
            files[role] = written
            say(report, f"Volumes: {written.name} "
                        f"({written.stat().st_size / 1e6:.1f} MB)")
        except Exception as exc:          # noqa: BLE001 - an export, not the result
            logger.exception("mesh_sim volume %s failed", role)
            say(report, f"Volumes: {role} skipped ({type(exc).__name__}: {exc})")

    # -- 6b. the statistics row ---------------------------------------------
    if result.tissue_stats:
        try:
            files["stats_csv"] = write_stats_csv(
                result, run.output(f"{subject}_field_stats.csv"))
            say(report, f"Statistics: {files['stats_csv'].name} — "
                        f"{len(result.percentiles)} percentile(s) x "
                        f"{len(result.tissue_stats)} tissue(s)")
        except Exception as exc:          # noqa: BLE001 - a log, not the result
            logger.exception("mesh_sim stats csv failed")
            say(report, f"Statistics: CSV skipped ({type(exc).__name__}: {exc})")

    # -- 7. report + zip ----------------------------------------------------
    # Written before the archive so a copy travels inside it; the report
    # does not mention the archive, so there is nothing to add afterwards.
    result.completed_at = datetime.now()
    result.elapsed_s = time.time() - t_start
    files["run_report"] = write_run_report(
        result, run.output(f"{subject}_run_report.docx"))
    # The archive is <subject>_mesh_sim_<model>___output.zip (the convention,
    # jobs.output_zip_name), flat inside: the same subject run iso / dm / vn /
    # tbe gives four archives whose contents are named identically, and the
    # model in the NAME is what keeps them apart once unpacked. tract_efield
    # reads the subject and the model off the white VTK's name inside.
    say(report, "Archiving outputs/ (the field mesh alone is a few hundred "
                "MB) …")
    result.zip_path = run.zip_outputs(subject_id=subject,
                                      model=result.conductivity_type)
    result.elapsed_s = time.time() - t_start
    result.summary = _summarize(result)
    return result


def _parse_fields_summary(path: Path) -> List[Dict]:
    """SimNIBS's fields_summary.txt as a list of tables.

    The file is a hand-formatted text report: a mesh name underlined with
    '=', a region line ("Gray Matter"), then per table a title underlined
    with '-', one line of description, and pipe-delimited rows with a
    '|---|' rule under the header. Each table comes back as {region, title,
    description, header, rows}, which is what the .docx report lays out and
    what the peak field is read from.
    """
    lines = path.read_text(errors="replace").splitlines()
    tables: List[Dict] = []
    region, current = "", None
    for i, raw in enumerate(lines):
        line = raw.rstrip()
        nxt = lines[i + 1].strip() if i + 1 < len(lines) else ""
        if not line.strip():
            current = None
            continue
        if set(line.strip()) <= {"=", "-"}:
            continue                                    # an underline
        if nxt and set(nxt) <= {"="}:
            continue                                    # the mesh name
        if nxt and set(nxt) <= {"-"}:
            current = {"region": region, "title": line.strip(),
                       "description": "", "header": [], "rows": []}
            tables.append(current)
            continue
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(set(c) <= {"-"} for c in cells):
                continue                                # the header rule
            if current is None:
                current = {"region": region, "title": "", "description": "",
                           "header": [], "rows": []}
                tables.append(current)
            if not current["header"]:
                current["header"] = cells
            else:
                current["rows"].append(cells)
            continue
        if current is not None and not current["header"]:
            current["description"] = line.strip()
        else:
            region = line.strip()                       # "Gray Matter"
    return tables


def _peak_field(summary: Sequence[Dict]) -> Optional[float]:
    """The 99.9th-percentile magnE from SimNIBS's own summary tables.

    Read rather than recomputed: it is the number SimNIBS itself reports, so
    the page and the report cannot disagree about what the peak was.
    """
    for table in summary:
        header = [h.lower() for h in table.get("header", [])]
        if "99.9%" not in header:
            continue
        column = header.index("99.9%")
        for row in table.get("rows", []):
            if row and row[0] == "magnE" and len(row) > column:
                match = re.match(r"\s*([0-9.eE+-]+)", row[column])
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return None
    return None


# ============================================================================
# The statistics row
# ----------------------------------------------------------------------------
# The E-field analysis plan, as a file: the 99 / 99.5 / 99.9th percentile of
# |E| in each tissue mask -- the threshold each cohort plot is drawn from.
#
# LONG format, one row per (tissue x percentile), not one wide row per run:
# that is the shape pandas/seaborn want for a plot grouped by tissue and by
# conductivity type, and concatenating a cohort is then `cat` rather than a
# reshape. Only the identifying columns and the threshold itself are kept;
# the tail's five-number summary, the tissue totals and the run settings
# that used to ride along are in the run report, where they are read rather
# than plotted. `percentile` stays because without it three thresholds per
# tissue could not be told apart.
#
# The threshold is VOLUME-WEIGHTED (see worker.tissue_field_stats): a
# tetrahedral mesh has wildly varying element sizes, so an unweighted
# percentile would over-count wherever the mesh happens to be fine.
# ============================================================================
STATS_COLUMNS = ("subject_id", "conductivity_type", "tissue", "percentile",
                 "threshold_v_per_m")


def write_stats_csv(result: JobResult, path: Path) -> Path:
    """One row per tissue and percentile, for cohort plots."""
    import csv
    percentiles = result.percentiles or task_settings.FIELD_PERCENTILES
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_COLUMNS)
        writer.writeheader()
        for tissue, entry in result.tissue_stats.items():
            for percentile in percentiles:
                tail = entry.get("percentiles", {}).get(f"{percentile:g}")
                if not tail:
                    continue
                writer.writerow({
                    "subject_id": result.subject_id,
                    "conductivity_type": result.conductivity_type,
                    "tissue": tissue,
                    "percentile": f"{percentile:g}",
                    "threshold_v_per_m": f"{tail['threshold']:.6g}",
                })
    return path


# ============================================================================
# Run report
# ----------------------------------------------------------------------------
# A .docx rather than a text file: the settings, the conductivity table, the
# percentile tables and the file guide are all tabular, and a Word table is
# read at a glance where a fixed-width text column had to be lined up by
# hand. What it holds: when and how long, the stimulation settings, every
# conductivity the solver ran with, the app's per-tissue percentiles and
# SimNIBS's own summary tables (fields_summary.txt, which is not kept as a
# separate file), the tensor's anisotropy for an anisotropic run, and what
# each result file is. Deliberately NOT in it: job ids, staged input paths,
# archive and log paths, the solver's eigenvalue QC line -- bookkeeping that
# belongs in the run folder and the console, not in a document that is read.
# ============================================================================
def write_run_report(result: JobResult, path: Path) -> Path:
    out = result.run.outputs
    when = result.completed_at or datetime.now()
    document = report_doc.new_document()

    document.add_heading(f"TMS simulation report — {result.subject_id}  "
                         f"[{result.conductivity_type}]", level=1)
    document.add_paragraph(
        f"Completed {when:%Y-%m-%d %H:%M:%S} · total execution "
        f"{result.elapsed_s / 60:.1f} min (simulation "
        f"{result.simulate_s / 60:.1f} min) · {result.simnibs}")

    # -- stimulation ----------------------------------------------------------
    document.add_heading("Stimulation", level=2)
    settings_rows = [
        ("Coil model", result.coil),
        ("EEG cap", result.eeg_cap),
        ("Coil centre", result.coil_center),
        ("Handle toward", result.coil_ydir),
        ("Coil–scalp gap", f"{result.distance:g} mm"),
        ("dI/dt", f"{result.didt:g} A/s"),
    ]
    if result.coil_centre_xyz:
        xyz = ", ".join(f"{v:.1f}" for v in result.coil_centre_xyz)
        settings_rows.append(("Coil placed at", f"({xyz}) mm, subject RAS"))
    settings_rows.append(("Conductivity type", result.conductivity_type))
    report_doc.add_table(document, ("Setting", "Value"), settings_rows,
                widths_cm=(4.5, 12.5))

    # -- conductivity ---------------------------------------------------------
    document.add_heading("Conductivity", level=2)
    if result.anisotropy == "scalar":
        document.add_paragraph(
            f"Isotropic in every tissue; values from "
            f"{result.conductivity_source}.")
    else:
        document.add_paragraph(
            f"Anisotropic ('{result.anisotropy}') from {result.tensor_name} in "
            f"white and gray matter, "
            + ("used as-is (no SimNIBS intensity correction, no eigenvalue "
               "caps)" if result.tensor_as_is else
               "rescaled and capped by SimNIBS (stock 'dir')")
            + f". Every other tissue: {result.conductivity_source}.")
    cond_rows = []
    for entry in result.conductivities:
        cond_rows.append((
            entry["simnibs_name"], entry["tag"],
            entry.get("table_tissue") or "—",
            f"{entry['value']:g}",
            "table" if entry.get("source") == "table"
            else "SimNIBS built-in (not in the table)"))
    if cond_rows:
        report_doc.add_table(document,
                    ("SimNIBS tissue", "Tag", "Table tissue", "σ (S/m)",
                     "Source"),
                    cond_rows, widths_cm=(3.5, 1.2, 3.5, 2.0, 6.8))

    # -- result ---------------------------------------------------------------
    document.add_heading("Result", level=2)
    if result.peak_field is not None:
        document.add_paragraph(
            f"Peak E-field: {result.peak_field:.3g} V/m in gray matter "
            f"(99.9th percentile) at dI/dt {result.didt:g} A/s.")
    if result.tissue_stats:
        percentiles = list(result.percentiles
                           or task_settings.FIELD_PERCENTILES)
        header = ["Tissue"] + [f"p{p:g} (V/m)" for p in percentiles] + [
            "max (V/m)", "mean (V/m)", "volume (mL)"]
        rows = []
        for tissue, entry in result.tissue_stats.items():
            row = [tissue.replace("_", " ")]
            for p in percentiles:
                tail = entry.get("percentiles", {}).get(f"{p:g}", {})
                row.append(f"{tail['threshold']:.3g}"
                           if "threshold" in tail else "—")
            row += [f"{entry.get('max', float('nan')):.3g}",
                    f"{entry.get('mean', float('nan')):.3g}",
                    f"{entry.get('volume_mm3', 0.0) / 1000:.0f}"]
            rows.append(row)
        document.add_paragraph(
            "|E| percentiles per tissue, volume-weighted on the mesh "
            "(the same numbers as the field_stats CSV):")
        report_doc.add_table(document, header, rows)
    for table in result.field_summary:
        if not table.get("header"):
            continue
        title = table.get("title") or "SimNIBS field summary"
        region = table.get("region")
        label = f"SimNIBS {title.lower()}" + (f" — {region.lower()}"
                                             if region else "")
        if table.get("description"):
            label += f": {table['description']}"
        document.add_paragraph(label)
        report_doc.add_table(document, table["header"], table["rows"])
    if result.tensor_stats:
        parts = [f"{tissue.replace('_', ' ')} {entry['mean']:.2g}"
                 for tissue, entry in result.tensor_stats.items()
                 if "mean" in entry]
        if parts:
            document.add_paragraph(
                "Conductivity tensor anisotropy index, mean per tissue: "
                + ", ".join(parts) + ".")

    # -- exposure guideline ---------------------------------------------------
    g = result.guideline
    if g and "at_limit" in g:
        document.add_heading("Exposure guideline (ICNIRP 2010)", level=2)
        freq = g["frequency_hz"]
        freq_text = (f"{freq / 1e3:g} kHz" if freq >= 1e3 else f"{freq:g} Hz")
        document.add_paragraph(
            f"Basic restriction on the internal electric field in CNS tissue "
            f"of the head at {freq_text} (Table 2; rms, the 99th percentile "
            f"of the tissue's field averaged over 2 mm cubes; for a pulse the "
            f"waveform's peak is compared with √2 × the rms value). This "
            f"run's 99th-percentile |E| at dI/dt {g['didt']:.4g} A/s: "
            + ", ".join(f"{t.replace('_', ' ')} {v:.3g} V/m"
                        for t, v in g["p99"].items())
            + ". |E| is linear in dI/dt, so the dI/dt at which the larger of "
              "the two would exactly meet each restriction follows by one "
              "ratio. ICNIRP does not cover medical exposure, and assesses a "
              "source this close to the head against the basic restrictions, "
              "not the reference levels.")
        rows = [(k.replace("_", " "), f"{v['rms']:.3g}", f"{v['peak']:.3g}",
                 f"{v['ratio']:.3g}", f"{v['didt']:.3g}",
                 f"{v['didt'] / task_settings.DIDT_100_MSO_DEFAULT_COIL:.1%}")
                for k, v in g["at_limit"].items()]
        report_doc.add_table(
            document, ("Exposure", "Restriction (V/m rms)",
                       "Weighted peak (V/m)", "This run / restriction",
                       "dI/dt at the restriction (A/s)",
                       "of MCF-B65 100% MSO"),
            rows, widths_cm=(3.0, 2.6, 2.6, 2.8, 3.2, 2.8))

    # -- files ----------------------------------------------------------------
    document.add_heading("Files", level=2)
    file_rows = [(str(Path(path_).relative_to(out)), meaning)
                 for _role, path_, meaning in result.guide(REPORT_ROLES)]
    for folder, meaning in result.folders():
        file_rows.append((str(folder.relative_to(out)) + "/", meaning))
    report_doc.add_table(document, ("File", "What it is"), file_rows,
                widths_cm=(7.0, 10.0))

    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path))
    return path


def _guideline_lines(result: JobResult) -> List[str]:
    """The guideline comparison as text lines (console, summary, report)."""
    g = result.guideline
    if not g or "at_limit" not in g:
        return []
    freq = g["frequency_hz"]
    freq_text = (f"{freq / 1e3:g} kHz" if freq >= 1e3 else f"{freq:g} Hz")
    lines = [f"ICNIRP 2010 basic restriction on the internal E-field in CNS "
             f"tissue of the head at {freq_text}: "
             + ", ".join(f"{v['rms']:.3g} V/m rms ({v['peak']:.3g} V/m "
                         f"weighted peak) {k.replace('_', ' ')}"
                         for k, v in g["at_limit"].items())]
    p99 = ", ".join(f"{t.replace('_', ' ')} {v:.3g}"
                    for t, v in g["p99"].items())
    lines.append(f"this run's 99th-percentile |E| at dI/dt {g['didt']:.3g} "
                 f"A/s: {p99} V/m")
    for k, v in g["at_limit"].items():
        lines.append(f"{k.replace('_', ' ')}: {v['ratio']:.3g} x the "
                     f"restriction; met at dI/dt {v['didt']:.3g} A/s "
                     f"({v['didt'] / task_settings.DIDT_100_MSO_DEFAULT_COIL:.1%} "
                     f"of the MCF-B65's 100% MSO)")
    return lines


def _summarize(result: JobResult) -> str:
    lines = [f"Done in {result.elapsed_s / 60:.1f} min — {result.coil} at "
             f"{result.placement}"]
    if result.peak_field is not None:
        lines.append(f"Peak E-field: {result.peak_field:.3g} V/m in gray "
                     f"matter (99.9th percentile) at dI/dt "
                     f"{result.didt:g} A/s — the field scales linearly with it")
    if result.coil_centre_xyz:
        xyz = ", ".join(f"{v:.1f}" for v in result.coil_centre_xyz)
        lines.append(f"Coil centre: ({xyz}) mm, projected onto the scalp")
    g = result.guideline
    if g and "at_limit" in g:
        lines.append("ICNIRP 2010 (CNS, p99, weighted peak): " + "; ".join(
            f"{k.replace('_', ' ')} {v['ratio']:.3g}x, met at dI/dt "
            f"{v['didt']:.3g} A/s" for k, v in g["at_limit"].items()))
    if result.conductivities:
        lines.append(
            f"Conductivity: {result.conductivity_source}"
            + (" + tensor as-is in WM/GM" if result.anisotropy != "scalar"
               and result.tensor_as_is else
               " + tensor (SimNIBS-rescaled) in WM/GM"
               if result.anisotropy != "scalar" else ""))
    if "field_mesh" in result.files:
        path = Path(result.files["field_mesh"])
        lines.append(f"Field mesh: {path.name}, "
                     f"{path.stat().st_size / 1e6:.0f} MB")
    if result.zip_path:
        lines.append(f"Archive: {Path(result.zip_path).name}, "
                     f"{Path(result.zip_path).stat().st_size / 1e6:.0f} MB")
    return "\n".join(lines)


# ============================================================================
# Views for the canvas
# ----------------------------------------------------------------------------
# The head model's own T1 is shown straight from the uploaded m2m folder
# (mesh_head's drawing code, unchanged); everything else comes from the
# finished run. One view per saved picture, no more: the plain cortex view,
# the normal component, the white-matter-only volume and the tissue
# segmentation were dropped from the page at the user's request -- the
# normal component still ships as a VTK, and the segmentation is mesh_head's
# to show.
# ============================================================================
VIEW_T1 = "T1 (from head model)"
VIEW_COIL = "Coil position (this subject)"
VIEW_FIELD_CORTEX_COIL = "E-field — cortex + coil position"
VIEW_FIELD_WM_SURFACE = "E-field — white matter (surface)"
VIEW_FIELD_VOLUME = "E-field magnitude — volume (GM + WM)"
VIEW_TENSOR_ORIENTATION = "Conductivity tensor — orientation (anisotropic only)"

VIEW_EXCEEDANCE = "ICNIRP exceedance — |E| / occupational peak (GM + WM)"

VIEW_LABELS = (VIEW_T1, VIEW_COIL, VIEW_FIELD_CORTEX_COIL,
               VIEW_FIELD_WM_SURFACE, VIEW_FIELD_VOLUME, VIEW_EXCEEDANCE,
               VIEW_TENSOR_ORIENTATION)
DEFAULT_VIEW = VIEW_T1
RESULT_VIEW = VIEW_FIELD_CORTEX_COIL     # what the job was started for


def view_labels() -> List[str]:
    return list(VIEW_LABELS)


def _figure_kwargs(saved: bool) -> Dict:
    """The one difference between a canvas picture and a saved one.

    Everything else -- the data, the layout, the colours, the annotation --
    is identical, which is the point: one function draws both.
    """
    if saved:
        return {"dpi": task_settings.FIGURE_DPI,
                "slice_fontsize": task_settings.SLICE_FONTSIZE,
                "linewidth": preview.SAVED_LINEWIDTH}
    return {"dpi": None,
            "slice_fontsize": task_settings.SLICE_FONTSIZE,
            "linewidth": preview.CANVAS_LINEWIDTH}


def _titled(result: JobResult, text: str) -> str:
    """`text`, with the conductivity type appended as one consistent tag.

    Every mosaic's SUBTITLE says something different -- one names the coil
    settings, one drops the conductivity type to make room for the coil
    overlay's own note, one never carried it at all -- so it cannot be the
    place a screenshot gets traced back to which model produced it. The
    title is: every mosaic this run draws goes through here.
    """
    return f"{text}  [{result.conductivity_type}]"


def _stats_annotation(result: JobResult) -> str:
    """The per-tissue percentile table, for the corner of a field figure."""
    if not result.tissue_stats:
        return ""
    return preview.percentile_table(
        result.tissue_stats,
        result.percentiles or task_settings.FIELD_PERCENTILES)


def _load_t1(result: JobResult):
    return preview.load_canonical(result.m2m_dir / "T1.nii.gz")


def _tensor_path(result: JobResult) -> Optional[Path]:
    """The uploaded conductivity tensor, staged in this run's inputs/."""
    entry = result.inputs.get("tensor")
    if not entry:
        return None
    path = Path(entry[1])
    return path if path.is_file() else None


def _tensor_grid_arrays(result: JobResult):
    """(principal, anisotropy, background T1, GM+WM mask, per-tissue masks),
    all on the TENSOR's own grid, or None when this run has no tensor.

    Pulled onto the tensor's grid rather than the other way round: it is
    usually the coarser of the two (task 2's DWI/b0 grid versus the head
    model's T1), so resampling the T1 and the tissue labels onto it
    (preview.resample_to_grid) is the cheaper direction.
    """
    tensor_file = _tensor_path(result)
    if tensor_file is None:
        return None
    tensor6, t_affine = preview.load_tensor6(tensor_file)
    principal, anisotropy = preview.tensor_eigen(tensor6)
    t1, t1_affine = _load_t1(result)
    background = preview.resample_to_grid(
        t1, t1_affine, tensor6.shape[:3], t_affine, order=1)

    tissue_masks: Dict[str, np.ndarray] = {}
    combined = None
    labels_path = result.m2m_dir / "final_tissues.nii.gz"
    if labels_path.is_file():
        labels, l_affine = preview.load_canonical(labels_path)
        labels_on_tensor = np.rint(preview.resample_to_grid(
            labels, l_affine, tensor6.shape[:3], t_affine, order=0)
            ).astype(np.int32)
        # A voxel outside where the model actually assigned a tensor (often
        # gray matter, if only white matter was made anisotropic) has to be
        # excluded rather than read as "isotropic": its eigenvectors are
        # meaningless, not evidence of zero anisotropy.
        has_tensor = np.abs(tensor6).sum(axis=-1) > 0
        for name, tag in task_settings.STAT_TISSUES:
            tissue_masks[name] = (labels_on_tensor == tag) & has_tensor
        combined = has_tensor & (labels_on_tensor > 0)
    return principal, anisotropy, background, combined, tissue_masks


def _surface_meshes(result: JobResult, quantity: str,
                    surface: str = "central"):
    """(vertices, faces, values) per hemisphere on one surface, from the
    run's overlays (central: SimNIBS's own; pial / white: the worker's)."""
    overlays = sorted(str(p) for p in
                      (result.run.outputs / "subject_overlays").glob("*"))
    meshes = []
    for hemi in ("lh", "rh"):
        gii = result.m2m_dir / "surfaces" / f"{hemi}.{surface}.gii"
        overlay = _overlay_for(overlays, hemi, quantity, surface)
        if not gii.is_file() or overlay is None:
            continue
        vertices, faces = preview.load_gifti_surface(gii)
        values = preview.read_curv_values(overlay, len(vertices))
        meshes.append((vertices, faces, values))
    return meshes


def _cortex_view(result: JobResult, quantity: str, saved: bool = False,
                 with_coil: bool = False):
    """The cortical surface over the T1, coloured by one field quantity."""
    meshes = _surface_meshes(result, quantity)
    if not meshes:
        return None
    t1, affine = _load_t1(result)
    signed = quantity == "normal"
    label = ("E normal component (V/m), + = out of the cortex" if signed
             else "|E| (V/m)")
    what = "normal component" if signed else "magnitude"
    subtitle = (f"{result.coil} at {result.placement} · dI/dt "
                f"{result.didt:g} A/s")
    coil = {}
    if with_coil and result.coil_centre_xyz:
        coil = {"coil_centre": result.coil_centre_xyz,
                "coil_normal": result.coil_normal_xyz}
        subtitle = (f"{result.coil} · coil at {result.coil_center} → "
                    f"{result.coil_ydir}, {result.distance:g} mm off the "
                    f"scalp · green marks the coil and its axis into the head")
    return preview.field_surface_mosaic(
        t1, affine, meshes,
        _titled(result, f"{result.subject_id}: E-field {what} on the "
                        f"central surface"),
        label, symmetric=signed,
        cmap="coolwarm" if signed else preview.FIELD_CMAP,
        annotation=_stats_annotation(result), subtitle=subtitle,
        **coil, **_figure_kwargs(saved))


def _field_volume(result: JobResult):
    """(field volume, affine) of |E| on the NIfTI grid, or None."""
    path = result.files.get("field_magn_nii")
    if not path or not Path(path).is_file():
        return None, None
    return preview.load_canonical(path)


def _white_surfaces(result: JobResult):
    """(vertices world-mm, faces) of the white surface per hemisphere."""
    out = []
    for hemi in ("lh", "rh"):
        gii = result.m2m_dir / "surfaces" / f"{hemi}.white.gii"
        if gii.is_file():
            out.append(preview.load_gifti_surface(gii))
    return out


def _volume_view(result: JobResult, saved: bool = False):
    """|E| in gray and white matter on the NIfTI grid, over the T1, with the
    white surface's cut drawn in white as the gray/white boundary."""
    volume, affine = _field_volume(result)
    if volume is None:
        return None
    t1, _t1_affine = _load_t1(result)
    kwargs = _figure_kwargs(saved)
    return preview.field_volume_mosaic(
        volume, t1,
        _titled(result, f"{result.subject_id}: E-field magnitude in gray "
                        f"and white matter"),
        "|E| (V/m)", annotation=_stats_annotation(result),
        subtitle=f"{result.coil} at {result.placement} · white line: the "
                 f"gray/white boundary (white surface)",
        boundaries=_white_surfaces(result), affine=affine,
        boundary_linewidth=(preview.BOUNDARY_LINEWIDTH * 1.5 if saved
                            else preview.BOUNDARY_LINEWIDTH),
        dpi=kwargs["dpi"], slice_fontsize=kwargs["slice_fontsize"])


def occupational_peak() -> Optional[float]:
    """The ICNIRP 2010 occupational weighted-peak restriction (V/m) at the
    conductivity table's frequency, or None if it cannot be evaluated."""
    freq = conductivity_frequency_hz()
    if freq is None:
        return None
    rms = icnirp_basic_restriction(freq).get(task_settings.EXCEEDANCE_GROUP)
    return None if rms is None else rms * task_settings.ICNIRP_PEAK_FACTOR


def _exceedance_view(result: JobResult, saved: bool = False):
    """|E| / the ICNIRP occupational peak restriction, in gray and white
    matter, drawn only where the ratio exceeds 1.

    The picture answers "where, and by how much, does the intended
    stimulation exceed the level the guideline sets for incidental
    exposure" -- deliberately not a compliance statement (ICNIRP excludes
    this exposure), but a map of the margin. The window starts at 1 (the
    restriction itself) and ends at the 99.5th percentile of the ratio, so
    the peak saturates like every other field picture here.
    """
    volume, affine = _field_volume(result)
    if volume is None:
        return None
    peak = occupational_peak()
    if not peak:
        return None
    field = np.where(np.isfinite(volume), volume, 0.0)
    ratio = field / peak
    tissue = field > 0
    above = ratio > 1.0
    # A picture and a number: how much of the tissue volume is over.
    n_tissue = int(tissue.sum())
    fraction = float(above.sum() / n_tissue) if n_tissue else 0.0
    voxel_ml = float(abs(np.linalg.det(affine[:3, :3]))) / 1000.0
    max_ratio = float(ratio[tissue].max()) if n_tissue else 0.0
    freq = conductivity_frequency()
    annotation = (f"ICNIRP 2010 {task_settings.EXCEEDANCE_GROUP}, peak, "
                  f"{freq}:  {peak:.3g} V/m\n"
                  f"GM+WM above it: {fraction:.1%} of the volume "
                  f"({above.sum() * voxel_ml:.0f} mL)\n"
                  f"max ratio {max_ratio:.3g}   dI/dt {result.didt:.4g} A/s")
    t1, _t1_affine = _load_t1(result)
    kwargs = _figure_kwargs(saved)
    return preview.field_volume_mosaic(
        ratio, t1,
        _titled(result, f"{result.subject_id}: |E| over the ICNIRP "
                        f"{task_settings.EXCEEDANCE_GROUP} restriction"),
        f"|E| / {peak:.3g} V/m (ICNIRP 2010 {task_settings.EXCEEDANCE_GROUP} "
        f"peak at {freq}); shown where > 1",
        mask=above, cmap=task_settings.EXCEEDANCE_CMAP, vmin=1.0,
        cbar_extend="max", annotation=annotation,
        subtitle=f"{result.coil} at {result.placement} · dI/dt "
                 f"{result.didt:.4g} A/s · white line: the gray/white "
                 f"boundary",
        boundaries=_white_surfaces(result), affine=affine,
        boundary_linewidth=(preview.BOUNDARY_LINEWIDTH * 1.5 if saved
                            else preview.BOUNDARY_LINEWIDTH),
        dpi=kwargs["dpi"], slice_fontsize=kwargs["slice_fontsize"])


def _wm_surface_view(result: JobResult, saved: bool = False):
    """|E| on the white surface -- the gray/white boundary.

    From the worker's white-surface overlay (the field read in the
    white-matter tetrahedra under each vertex, the same values the white
    VTK carries). If a run has no such overlay, the field volume is sampled
    at the vertices instead, the way this view was drawn before the worker
    wrote one.
    """
    meshes = _surface_meshes(result, "magn", surface="white")
    source = "field in the white-matter elements under each vertex"
    if not meshes:
        volume, affine = _field_volume(result)
        if volume is None:
            return None
        meshes = [(v, f, preview.sample_volume_at(volume, affine, v))
                  for v, f in _white_surfaces(result)]
        source = "field volume sampled at the vertices"
    if not meshes:
        return None
    t1, t1_affine = _load_t1(result)
    return preview.field_surface_mosaic(
        t1, t1_affine, meshes,
        _titled(result, f"{result.subject_id}: E-field magnitude on the "
                        f"white surface"),
        "|E| (V/m)", cmap=preview.FIELD_CMAP,
        annotation=_stats_annotation(result),
        subtitle=f"the gray/white boundary · {source} · {result.coil} at "
                 f"{result.placement}",
        **_figure_kwargs(saved))


def _coil_view(result: JobResult, saved: bool = False):
    t1, affine = _load_t1(result)
    xyz = result.coil_centre_xyz or [0.0, 0.0, 0.0]
    kwargs = _figure_kwargs(saved)
    return preview.coil_position_slice(
        t1, affine, xyz, result.coil_normal_xyz,
        title=_titled(result, f"{result.subject_id}: coil at "
                              f"{result.coil_center}, handle toward "
                              f"{result.coil_ydir}"),
        subtitle=f"{result.coil} · {result.distance:g} mm off the scalp · "
                 f"centre ({xyz[0]:.0f}, {xyz[1]:.0f}, {xyz[2]:.0f}) mm",
        dpi=kwargs["dpi"], slice_fontsize=kwargs["slice_fontsize"],
        linewidth=(preview.COIL_LINEWIDTH if saved else None))


def _tensor_orientation_view(result: JobResult, saved: bool = False):
    """The uploaded conductivity tensor's principal axis, direction-encoded.

    Only produces anything for an anisotropic run: an isotropic one has no
    tensor to check.
    """
    arrays = _tensor_grid_arrays(result)
    if arrays is None:
        return None
    principal, anisotropy, background, mask, tissue_masks = arrays
    stats = {name: preview.tensor_tissue_stats(anisotropy, tissue_mask)
             for name, tissue_mask in tissue_masks.items()}
    kwargs = _figure_kwargs(saved)
    return preview.tensor_orientation_mosaic(
        principal, anisotropy, background, mask,
        _titled(result, f"{result.subject_id}: conductivity tensor "
                        f"orientation"),
        annotation=preview.anisotropy_table(stats),
        subtitle=f"{result.tensor_name} · {result.anisotropy} mapping · "
                 f"principal eigenvector: R=left–right, "
                 f"G=anterior–posterior, B=superior–inferior · opacity = "
                 f"anisotropy index",
        dpi=kwargs["dpi"], slice_fontsize=kwargs["slice_fontsize"])


def render_view(view_label: str, result: Optional[JobResult]):
    """Picture for one view, or None when this run has nothing to draw."""
    if result is None:
        return None
    try:
        if view_label == VIEW_T1:
            volume, _affine = preview.load_canonical(result.m2m_dir / "T1.nii.gz")
            return preview.anatomy_mosaic(
                volume, _titled(result, f"{result.subject_id}: T1 (from "
                                        f"the head model)"),
                "T1 raw intensity (a.u.)")
        if view_label == VIEW_COIL:
            return _coil_view(result) if result.coil_centre_xyz else None
        if view_label == VIEW_FIELD_CORTEX_COIL:
            return _cortex_view(result, "magn", with_coil=True)
        if view_label == VIEW_FIELD_WM_SURFACE:
            return _wm_surface_view(result)
        if view_label == VIEW_FIELD_VOLUME:
            return _volume_view(result)
        if view_label == VIEW_EXCEEDANCE:
            return _exceedance_view(result)
        if view_label == VIEW_TENSOR_ORIENTATION:
            return _tensor_orientation_view(result)
    except Exception:
        logger.exception("Could not render %s", view_label)
        return None
    return None
