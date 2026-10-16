# ============================================================================
# tasks/mesh_head/service.py — T1 (+T2) -> SimNIBS head model with charm
# ----------------------------------------------------------------------------
# THE seam. UI-agnostic: run_job() takes paths and returns paths, so the web
# page and a later cli.py produce byte-identical run folders.
#
# What one job does
#   1. probe        SimNIBS present, charm present, version >= 4      (once)
#   2. stage        the uploads into inputs/, named <sub>_T1 / _T2 / _b0
#   3. align        ONLY if the user says the T1/T2 are not yet in the DTI
#                   b0's space: rigid T1 -> b0 (task 2's parameters), then the
#                   transform is written into the T1 and T2 HEADERS. Nothing
#                   is resampled -- charm gets the full-resolution T1, in b0
#                   world coordinates, which is what the tensors of task 2 and
#                   the simulation of task 4 are in. An alignment-check mosaic
#                   (T1 contours over the b0) is written for the eye.
#   4. charm        `charm <sub> T1 [T2] --forcerun --usesettings run.ini`,
#                   cwd = outputs/, so m2m_<sub>/ lands in the run. stdout is
#                   streamed line by line into the transcript; a run.ini with
#                   the task's thread count is derived from SimNIBS's charm.ini.
#   5. vtk          the central / pial / white surfaces as one .vtk each with
#                   both hemispheres merged (vendor/gii2vtk), tagged SPACE=RAS
#                   so 3D Slicer drops them onto the T1 with no transform.
#   6. previews     tissue segmentation and cortical-surface mosaics, saved
#                   next to m2m_<sub>/ and re-drawn on demand by the page.
#   7. report + zip a one-page run report, then everything under outputs/.
#
# Why a subprocess and not the simnibs API: this app's env cannot import
# simnibs (its own Python 3.11 ships with the install), and charm IS a program
# already -- wrapping its CLI costs nothing and inherits every flag.
# ============================================================================
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk

import settings as global_settings
from app import jobs
from tasks.mesh_head import preview
from tasks.mesh_head import settings as task_settings
from vendor.gii2vtk import convert_m2m_surfaces

logger = logging.getLogger(__name__)

TASK_SLUG = "mesh_head"


# ============================================================================
# SimNIBS availability
# ----------------------------------------------------------------------------
# Probed once per process, at page build. The page reads the result to decide
# between a working form and a "not supported on this server" notice.
# ============================================================================
@dataclass(frozen=True)
class SimnibsStatus:
    available: bool
    version: Optional[str]
    message: str


def _probe_simnibs() -> SimnibsStatus:
    root = Path(global_settings.SIMNIBS_ROOT)
    python = Path(global_settings.SIMNIBS_PYTHON)
    charm = Path(task_settings.CHARM_BIN)
    if not root.is_dir():
        return SimnibsStatus(False, None,
                             f"SimNIBS not found: {root} does not exist "
                             f"(SIMNIBS_ROOT in settings.py).")
    if not python.is_file():
        return SimnibsStatus(False, None,
                             f"SimNIBS's Python not found at {python}.")
    if not (charm.is_file() and os.access(charm, os.X_OK)):
        return SimnibsStatus(False, None,
                             f"charm not found or not executable at {charm}.")
    try:
        proc = subprocess.run(
            [str(python), "-c", "import simnibs; print(simnibs.__version__)"],
            capture_output=True, text=True, env=_clean_env(),
            timeout=task_settings.PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return SimnibsStatus(False, None,
                             "SimNIBS did not answer within "
                             f"{task_settings.PROBE_TIMEOUT_S}s (import simnibs).")
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["?"]
        return SimnibsStatus(False, None,
                             f"`import simnibs` failed under {python}: {tail[0]}")
    version = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    numbers = tuple(int(n) for n in re.findall(r"\d+", version)[:2]) or (0,)
    if numbers < task_settings.MIN_SIMNIBS_VERSION:
        want = ".".join(map(str, task_settings.MIN_SIMNIBS_VERSION))
        return SimnibsStatus(False, version,
                             f"SimNIBS {version} is too old: charm needs "
                             f"{want} or newer.")
    return SimnibsStatus(True, version, f"SimNIBS {version} at {root}")


@lru_cache(maxsize=1)
def simnibs_status() -> SimnibsStatus:
    status = _probe_simnibs()
    (logger.info if status.available else logger.warning)(
        "mesh_head: %s", status.message)
    return status


def _clean_env() -> Dict[str, str]:
    """The environment charm runs in: ours, minus anything that would point
    SimNIBS's interpreter at this app's packages, plus SimNIBS's bin on PATH."""
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"):
        env.pop(key, None)
    env["PATH"] = f"{global_settings.SIMNIBS_ROOT / 'bin'}:{env.get('PATH', '')}"
    return env


# ============================================================================
# What a run produces
# ----------------------------------------------------------------------------
# Roles rather than filenames, so the page can name a file by what it IS.
# (role, path relative to outputs/ with {sub} for the subject id, meaning).
# Order is display order; a role whose file a run did not write is skipped.
# ============================================================================
FILE_GUIDE: List[Tuple[str, str, str]] = [
    ("run_report", "{sub}_run_report.txt",
     "this run in one page: inputs, timing, and what every file and folder "
     "below is"),
    ("mesh", "m2m_{sub}/{sub}.msh",
     "tetrahedral head mesh (gmsh v2), tissue-tagged — the input of the "
     "simulation task"),
    ("tissues", "m2m_{sub}/final_tissues.nii.gz",
     "tissue label volume on the T1 grid; ids in final_tissues_LUT.txt"),
    ("tissues_lut", "m2m_{sub}/final_tissues_LUT.txt",
     "id → tissue name and colour for final_tissues"),
    ("lh_central", "m2m_{sub}/surfaces/lh.central.gii",
     "left central cortical surface (mid-thickness), world mm"),
    ("rh_central", "m2m_{sub}/surfaces/rh.central.gii",
     "right central cortical surface"),
    ("lh_pial", "m2m_{sub}/surfaces/lh.pial.gii", "left pial surface"),
    ("rh_pial", "m2m_{sub}/surfaces/rh.pial.gii", "right pial surface"),
    ("lh_white", "m2m_{sub}/surfaces/lh.white.gii", "left white surface"),
    ("rh_white", "m2m_{sub}/surfaces/rh.white.gii", "right white surface"),
    ("vtk_central", "vtk/both.central.vtk",
     "central surface, both hemispheres in one model, for 3D Slicer "
     "(title line tagged SPACE=RAS: loads onto the T1 with no transform)"),
    ("vtk_pial", "vtk/both.pial.vtk",
     "pial surface, both hemispheres; charm's per-vertex sigma (mm) attached "
     "as a point scalar"),
    ("vtk_white", "vtk/both.white.vtk",
     "white surface, both hemispheres; sigma attached"),
    ("t1_used", "m2m_{sub}/T1.nii.gz",
     "the T1 charm worked on (header in b0 space when registration ran)"),
    ("t2_used", "m2m_{sub}/T2_reg.nii.gz", "the T2, registered to that T1"),
    ("tissues_mni", "m2m_{sub}/toMNI/final_tissues_MNI.nii.gz",
     "tissue labels warped to MNI; the warps are next to it in toMNI/"),
    ("report", "m2m_{sub}/charm_report.html",
     "charm's own QA report: segmentation and surfaces drawn over the MRI"),
    ("charm_log", "m2m_{sub}/charm_log.html", "charm's full log"),
    ("transform", "{sub}_T1_to_b0.tfm",
     "rigid transform (ITK, b0 → T1 physical) that was written into the headers"),
    ("t1_charm", "{sub}_T1_charm.nii.gz",
     "the input T1 as handed to charm: qform = sform, and in b0 world "
     "coordinates when registration ran"),
    ("t2_charm", "{sub}_T2_charm.nii.gz", "the input T2, same header treatment"),
    ("align_png", "{sub}_alignment_check.png",
     "registered T1 as contours over the b0"),
    ("tissues_png", "{sub}_tissues_mosaic.png", "tissue segmentation mosaic"),
    ("surfaces_png", "{sub}_surfaces_mosaic.png",
     "white / central / pial surfaces over the T1"),
]
FILE_MEANING = {role: meaning for role, _rel, meaning in FILE_GUIDE}
FILE_GUIDE_REL = {role: rel for role, rel, _meaning in FILE_GUIDE}

# Folders worth pointing at as a whole (they are in the archive).
FOLDER_GUIDE = [
    ("vtk/", "the cortical surfaces as VTK PolyData, lh + rh merged, one "
             "colour; load in 3D Slicer as Model (no transform needed)"),
    ("m2m_{sub}/eeg_positions/", "EEG electrode positions (10-10, 10-20, …) "
                                 "on this head, csv + geo"),
    ("m2m_{sub}/toMNI/", "non-linear warps subject ↔ MNI"),
    ("m2m_{sub}/segmentation/", "charm's intermediate segmentation and the "
                                "bias-corrected T1/T2"),
    ("m2m_{sub}/label_prep/", "upsampled labels charm meshed; edit these and "
                              "re-run `charm <sub> --mesh` to fix a mesh"),
]


@dataclass
class JobResult:
    """What one completed job produced. Paths only -- no widgets, no arrays."""
    run: jobs.RunDir
    subject_id: str
    m2m_dir: Path
    registered: bool                      # was the T1 moved into b0 space?
    files: Dict[str, Path] = field(default_factory=dict)   # role -> path
    # label -> (name of the uploaded file, staged copy in inputs/)
    inputs: Dict[str, Tuple[str, Path]] = field(default_factory=dict)
    simnibs: str = ""
    zip_path: Optional[Path] = None
    charm_s: float = 0.0
    elapsed_s: float = 0.0
    completed_at: Optional[datetime] = None
    summary: str = ""

    def guide(self) -> List[Tuple[str, Path, str]]:
        """(role, path, meaning) for every guided file this run wrote."""
        return [(role, self.files[role], FILE_MEANING[role])
                for role, _rel, _m in FILE_GUIDE if role in self.files]

    def folders(self) -> List[Tuple[Path, str]]:
        out = []
        for rel, meaning in FOLDER_GUIDE:
            p = self.run.outputs / rel.format(sub=self.subject_id)
            if p.is_dir():
                out.append((p, meaning))
        return out


# ============================================================================
# Small helpers
# ============================================================================
def say(report: jobs.Reporter, message: str) -> None:
    """Send one progress line, if anybody is listening, and always log it."""
    logger.info(message)
    if report is not None:
        report(message)


def default_subject_id(t1_path) -> str:
    """First CASE_NAME_CHARS characters of the T1's filename (IXI034_t1 ->
    IXI034), the study's naming. Falls back to the stem, then to 'subject'."""
    if not t1_path:
        return "subject"
    stem = Path(str(t1_path)).name.split(".")[0]
    return sanitize_subject_id(stem[:task_settings.CASE_NAME_CHARS])


def sanitize_subject_id(text) -> str:
    """charm turns the id into a folder name (m2m_<id>) and a file name
    (<id>.msh), so only letters, digits, '-' and '_' survive."""
    clean = re.sub(r"[^A-Za-z0-9_-]", "", str(text or "").strip())
    return clean or "subject"


def _for_registration(img: sitk.Image) -> sitk.Image:
    """float32, one 3D scalar volume (a 4D DWI keeps its first frame)."""
    img = sitk.Cast(img, sitk.sitkFloat32)
    if img.GetDimension() == 4:
        img = img[:, :, :, 0]
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    return img


# ============================================================================
# Alignment to the b0
# ============================================================================
def register_t1_to_b0(t1_path: Path, b0_path: Path,
                      report: jobs.Reporter = None) -> sitk.Transform:
    """Rigid T1 -> b0, task 2's parameters (Mattes MI, gradient descent,
    geometry-centred Euler3D start). Returns the ITK transform, which maps
    b0 physical points to T1 physical points (ITK's resampling convention)."""
    fixed = _for_registration(sitk.ReadImage(str(b0_path)))
    moving = _for_registration(sitk.ReadImage(str(t1_path)))

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=task_settings.REG_HISTOGRAM_BINS)
    registration.SetOptimizerAsGradientDescent(
        learningRate=task_settings.REG_LEARNING_RATE,
        numberOfIterations=task_settings.REG_ITERATIONS,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetInitialTransform(
        sitk.CenteredTransformInitializer(
            fixed, moving, sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY),
        inPlace=False)

    say(report, "Registration: rigid T1 -> b0 (Mattes MI) …")
    t0 = time.time()
    transform = registration.Execute(fixed, moving)
    say(report, f"Registration: "
                f"{registration.GetOptimizerStopConditionDescription()} — "
                f"final metric {registration.GetMetricValue():.4f} in "
                f"{time.time() - t0:.0f}s")
    return transform


def _affine_of(transform: sitk.Transform) -> np.ndarray:
    """4x4 matrix (LPS) of an affine-like ITK transform, found by pushing the
    origin and the three unit vectors through it -- so a Composite, Euler or
    Affine transform all come out the same way."""
    o = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)))
    m = np.eye(4)
    for i in range(3):
        e = [0.0, 0.0, 0.0]
        e[i] = 1.0
        m[:3, i] = np.asarray(transform.TransformPoint(tuple(e))) - o
    m[:3, 3] = o
    return m


def write_with_affine(src: Path, dst: Path,
                      affine: Optional[np.ndarray] = None) -> np.ndarray:
    """Copy `src` to `dst` with qform AND sform set to `affine` (default: the
    file's own best affine, i.e. its sform when coded, else its qform).

    charm refuses a scan whose qform and sform disagree beyond 1e-6 and asks
    for --forceqform; scanner exports routinely disagree by more than that,
    and a file with no qform at all would make --forceqform pick up garbage.
    Writing both from one matrix, then letting charm force the qform, gives it
    one geometry whatever the upload looked like. Voxel data are untouched.
    """
    img = nib.load(str(src))
    affine = img.affine if affine is None else np.asarray(affine, dtype=float)
    out = nib.Nifti1Image(np.asanyarray(img.dataobj), affine, img.header)
    out.set_qform(affine, code=1)
    out.set_sform(affine, code=1)
    nib.save(out, str(dst))
    return affine


def b0_space_affine(src: Path, transform: sitk.Transform) -> np.ndarray:
    """The affine that puts `src`'s voxels into the b0's world.

    A voxel v sits at RAS point A·v. ITK works in LPS, and the transform T
    maps b0-physical -> T1-physical, so the same voxel's b0-world position is
    D·T⁻¹·D·A·v with D = diag(-1, -1, 1). That product is the new affine.
    """
    img = nib.load(str(src))
    D = np.diag([-1.0, -1.0, 1.0, 1.0])
    return D @ np.linalg.inv(_affine_of(transform)) @ D @ img.affine


def _alignment_check(t1_b0space: Path, b0_path: Path, out_png: Path) -> None:
    """Resample the header-rewritten T1 onto the b0 grid with the IDENTITY --
    if the header math is right the contours land on the b0's anatomy."""
    b0 = _for_registration(sitk.ReadImage(str(b0_path)))
    t1 = _for_registration(sitk.ReadImage(str(t1_b0space)))
    on_b0 = sitk.Resample(t1, b0, sitk.Transform(), sitk.sitkLinear, 0.0)
    b0_arr = np.transpose(sitk.GetArrayFromImage(b0), (2, 1, 0))
    t1_arr = np.transpose(sitk.GetArrayFromImage(on_b0), (2, 1, 0))
    image = preview.alignment_mosaic(
        b0_arr, t1_arr, "Alignment check: registered T1 (red) over the b0")
    preview.save_png(image, out_png)


# ============================================================================
# charm
# ============================================================================
def _write_charm_settings(dst: Path) -> Path:
    """SimNIBS's charm.ini with the task's thread count. Only that one key is
    changed, so an upgrade's new defaults carry through untouched."""
    src = Path(global_settings.SIMNIBS_ROOT) / "simnibs_env" / "lib"
    candidates = list(src.glob("python3*/site-packages/simnibs/charm.ini"))
    if not candidates:
        raise FileNotFoundError("charm.ini not found under the SimNIBS install")
    text = candidates[0].read_text()
    text, n = re.subn(r"(?m)^(\s*threads\s*=\s*)\d+",
                      rf"\g<1>{int(task_settings.CHARM_THREADS)}", text)
    if n != 1:
        raise ValueError("charm.ini: expected exactly one `threads =` line")
    dst.write_text(text)
    return dst


# charm's stdout is mostly samseg's optimiser chatter (histograms, ITK object
# dumps, per-iteration deformations): thousands of lines per run, none of
# which a user acts on. Only lines matching these reach the console; every
# line goes to charm_stdout.log in the run folder, and on failure the last
# RAW_TAIL_ON_ERROR raw lines are pushed to the console as well.
CONSOLE_MARKERS = ("[ simnibs ]", "Time to", "Total time spent", "WARNING",
                   "ERROR", "Error", "Traceback", "Exception", "CRITICAL")
SWALLOWED_HEARTBEAT = 250          # "… N more lines" every this many hidden
RAW_TAIL_ON_ERROR = 40


def _console_worthy(line: str) -> bool:
    return any(marker in line for marker in CONSOLE_MARKERS)


def run_charm(subject_id: str, t1: Path, t2: Optional[Path], cwd: Path,
              settings_ini: Path, raw_log: Path,
              report: jobs.Reporter = None) -> None:
    """Run charm; its informative stdout lines stream into the transcript,
    all of it into `raw_log`.

    `--forcerun` so a re-run into the same folder does not stop on the
    existing m2m_ folder (never the case for a fresh run dir, but harmless).
    A watchdog kills a run past CHARM_TIMEOUT_S so the queue is given back.
    """
    cmd = [str(task_settings.CHARM_BIN), subject_id, str(t1)]
    if t2 is not None:
        cmd.append(str(t2))
    # --forceqform: charm's preferred answer to qform/sform disagreement; the
    # copies it is given were written with both from one matrix, so this only
    # absorbs the quaternion rounding that would otherwise trip its 1e-6 check.
    cmd += ["--forcerun", "--forceqform", "--usesettings", str(settings_ini)]
    say(report, "charm: " + " ".join(cmd[1:]))
    say(report, f"charm: working folder {cwd}; full stdout in {raw_log.name}")

    proc = subprocess.Popen(cmd, cwd=str(cwd), env=_clean_env(),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, errors="replace", bufsize=1)
    timed_out = threading.Event()

    def _kill():
        timed_out.set()
        proc.kill()

    watchdog = threading.Timer(task_settings.CHARM_TIMEOUT_S, _kill)
    watchdog.daemon = True
    watchdog.start()
    tail: List[str] = []
    swallowed = 0
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
                    say(report, f"charm | {line}")
                else:
                    swallowed += 1
                    if swallowed % SWALLOWED_HEARTBEAT == 0:
                        say(report, f"charm | … working ({swallowed:,} detail "
                                    f"lines so far, kept in {raw_log.name})")
        proc.wait()
    finally:
        watchdog.cancel()
    if timed_out.is_set():
        raise TimeoutError(f"charm exceeded {task_settings.CHARM_TIMEOUT_S}s "
                           f"and was killed")
    if proc.returncode != 0:
        say(report, f"charm | --- last {len(tail)} raw lines ---")
        for line in tail:
            say(report, f"charm | {line}")
        raise RuntimeError(f"charm exited with code {proc.returncode}; see the "
                           f"console above, {raw_log.name} and "
                           f"m2m_{subject_id}/charm_log.html")


# ============================================================================
# The job
# ============================================================================
def run_job(t1_path, t2_path=None, subject_id: Optional[str] = None,
            aligned_to_b0: bool = True, b0_path=None,
            run: Optional[jobs.RunDir] = None,
            report: jobs.Reporter = None) -> JobResult:
    """Build the head model. Blocks for the duration (30-60 min typically)."""
    t_start = time.time()
    status = simnibs_status()
    if not status.available:
        raise RuntimeError(f"This function is not supported on this server: "
                           f"{status.message}")
    if not t1_path:
        raise ValueError("A T1 is required.")
    if not aligned_to_b0 and not b0_path:
        raise ValueError("The T1/T2 are declared not to be in the DTI b0's "
                         "space, so a b0 is required to register them to.")
    sub = sanitize_subject_id(subject_id) if subject_id else default_subject_id(t1_path)
    run = run or jobs.new_run(TASK_SLUG)
    say(report, f"Job {run.job_id}: subject id {sub}, {status.message}")

    # -- 2. stage -----------------------------------------------------------
    t1 = jobs.stage_upload(t1_path, run, f"{sub}_T1.nii.gz")
    t2 = jobs.stage_upload(t2_path, run, f"{sub}_T2.nii.gz") if t2_path else None
    b0 = jobs.stage_upload(b0_path, run, f"{sub}_b0.nii.gz") if b0_path else None
    staged = {label: (Path(str(src)).name, dst)
              for label, src, dst in (("T1", t1_path, t1), ("T2", t2_path, t2),
                                      ("b0", b0_path, b0)) if dst is not None}
    for label, path in (("T1", t1), ("T2", t2), ("b0", b0)):
        if path is not None:
            img = nib.load(str(path))
            say(report, f"{label}: {img.shape} voxels, "
                        f"{tuple(round(float(z), 3) for z in img.header.get_zooms()[:3])} mm, "
                        f"{''.join(nib.aff2axcodes(img.affine))}")

    files: Dict[str, Path] = {}
    registered = False

    # -- 3. align -----------------------------------------------------------
    # Either way charm gets a copy in outputs/ with one consistent header.
    t1_charm = run.output(f"{sub}_T1_charm.nii.gz")
    t2_charm = run.output(f"{sub}_T2_charm.nii.gz") if t2 is not None else None
    if not aligned_to_b0:
        assert b0 is not None
        transform = register_t1_to_b0(t1, b0, report)
        tfm = run.output(f"{sub}_T1_to_b0.tfm")
        sitk.WriteTransform(transform, str(tfm))
        files["transform"] = tfm
        write_with_affine(t1, t1_charm, b0_space_affine(t1, transform))
        say(report, "T1: header rewritten into the b0's world coordinates "
                    "(no resampling; voxel data unchanged)")
        if t2 is not None:
            write_with_affine(t2, t2_charm, b0_space_affine(t2, transform))
            say(report, "T2: same header rewrite (charm registers it to the T1 "
                        "itself)")
        align_png = run.output(f"{sub}_alignment_check.png")
        _alignment_check(t1_charm, b0, align_png)
        files["align_png"] = align_png
        say(report, "Alignment check written — see the canvas view "
                    f"'{VIEW_ALIGN}'")
        registered = True
    else:
        say(report, "T1/T2 declared to be in the DTI b0's space already: "
                    "no registration")
        write_with_affine(t1, t1_charm)
        if t2 is not None:
            write_with_affine(t2, t2_charm)
        say(report, "T1/T2: copied with qform = sform (charm requires the two "
                    "to agree)")
    files["t1_charm"] = t1_charm
    if t2_charm is not None:
        files["t2_charm"] = t2_charm
    t1, t2 = t1_charm, t2_charm

    # -- 4. charm -----------------------------------------------------------
    ini = _write_charm_settings(run.inputs / "charm_settings.ini")
    say(report, f"charm: threads = {task_settings.CHARM_THREADS or 'all'} "
                f"(charm_settings.ini)")
    say(report, "charm: this takes 30-60 minutes on a 1 mm T1. Progress lines "
                "below are charm's own.")
    t_charm = time.time()
    run_charm(sub, t1, t2, run.outputs, ini, run.root / "charm_stdout.log", report)
    charm_s = time.time() - t_charm
    say(report, f"charm: finished in {charm_s / 60:.1f} min")

    m2m = run.outputs / f"m2m_{sub}"
    for role, rel, _meaning in FILE_GUIDE:
        p = run.outputs / rel.format(sub=sub)
        if p.is_file() and role not in files:
            files[role] = p
    for must in ("mesh", "tissues", "lh_central", "rh_central"):
        if must not in files:
            raise RuntimeError(f"charm ended without writing "
                               f"{FILE_GUIDE_REL[must].format(sub=sub)}; "
                               f"see m2m_{sub}/charm_log.html")

    result = JobResult(run=run, subject_id=sub, m2m_dir=m2m,
                       registered=registered, files=files, inputs=staged,
                       simnibs=status.message, charm_s=charm_s)

    # -- 5. vtk -------------------------------------------------------------
    say(report, "VTK: converting central / pial / white surfaces, lh + rh "
                "merged …")
    try:
        vtk_dir = run.outputs / "vtk"
        written = convert_m2m_surfaces(m2m / "surfaces", vtk_dir,
                                       surfaces=SURFACE_NAMES, merge=True,
                                       with_sigma=True, hemisphere_scalar=False,
                                       space="RAS")
        for stem, path in written.items():
            role = f"vtk_{stem.split('.')[-1]}"
            if role in FILE_MEANING:
                files[role] = path
            say(report, f"VTK: {path.relative_to(run.outputs)} "
                        f"({path.stat().st_size / 1e6:.0f} MB)")
    except Exception as exc:                  # noqa: BLE001 - an export, not the result
        logger.exception("mesh_head vtk export failed")
        say(report, f"VTK: skipped ({type(exc).__name__}: {exc})")

    # -- 6. previews --------------------------------------------------------
    say(report, "Previews: tissue segmentation and cortical surfaces …")
    try:
        tissues_png = run.output(f"{sub}_tissues_mosaic.png")
        preview.save_png(_tissue_mosaic(result), tissues_png)
        files["tissues_png"] = tissues_png
        surfaces_png = run.output(f"{sub}_surfaces_mosaic.png")
        preview.save_png(_surface_mosaic(result, SURFACE_NAMES,
                                         linewidth=SAVED_LINEWIDTH),
                         surfaces_png)
        files["surfaces_png"] = surfaces_png
    except Exception as exc:                  # noqa: BLE001 - a picture, not the result
        logger.exception("mesh_head previews failed")
        say(report, f"Previews: skipped ({type(exc).__name__}: {exc})")

    # -- 7. report + zip ----------------------------------------------------
    # Written twice: once before the archive so a copy travels inside it,
    # once after so the copy on disk carries the archive line and the total
    # including the archiving step.
    result.completed_at = datetime.now()
    result.elapsed_s = time.time() - t_start
    files["run_report"] = write_run_report(result, run.output(f"{sub}_run_report.txt"))
    say(report, "Archiving outputs/ (the m2m folder is several hundred MB) …")
    result.zip_path = run.zip_outputs(f"{sub}_m2m")
    result.completed_at = datetime.now()
    result.elapsed_s = time.time() - t_start
    write_run_report(result, files["run_report"])
    result.summary = _summarize(result)
    return result


# ============================================================================
# Run report
# ----------------------------------------------------------------------------
# The middle-left panel of the page, as a text file that travels with the
# outputs: what went in, what came out and what each file is -- so a folder
# found on a disk a year later still explains itself.
# ============================================================================
def write_run_report(result: JobResult, path: Path) -> Path:
    out = result.run.outputs
    when = result.completed_at or datetime.now()
    lines = [
        f"mesh_head run report — subject {result.subject_id}",
        "=" * 72,
        f"Job id            : {result.run.job_id}",
        f"Completed         : {when:%Y-%m-%d %H:%M:%S}",
        f"Total execution   : {result.elapsed_s / 60:.1f} min "
        f"(charm {result.charm_s / 60:.1f} min"
        + (", archive included)" if result.zip_path else
           ", before archiving)"),
        f"SimNIBS           : {result.simnibs}",
        f"Registration      : "
        + ("rigid T1 -> DTI b0 (Mattes MI); transform written into the "
           "T1/T2 headers, no resampling" if result.registered else
           "none (T1/T2 declared already in the DTI b0's space)"),
        f"charm settings    : {result.run.inputs / 'charm_settings.ini'} "
        f"(threads = {task_settings.CHARM_THREADS or 'all'})",
        "",
        "Inputs (uploaded file -> staged copy)",
        "-" * 72,
    ]
    for label, (name, staged) in result.inputs.items():
        lines.append(f"  {label:<3} {name}  ->  {staged}")
    lines += ["", f"Output folder: {out}", "-" * 72]
    width = max((len(str(Path(p).relative_to(out)))
                 for _r, p, _m in result.guide()), default=20)
    for _role, p, meaning in result.guide():
        lines.append(f"  {str(Path(p).relative_to(out)):<{width}}  {meaning}")
    lines += ["", f"Folders inside m2m_{result.subject_id}/", "-" * 72]
    for folder, meaning in result.folders():
        lines.append(f"  {str(folder.relative_to(out)) + '/':<{width}}  {meaning}")
    lines += ["", "Archive", "-" * 72]
    if result.zip_path and Path(result.zip_path).is_file():
        lines.append(f"  {result.zip_path}  "
                     f"({Path(result.zip_path).stat().st_size / 1e6:.0f} MB, "
                     f"everything under outputs/)")
    else:
        lines.append("  (being written — this copy of the report is the one "
                     "inside the archive)")
    lines += ["", f"Full transcript: {result.run.log_file}",
              f"charm stdout   : {result.run.root / 'charm_stdout.log'}", ""]
    path.write_text("\n".join(lines))
    return path


def _summarize(result: JobResult) -> str:
    lines = [f"Done in {result.elapsed_s / 60:.1f} min — m2m folder: "
             f"{result.m2m_dir}"]
    try:
        lab, _ = preview.load_canonical(result.files["tissues"])
        lut = preview.read_lut(result.files["tissues_lut"]) \
            if "tissues_lut" in result.files else {}
        ids, counts = np.unique(lab[lab != 0].astype(np.int32),
                                return_counts=True)
        parts = [f"{lut[int(i)][0] if int(i) in lut else int(i)}: {int(c):,}"
                 for i, c in zip(ids, counts)]
        lines.append("Tissue voxels — " + "; ".join(parts))
    except Exception:                          # noqa: BLE001 - summary only
        pass
    for hemi in ("lh", "rh"):
        role = f"{hemi}_central"
        if role in result.files:
            try:
                v, f = preview.load_gifti_surface(result.files[role])
                lines.append(f"{hemi}.central: {len(v):,} vertices, "
                             f"{len(f):,} triangles")
            except Exception:                  # noqa: BLE001
                pass
    if "mesh" in result.files:
        lines.append(f"Head mesh: {result.files['mesh'].name}, "
                     f"{result.files['mesh'].stat().st_size / 1e6:.0f} MB")
    if result.zip_path:
        lines.append(f"Archive: {result.zip_path.name}, "
                     f"{result.zip_path.stat().st_size / 1e6:.0f} MB")
    return "\n".join(lines)


# ============================================================================
# Views for the canvas
# ----------------------------------------------------------------------------
# The input views are drawn from the UPLOADED files whenever selected -- even
# after a run -- so swapping a T1 and looking at "T1 loaded" never shows the
# previous subject. Everything else comes from the finished run's folder.
# ============================================================================
VIEW_T1 = "T1 loaded"
VIEW_T2 = "T2 loaded"
VIEW_B0 = "DTI b0 loaded"
VIEW_ALIGN = "Alignment check: registered T1 over b0"
VIEW_TISSUES = "Tissue segmentation (final_tissues)"
VIEW_SURF_ALL = "Cortical surfaces over T1: white / central / pial"

VIEW_LABELS = (VIEW_T1, VIEW_T2, VIEW_B0, VIEW_ALIGN, VIEW_TISSUES,
               VIEW_SURF_ALL)
INPUT_VIEWS = {VIEW_T1: "t1", VIEW_T2: "t2", VIEW_B0: "b0"}
DEFAULT_VIEW = VIEW_T1
RESULT_VIEW = VIEW_SURF_ALL           # what the job was started for

SURFACE_NAMES = ("white", "central", "pial")
SAVED_LINEWIDTH = 1.2                 # the saved mosaic; the canvas keeps 0.5


def view_labels() -> List[str]:
    return list(VIEW_LABELS)


def is_input_view(view_label: str) -> Optional[str]:
    return INPUT_VIEWS.get(view_label)


def render_input_preview(path, label: str):
    if not path:
        return None
    volume, _affine = preview.load_canonical(path)
    return preview.anatomy_mosaic(volume, f"{label} — {Path(str(path)).name}",
                                  f"{label} raw intensity (a.u.)")


def _tissue_mosaic(result: JobResult):
    lab, _ = preview.load_canonical(result.files["tissues"])
    lut = preview.read_lut(result.files["tissues_lut"]) \
        if "tissues_lut" in result.files else {}
    return preview.label_mosaic(
        lab, f"{result.subject_id}: tissue segmentation (charm final_tissues)",
        lut)


def _surface_mosaic(result: JobResult, names=SURFACE_NAMES,
                    linewidth: float = 0.5):
    t1, affine = preview.load_canonical(result.files["t1_used"])
    surfaces = {}
    for name in names:
        meshes = [preview.load_gifti_surface(result.files[f"{h}_{name}"])
                  for h in ("lh", "rh") if f"{h}_{name}" in result.files]
        if meshes:
            surfaces[name] = meshes
    return preview.surface_mosaic(
        t1, affine, surfaces,
        f"{result.subject_id}: {' / '.join(surfaces)} surface(s) over the T1",
        linewidth=linewidth)


def render_result_view(view_label: str, result: Optional[JobResult]):
    """Picture for a result view, or None if this run has no such view."""
    if result is None:
        return None
    if view_label == VIEW_ALIGN:
        p = result.files.get("align_png")
        return str(p) if p and Path(p).is_file() else None
    if view_label == VIEW_TISSUES:
        return _tissue_mosaic(result)
    if view_label == VIEW_SURF_ALL:
        return _surface_mosaic(result, SURFACE_NAMES)
    return None
