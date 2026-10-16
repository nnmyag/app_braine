# ============================================================================
# tasks/anicond/service.py — DTI -> anisotropic conductivity tensor
# ----------------------------------------------------------------------------
# THE seam. UI-agnostic: no gradio, no Qt, no print-driven control flow. Every
# front-end (web.py now; cli.py / gui.py later) calls run_job() and gets back
# paths and values -- never a widget.
#
# What it does, and where each step comes from in PROJ_DCONDNET/scripts:
#
#   1. stage the upload                            (new: a job is not a cohort)
#   2. N4 bias field from the b0, reused for       p1_common/b3
#      every volume
#   3. segmentation onto the b0 grid               p1_common/b4, minus the
#                                                  registration (see below)
#   4. WLS diffusion tensor fit, Otsu brain mask   p1_common/b5
#   5. whole-brain CSA tractography, FA >= 0.2     p1_common/b6
#   6. conductivity model(s) by name               p2_condmodel/c1
#   7. 3x3 -> 6x1 upper-triangular tensor          tools/convert_condtens3x3_
#                                                  to_condtens6x1.py
#
# The batch scripts themselves are not vendored: they iterate a BIDS cohort,
# skip already-logged subjects and write into derivative trees. What they DO
# per scan is the sequence above, calling the same vendored library functions in
# the same order, so a served result and a study result agree.
#
# THE SEGMENTATION'S SPACE is the one thing a caller has to answer for. Two
# ways in, and the caller says which:
#
#   already in DWI space -> only RESAMPLED onto the b0 grid (nearest-neighbour,
#                           labels are never interpolated)
#   not yet              -> the T1 it was drawn on is registered to the b0
#                           rigidly and that transform is applied to the labels
#                           -- b4 exactly. No T1, nothing to register through,
#                           so the job stops rather than guessing.
#
# Either way the overlap with the brain mask is measured and reported, so a
# segmentation that is still in the wrong space is loud rather than silent.
# ============================================================================

# --- repo root + vendor/anicond on sys.path, so `core` / `cond_models`
# --- import from any working directory (same bootstrap as PROJ_DCONDNET)
import os as _os, sys as _sys
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
for _p in (_root, _os.path.join(_root, "vendor", "anicond")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)

import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import SimpleITK as sitk
from matplotlib.colors import LinearSegmentedColormap
from dipy.core.gradients import gradient_table

from core import general as gnr
from core.cond import lib_cond as lcond
from core.cond.lib_tractbased_core import eff_cond_spherical_mean
from core.med import lib_dwi as ldwi
from core.med import lib_mri as lmri
from core.med import lib_tract as ltract
from core.tens import lib_tsio as ltsio

from app import jobs
from tasks.anicond import preview, runctx, tractio
from tasks.anicond import settings as task_settings
from vendor import tractviz

logger = logging.getLogger(__name__)

# A bval at or below this counts as a b0 (the same threshold the vendored fit
# applies when it checks that a b0 exists at all).
B0_MAX_BVAL = 50.0

# Below this the eigenvalue triple is clipped before G_eff is formed, matching
# core.cond.model_scoring.EPS -- a zero sigma_perp is a pole of the closed form.
EPS = 1e-12


# ============================================================================
# What a finished job can be looked at
# ----------------------------------------------------------------------------
# The label IS the key: a front-end offers these strings and hands one back to
# render_view(). Each is drawn from a FILE the run wrote (VIEW_SOURCE below), so
# the canvas still works after the page is reloaded, and two tabs cannot
# overwrite each other's preview the way a shared in-memory cache would.
# ============================================================================
VIEW_B0 = "DWI b0 (first volume)"
VIEW_SEG = "Tissue segmentation, in DWI space"
VIEW_ANISO = "Anisotropic tissues only"
VIEW_FA = "FA map"
VIEW_EFF = "Effective conductivity (G_eff)"
VIEW_EFF_WMGM = "Effective conductivity within WM-GM"
VIEW_HIST = "Eigenvalue histogram (anisotropic tissue)"

VIEW_LABELS = (VIEW_B0, VIEW_SEG, VIEW_ANISO, VIEW_FA, VIEW_EFF, VIEW_EFF_WMGM,
               VIEW_HIST)
DEFAULT_VIEW = VIEW_B0          # before a run: what an input check would use
RESULT_VIEW = VIEW_EFF_WMGM     # after one: what the job was started for

# view -> the role of the file it is drawn from. VIEW_SEG and VIEW_ANISO share
# one file: the second is the first with every other label dropped, so the two
# pictures cannot disagree about what was segmented.
VIEW_SOURCE = {
    VIEW_B0: "b0",
    VIEW_SEG: "seg_dwi",
    VIEW_ANISO: "seg_dwi",
    VIEW_FA: "fa",
    VIEW_EFF: "eff_cond",
    VIEW_EFF_WMGM: "eff_wmgm_mosaic",
    VIEW_HIST: "eval_hist",
}

# What the page lists as intermediates, in the order it lists them. Everything
# else a run writes (eigenvectors, MD, the diffusion tensor) stays in the
# archive rather than on the page. "tracts" is only present when a tract-based
# model ran -- the other models never track.
INTERMEDIATE_ROLES = ("seg_dwi", "aniso_mask", "fa", "d_tensor", "tracts",
                      "tracts_viz", "eff_cond", "eff_mosaic",
                      "eff_wmgm_mosaic", "eval_hist")


@dataclass
class JobResult:
    """What one completed job produced. Paths only -- no widgets, no arrays."""
    run: jobs.RunDir
    ctx: runctx.JobContext
    tensors_6x1: List[Path] = field(default_factory=list)   # the deliverable
    all_outputs: List[Path] = field(default_factory=list)
    # {role: path} for the files a front-end names or draws (see the roles
    # above). With more than one model -- only reachable from a cli -- the
    # per-model roles are the FIRST model's; every model's files are in
    # all_outputs and in the archive regardless.
    files: Dict[str, Path] = field(default_factory=dict)
    zip_path: Optional[Path] = None
    elapsed_s: float = 0.0
    summary: str = ""

    @property
    def intermediates(self) -> List[Tuple[str, Path]]:
        return [(role, self.files[role]) for role in INTERMEDIATE_ROLES
                if role in self.files]

    def short_name(self, path) -> str:
        """Drop the '<case>_' prefix every filename repeats."""
        name = Path(path).name
        prefix = f"{self.ctx.case_name}_"
        return name[len(prefix):] if name.startswith(prefix) else name


# ============================================================================
# Input loading
# ============================================================================
def _volume_sort_key(path: Path):
    """Order per-volume files by the LAST number in the name (DTI-07 -> 7).

    Falls back to the name itself when there is no number, so a set of files
    with no index still loads in a stable, reproducible order.
    """
    nums = re.findall(r"(\d+)", path.stem)
    return (0, int(nums[-1]), path.name) if nums else (1, 0, path.name)


def load_dwi_series(paths: Sequence, n_expected: Optional[int] = None,
                    report: jobs.Reporter = None
                    ) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """Load the uploaded DWI into one [X, Y, Z, N] array.

    Accepts either a single 4D NIfTI, or several 3D volumes (one per gradient
    direction, as PROJ_DCONDNET's `<subj>_DTI-NN.nii.gz` files) which are
    stacked in filename-number order. Returns (dwi_4d, affine, reference image);
    the reference is the first volume, and supplies the affine and header every
    output is written with.

    `n_expected` is the number of b-values. A multi-file upload may carry more
    volumes than the gradient table describes -- IXI ships a trailing QC volume
    -- so files whose filename index is past the end of the table are dropped,
    exactly as p1_common/b3 does with its `extra_indices`. Dropping is by INDEX,
    never by position, so it cannot silently discard a real direction.
    """
    files = sorted((Path(str(p)) for p in paths), key=_volume_sort_key)
    if not files:
        raise ValueError("No DWI file given")

    if n_expected is not None and len(files) > n_expected > 0:
        indexed = [(f, _volume_sort_key(f)) for f in files]
        numbered = [(f, key[1]) for f, key in indexed if key[0] == 0]
        if len(numbered) == len(files):
            extras = [f for f, idx in numbered if idx >= n_expected]
            if extras and len(files) - len(extras) == n_expected:
                files = [f for f, idx in numbered if idx < n_expected]
                jobs_report(report,
                            f"DWI: dropped {len(extras)} volume(s) past the end "
                            f"of the gradient table (likely QC): "
                            f"{', '.join(f.name for f in extras)}")

    first = nib.load(str(files[0]))
    if len(files) == 1:
        data = first.get_fdata()
        if data.ndim == 3:
            raise ValueError(
                f"{files[0].name} is a single 3D volume. Upload the whole DWI "
                "series -- either one 4D file, or every per-direction volume at "
                "once. A b0 alone cannot give a diffusion tensor.")
        if data.ndim != 4:
            raise ValueError(f"{files[0].name}: expected a 4D DWI, got shape "
                             f"{data.shape}")
        jobs_report(report, f"DWI: 1 file, {data.shape[3]} volumes, "
                            f"grid {data.shape[:3]}")
        return data, first.affine, first

    vols = []
    for path in files:
        img = nib.load(str(path))
        arr = img.get_fdata()
        if arr.ndim == 4 and arr.shape[3] == 1:
            arr = arr[..., 0]
        if arr.ndim != 3:
            raise ValueError(f"{path.name}: expected a 3D volume in a multi-file "
                             f"series, got shape {arr.shape}")
        if arr.shape != first.shape[:3]:
            raise ValueError(f"{path.name}: grid {arr.shape} does not match "
                             f"{files[0].name} {first.shape[:3]}")
        vols.append(arr)

    data = np.stack(vols, axis=-1)
    jobs_report(report, f"DWI: {len(files)} files stacked in name order "
                        f"({files[0].name} … {files[-1].name}), grid {data.shape[:3]}")
    return data, first.affine, first


def default_case_name(seg_path) -> str:
    """Subject id from the segmentation filename: its first CASE_NAME_CHARS
    characters, which is how the study names its subjects (IXI015_seg -> IXI015).
    Falls back to the whole stem for a shorter name, and to 'case' for none."""
    if not seg_path:
        return "case"
    stem = Path(str(seg_path)).name.split(".")[0]
    return stem[:task_settings.CASE_NAME_CHARS].strip() or "case"


def jobs_report(report: jobs.Reporter, message: str) -> None:
    """Send one progress line, if anybody is listening, and always log it."""
    logger.info(message)
    if report is not None:
        report(message)


def _write_nii(arr: np.ndarray, affine: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lmri.nib_save_from_arr(arr, affine, str(path.parent), path.name)
    return path


# ============================================================================
# Preprocessing
# ============================================================================
def bias_correct(dwi_4d: np.ndarray, b0_path: Path,
                 report: jobs.Reporter = None) -> np.ndarray:
    """N4 bias field estimated once from the b0, divided out of every volume.

    Exactly b3's strategy (estimate on the b0, reuse the field), just applied to
    the in-memory array instead of re-reading each volume from disk.
    """
    _corrected_b0, bias_field = ldwi.n4_bias_field(str(b0_path))
    # SimpleITK indexes [z, y, x]; NIfTI arrays are [x, y, z].
    bias = np.transpose(sitk.GetArrayFromImage(bias_field), (2, 1, 0))
    bias = np.where(np.abs(bias) < 1e-9, 1.0, bias)
    out = dwi_4d / bias[..., None]
    jobs_report(report, f"Bias correction: N4 field from the b0 applied to all "
                        f"{dwi_4d.shape[-1]} volumes "
                        f"(field range {bias.min():.3g}–{bias.max():.3g})")
    return out


def _for_registration(img: sitk.Image) -> sitk.Image:
    """b4's preprocess_for_registration: float32, one 3D scalar volume."""
    img = sitk.Cast(img, sitk.sitkFloat32)
    if img.GetDimension() == 4:
        img = img[:, :, :, 0]
    if img.GetNumberOfComponentsPerPixel() > 1:
        img = sitk.VectorIndexSelectionCast(img, 0)
    return img


def register_segmentation_via_t1(seg_path: Path, t1_path: Path, b0_path: Path,
                                 transform_path: Optional[Path] = None,
                                 report: jobs.Reporter = None) -> np.ndarray:
    """Rigidly register the T1 to the b0 and carry the labels along.

    p1_common/b4, parameter for parameter (Mattes MI, gradient descent, a
    geometry-centred Euler3D start), minus the anatomical resampling b4 also
    does -- this job needs the labels in DWI space, not the T1.

    The T1 is what registers: it is what the segmentation was drawn on, and a
    label map has no intensity structure a mutual-information metric could use.
    Nearest-neighbour for the labels themselves, as always.
    """
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

    jobs_report(report, "Registration: rigid T1 -> b0 (Mattes MI), the "
                        "transform the segmentation will be carried by …")
    t0 = time.time()
    transform = registration.Execute(fixed, moving)
    jobs_report(report,
                f"Registration: {registration.GetOptimizerStopConditionDescription()} "
                f"— final metric {registration.GetMetricValue():.4f} in "
                f"{time.time() - t0:.0f}s")

    if transform_path is not None:
        transform_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteTransform(transform, str(transform_path))

    seg = sitk.ReadImage(str(seg_path))
    b0 = sitk.ReadImage(str(b0_path))
    resampled = sitk.Resample(seg, b0, transform, sitk.sitkNearestNeighbor, 0,
                              seg.GetPixelID())
    segs = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))
    jobs_report(report, f"Segmentation: registered through the T1 and resampled "
                        f"onto the b0 grid {segs.shape} (nearest-neighbour)")
    return np.rint(segs).astype(np.int32)


def align_segmentation(seg_path: Path, b0_path: Path, ref_img,
                       report: jobs.Reporter = None) -> np.ndarray:
    """Put an ALREADY COREGISTERED segmentation on the b0 voxel grid.

    Same grid already -> loaded as is. Different grid -> resampled with
    NEAREST-NEIGHBOUR through the two headers' physical spaces, which assumes
    the segmentation is already coregistered with the DWI (see the module
    docstring). Labels are never interpolated.
    """
    seg_img = nib.load(str(seg_path))
    same_grid = (seg_img.shape[:3] == ref_img.shape[:3]
                 and np.allclose(seg_img.affine, ref_img.affine, atol=1e-4))
    if same_grid:
        jobs_report(report, "Segmentation: already on the b0 grid, used as is")
        return np.rint(seg_img.get_fdata()).astype(np.int32)

    sitk_seg = sitk.ReadImage(str(seg_path))
    sitk_ref = sitk.ReadImage(str(b0_path))
    resampled = sitk.Resample(sitk_seg, sitk_ref, sitk.Transform(),
                              sitk.sitkNearestNeighbor, 0, sitk_seg.GetPixelID())
    segs = np.transpose(sitk.GetArrayFromImage(resampled), (2, 1, 0))
    jobs_report(report,
                f"Segmentation: resampled {seg_img.shape[:3]} -> {segs.shape} "
                f"onto the b0 grid (nearest-neighbour). This assumes it is "
                f"already coregistered with the DWI -- no registration is done.")
    return np.rint(segs).astype(np.int32)


def check_segmentation(segs: np.ndarray, brain_mask: np.ndarray,
                       ctx: runctx.JobContext,
                       report: jobs.Reporter = None) -> None:
    """Report how well the segmentation lands on the brain, and warn if not.

    A segmentation in the wrong space still produces a full-looking tensor
    field, so this is the check that turns a silent wrong answer into a visible
    one. The number that matters is the ANISOTROPIC tissue's overlap with the
    brain mask -- a whole-head label map legitimately puts scalp and skull
    outside it, but white matter that is not inside the brain means the two
    volumes are not in the same space.

    Never aborts on a low overlap: the user may be running something unusual on
    purpose. It does abort when the anisotropic mask is empty, since there would
    be nothing for a model to refine.
    """
    labelled = segs > 0
    n_labelled = int(labelled.sum())
    if n_labelled == 0:
        raise ValueError("The segmentation is empty (no voxel above 0)")

    brain = brain_mask.astype(bool)
    jobs_report(report, f"Segmentation check: {n_labelled} labelled voxels, "
                        f"{float((labelled & brain).sum()) / n_labelled:.0%} of "
                        f"them inside the b0 brain mask (a whole-head label map "
                        f"is expected to sit partly outside)")

    present = set(int(v) for v in np.unique(segs))
    missing = [i for i in ctx.aniso_ids if i not in present]
    if missing:
        # Not an error: a cohort's WM ids are a superset of what any one
        # segmentation happens to contain, and np.isin simply skips the rest.
        jobs_report(report, f"Segmentation check: no voxel carries anisotropic "
                            f"tissue id(s) {missing}; they contribute nothing")

    aniso_mask = np.isin(segs, list(ctx.aniso_ids))
    n_aniso = int(aniso_mask.sum())
    if n_aniso == 0:
        raise ValueError(
            f"The segmentation contains no voxel with any of the anisotropic "
            f"tissue id(s) {list(ctx.aniso_ids)}, so there is nothing to make "
            f"anisotropic. Present ids: {sorted(present)}")

    aniso_overlap = float((aniso_mask & brain).sum()) / n_aniso
    line = (f"Segmentation check: anisotropic tissue = {n_aniso} voxels, "
            f"{aniso_overlap:.0%} inside the brain mask")
    if aniso_overlap < task_settings.MIN_SEG_BRAIN_OVERLAP:
        line += ("  ** LOW ** — the segmentation looks misaligned with the DWI. "
                 "The result will be wrong. Coregister it to the b0 first.")
    jobs_report(report, line)


# ============================================================================
# Tractography
# ============================================================================
def whole_brain_tracts(dwi_4d, gtab, FA, affine, ref_img, out_path: Path,
                       report: jobs.Reporter = None) -> list:
    """Track the whole brain and return the streamlines in VOX space.

    The .vtk is written and read back rather than converted in memory, on
    purpose: tracking returns RASMM coordinates and the models index voxels, and
    this is the exact round-trip b6 -> c1 performs. Doing the conversion a
    second way here is how the two would eventually disagree. The file is a
    deliverable anyway.

    The file goes through tasks/anicond/tractio rather than the vendored
    export/load pair: those hand .vtk to fury via dipy, whose version pin does
    not survive a fury upgrade. tractio writes and reads the same convention.
    """
    tracking_mask = ldwi.get_fa_map(FA, fa_thr=task_settings.FA_THR)
    n_seed_vox = int(np.count_nonzero(tracking_mask))
    jobs_report(report, f"Tractography: FA >= {task_settings.FA_THR} mask has "
                        f"{n_seed_vox} voxels; tracking (this is the long step)")

    t0 = time.time()
    streamlines_world = ltract.run_tractography(
        dwi_4d, gtab, affine=affine, mask=tracking_mask,
        seed_density=task_settings.SEED_DENSITY,
    )
    jobs_report(report, f"Tractography: {len(streamlines_world)} streamlines in "
                        f"{time.time() - t0:.0f}s")
    if not streamlines_world:
        raise RuntimeError(
            "Tractography produced no streamline. Check the gradient table and "
            f"that the FA mask is not empty (FA >= {task_settings.FA_THR}).")

    n_written = tractio.save_tractogram(streamlines_world, ref_img, out_path)
    streamlines = tractio.load_tractogram(out_path, ref_img)
    jobs_report(report, f"Tractography: saved {out_path.name} ({n_written} "
                        f"streamlines inside the grid) and reloaded "
                        f"{len(streamlines)} tracts in VOX space")
    return streamlines


def simplified_tracts(streamlines_vox, ref_img, out_path: Path,
                      report: jobs.Reporter = None) -> Path:
    """A lighter twin of the tractogram FOR VIEWING: vendor/tractviz with
    its defaults (healed, short fragments dropped, every bundle thinned,
    smoothed). Written through tractio like the tractogram itself, so it
    lands on the same b0 in 3D Slicer. Not an input to any model, which
    is why it is written after the tractogram and never read back."""
    streamlines_mm = [nib.affines.apply_affine(ref_img.affine,
                                               np.asarray(s, dtype=np.float64))
                      for s in streamlines_vox]
    simplified = tractviz.simplify_tracts(
        streamlines_mm,
        report=lambda line: jobs_report(report, f"Tracts (viz): {line}"))
    if not simplified.streamlines:
        raise ValueError("no streamline survives the simplification")
    n_written = tractio.save_tractogram(simplified.streamlines, ref_img,
                                        out_path)
    jobs_report(report, f"Tracts (viz): saved {out_path.name} ({n_written} "
                        f"streamlines, {out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def load_uploaded_tracts(tract_path: Path, ref_img, grid_shape,
                         report: jobs.Reporter = None) -> list:
    """Streamlines from a user-supplied tractogram, in VOX space of this DWI.

    Read exactly the way whole_brain_tracts reads back what it wrote, so a
    file from an earlier run of the same DWI reproduces that run's tensor. The
    file has no way of saying which DWI it belongs to, so the one check that
    can be made is made: a tractogram of another scan, or one saved in another
    space, lands mostly outside this grid and is refused rather than fed to a
    model that would silently produce an isotropic map.
    """
    t0 = time.time()
    streamlines = tractio.load_tractogram(tract_path, ref_img)
    if not streamlines:
        raise ValueError(f"{tract_path.name} contains no streamline")

    points = np.vstack(streamlines)
    inside = np.all((points >= -0.5) & (points < np.asarray(grid_shape) - 0.5),
                    axis=1)
    frac = float(inside.mean())
    jobs_report(report, f"Tractography: skipped — using {tract_path.name}: "
                        f"{len(streamlines)} streamlines loaded in VOX space in "
                        f"{time.time() - t0:.0f}s, {frac:.0%} of their points "
                        f"inside the DWI grid")
    if frac < task_settings.MIN_TRACT_GRID_OVERLAP:
        raise ValueError(
            f"{tract_path.name}: only {frac:.0%} of its points fall inside this "
            f"DWI's grid {tuple(grid_shape)}. It was tracked on another scan, or "
            f"saved in another coordinate space. Upload the tractogram a "
            f"previous run of THIS DWI wrote, or leave the box empty to track.")
    return streamlines


# ============================================================================
# The job
# ============================================================================
def run_job(dwi_paths: Sequence,
            bval_path,
            bvec_path,
            seg_path,
            model_names: Sequence[str],
            aniso_ids_text: Optional[str] = None,
            iso_ref_ids_text: Optional[str] = None,
            conductivity_csv=None,
            freq: Optional[str] = None,
            seg_registered: bool = True,
            t1_path=None,
            tracts_path=None,
            case_name: Optional[str] = None,
            run: Optional[jobs.RunDir] = None,
            report: jobs.Reporter = None) -> JobResult:
    """Run the whole DTI -> conductivity pipeline for one case.

    Params:
        dwi_paths        : one 4D NIfTI, or the per-direction 3D volumes
        bval_path        : b-values, one per volume
        bvec_path        : gradient directions (3 x N or N x 3)
        seg_path         : tissue label map
        model_names      : conductivity models to run, by name
        aniso_ids_text   : tissue id(s) given an anisotropic tensor, e.g. "13,14,15";
                           None -> the WM-related ids of the conductivity table
        iso_ref_ids_text : reference tissue id(s) for directmapping's global scale;
                           None -> the GM-related ids of that table
        conductivity_csv : target conductivity table; None -> the shipped default
        freq             : which conductivity_iso_<freq> column of that table to
                           use, e.g. "10khz"; None -> the config's
                           conductivity_freq_in_use if the table has it, else
                           the table's first frequency
        seg_registered   : True  -> the labels are already in the DWI's space and
                                    are only resampled onto the b0 grid
                           False -> `t1_path` is required and the labels are
                                    registered through a rigid T1 -> b0 fit
        t1_path          : the T1 the segmentation was drawn on; only read when
                           `seg_registered` is False
        tracts_path      : a whole-brain tractogram of THIS DWI (.vtk, as a
                           previous run wrote it), used instead of tracking;
                           only read when a selected model needs tracts
        case_name        : subject id, the prefix of every output file;
                           None -> default_case_name(seg_path)
        run              : an existing run folder; None -> a fresh one
        report           : callable taking one progress line

    Returns a JobResult. Raises ValueError for bad input, RuntimeError for a
    step that cannot produce a result.
    """
    t_start = time.time()
    run = run or jobs.new_run("anicond")
    case_name = (case_name or "").strip() or default_case_name(seg_path)

    if not seg_registered and not t1_path:
        # Refused here rather than at the widget, so a cli run cannot skip the
        # check: without a T1 there is no transform, and resampling a
        # segmentation that is in another space silently produces a full-looking
        # tensor field that is wrong everywhere.
        raise ValueError(
            "The segmentation is not in the DWI's space and no T1 was given. "
            "Upload the T1 it was drawn on (it is registered to the b0 and the "
            "labels are carried along), or tick the box saying the "
            "segmentation is already coregistered.")

    ctx = runctx.build(case_name, model_names, aniso_ids_text,
                       iso_ref_ids_text, conductivity_csv, freq)
    jobs_report(report, f"Job {run.job_id} — {len(ctx.model_names)} model(s)")
    for line in ctx.describe().splitlines():
        jobs_report(report, "  " + line)

    # --- 1. stage the upload -------------------------------------------------
    dwi_staged = [jobs.stage_upload(p, run) for p in dwi_paths]
    bval_staged = jobs.stage_upload(bval_path, run)
    bvec_staged = jobs.stage_upload(bvec_path, run)
    seg_staged = jobs.stage_upload(seg_path, run)
    t1_staged = (jobs.stage_upload(t1_path, run)
                 if (t1_path and not seg_registered) else None)
    tracts_staged = (jobs.stage_upload(tracts_path, run)
                     if (tracts_path and ctx.needs_tracts) else None)
    if conductivity_csv:
        jobs.stage_upload(conductivity_csv, run)
    (run.root / "run_config.txt").write_text(ctx.describe() + "\n")

    # --- 2. gradients and the DWI series ------------------------------------
    bvals, bvecs, _bvecs_norm = ldwi.load_bvals_bvecs(str(bval_staged),
                                                      str(bvec_staged))
    dwi_4d, affine, ref_img = load_dwi_series(dwi_staged, len(bvals), report)
    if dwi_4d.shape[-1] != len(bvals):
        raise ValueError(f"{dwi_4d.shape[-1]} DWI volume(s) but {len(bvals)} "
                         f"b-values — they must match, in the same order")

    b0_indices = np.where(np.asarray(bvals) <= B0_MAX_BVAL)[0]
    if b0_indices.size == 0:
        raise ValueError(f"No b0 volume: every b-value is above {B0_MAX_BVAL:g}")
    b0_idx = int(b0_indices[0])
    jobs_report(report, f"Gradients: {len(bvals)} volumes, b0 at index {b0_idx}, "
                        f"b-values {np.min(bvals):g}–{np.max(bvals):g}")

    # In inputs/, not outputs/: it is the uploaded DWI's own first b0, kept so
    # the page can show what everything else was computed from.
    b0_path = run.inputs / f"{case_name}_b0.nii.gz"
    b0 = dwi_4d[..., b0_idx].astype(np.float32).copy()   # the mosaics' backdrop
    _write_nii(b0, affine, b0_path)
    result = JobResult(run=run, ctx=ctx)
    result.files["b0"] = b0_path

    # --- 3. bias correction (b3) --------------------------------------------
    if task_settings.APPLY_BIAS_CORRECTION:
        dwi_4d = bias_correct(dwi_4d, b0_path, report)

    # --- 4. diffusion tensor (b5) -------------------------------------------
    jobs_report(report, "Diffusion tensor: fitting (WLS) …")
    gtab = gradient_table(bvals=bvals, bvecs=bvecs)
    tenfit, FA, MD, brain_mask, dwi_4d, gtab = ldwi.fit_dti_tensor_from_arrays(
        dwi_4d, gtab, dwi_4d[..., b0_idx], return_dat_for_tracking=True)

    d_evals = np.asarray(tenfit.evals, dtype=np.float64)
    d_evecs = np.asarray(tenfit.evecs, dtype=np.float64)
    n_bad = int(np.count_nonzero(~np.isfinite(d_evals)))
    if n_bad:
        # b5 saves the raw fit and lets the study notice; a served job cannot,
        # so non-finite eigenvalues become 0 (an unfittable voxel) and are named.
        jobs_report(report, f"Diffusion tensor: {n_bad} non-finite eigenvalue(s) "
                            f"set to 0")
        d_evals = np.nan_to_num(d_evals, nan=0.0, posinf=0.0, neginf=0.0)
        d_evecs = np.nan_to_num(d_evecs, nan=0.0, posinf=0.0, neginf=0.0)
    jobs_report(report, f"Diffusion tensor: fitted, brain mask "
                        f"{int(brain_mask.sum())} voxels, "
                        f"FA {np.nanmin(FA):.2f}–{np.nanmax(FA):.2f}")

    outputs: List[Path] = []
    diff_dir = run.outputs / "diffusion"
    fa_path = _write_nii(FA.astype(np.float32), affine,
                         diff_dir / f"{case_name}_d_fa.nii.gz")
    result.files["fa"] = fa_path
    outputs.append(fa_path)
    if task_settings.SAVE_DIFFUSION_TENSOR_6X1:
        # The same 6x1 layout as the conductivity tensor (FSL dtifit order),
        # so the two files are read by the same tools the same way.
        quad = np.nan_to_num(np.asarray(tenfit.quadratic_form, dtype=np.float64),
                             nan=0.0, posinf=0.0, neginf=0.0)
        sym, _asym = ltsio.as_tensor3x3(quad)
        d_path = _write_nii(ltsio.to_upper_triangular6(sym).astype(np.float32),
                            affine, diff_dir / f"{case_name}_d_tensor6x1.nii.gz")
        result.files["d_tensor"] = d_path
        outputs.append(d_path)

    # --- 5. segmentation in the b0's space (b4) -----------------------------
    if seg_registered:
        segs = align_segmentation(seg_staged, b0_path, ref_img, report)
    else:
        segs = register_segmentation_via_t1(
            seg_staged, t1_staged, b0_path,
            transform_path=run.outputs / f"{case_name}_t1_to_b0.tfm",
            report=report)
        outputs.append(run.outputs / f"{case_name}_t1_to_b0.tfm")
    if segs.shape != FA.shape:
        raise ValueError(f"Segmentation grid {segs.shape} does not match the "
                         f"DWI grid {FA.shape} after resampling")
    check_segmentation(segs, brain_mask, ctx, report)

    seg_path_out = _write_nii(segs.astype(np.int16), affine,
                              run.outputs / f"{case_name}_seg_match_dwi.nii.gz")
    result.files["seg_dwi"] = seg_path_out
    outputs.append(seg_path_out)

    # The voxels the models will actually make anisotropic. Written as its own
    # volume because it is the mask every later number is quoted over -- the
    # histogram, the overlap check, the conductivity window.
    aniso_mask = np.isin(segs, list(ctx.aniso_ids))
    if task_settings.SAVE_ANISO_MASK:
        mask_path = _write_nii(aniso_mask.astype(np.uint8), affine,
                               run.outputs / f"{case_name}_aniso_mask.nii.gz")
        result.files["aniso_mask"] = mask_path
        outputs.append(mask_path)

    # --- 6. whole-brain tractography (b6), only if a model needs it ---------
    # Tracking is the long step, and it depends on nothing a user changes
    # between runs of the same DWI (the model, the tissue ids, the table), so a
    # tractogram from an earlier run can stand in for it.
    streamlines = None
    if not ctx.needs_tracts:
        jobs_report(report, "Tractography: skipped — no selected model needs "
                            "tracts" + (" (the uploaded tractogram is not used)"
                                        if tracts_path else ""))
    elif tracts_staged is not None:
        streamlines = load_uploaded_tracts(tracts_staged, ref_img, FA.shape,
                                           report)
    else:
        tract_path = run.outputs / f"{case_name}_wb_tracts.vtk"
        streamlines = whole_brain_tracts(dwi_4d, gtab, FA, affine, ref_img,
                                         tract_path, report)
        if task_settings.EXPORT_TRACTS_VTK:
            outputs.append(tract_path)
            result.files["tracts"] = tract_path
            # ... and its simplified twin for viewing. An export, not the
            # result: if it fails the job goes on without it.
            try:
                viz_path = simplified_tracts(
                    streamlines, ref_img,
                    run.outputs / f"{case_name}_wb_tracts_viz.vtk", report)
                outputs.append(viz_path)
                result.files["tracts_viz"] = viz_path
            except Exception as exc:      # noqa: BLE001 - an export, not the result
                logger.exception("anicond simplified tractogram failed")
                jobs_report(report, f"Tracts (viz): skipped "
                                    f"({type(exc).__name__}: {exc})")

    # --- 7. conductivity models (c1) + 6x1 conversion -----------------------
    for model_name in ctx.model_names:
        model = runctx.build_model(model_name, ctx)
        jobs_report(report, f"[{model.short_name}] {model_name}: estimating …")
        t0 = time.time()
        C, c_evals, c_evecs = model(
            d_evals, d_evecs, segs=segs, streamlines=streamlines, affine=affine)
        jobs_report(report, f"[{model.short_name}] done in {time.time() - t0:.0f}s")

        region_mask, region_label = model_region(model_name, segs, ctx)
        if task_settings.RESTRICT_TO_MODEL_REGION:
            C, c_evals, c_evecs = restrict_to_region(
                C, c_evals, c_evecs, segs, region_mask, region_label,
                ctx.dict_cond, model_name, report)

        model_dir = run.outputs / model_name
        made = _save_model_outputs(C, c_evals, c_evecs, affine, model_dir,
                                   case_name, model_name, region_mask,
                                   region_label, aniso_mask,
                                   np.isin(segs, list(ctx.iso_ref_ids)), b0,
                                   eff_mosaic_cap(ctx.dict_cond), report)
        outputs += made["files"]
        result.tensors_6x1 += made["tensors_6x1"]
        # First model wins the named roles; a second one (cli only) still gets
        # its own folder, its own files and its place in the archive.
        for role, path in made["files_by_role"].items():
            result.files.setdefault(role, path)

    # --- 8. package ----------------------------------------------------------
    result.all_outputs = outputs
    result.zip_path = run.zip_outputs(f"{case_name}_anicond")
    result.elapsed_s = time.time() - t_start
    result.summary = _summarize(result)
    jobs_report(report, f"Finished in {result.elapsed_s:.0f}s — "
                        f"{len(outputs)} file(s), zipped as "
                        f"{result.zip_path.name}")
    return result


def effective_conductivity(c_evals) -> np.ndarray:
    """G_eff: the effective conductivity averaged over the unit sphere.

    sigma_para = sigma_1 and sigma_perp = (sigma_2 + sigma_3) / 2 -- the
    collapse core.cond.model_scoring._sigma_para_perp makes, and the quantity
    the TBe model calibrates against, so the map on the page is the same number
    the study scores. The closed form itself is the vendored one; re-typing it
    here is how the two would eventually disagree.
    """
    evals = np.clip(np.asarray(c_evals, dtype=np.float64), EPS, None)
    return np.asarray(eff_cond_spherical_mean(
        evals[..., 0], 0.5 * (evals[..., 1] + evals[..., 2]), eps=EPS))


def model_region(model_name: str, segs: np.ndarray,
                 ctx: runctx.JobContext) -> Tuple[np.ndarray, str]:
    """(mask, label) of the voxels `model_name` may make anisotropic.

    Tract-based models work on the anisotropic tissue ids alone (WM); the
    diffusion-only models on those plus the reference ids (GM) -- the two
    tissues their calibration is built on. See RESTRICT_TO_MODEL_REGION.
    """
    if model_name in task_settings.TRACT_BASED_MODELS:
        return np.isin(segs, list(ctx.aniso_ids)), "WM"
    ids = list(dict.fromkeys(ctx.aniso_ids + ctx.iso_ref_ids))
    return np.isin(segs, ids), "WM+GM"


def restrict_to_region(C, c_evals, c_evecs, segs, region: np.ndarray,
                       region_label: str, dict_cond: Dict[int, float],
                       model_name: str, report: jobs.Reporter = None):
    """Reset every voxel outside `region` to its isotropic tensor.

    sigma_iso * I from the table for a labelled voxel, 0 for an unlabelled
    one; eigenvectors become the identity. The vendored DM and VN models
    scale the diffusion tensor everywhere it was fitted, which puts an
    anisotropic tensor into scalp, CSF and noise outside the head; this
    confines the result to where the model is meant to apply. A no-op for
    a model that is already isotropic outside the region (TBe).
    """
    outside = ~np.asarray(region, dtype=bool)
    iso = np.zeros(segs.shape, dtype=np.float64)
    for tid, cond in dict_cond.items():
        iso[segs == tid] = cond

    c_evals = np.array(c_evals, dtype=np.float32, copy=True)
    c_evecs = np.array(c_evecs, dtype=np.float32, copy=True)
    C = np.array(C, dtype=np.float32, copy=True)
    was_aniso = outside & (np.ptp(c_evals, axis=-1) > 1e-9)
    c_evals[outside] = iso[outside, None]
    c_evecs[outside] = np.eye(3, dtype=np.float32)
    C[outside] = iso[outside, None, None] * np.eye(3, dtype=np.float32)
    jobs_report(report, f"[{model_name}] anisotropic in {region_label} only: "
                        f"{int(was_aniso.sum())} voxel(s) outside it reset to "
                        f"their isotropic value")
    return C, c_evals, c_evecs


def eff_mosaic_cap(dict_cond: Dict[int, float]) -> float:
    """Top of the effective-conductivity colour scale, in S/m: the table's
    value for EFF_MOSAIC_CAP_TISSUE (matched by name through the tissue
    list), or the table's maximum if that tissue has no row."""
    wanted = task_settings.EFF_MOSAIC_CAP_TISSUE.strip().lower()
    for tid, cond in dict_cond.items():
        if gnr.dict_tissue.get(tid, "").strip().lower() == wanted:
            return float(cond)
    return float(max(dict_cond.values()))


def eff_cond_mosaic(g_eff, aniso_mask, b0, case_name: str, model_label: str,
                    max_cond: float):
    """The G_eff picture: every tissue in colour over the b0.

    Anisotropic voxels carry G_eff, isotropic ones their isotropic value (the
    same number the tensor holds there). One LINEAR scale, [0, max_cond], with
    max_cond from eff_mosaic_cap() -- the table's blood conductivity -- the
    same fixed-range, jet-coloured convention task 1 draws its conductivity
    map with, so the two pages read alike. Anything above the ceiling (CSF,
    eyes, a runaway VN-fallback voxel) is capped to it rather than stretching
    the scale. The background (no tissue label) stays the grey b0.
    """
    g_eff = np.asarray(g_eff, dtype=np.float64)
    tissue = np.isfinite(g_eff) & (g_eff > 1e-6)          # a labelled voxel
    max_cond = float(max_cond)
    return preview.mosaic(
        np.clip(g_eff, 0.0, max_cond),
        f"{case_name} — effective conductivity, {model_label} "
        f"(G_eff where the model is anisotropic, σ_iso elsewhere; white "
        f"outline = WM)",
        "conductivity [S/m]", cmap=task_settings.MOSAIC_CMAP, mask=tissue,
        background=b0, vmin=0.0, vmax=max_cond, cap_top=True,
        contour=(np.asarray(aniso_mask).astype(bool)
                 if aniso_mask is not None else None),
        n_slices=task_settings.MOSAIC_SLICES, n_cols=task_settings.MOSAIC_COLS)


def _save_model_outputs(C, c_evals, c_evecs, affine, model_dir: Path,
                        case_name: str, model_name: str,
                        region_mask: Optional[np.ndarray] = None,
                        region_label: str = "region",
                        wm_mask: Optional[np.ndarray] = None,
                        gm_mask: Optional[np.ndarray] = None,
                        b0: Optional[np.ndarray] = None,
                        max_cond: float = 1.0,
                        report: jobs.Reporter = None) -> dict:
    """Write one model's results and return what was written.

    `region_mask` is where the model's tensor is anisotropic (G_eff and the
    histogram are taken over it); `wm_mask` is the anisotropic tissue itself,
    outlined on the mosaic for every model so the pictures compare;
    `gm_mask` the reference tissue, shown with WM on the second mosaic.
    """
    aniso_mask = region_mask
    files: List[Path] = []
    tensors_6x1: List[Path] = []
    by_role: Dict[str, Path] = {}

    if task_settings.SAVE_TENSOR_6X1:
        sym, asym = ltsio.as_tensor3x3(np.asarray(C))
        six = ltsio.to_upper_triangular6(sym)
        if asym > 1e-5:
            jobs_report(report, f"[{model_name}] input tensor was asymmetric "
                                f"(max |C - C.T| = {asym:.3e}); symmetrised")
        # No model name in the file: the folder it sits in is the model.
        path = _write_nii(six, affine, model_dir / f"{case_name}_c_tensor6x1.nii.gz")
        # Read back and expand, the same check the batch converter runs -- the
        # 6x1 file is the deliverable, so it is verified rather than assumed.
        chk = ltsio.from_upper_triangular6(nib.load(str(path)).get_fdata())
        scale = float(np.max(np.abs(sym))) or 1.0
        err = float(np.max(np.abs(chk - sym))) / scale
        jobs_report(report, f"[{model_name}] 6x1 tensor {six.shape} written; "
                            f"round-trip max|reload - C|/max|C| = {err:.2e}")
        files.append(path)
        tensors_6x1.append(path)
        by_role["tensor_6x1"] = path

    if task_settings.SAVE_EFF_COND:
        # G_eff is a statement about an ANISOTROPIC tensor. Outside the
        # anisotropic tissue the tensor is sigma_iso * I, whose sphere average
        # is sigma_iso itself -- so the map carries the plain isotropic value
        # there (the eigenvalue mean), by construction and not by coincidence.
        g_eff = effective_conductivity(c_evals)
        if aniso_mask is not None:
            g_eff = np.where(aniso_mask.astype(bool), g_eff,
                             np.asarray(c_evals, dtype=np.float64).mean(axis=-1))
        path = _write_nii(g_eff.astype(np.float32), affine,
                          model_dir / f"{case_name}_c_eff.nii.gz")
        if aniso_mask is not None and aniso_mask.any():
            inside = g_eff[aniso_mask.astype(bool)]
            jobs_report(report, f"[{model_name}] G_eff over the anisotropic "
                                f"tissue: median {np.median(inside):.3g}, range "
                                f"{inside.min():.3g}–{inside.max():.3g} S/m")
        files.append(path)
        by_role["eff_cond"] = path

        # The same picture the canvas draws on demand, written once so it
        # travels with the volume (and in the archive) without a page.
        if task_settings.SAVE_EFF_MOSAIC:
            path = model_dir / f"{case_name}_c_eff_mosaic.png"
            preview.save_png(eff_cond_mosaic(g_eff, wm_mask, b0, case_name,
                                             model_name, max_cond), path)
            files.append(path)
            by_role["eff_mosaic"] = path

        # The same map over WM ALONE, no outline: the
        # whole-head mosaic above is capped for comparison across tissues,
        # which hides the within-region structure (TBe's WM spans 0.070-0.083
        # on a 0-1.5 scale). Here the scale is a fixed brain-tissue range
        # (EFF_WMGM_RANGE), clipped at both ends, so that structure -- and
        # the WM/GM step -- is what is seen, and two runs compare.
        if (task_settings.SAVE_EFF_WMGM_MOSAIC and wm_mask is not None
                and gm_mask is not None):
            path = model_dir / f"{case_name}_c_eff_wmgm_mosaic.png"
            wm = np.asarray(wm_mask).astype(bool)
            gm = np.asarray(gm_mask).astype(bool)
            lo, hi = (float(v) for v in task_settings.EFF_WMGM_RANGE)
            wm_mean = float(g_eff[wm].mean()) if wm.any() else float("nan")
            gm_mean = float(g_eff[gm].mean()) if gm.any() else float("nan")
            short = lcond.model_short_name(model_name)
            ramp = LinearSegmentedColormap.from_list(
                "wmgm", list(task_settings.EFF_WMGM_COLORS))
            preview.save_png(
                preview.mosaic(
                    np.clip(g_eff, lo, hi),
                    f"{case_name}/{short} - Effective Conductivity within WM",
                    "$G_{eff}$ [S/m]", cmap=ramp, vmin=lo, vmax=hi,
                    cap_top=True, cap_bottom=True, mask=wm, background=b0,
                    subtitle=f"{g_eff.shape[0]}×{g_eff.shape[1]}×{g_eff.shape[2]}, "
                             f"WM Mean {wm_mean:.3g}",
                    background_dim=task_settings.EFF_WMGM_BACKGROUND_DIM,
                    n_slices=task_settings.MOSAIC_SLICES,
                    n_cols=task_settings.MOSAIC_COLS),
                path)
            files.append(path)
            by_role["eff_wmgm_mosaic"] = path
    else:
        g_eff = None

    # The one figure the job draws rather than renders on demand: it is also a
    # deliverable (the distribution is what a reviewer asks for), and unlike a
    # mosaic it cannot be redrawn from a single volume on disk -- it needs the
    # eigenvalues, which are three files and a mask away.
    if task_settings.SAVE_EVAL_HISTOGRAM and aniso_mask is not None:
        path = model_dir / f"{case_name}_c_evals_hist.png"
        preview.save_png(
            preview.eigenvalue_hist(
                c_evals, aniso_mask,
                title=f"{case_name} — {model_name}, {region_label}",
                eff_cond=g_eff),
            path)
        files.append(path)
        by_role["eval_hist"] = path

    return {"files": files, "tensors_6x1": tensors_6x1,
            "files_by_role": by_role}


def _summarize(result: JobResult) -> str:
    lines = [
        f"Case            : {result.ctx.case_name}",
        f"Job             : {result.run.job_id}",
        f"Models          : {', '.join(result.ctx.model_names)}",
        f"Elapsed         : {result.elapsed_s:.0f}s",
        f"Output folder   : {result.run.outputs}",
        "",
        "Conductivity tensors (6 components: Cxx, Cxy, Cxz, Cyy, Cyz, Czz):",
    ]
    lines += [f"  {p.relative_to(result.run.outputs)}" for p in result.tensors_6x1]
    return "\n".join(lines)


# ============================================================================
# Rendering the canvas
# ----------------------------------------------------------------------------
# Every view is drawn from a file the run wrote, on demand. Nothing here holds
# an array between calls: the volumes are a few MB, and a cache would either be
# per-process (two tabs overwriting each other) or per-result (the whole run in
# memory for as long as the tab stays open).
#
# Returns a PIL image, or a path for a view that is already a picture, so a Qt
# front-end could show the same thing.
# ============================================================================
def view_labels() -> List[str]:
    """The selectable views, in display order."""
    return list(VIEW_LABELS)


def render_view(view_label: str, result: Optional[JobResult]):
    """One view of a finished run, or None if this run has no such file."""
    if result is None:
        return None
    path = result.files.get(VIEW_SOURCE.get(view_label, ""))
    if not path or not Path(path).is_file():
        return None

    case = result.ctx.case_name
    if view_label in (VIEW_HIST, VIEW_EFF_WMGM):
        return str(path)                    # drawn by the job, already a PNG

    volume, _affine = lmri.nib_load(str(path))
    volume = np.asarray(volume)

    if view_label == VIEW_B0:
        return preview.mosaic(volume, f"{case} — b0", "b0 intensity (a.u.)",
                              cmap="gray", n_slices=task_settings.MOSAIC_SLICES,
                              n_cols=task_settings.MOSAIC_COLS)
    if view_label == VIEW_SEG:
        return preview.label_mosaic(
            volume, f"{case} — tissue segmentation in DWI space",
            names=gnr.dict_tissue, n_slices=task_settings.MOSAIC_SLICES,
            n_cols=task_settings.MOSAIC_COLS)
    if view_label == VIEW_ANISO:
        return preview.label_mosaic(
            volume, f"{case} — anisotropic tissues "
                    f"({runctx.describe_ids(result.ctx.aniso_ids)})",
            names=gnr.dict_tissue, highlight=result.ctx.aniso_ids,
            n_slices=task_settings.MOSAIC_SLICES,
            n_cols=task_settings.MOSAIC_COLS)
    if view_label == VIEW_FA:
        # Fixed 0-1: FA has a meaning at a given value, so an auto window would
        # make two runs incomparable at a glance.
        return preview.mosaic(volume, f"{case} — FA", "FA", cmap="gray",
                              vmin=0.0, vmax=1.0,
                              n_slices=task_settings.MOSAIC_SLICES,
                              n_cols=task_settings.MOSAIC_COLS)
    if view_label == VIEW_EFF:
        mask = None
        mask_path = result.files.get("aniso_mask")
        if mask_path and Path(mask_path).is_file():
            mask = np.asarray(lmri.nib_load(str(mask_path))[0]).astype(bool)
        b0 = None
        b0_path = result.files.get("b0")
        if b0_path and Path(b0_path).is_file():
            b0 = np.asarray(lmri.nib_load(str(b0_path))[0])
        return eff_cond_mosaic(volume, mask, b0, case,
                               ", ".join(result.ctx.model_names),
                               eff_mosaic_cap(result.ctx.dict_cond))
    return None
