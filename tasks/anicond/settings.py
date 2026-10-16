# ============================================================================
# tasks/anicond/settings.py — how the DTI -> conductivity task runs
# ----------------------------------------------------------------------------
# Task-local. The global settings.py owns the server and the run folders; this
# file owns the pipeline. Anything a USER should be able to change per job is a
# widget on the page instead of a constant here.
# ============================================================================
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TASK_DIR / "assets"

# --- Models -----------------------------------------------------------------
# A gate rather than the whole vendored registry: a model can exist in
# vendor/cond_models/ for a CLI run without appearing on the page until it is
# confirmed. `isotropic` is deliberately absent -- it has no __init__, so a job
# cannot hand it the uploaded conductivity table (see vendor/VERSION.md).
ENABLED_MODELS = ("directmapping", "volumenormalized", "tractbased_e")
DEFAULT_MODEL = "tractbased_e"

# Models needing whole-brain tracts. For a job that selects none of these, the
# tractography step -- by far the longest -- is skipped entirely.
TRACT_BASED_MODELS = ("tractbased_e", "tractbased_g")

# Where a model's tensor is allowed to be anisotropic. Every labelled voxel
# outside that region is reset to sigma_iso * I (its table value) after the
# model runs:
#   tract-based models      -> the anisotropic tissue ids (WM)
#   diffusion-only (DM, VN) -> those plus the reference (GM) ids, the tissues
#                              their calibration is built on
# The vendored DM and VN scale the diffusion tensor in EVERY voxel, scalp and
# CSF included; this is what confines them, in the task layer (service.py).
RESTRICT_TO_MODEL_REGION = True

# --- Preprocessing ----------------------------------------------------------
# N4 bias field estimated from the b0 and reused for every volume, matching
# p1_common/b3 (USE_BIAS_CORRECTION = True there).
APPLY_BIAS_CORRECTION = True

# b3's other switches (gradient subsampling, intensity normalization,
# resampling) are all OFF in the study and are not offered here.

# --- Tensor fit and tractography -------------------------------------------
FA_THR = 0.2                # FA threshold of the tracking mask (p1_common/b6)
SEED_DENSITY = 1            # seeds per voxel
EXPORT_TRACTS_VTK = True    # keep the tractogram as a downloadable output,
                            # plus its simplified twin for viewing
                            # (<case>_wb_tracts_viz.vtk, vendor/tractviz)

# A user may upload that tractogram back to skip tracking. Fraction of its
# points that must land inside the DWI grid before it is accepted: a file from
# another scan or another coordinate space lands mostly outside, and would
# otherwise give an isotropic map without a word.
MIN_TRACT_GRID_OVERLAP = 0.9

# --- Segmentation -----------------------------------------------------------
# The segmentation must end up describing the SAME physical space as the DWI.
# Two ways in, chosen by the caller (the page's "already in DWI space" box):
#
#   already coregistered -> resampled onto the b0 grid with nearest-neighbour
#                           (never interpolated -- these are labels)
#   not coregistered     -> the T1 it was made from is registered to the b0
#                           rigidly and that transform is applied to the labels,
#                           which is what p1_common/b4 does. Without a T1 there
#                           is nothing to register through, so the job stops.
#
# Either way the overlap with the brain mask is checked and reported.
# Fraction of the ANISOTROPIC tissue's voxels that must fall inside the b0
# brain mask before the job stops warning. Measured on that tissue rather than
# on every label, because a whole-head map legitimately puts scalp and skull
# outside the brain -- but white matter outside it means the wrong space.
MIN_SEG_BRAIN_OVERLAP = 0.8

# --- Registration (only when the segmentation is NOT already in DWI space) --
# b4's rigid T1 -> b0 fit, parameter for parameter. Unlike task 1's Mattes-MI
# registration this one samples every voxel (SimpleITK's default), so it does
# not inherit that task's thread-dependent RANDOM sampling.
REG_HISTOGRAM_BINS = 50
REG_LEARNING_RATE = 1.0
REG_ITERATIONS = 200

# --- Outputs ----------------------------------------------------------------
# Both tensors are written in FSL's dtifit layout: 6 volumes, upper triangle
# in row order (xx, xy, xz, yy, yz, zz). The eigen-decompositions, the 3x3
# forms and MD are NOT written -- they are derivable from these, and the FA
# map (always written: it is a view and an intermediate) covers the review.
SAVE_TENSOR_6X1 = True          # the deliverable, outputs/<model>/<case>_c_tensor6x1
SAVE_DIFFUSION_TENSOR_6X1 = True    # outputs/diffusion/<case>_d_tensor6x1
SAVE_EFF_COND = True        # G_eff, the scalar map drawn on the canvas
SAVE_EFF_MOSAIC = True      # ... and its axial mosaic as a .png, next to it
SAVE_EFF_WMGM_MOSAIC = True     # ... and a second mosaic of WM and GM alone,
                                # scaled to their own values
# How much darker than the whole-head mosaic that second mosaic's b0 backdrop
# is drawn (a factor on its display window), so the colours stand out.
EFF_WMGM_BACKGROUND_DIM = 2.5
# Its colour scale: blue -> cyan -> green -> yellow -> orange over a FIXED
# range in S/m, the same for every run so mosaics compare; values beyond
# either end are clipped to it (the bar's end ticks say so). Deliberately not
# jet, so it cannot be mistaken for the whole-head mosaic.
EFF_WMGM_COLORS = ("#0000ff", "#00ffff", "#00ff00", "#ffff00", "#ff8c00")
EFF_WMGM_RANGE = (0.03, 0.2)
# Ceiling of that mosaic's colour scale: the conductivity of THIS tissue in the
# table the job runs with (Eye_humour: 1.5 S/m at 10 kHz). Scaled to the table's
# maximum (CSF, 2 S/m) WM and GM both sat at the foot of the scale in the same
# blue; scaled to blood they sit apart. Values above the ceiling are capped.
# Falls back to the table's maximum when the tissue is not in the table.
EFF_MOSAIC_CAP_TISSUE = "Eye_humour"
SAVE_ANISO_MASK = True      # which voxels the model actually made anisotropic
SAVE_EVAL_HISTOGRAM = True  # sigma_1/2/3 inside that mask, as a .png

# --- Preview mosaic ---------------------------------------------------------
# Mosaics are rendered ON DEMAND from the files above when a view is selected,
# not written by the job: six views x a minute-long job is a lot of figures
# nobody may look at, and the volumes are small enough to re-read.
MOSAIC_SLICES = 20
MOSAIC_COLS = 5
MOSAIC_CMAP = "jet"         # scalar maps (b0 uses gray, FA its own scale)

# --- Defaults offered on the page -------------------------------------------
# The tissue ids are NOT constants: they are read off the conductivity table in
# use (runctx.default_aniso_ids), so uploading a table with a different
# labelling moves the defaults with it.
DEFAULT_CONDUCTIVITY_CSV = ASSETS_DIR / "default_target_conductivity.csv"

# The subject id, which prefixes every output file, is taken from the
# segmentation's filename: its first N characters (IXI015_seg.nii.gz -> IXI015,
# the study's own naming). Editable on the page.
CASE_NAME_CHARS = 6
