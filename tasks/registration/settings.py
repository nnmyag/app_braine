# ============================================================================
# tasks/registration/settings.py — affine registration of one NIfTI onto another
# ----------------------------------------------------------------------------
# Task-local. SimpleITK, Mattes mutual information (works across modalities),
# coarse-to-fine, as isocond's T2->T1 registration -- but with a
# regular-step optimizer instead of isocond's plain gradient descent, whose
# convergence check stops after a few iterations at full resolution: on a
# known 8-degree / 6 % scale / 12 mm distortion of IXI258's T1, isocond's
# setup recovered labels to Dice 0.67, this one to 0.98 (2026-10-09).
# ============================================================================

# Transform choices offered on the page -> stages run, in order.
MODES = {
    "Rigid (6 DOF)": ("rigid",),
    "Affine (12 DOF; rigid, then affine)": ("rigid", "affine"),
}
DEFAULT_MODE = "Affine (12 DOF; rigid, then affine)"

# Viewer: each slice is drawn at the integer zoom that brings its longer side
# closest to this many screen pixels (1:1 or 2:1 ..., never a fraction, so the
# pixel under the cursor is the one whose value is shown).
VIEW_TARGET_PX = 256
# Displacement co-histogram: bins per axis, the intensity range shown
# (0 to this percentile of the non-zero voxels; brighter ones go in the last
# bin), figure size (inches) and resolution.
COHIST_BINS = 128
COHIST_TOP_PERCENTILE = 99.5
COHIST_FIGSIZE = (6.4, 7.4)
COHIST_DPI = 100
# Display window of the grey panels: these percentiles of the non-zero voxels.
WINDOW_PERCENTILES = (1.0, 99.0)

# --- Registration -----------------------------------------------------------
HISTOGRAM_BINS = 50
SAMPLING_PERCENTAGE = 0.2          # random voxel sample per iteration
SAMPLING_SEED = 42                 # fixed: the same inputs give the same result
SHRINK_FACTORS = (4, 2, 1)         # coarse-to-fine pyramid
SMOOTHING_SIGMAS_MM = (2, 1, 0)
LEARNING_RATE = 2.0                # first step, in mm (scales from physical shift)
MIN_STEP = 1e-4                    # stop when the step shrinks below this
MAX_ITERATIONS = 500               # per pyramid level
