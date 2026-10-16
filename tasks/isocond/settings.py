# ============================================================================
# tasks/isocond/settings.py — how the T1/T2 -> conductivity map task runs
# ----------------------------------------------------------------------------
# Task-local. vendor/isocond/settings.py owns the ALGORITHM (target size,
# spacing, normalization, colormaps, checkpoint folder); this file owns what an
# operator picks for the service.
# ============================================================================

# --- Models -----------------------------------------------------------------
# A gate rather than the whole vendored registry: a model can exist in
# vendor/isocond/nets/ without appearing on the page until it is confirmed.
# Entries are model FAMILIES: "condnettart" enables every CondNet-TART
# frequency whose checkpoint pair is in vendor/isocond/models
# (condnettart_<freq>_stage0/1.pth -> "CondNet-TART (<freq>)"); the page's
# list rescans that folder whenever it is opened, so a new pair needs no
# restart. "condnet" is the one fixed CondNet (900 MHz) model.
#
# NOTE on comparing them: CondNet-TART predicts S/m at the frequency in its
# name, CondNet predicts normalized [0, 1] values at 900 MHz. Each mosaic is
# labelled with its own model's unit, but the numbers are not comparable.
ENABLED_MODELS = ("condnettart", "condnet")

# Preferred model on the page and the CLI. If it is not present, the first
# CondNet-TART found (lowest frequency) is used.
DEFAULT_MODEL = "condnettart_10kHz"

# --- Inference --------------------------------------------------------------
DEVICE = "cpu"          # "cpu" or "cuda"

# Keep loaded checkpoints between jobs. CondNet-TART's stage-0 file is 107 MB,
# so re-reading it per job would dominate a run that otherwise takes seconds.
# Safe because the shared queue runs one job at a time -- see service.py.
CACHE_LOADED_MODELS = True

# --- Outputs ----------------------------------------------------------------
# The conductivity map is always written. This adds what the original CLI's
# --save-intermediate wrote: the segmentation with its mosaic and legend (for
# models that segment) and the exact normalized volumes the model saw.
SAVE_INTERMEDIATE = True

# With the intermediates, also a histogram of the predicted conductivity
# inside the anisotropic tissues against the isotropic rest (task 2's style),
# for models that segment. The tissues are matched by NAME against the
# snapshot's tissue_labels.csv (substring, case-insensitive), so a legend
# that spells them differently still finds them.
SAVE_VALUE_HISTOGRAM = True
ANISOTROPIC_TISSUE_NAMES = ("white matter", "brainstem", "thalamus")

# --- Canvas -----------------------------------------------------------------
# Ceiling of the conductivity view's colour scale, in S/m: eye humour at
# 10 kHz, the same tissue task 2 caps its mosaic at (EFF_MOSAIC_CAP_TISSUE),
# so the two pages' colour bars agree. The snapshot scales the view to the
# model's MAX_TISSUE_CONDUCTIVITY (2 S/m, CSF); the cap keeps the brain
# tissues off the foot of the scale. Values above it are capped. Applies to
# models whose output is in S/m; a normalized model (CondNet, [0, 1] at
# 900 MHz) keeps its own scale. None -> the model's own ceiling.
CONDUCTIVITY_VMAX = 1.5
