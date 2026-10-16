# ============================================================================
# tasks/isocond_debug/settings.py — the isocond result check page
# ----------------------------------------------------------------------------
# Task-local. The page checks a conductivity result that already exists
# (usually task 1's) against its scans; it estimates nothing. Registration
# and grid geometry are task 1's own, read from vendor/isocond.
# ============================================================================

# --- Display ----------------------------------------------------------------
ANATOMY_CMAP = "gray"            # T1/T2, raw intensities, volume percentile window
CONDUCTIVITY_CMAP = "jet"

# Colour scale of an S/m conductivity map: [0, this], the top tick reading
# "≥". The same cap as task 1's page (tasks/isocond/settings.py), so the two
# look alike; hover values are never capped. A normalized map (CondNet) uses
# [0, its maximum] instead.
CONDUCTIVITY_VMAX = 1.5

# A tissue gets a row in the per-slice table only when it covers at least
# this many pixels on the slice; fewer is noise.
MIN_TISSUE_PIXELS = 20

# Without Seg, the table gives each panel's min/median/max over the slice's
# head: pixels whose conductivity exceeds this fraction of the map's maximum.
HEAD_FRACTION = 0.05

# --- Architecture panel -----------------------------------------------------
# The panel shows the CondNet-TART checkpoints of the frequency in the Cond
# file name; this model is the fallback when the name gives none.
DEFAULT_MODEL_KEY = "condnettart_10kHz"
DEVICE = "cpu"          # "cpu" or "cuda"

# The dummy input pushed through the network to trace each layer's output
# shape: one slice, the model's own size.
TRACE_INPUT_SHAPE = (1, 1, 256, 256)
