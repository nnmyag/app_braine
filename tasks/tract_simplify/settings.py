# ============================================================================
# tasks/tract_simplify/settings.py — how the tractogram -> visual tracts task runs
# ----------------------------------------------------------------------------
# Task-local. Everything a user changes PER RUN is a widget on the page (the
# length cut, the thinning, the smoothing, the healing gap and angle); this
# file holds what they should not have to know.
# ============================================================================
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TASK_DIR / "assets"

# --- The post-processing: vendor/tractviz --------------------------------
# Healing, the length cut, QuickBundles thinning and the smoothing are
# vendor/tractviz/simplify.py, shared with the simplified tractograms tasks
# 2 and 5 export; the reasoning behind each knob is documented there. The
# page's defaults ARE that module's defaults, read from it so the two cannot
# drift; only the slider ranges are decided here.
from vendor import tractviz

DEFAULT_HEAL = tractviz.DEFAULT_HEAL
DEFAULT_HEAL_GAP_MM = tractviz.DEFAULT_HEAL_GAP_MM
HEAL_GAP_MIN, HEAL_GAP_MAX = 0.5, 5.0
DEFAULT_HEAL_ANGLE_DEG = tractviz.DEFAULT_HEAL_ANGLE_DEG
HEAL_ANGLE_MIN, HEAL_ANGLE_MAX = 10.0, 90.0
TANGENT_POINTS = tractviz.TANGENT_POINTS
GAP_STEP_MM = tractviz.GAP_STEP_MM

# Absolute length, mm, applied after healing (a relative cut would shift
# its meaning after healing makes tracts longer).
DEFAULT_MIN_LENGTH_MM = tractviz.DEFAULT_MIN_LENGTH_MM
MIN_LENGTH_MIN, MIN_LENGTH_MAX = 0.0, 200.0

QB_THRESHOLD_MM = tractviz.QB_THRESHOLD_MM
QB_POINTS = tractviz.QB_POINTS
DEFAULT_KEEP_PERCENT = tractviz.DEFAULT_KEEP_PERCENT      # of each cluster
KEEP_PERCENT_MIN, KEEP_PERCENT_MAX = 1.0, 100.0
THIN_SEED = tractviz.THIN_SEED

DEFAULT_SMOOTH_MM = tractviz.DEFAULT_SMOOTH_MM            # 0 = off
SMOOTH_MIN_MM, SMOOTH_MAX_MM = 0.0, 3.0

# --- Naming -----------------------------------------------------------------
# Subject id: the first characters of the tractogram's name (task 2's rule).
CASE_NAME_CHARS = 6

# --- Canvas -----------------------------------------------------------------
# One figure, two tiles side by side at the SAME axial slice -- the middle
# of the slices that carry anatomy in the reference image (app/mosaic.py's
# occupied span) -- the original tractogram on the left, the current
# revision on the right (preview.tract_pair_view()). No per-tile slice
# number and no dropdown: the slice index, the settings and the colour key
# are all in the one figure title instead.
# Each tile drawn this wide (inches, at CANVAS_DPI); two tiles side by side
# fill the canvas.
CANVAS_TILE_IN = 5.0
CANVAS_DPI = 150
# More than this many streamlines are subsampled for DRAWING only (the VTK
# and the numbers use all of them), the same cap task 5 uses.
MAX_DRAWN_STREAMLINES = 4000
DRAW_SEED = 0
# 50% thicker than task 5's canvas default (0.35): two tracings side by
# side, half the width each, read better a little heavier.
TRACT_LINEWIDTH_CANVAS = 0.525

# --- The 3D viewer ------------------------------------------------------------
# NiiVue (BSD-2), one JavaScript bundle in assets/, loaded by viewer.html in
# an iframe under the canvas. No Python dependency: if the bundle is missing
# the page says so and the mosaic canvas carries on. To install or update:
#     curl -L -o tasks/tract_simplify/assets/niivue.js \
#          https://unpkg.com/@niivue/niivue@0.69.0/dist/index.js
VIEWER_BUNDLE = ASSETS_DIR / "niivue.js"
VIEWER_PAGE = ASSETS_DIR / "viewer.html"
VIEWER_HEIGHT_PX = 540
