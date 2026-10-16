# ============================================================================
# tasks/tract_efield/settings.py — how the field -> tracts task runs
# ----------------------------------------------------------------------------
# The question this task answers: which tract streamlines reach the part of
# the white-matter surface where the TMS field is strongest? Everything below
# is a choice about "strongest" and "reach"; the page exposes the two that
# matter (the percentile and the contact distance) and the rest is fixed
# here so every run of a cohort is selected the same way.
# ============================================================================
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TASK_DIR / "assets"

# --- The hot spot ----------------------------------------------------------
# Vertices of the white surface whose |E| is at or above this percentile of
# the surface's own distribution. Area-weighted (each vertex carries a third
# of its triangles' area), the same cumulative-weight rule SimNIBS and
# mesh_sim's worker use for the volume percentiles, so a coarse patch of the
# surface does not count for more than a fine one. 99.5 is the study's
# "percentile-to-maximum" choice.
DEFAULT_PERCENTILE = 99.5
PERCENTILE_MIN, PERCENTILE_MAX = 50.0, 99.99
# An absolute threshold (V/m) overrides the percentile when > 0 -- for a
# cohort compared at one fixed field strength rather than each subject's own
# tail. 0 = use the percentile.
DEFAULT_ABSOLUTE_V_PER_M = 0.0

# --- "Reaching" the hot spot ----------------------------------------------
# A streamline is connected to the hot spot when a point of it lies within
# this many mm of a hot-spot vertex. Whole-brain CSA streamlines stop where
# FA drops, i.e. at or just short of the gray/white boundary, so an endpoint
# typically sits 1-3 mm inside the white surface; 3 mm reaches those without
# also catching fibres merely passing under the spot.
DEFAULT_CONTACT_MM = 3.0
CONTACT_MM_MIN, CONTACT_MM_MAX = 0.5, 15.0
# Which points count: any point along the streamline within the distance (a
# tract merely passing beneath the hot spot counts, not only one that
# terminates there) -- a fibre running under the hot spot is exposed to the
# field whether or not it ends there. Fixed by the user's choice
# (2026-09-14); not a page control -- select_streamlines() still takes a
# mode and honours "endpoint" from a future cli call, the page always asks
# for this one.
CONTACT_MODES = (("either endpoint within the distance", "endpoint"),
                 ("any point along the streamline within the distance", "any"))
DEFAULT_CONTACT_MODE = "any"

# --- Space check -----------------------------------------------------------
# The tractogram is in the DWI's space (task 2) and the simulation in the
# head model's, which task 3 put into the DWI's b0 space -- so they overlay
# with no transform, PROVIDED the head model was built that way. Two cheap
# witnesses, both reported and both warned about below these fractions:
# the fraction of streamline points inside the field volume's grid, and
# the fraction of streamline endpoints within ENDPOINT_NEAR_MM of the white
# surface (a tractogram in the wrong space has its endpoints anywhere).
MIN_GRID_OVERLAP = 0.9
ENDPOINT_NEAR_MM = 5.0
MIN_ENDPOINT_NEAR_FRACTION = 0.5

# --- Healing broken tracts (page label: "Heal broken tracts") --------------
# Tractography breaks a long tract into pieces (a stopping criterion fires
# mid-tract, the next seed picks it up a step later), so a streamline that
# reaches the hot spot may be only the first piece and say nothing about
# where the tract goes. With healing on, a streamline whose endpoint lies
# within JOIN_MM of an endpoint of a selected streamline, AND continues in
# its direction (the two exit tangents within JOIN_MAX_ANGLE_DEG of
# head-to-tail), is added; repeated up to MAX_HOPS times from the newly
# added pieces. The direction test is what stops this from swallowing the
# whole cortex: near the gray/white boundary thousands of streamlines END
# side by side, and two neighbours that both terminate there point the same
# way, not head-to-tail. Endpoint gap and angle sit on one line with the
# checkbox on the page, task 6's layout; the hop count is fixed (below), not
# a control -- three hops was never worth a slider in practice.
DEFAULT_FOLLOW_BREAKS = False
DEFAULT_JOIN_MM = 2.0
JOIN_MM_MIN, JOIN_MM_MAX = 0.5, 10.0
DEFAULT_MAX_HOPS = 3           # fixed; was a page control, no longer one
MAX_HOPS_MIN, MAX_HOPS_MAX = 1, 10
JOIN_MAX_ANGLE_DEG = 45.0      # default; the page's angle control starts here
JOIN_ANGLE_MIN, JOIN_ANGLE_MAX = 10.0, 90.0
# How far in from an endpoint the exit tangent is measured (points): a
# single step is noisy on a 0.5 mm step size.
TANGENT_POINTS = 4

# --- The simplified copy: the *___simplified.vtk files ---------------------
# vendor/tractviz's simplification of the selection, for a viewer: its own
# healing of the whole selection (gap and angle below -- the hop-following
# above is a different thing, it grows the selection), short fragments
# dropped, every bundle thinned, smoothed. Gentler than tract_simplify's
# page defaults (100 mm, 10 %, σ 1.5 mm): a selection is already a small
# set of tracts, and a tenth of it was too few to look at. The pictures and
# the numbers never use this copy.
SIMPLIFY_HEAL = True
SIMPLIFY_HEAL_GAP_MM = 2.0
SIMPLIFY_HEAL_ANGLE_DEG = 45.0
SIMPLIFY_MIN_LENGTH_MM = 20.0
SIMPLIFY_KEEP_PERCENT = 50.0
SIMPLIFY_SMOOTH_MM = 1.0

# --- Activating function --------------------------------------------------
# What a fibre in the field "feels" (Rattay's activating function): with
# E_s = E · t the field's component along the streamline, f = -dE_s/ds
# (V/m^2) is the cable equation's source term -- positive depolarises.
# It is invariant to the streamline's arbitrary direction (E_s and s flip
# together), so task 2's streamlines serve as they are. At a fibre's END
# the driver is the field itself, E · t_out with t_out the tangent
# pointing out of the fibre through that end: positive depolarises the
# terminal (the end the field points toward is the one whose membrane
# potential rises). Both are computed from the E VECTOR volume in the
# simulation archive, sampled along the streamline.
#
# The field comes off a ~1 mm grid, so E_s is smoothed along the fibre
# before it is differentiated; without this the derivative of a
# piecewise-linear interpolant is a staircase. Sigma in mm.
ACTIVATION_SMOOTH_MM = 1.5
# Sampling near the gray/white boundary blends the field with the zero the
# NIfTI carries outside GM+WM; the interpolation is mask-aware (E*mask and
# mask interpolated separately, divided) and a point where the mask weight
# is below this is outside the tissue and gets no field.
ACTIVATION_MIN_MASK_WEIGHT = 0.05
# The activation colour window: symmetric, +- this percentile of |f| along
# the selected streamlines (the picture and the VTK share it). f is spiky
# -- a few bends carry values ten times the bulk -- so the window is set
# well below the maximum, or nearly every segment sits in the map's
# neutral middle and the picture shows nothing.
ACTIVATION_COLOR_PCTL = 95.0
# Blue (hyperpolarising) -> mid grey (zero) -> red (depolarising): a
# diverging map with a GREY middle rather than white, so a near-zero
# segment is still a visible fibre yet cannot be taken for the hot spot's
# white; light enough to read on the darkened T1.
ACTIVATION_RAMP = ((0.2, 0.45, 1.0), (0.62, 0.62, 0.62), (1.0, 0.2, 0.15))

# --- Figures ---------------------------------------------------------------
# The field mosaic is read zoomed into one slice at a time, so it is saved
# far above mesh_sim's 200 dpi, and the canvas copy above the default too
# (the page's image viewer zooms). The T1 mosaic keeps the shared default.
FIGURE_DPI = 320
CANVAS_DPI = 150
SLICE_FONTSIZE = 10.5
FIELD_CMAP = "jet"
# Streamlines are drawn where they pass through a slab this thick around
# each drawn slice: thinner and a tract crossing the slice obliquely is a
# few dots, thicker and every slice shows the same tangle. 5 mm with the
# depth fade (TRACT_ALPHA_*) shows a run of each fibre without flattening
# the slab into a smear.
SLAB_MM = 5.0
# The T1 behind the picture: pushed darker than mesh_sim's 1.7 (how much
# brighter than its own 99th percentile the window's top is set), so the
# direction-coloured tracts carry the picture.
BACKGROUND_DIM = 3.2
# Streamlines are coloured by direction the way every DTI viewer encodes
# it: red = left-right, green = anterior-posterior, blue = superior-inferior,
# per segment. Nothing else on the picture is in colour: the white surface
# is a thin neutral line (the gray/white boundary, for orientation) and the
# hot spot the same line in white, a little thicker.
#
# Why the strokes are this thin: at FIGURE_DPI a tile is ~2.75 px per mm,
# so a 1 pt stroke is ~1.6 mm wide -- wider than the spacing between
# neighbouring streamlines -- and opaque neighbours fuse into filled
# patches instead of reading as lines. ~0.45 pt (about 2 px) keeps them
# apart; the depth fade below does the rest.
TRACT_LINEWIDTH_CANVAS = 0.35
TRACT_LINEWIDTH_SAVED = 0.45
# Opacity of a streamline segment falls from TRACT_ALPHA_NEAR on the slice
# plane to TRACT_ALPHA_FAR at the slab's edge, so a bundle reads as depth
# rather than as one flat block, and overlaps add up instead of covering.
TRACT_ALPHA_NEAR = 0.9
TRACT_ALPHA_FAR = 0.25
# A piece reached by following a break (hop >= 1) is drawn this fraction as
# thick as a piece that touches the hot spot, so the direct hits stand out.
FOLLOWED_LINEWIDTH_FACTOR = 0.6
SURFACE_LINEWIDTH_CANVAS = 0.3
SURFACE_LINEWIDTH_SAVED = 0.45
SURFACE_COLOR = "#8a8a8a"             # the gray/white boundary
HOTSPOT_COLOR = "white"
HOTSPOT_LINEWIDTH_FACTOR = 1.8        # the hot spot's stroke, x the surface's
# The E-field-coloured tract VTK: jet from 0 to this percentile of |E|
# along the selected streamlines (one outlier point must not compress the
# rest of the scale into blue); red is the strongest.
TRACT_COLOR_PCTL = 99.0
# Drawing every one of a large selection makes the mosaic a smear and the
# PNG slow; above this many, a random subset (fixed seed) is drawn. The
# VTK and the numbers always use the whole selection.
MAX_DRAWN_STREAMLINES = 4000
DRAW_SEED = 0
