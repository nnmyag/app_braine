# ============================================================================
# tasks/mesh_head/settings.py — how the charm head-model task runs
# ----------------------------------------------------------------------------
# Task-local knobs. The global settings.py owns where SimNIBS lives
# (SIMNIBS_ROOT / SIMNIBS_PYTHON), because that is a machine fact rather than
# a task choice; this file owns what an operator picks for the service.
# ============================================================================
from pathlib import Path

import settings as global_settings

TASK_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TASK_DIR / "assets"

# --- SimNIBS ----------------------------------------------------------------
# charm is SimNIBS's own command-line program, so the task shells out to it
# rather than importing the simnibs API (which this app's env cannot do).
CHARM_BIN = global_settings.SIMNIBS_ROOT / "bin" / "charm"

# charm appeared in SimNIBS 4.0; the folder layout the page reads back
# (final_tissues.nii.gz, surfaces/*.central.gii, charm_report.html) is 4.x's.
# An older install is reported as "not supported" rather than run.
MIN_SIMNIBS_VERSION = (4, 0)

# How long the availability probe (`simnibs_python -c "import simnibs"`) may
# take before the install is declared unusable. Cold imports of simnibs take
# a few seconds; a hung NFS mount should not hang the page build.
PROBE_TIMEOUT_S = 60

# --- charm ------------------------------------------------------------------
# Threads for the segmentation (GEMS/samseg). SimNIBS's charm.ini defaults to
# 8; 0 means "all available". Jobs run one at a time across every task (see
# MAX_CONCURRENT_JOBS), so a charm run may take the whole machine. Applied by
# writing a patched copy of charm.ini into the run and passing --usesettings.
CHARM_THREADS = 0

# Hard ceiling on one charm run. A healthy run on a 1 mm T1 takes 30-60 min;
# anything beyond this is a hang, and the queue should be given back.
CHARM_TIMEOUT_S = 6 * 3600

# --- Subject id -------------------------------------------------------------
# Default subject id = the first CASE_NAME_CHARS characters of the T1's
# filename, which is how the study names its subjects (IXI034_t1 -> IXI034).
# Same convention as task 2's case name.
CASE_NAME_CHARS = 6

# --- Registration to the DTI b0 --------------------------------------------
# Only when the user says the T1/T2 are NOT yet in the b0's space. Rigid
# T1 -> b0 with Mattes MI, the parameters task 2 registers its T1 with (its
# p1_common/b4 lineage), so the two pages agree on what "in DWI space" means.
# The result is written into the T1/T2 HEADERS (qform + sform), never by
# resampling: charm gets the full-resolution T1, just in b0 world coordinates.
REG_HISTOGRAM_BINS = 50
REG_LEARNING_RATE = 1.0
REG_ITERATIONS = 200

# The mosaics' grid follows the shared rule in app/mosaic.py.
