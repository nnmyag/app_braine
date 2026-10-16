# ============================================================================
# one place to tune how APP_BRAINE runs
# ============================================================================
from pathlib import Path

# --- Repository layout ------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VENDOR_DIR = PROJECT_ROOT / "vendor"        # frozen PROJ_DCONDNET libraries
DATA_DIR = PROJECT_ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"                # data/runs/<task>/<job id>/
# One row per calculation request, every task: job id, task, browser session,
# IP, when it was requested / started / finished, how it ended. Appended by
# app/jobs.py (usage_start) and completed in place (usage_finish).
USAGE_LOG = DATA_DIR / "usage_log.csv"
TASKS_DIR = PROJECT_ROOT / "tasks"

# Files a page offers for download that are NOT a run's output -- a default
# table to edit and re-upload, an example coil. Globbed rather than listed, so
# adding a task still costs one folder and one line in app/registry.py.
ASSET_DIRS = sorted(TASKS_DIR.glob("*/assets"))

# --- Server -----------------------------------------------------------------
# Overridable from the command line: run_web.py --host 0.0.0.0 --port 7778
HOST = "127.0.0.1"
PORT = 7777

# --- Tasks 3 & 4 -------------------------------------------------------------
# SimNIBS ships its own Python 3.11; the app env cannot import `simnibs`, so
# tasks/mesh_head (charm) and tasks/mesh_sim (simulation) run their workers
# under this interpreter. See those folders.
SIMNIBS_ROOT = Path("/home/my/SimNIBS-4.6")
SIMNIBS_PYTHON = SIMNIBS_ROOT / "bin" / "simnibs_python"

# --- Jobs -------------------------------------------------------------------
# One heavy job at a time, across ALL tasks: tractography and charm each want
# the whole machine, and two at once would only make both slower. Later jobs
# queue and are told their position.
MAX_CONCURRENT_JOBS = 1

# Runs older than this are deleted when a new job starts. 0 disables pruning.
KEEP_RUNS_DAYS = 7

# --- Logging ----------------------------------------------------------------
LOG_LEVEL = "INFO"

# Loggers mirrored into a page's console box, so the browser sees the same
# detail the terminal does. "vendor" covers both snapshots' own logging
# (vendor.isocond.pipeline / .nets); the anicond snapshot logs under the bare
# `core` / `cond_models` names it is imported by.
CAPTURED_LOGGERS = ("tasks", "app", "core", "cond_models", "vendor")

# ...except these, which are dropped from the page's box (the terminal still
# gets them). Each already sends every line through the job's reporter, so
# capturing their logging too would print each step twice. Measured: without
# this, one `say()` puts two identical rows in the transcript.
UNCAPTURED_LOGGERS = (
    "vendor.isocond.pipeline.session",   # reports through its progress callback
    "tasks.isocond.service",             # say() reports every line
    "tasks.anicond.service",             # jobs_report() reports every line
    "tasks.mesh_head.service",           # say() reports every line
    "tasks.mesh_sim.service",            # the same
    "tasks.tract_efield.service",        # the same
    "tasks.tract_simplify.service",      # the same
    "tasks.isocond_debug.service",       # the same
    "tasks.registration.service",        # the same
    "tasks.resample.service",            # the same
)

# --- Page chrome ------------------------------------------------------------
APP_TITLE = "BRAINE - Portal for Brain & Electromag. Studies"
DEVELOPED_BY = "Hirata Laboratory"
LAST_UPDATE = "2026/9/15"
