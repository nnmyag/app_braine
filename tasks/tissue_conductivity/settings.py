# ============================================================================
# tasks/tissue_conductivity/settings.py — conductivity per tissue at any frequency
# ----------------------------------------------------------------------------
# Task-local. The physics (Gabriel 1996 parameters) and the class list are
# data files in assets/; this file holds the rules around them.
# ============================================================================
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TASK_DIR / "assets"

# Gabriel, Lau & Gabriel 1996 (Part III) four-dispersion Cole-Cole
# parameters, 57 tissues, and the 15 classes with the Gabriel tissue each one
# follows -- both editable as plain CSV.
PARAMETERS_CSV = ASSETS_DIR / "gabriel1996_parameters.csv"
CLASSES_CSV = ASSETS_DIR / "classes.csv"

# The anchor: every class takes its 10 kHz value from this table (by
# tissue_id), so the result at 10 kHz IS that table, whatever Gabriel says
# there. Read at run time, so an edit to it carries over.
from tasks.anicond.settings import DEFAULT_CONDUCTIVITY_CSV as ANCHOR_CSV  # noqa: E402
ANCHOR_COLUMN = "conductivity_iso_10khz"
ANCHOR_HZ = 10e3

# Frequencies the page accepts. The paper fitted its parameters over
# 10 Hz - 100 GHz (FIT_MIN_HZ); down to MIN_HZ the same formula is
# EXTRAPOLATED, and the page says so.
MIN_HZ, MAX_HZ = 1.0, 100e9
FIT_MIN_HZ = 10.0

# Below this frequency the page warns: Gabriel et al. state their parametric
# model is least certain at low frequencies (sparse, scattered data; below
# ~1 MHz, and most of all below ~100 Hz).
LOW_FREQUENCY_WARN_HZ = 1e6

DEFAULT_VALUE, DEFAULT_UNIT = 10.0, "kHz"

# One-click frequencies (value, unit), shown as buttons above the inputs.
QUICK_FREQUENCIES = ((1, "Hz"), (10, "Hz"), (50, "Hz"), (2, "kHz"), (10, "kHz"))
UNITS = {"Hz": 1.0, "kHz": 1e3, "MHz": 1e6, "GHz": 1e9}

# The YAML's normalization block: sigma_max = the largest class conductivity
# at the frequency (CSF, at every frequency), and this headroom tau, so the
# output scale is sigma_max / (1 - tau) -- as in the 10 kHz table.
TAU = 0.1
