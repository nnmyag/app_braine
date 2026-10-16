# ============================================================================
# tasks/condnettart_train/settings.py — train CondNet-TART at a new frequency
# ----------------------------------------------------------------------------
# Task-local. The training itself is the condnettart_training project's own
# command line, run in its own environment; nothing of it is re-implemented
# here.
# ============================================================================
from pathlib import Path

TRAINING_PROJECT = Path("/home/my/Documents/PROJ_CONDNETS/condnettart_training")
# Its environment (torch with CUDA); the app's own env has CPU-only torch.
TRAINING_PYTHON = Path("/home/my/anaconda3/envs/condnettart/bin/python")
# The training file whose data, model and schedule are used; only the
# conductivity table is replaced by the uploaded one (03 --table).
EXPERIMENT = "config/train_10kHz.yaml"

# How often the page shows the new lines of the running training's log.
POLL_SECONDS = 2
# Console: the last this-many lines of the log.
CONSOLE_TAIL_LINES = 200
