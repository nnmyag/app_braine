# -*- coding: utf-8 -*-
"""
Trimmed stand-in for PROJ_DCONDNET/scripts/core/general.py.

The original is a COHORT DRIVER: it resolves the IXI BIDS roots, walks subject
folders, reads the accepted-subject list and hands batch scripts one scan at a
time. A web job has none of that -- it has one uploaded case in one run folder.

So everything path-, dataset- and cohort-shaped is gone, and what remains is the
~25 names the vendored libraries actually read off `gnr`:

    tract_window, tract_min_center_tracts, tract_min_tangent_tracts
    tract_min_len, tract_min_len_for_wholebrain_tracking
    sh_order_max, dtensor_fit_method
    dict_tissue, ls_gm_seg, ls_wm_seg, dict_cond_10khz, dict_cond_in_use
    WM_ISO_COND, cond_model_fdpath, mkdir_ifne

These are DEFAULTS ONLY. Anything a job varies -- the target conductivity table
and which label is the anisotropic tissue -- is passed explicitly to the model
constructors (`dict_cond=`, `wm_id=`, `WM_segs=`) by tasks/anicond/runctx.py, so
this module is never mutated at run time and two jobs can never see each other's
settings.

Values come from vendor/core/config.yaml, which is the `labels:`/`tractography:`
part of the original core/config.yaml with the `paths:` section dropped.
"""
import os

import yaml

# --- where the vendored snapshot lives -------------------------------------
VENDOR_FDPATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
cond_model_fdpath = os.path.join(VENDOR_FDPATH, "cond_models")
config_fpath = os.path.join(VENDOR_FDPATH, "core", "config.yaml")

with open(config_fpath, "r") as f:
    _cfg = yaml.safe_load(f)


# --- Tract window ----------------------------------------------------------
# SINGLE SOURCE OF TRUTH, as in the original: the tract-based models and the
# tract reference must see the same window, or a model is scored against a
# different neighbourhood than it computed on. Callers pass None to take these.
tract_window = _cfg["tract_window"]["edge_voxels"]
tract_min_center_tracts = _cfg["tract_window"]["min_center_tracts"]
tract_min_tangent_tracts = _cfg["tract_window"]["min_tangent_tracts"]

# --- Tractography / tensor fitting -----------------------------------------
tract_min_len = _cfg["tractography"]["min_len_mm"]
tract_min_len_for_wholebrain_tracking = _cfg["tractography"]["min_len_whole_brain_mm"]
sh_order_max = _cfg["tractography"]["sh_order_max"]
dtensor_fit_method = _cfg["tractography"]["dtensor_fit_method"]

# --- Tissue table and its isotropic conductivities -------------------------
dict_tissue = {int(k): v for k, v in _cfg["labels"]["tissue"].items()}

ls_gm_seg = _cfg["labels"]["GM_segs"]
ls_gm_seg = [int(x) for x in (ls_gm_seg if isinstance(ls_gm_seg, list)
                              else str(ls_gm_seg).split(","))]
ls_wm_seg = _cfg["labels"]["WM_segs"]
ls_wm_seg = [int(x) for x in (ls_wm_seg if isinstance(ls_wm_seg, list)
                              else str(ls_wm_seg).split(","))]

dict_cond_10khz = {int(k): v for k, v in _cfg["labels"]["conductivity_iso_10khz"].items()}

# The frequency whose table is in use, named by its conductivity_iso_<freq>
# suffix. A served job reads that same column off its CSV (tasks/anicond/runctx).
cond_freq_in_use = str(_cfg["labels"].get("conductivity_freq_in_use", "10khz")).strip().lower()
dict_cond_in_use = {int(k): v for k, v in
                    _cfg["labels"][f"conductivity_iso_{cond_freq_in_use}"].items()}

# --- WM isotropic conductivity (fallback for the tract-based models) -------
wm_id = [k for k, v in dict_tissue.items() if "wm" in v.lower()]
WM_ISO_COND = dict_cond_in_use[wm_id[0]] if wm_id else 0.07


# --- the one helper the vendored libs call ---------------------------------
def mkdir_ifne(fd_path):
    """Create a folder if it does not exist (parents included)."""
    os.makedirs(fd_path, exist_ok=True)
