# -*- coding: utf-8 -*-
"""
Conductivity-model dispatcher -- VENDORED, TRIMMED. See vendor/VERSION.md.

Kept byte-identical from PROJ_DCONDNET/scripts/core/cond/lib_cond.py:
    get_available_cond_models, load_cond_model, model_short_name,
    get_model_short_names

This is the API layer the study's rule points at: a script never imports a model
module, it asks for one by name and gets an instance back. The service keeps
that rule -- tasks/anicond/service.py knows model NAMES, never model classes.

Dropped: generate_iso_cond_dict (its CSV branch is superseded by
tasks/anicond/runctx.parse_conductivity_csv, which reads the uploaded table and
validates it against the tissue list) and every scan-IO helper --
cond_output_paths, load_scan_inputs, load_streamlines, the tract_reference_*
family, saved_conductivity_exists, load_saved_conductivity. All of those address
files by BIDS subject folder and derivative tag; a job addresses files inside
its own run folder, and service.py owns those paths.
"""
import glob
import importlib
import inspect
import os
import sys

from core import general as gnr


def get_available_cond_models(cond_models_fdpath):
    """
    Return list of available conductivity model names.
    Param: path to cond_models/ directory
    """
    py_files = glob.glob(os.path.join(cond_models_fdpath, "*.py"))

    model_names = []
    for fpath in py_files:
        fname = os.path.basename(fpath)

        # skip non-model files
        if fname in ["__init__.py", "base.py", "registry.py"]:
            continue

        model_name = os.path.splitext(fname)[0]
        model_names.append(model_name)

    return sorted(model_names)

def load_cond_model(model_name):
    """
    Dynamically load a conductivity model.
    Workflow: model name → load cond_models/xxx.py → find the model class → return an instance
    Returns: ConductivityModel
    """
    module_path = f"cond_models.{model_name}"

    if module_path in sys.modules:
        module = importlib.reload(sys.modules[module_path])
    else:
        module = importlib.import_module(module_path)

    from cond_models.base import ConductivityModel

    model_classes = []

    # find subclasses of ConductivityModel DEFINED in this module
    # (obj.__module__ guard ignores models merely imported for reuse,
    #  e.g. the tract-based models importing VolumeNormalizedModel as a fallback)
    for _, obj in inspect.getmembers(module):
        if (inspect.isclass(obj) and issubclass(obj, ConductivityModel)
                and obj is not ConductivityModel
                and obj.__module__ == module.__name__):
            model_classes.append(obj)

    if len(model_classes) == 0:
        raise ValueError(f"No ConductivityModel found in '{model_name}.py'")

    if len(model_classes) > 1:
        raise ValueError(
            f"Multiple ConductivityModel classes found in '{model_name}.py'. "
            f"Please keep only one."
        )

    return model_classes[0]()  # instantiate


def model_short_name(model_name):
    """Resolve a model NAME string ('tractbased_e') to its short tag ('TBe').

    The tag is declared once, as `short_name` on the model class, so this reads it
    back for the callers that only ever hold the name string (log rows, figure
    titles, config). Falls back to `model_name` if the module cannot be loaded or
    declares no tag, so a label never crashes a plot.
    """
    try:
        return getattr(load_cond_model(model_name), "short_name", model_name)
    except Exception:
        return model_name


def get_model_short_names(cond_models_fdpath=None):
    """{model_name: short_name} for every model in cond_models/, name-sorted.

    Useful as the `model_name_map` of the c3 plots. Loading every model is only
    worth it for a whole-cohort figure -- for a single label use model_short_name.
    """
    fdpath = cond_models_fdpath if cond_models_fdpath is not None else gnr.cond_model_fdpath
    return {n: model_short_name(n) for n in get_available_cond_models(fdpath)}

