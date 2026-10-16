# tasks/anicond — DTI -> anisotropic conductivity tensor.
#
# The vendored study code (`core`, `cond_models`) lives in vendor/anicond and
# is imported as top-level packages, so the repo root and that folder go on
# sys.path here, at package level: any `tasks.anicond.<module>` import then
# works from any working directory and any importer (service.py, cli.py, and
# mesh_sim, which reads the conductivity table through runctx).
import os as _os
import sys as _sys

_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
for _p in (_root, _os.path.join(_root, "vendor", "anicond")):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
