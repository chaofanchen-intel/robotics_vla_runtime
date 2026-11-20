"""What-if hook: run M*'s Cosmos3 classifier-free-guidance branches as two sequential forwards, like vLLM-Omni does.

M*'s ``Cosmos3DiTSubmodule.batched_cfg`` (class attribute, default True) packs the conditional and the unconditional branch
into ONE forward per denoise step ("the math is the same; set False to fall back to the sequential path"). vLLM-Omni's
``pipeline_cosmos3`` runs the two branches as two forwards. This hook flips the attribute before any submodule is built, so
the only thing it changes is how many forwards a step makes. Read from the environment of the M* worker:

* ``MSTAR_COSMOS3_BATCHED_CFG``  ``0`` to fall back to the sequential path; anything else leaves M* untouched (a control arm
  may load the hook too).
* ``MSTAR_COSMOS3_CFG_LOG``      path; one JSON line (``applied`` true/false, with the reason when false) per process that
  imported the module: proof the hook ran inside the worker. A cell without that line did not test anything.

Counterfactual only; outputs are not compared here (M*'s docs say the two paths do the same math).
"""
from __future__ import annotations

import os

_FLAG = os.environ.get("MSTAR_COSMOS3_BATCHED_CFG")

if _FLAG == "0":
    import importlib.abc
    import importlib.util
    import json
    import sys

    _TARGET = "mstar.model.cosmos3.submodules"
    _LOG = os.environ.get("MSTAR_COSMOS3_CFG_LOG")

    def _log(**fields) -> None:
        if _LOG:
            with open(_LOG, "a") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "hook": "cosmos3_batched_cfg", **fields}) + "\n")

    def _apply(module) -> None:
        cls = getattr(module, "Cosmos3DiTSubmodule", None)
        if cls is None or not hasattr(cls, "batched_cfg"):
            _log(applied=False, refused="Cosmos3DiTSubmodule.batched_cfg not found; the source changed")
            return
        before = cls.batched_cfg
        cls.batched_cfg = False
        _log(applied=True, before=bool(before), after=bool(cls.batched_cfg))

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                _apply(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
