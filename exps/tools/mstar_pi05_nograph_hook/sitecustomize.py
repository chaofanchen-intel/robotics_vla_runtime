"""What-if hook: serve M*'s Pi0.5 without any accelerator (CUDA) graph, eager forwards only.

M* captures three graph families for Pi0.5: the SigLIP encoder (bs 1/2/4), the LLM prefill walk and the 10-step action
expert (``action_gen``), all through ``get_accelerator_graph_configs`` on the submodules in ``mstar.model.pi05.submodules``.
This hook makes every such method return an empty list, so ``can_use_accelerator_graphs`` is False for every walk and nothing
is captured. ``torch.compile`` is untouched (switch it off separately with ``TORCHDYNAMO_DISABLE=1``): the two arms together
give M* with neither graphs nor compile, the closest M* has to vLLM-Omni's eager recipe. Read from the environment of the worker:

* ``MSTAR_PI05_NO_GRAPH``      ``1`` to apply; anything else leaves M* untouched.
* ``MSTAR_PI05_NO_GRAPH_LOG``  path; one JSON line (``applied`` true/false, the classes patched or the refusal reason) per process
  that imported the module: proof the hook ran in the worker. A cell without that line did not test anything.

Counterfactual only. Outputs are not compared (graphs do not change the math).
"""
from __future__ import annotations

import os

if os.environ.get("MSTAR_PI05_NO_GRAPH") == "1":
    import importlib.abc
    import importlib.util
    import json
    import sys

    _TARGET = "mstar.model.pi05.submodules"
    _LOG = os.environ.get("MSTAR_PI05_NO_GRAPH_LOG")
    _METHODS = ("get_accelerator_graph_configs", "get_piecewise_accelerator_graph_configs")

    def _log(**fields) -> None:
        if _LOG:
            with open(_LOG, "a") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "hook": "mstar_pi05_no_graph", **fields}) + "\n")

    def _apply(module) -> None:
        patched = []
        for name, cls in vars(module).items():
            if not isinstance(cls, type) or cls.__module__ != module.__name__:
                continue
            for method in _METHODS:
                if method in vars(cls):
                    setattr(cls, method, lambda self, *args, **kwargs: [])
                    patched.append(f"{name}.{method}")
        if not any(item.endswith("get_accelerator_graph_configs") for item in patched):
            _log(applied=False, refused="no get_accelerator_graph_configs found in mstar.model.pi05.submodules; the source changed")
            return
        _log(applied=True, patched=patched)

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
