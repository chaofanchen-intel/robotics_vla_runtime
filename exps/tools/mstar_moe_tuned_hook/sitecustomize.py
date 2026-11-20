"""What-if / trace hook for M*'s fused-MoE tile choice (never modifies external/).

Two independent switches, both read from the environment of the M* worker:

* ``MSTAR_MOE_TUNED_JSON``  path of a vLLM ``fused_moe/configs/E=...json``. M* calls
  ``runner.get_default_config`` (fixed tiles) for every MoE layer; this replaces that name inside the
  process, after ``mstar.utils.fused_moe.runner`` is imported, with vLLM's nearest-key lookup. A
  counterfactual to test where M*'s Qwen3-Omni Thinker time goes, not a configuration any of the three
  frameworks ships. ``MSTAR_MOE_TUNED_LOG`` (optional) records one JSON line per distinct token count it
  re-tiled -- proof the hook ran inside the worker.
* ``MSTAR_MOE_TRACE``       path; append one ``{"M": tokens}`` line per MoE call (first 40,000 calls),
  with or without the tuned table. Shows how many passes a request makes through the MoE layers and
  with how many tokens each (a forward of M tokens through 48 layers = 48 lines).
"""
from __future__ import annotations

import os

_PATH = os.environ.get("MSTAR_MOE_TUNED_JSON")
_TRACE = os.environ.get("MSTAR_MOE_TRACE")

if _PATH or _TRACE:
    import importlib.abc
    import importlib.util
    import json
    import sys

    _TARGET = "mstar.utils.fused_moe.runner"
    _TABLE = {int(k): v for k, v in json.load(open(_PATH)).items()} if _PATH else None
    _LOG = os.environ.get("MSTAR_MOE_TUNED_LOG")
    _SEEN: set[int] = set()
    _CALLS = [0]
    _TRACE_CAP = 40_000

    def _wrap(original):
        def config(M: int, E: int, N: int, K: int, top_k: int) -> dict:
            if _TRACE and _CALLS[0] < _TRACE_CAP:
                _CALLS[0] += 1
                with open(_TRACE, "a") as handle:
                    handle.write(json.dumps({"M": M, "N": N}) + "\n")
            if _TABLE is None:
                return original(M, E, N, K, top_k)
            chosen = dict(_TABLE[min(_TABLE, key=lambda x: abs(x - M))])
            if _LOG and M not in _SEEN:
                _SEEN.add(M)
                with open(_LOG, "a") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "M": M, "E": E, "N": N, "K": K, "config": chosen}) + "\n")
            return chosen

        return config

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                module.get_default_config = _wrap(module.get_default_config)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
