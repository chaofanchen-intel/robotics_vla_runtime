"""CPU-profile hook for one M* engine step type (never modifies external/).

`Engine.exec` is the span the NVTX range ``engine.<node>.<walk>.bs<N>`` covers: declare -> admit -> plan -> forward ->
commit -> collect outputs. nsys says how long it takes on the host, not where. This hook wraps ``Engine.exec`` and runs
``cProfile`` around selected calls of one ``node.walk``, so the cumulative-time table names the Python frames
(dynamo guard / dispatch, planner, model forward, output collection). Read from the environment of the M* worker:

* ``MSTAR_CPROFILE_TARGET``  ``<node>.<walk>`` to profile, e.g. ``LLM.prefill_text``.
* ``MSTAR_CPROFILE_OUT``     directory; one ``<target>.<k>.pstats`` per profiled call, plus ``hook_proof.jsonl``.
* ``MSTAR_CPROFILE_SKIP``    calls of the target to let through first (warm-up); default 3.
* ``MSTAR_CPROFILE_COUNT``   calls to profile after that; default 3.

cProfile follows the calling thread only (the engine's GPU thread) and inflates Python-heavy code, so compare two
profiles taken the same way (compile on / off) rather than reading absolute milliseconds.
"""
from __future__ import annotations

import os

_TARGET = os.environ.get("MSTAR_CPROFILE_TARGET")
_OUT = os.environ.get("MSTAR_CPROFILE_OUT")

if _TARGET and _OUT:
    import cProfile
    import importlib.abc
    import importlib.util
    import json
    import sys
    import time

    _MODULE = "mstar.engine.engine"
    _SKIP = int(os.environ.get("MSTAR_CPROFILE_SKIP", "3"))
    _COUNT = int(os.environ.get("MSTAR_CPROFILE_COUNT", "3"))
    _SEEN = [0]

    def _wrap(original):
        def exec_(self, batch):
            name = f"{batch.node_name}.{batch.step_context.graph_walk}"
            if name != _TARGET:
                return original(self, batch)
            _SEEN[0] += 1
            k = _SEEN[0] - _SKIP
            if k < 1 or k > _COUNT:
                return original(self, batch)
            profile = cProfile.Profile()
            start = time.perf_counter()
            profile.enable()
            try:
                return original(self, batch)
            finally:
                profile.disable()
                os.makedirs(_OUT, exist_ok=True)
                profile.dump_stats(os.path.join(_OUT, f"{_TARGET}.{k}.pstats"))
                with open(os.path.join(_OUT, "hook_proof.jsonl"), "a") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "target": _TARGET, "call": _SEEN[0], "k": k,
                                             "wall_ms": (time.perf_counter() - start) * 1e3}) + "\n")

        return exec_

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _MODULE:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                module.Engine.exec = _wrap(module.Engine.exec)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
