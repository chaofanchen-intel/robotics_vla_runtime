"""CPU-profile hook for one vLLM-Omni BAGEL denoise step (never modifies external/).

M*'s side is `tools/mstar_cprofile_hook` (one `Engine.exec` of `LLM.image_gen`). This is its counterpart for the vLLM-Omni
single-stage BAGEL path: it wraps `Bagel.forward` in `vllm_omni.diffusion.models.bagel.bagel_transformer` (the model forward of
one denoise step: batched-sequence build, `NaiveCache.merge`, the language model, the CFG combine) and runs `cProfile` around
selected calls, so `tools/analyze_cprofile.py` can put the Python frames of the two engines side by side. Read from the
environment of the diffusion worker (spawned from the server, so it inherits it):

* ``VLLM_BAGEL_CPROFILE_OUT``    directory; one ``Bagel.forward.<k>.pstats`` per profiled call, plus ``hook_proof.jsonl``.
* ``VLLM_BAGEL_CPROFILE_SKIP``   calls to let through first (warm-up); default 100 (two 50-step requests).
* ``VLLM_BAGEL_CPROFILE_COUNT``  calls to profile after that; default 10.

The span is narrower than M*'s `Engine.exec` (no timestep tensor, no scheduler update, no engine planning), which makes the
vLLM side look cheaper than it is: a conservative direction for a claim that vLLM spends more host time per step. cProfile
follows the calling thread only and inflates Python-heavy code, so compare profiles taken the same way, as ratios.
"""
from __future__ import annotations

import os

_OUT = os.environ.get("VLLM_BAGEL_CPROFILE_OUT")

if _OUT:
    import cProfile
    import importlib.abc
    import importlib.util
    import json
    import sys
    import time

    _MODULE = "vllm_omni.diffusion.models.bagel.bagel_transformer"
    _SKIP = int(os.environ.get("VLLM_BAGEL_CPROFILE_SKIP", "100"))
    _COUNT = int(os.environ.get("VLLM_BAGEL_CPROFILE_COUNT", "10"))
    _SEEN = [0]

    def _wrap(original):
        def forward(self, *args, **kwargs):
            _SEEN[0] += 1
            k = _SEEN[0] - _SKIP
            if k < 1 or k > _COUNT:
                return original(self, *args, **kwargs)
            profile = cProfile.Profile()
            start = time.perf_counter()
            profile.enable()
            try:
                return original(self, *args, **kwargs)
            finally:
                profile.disable()
                os.makedirs(_OUT, exist_ok=True)
                profile.dump_stats(os.path.join(_OUT, f"Bagel.forward.{k}.pstats"))
                with open(os.path.join(_OUT, "hook_proof.jsonl"), "a") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "target": "Bagel.forward", "call": _SEEN[0], "k": k,
                                             "wall_ms": (time.perf_counter() - start) * 1e3}) + "\n")

        return forward

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _MODULE:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                module.Bagel.forward = _wrap(module.Bagel.forward)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
