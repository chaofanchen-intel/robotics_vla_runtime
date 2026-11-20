"""What-if hook: keep M*'s BAGEL sampler out of torch.compile, so a new request id no longer forces a recompile.

Round 7 found (TORCH_LOGS=recompiles,graph_breaks, cProfile) that with torch.compile on every BAGEL request recompiles five
dynamo frames -- ``_sample_per_request``, ``_sample_tokens``, ``sample``, the resume frame inside ``_sample_per_request`` and the
resume frame inside ``forward_batched`` -- because the traced code builds ``{rid: {"new_token": [token]}}`` from
``request_ids`` and so guards on ``request_ids[0] == '<this request's uuid>'``. That is the ~280 ms the compiled
``LLM.prefill_text`` costs per request. M* raises ``recompile_limit`` to 84, so the cache grows by one entry per request.

This hook wraps ``LLMSubmodule._sample_per_request`` with ``torch.compiler.disable`` (the sampler then runs eagerly; it is
already CUDA-graph capturable), leaving everything else compiled. Read from the environment of the M* worker:

* ``MSTAR_BAGEL_SAMPLER_EAGER``  ``1`` to apply the sampler-only variant, ``2`` for the full variant (below); anything else
  leaves M* untouched (the control arm also loads the hook).
* ``MSTAR_BAGEL_SAMPLER_LOG``    optional path; one JSON line per process that imported the module, proof the hook ran
  inside the worker. A cell without that line did not test anything.

Round 8 showed variant 1 removes four of the five recompiles and not the fifth: the resume frame of ``forward_batched`` that
holds the ``return self._sample_per_request(request_ids, ...)`` tail still guards on ``request_ids[0]`` (1 recompile per
request, ~170-240 ms at 512^2 left in B1). Variant 2 therefore also takes the *tail* out of dynamo: ``forward_batched`` becomes a
frame dynamo skips (``skip_code``), which runs the model part through a lazily built ``torch.compile`` of the same dispatch
(``_forward_decode_batched`` / ``_forward_prefill_text`` / ``_forward_prefill_vit``, no sampling) and then calls the eager
sampler. A CPU toy of exactly this shape went from one recompile per new request id to none (see tests/test_round8_tools.py).
The variant refuses to apply, and logs why, when the source of ``forward_batched`` is not the one it was written against.

Counterfactual only: it changes which code dynamo traces, not what the sampler computes. Sampled tokens are not compared
here. The proof of effect is the TORCH_LOGS recompile count inside the measured requests (5 -> 0) and the E2E difference.
"""
from __future__ import annotations

import os

_ARMED = os.environ.get("MSTAR_BAGEL_SAMPLER_EAGER")

if _ARMED is not None:
    import importlib.abc
    import importlib.util
    import json
    import sys

    _TARGET = "mstar.model.bagel.submodules"
    _LOG = os.environ.get("MSTAR_BAGEL_SAMPLER_LOG")

    _TAIL = "return self._sample_per_request(request_ids, out.pop(_LOGITS)[0])"
    _WALKS = ("_forward_decode_batched", "_forward_prefill_text", "_forward_prefill_vit")

    def _patch_forward_batched(cls, torch):
        """Variant 2. Returns None when applied, or the reason it was not."""
        import inspect
        import torch._dynamo.eval_frame as eval_frame
        original = cls.forward_batched
        source = inspect.getsource(original)
        missing = [token for token in (_TAIL, *_WALKS, "sample_prefill_token=True") if token not in source]
        if missing:
            return "forward_batched source differs from the one this hook was written against: no " + ", ".join(missing)
        logits_key = inspect.getmodule(original)._LOGITS

        def model_part(self, graph_walk, input_ids=None, input_embeds=None, sample_token=False, requires_cfg=False, **kwargs):
            if graph_walk == "decode":
                return self._forward_decode_batched(input_ids=input_ids, requires_cfg=requires_cfg)
            if graph_walk == "prefill_text":
                return self._forward_prefill_text(input_ids=input_ids, requires_cfg=requires_cfg,
                                                  sample_prefill_token=True, **kwargs)
            if graph_walk == "prefill_vit":
                return self._forward_prefill_vit(input_embeds=input_embeds, requires_cfg=requires_cfg,
                                                 sample_prefill_token=sample_token, **kwargs)
            raise ValueError(f"Batched forward not supported for graph walk: {graph_walk!r}")

        def forward_batched(self, graph_walk, engine_inputs, input_ids=None, input_embeds=None, sample_token=False,
                            requires_cfg=False, **kwargs):
            compiled = self.__dict__.get("_hook_model_part")
            if compiled is None:
                compiled = torch.compile(lambda *a, **k: model_part(self, *a, **k), fullgraph=False, dynamic=None)
                self.__dict__["_hook_model_part"] = compiled
            out = compiled(graph_walk, input_ids=input_ids, input_embeds=input_embeds, sample_token=sample_token,
                           requires_cfg=requires_cfg, **kwargs)
            if logits_key in out:
                return self._sample_per_request(engine_inputs.request_ids, out.pop(logits_key)[0])
            return out

        eval_frame.skip_code(forward_batched.__code__)
        cls.forward_batched = forward_batched
        return None

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                applied = _ARMED in ("1", "2")
                patched = []
                refused = None
                if applied:
                    import torch
                    cls = module.LLMSubmodule
                    if _ARMED == "2":
                        refused = _patch_forward_batched(cls, torch)
                        applied = refused is None
                        if applied:
                            patched.append("LLMSubmodule.forward_batched")
                    if applied:
                        cls._sample_per_request = torch.compiler.disable(cls._sample_per_request)
                        patched.append("LLMSubmodule._sample_per_request")
                if _LOG:
                    with open(_LOG, "a") as handle:
                        handle.write(json.dumps({"pid": os.getpid(), "armed": _ARMED, "applied": applied,
                                                 "patched": ", ".join(patched) or None, "refused": refused}) + "\n")

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
