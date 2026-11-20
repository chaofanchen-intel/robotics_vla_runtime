"""What-if hook: run vLLM-Omni's Pi0.5 attention through ``F.scaled_dot_product_attention`` instead of its hand-written matmul + softmax.

``vllm_omni/diffusion/models/pi05/modeling_pi05.py::_attend`` is a manual eager attention (``softmax(Q K^T * scale + mask) V``, fp32 softmax),
called for every layer of the prefix pass and of each of the 10 action-expert denoise steps. That is independent of ``--enforce-eager``: it is
how the model is written. This hook replaces ``_attend`` with the same math through SDPA (additive mask, same KV-group repeat), so the only
change is the kernel family. The softmax runs in the SDPA kernel's own precision instead of an explicit fp32 upcast: a rounding-level
difference, outputs are not compared. Read from the environment of the stage worker:

* ``VLLM_PI05_SDPA``      ``1`` to apply; anything else leaves vLLM-Omni untouched.
* ``VLLM_PI05_SDPA_LOG``  path; one JSON line (``applied`` true/false, refusal reason) per process that imported the module.

Counterfactual only.
"""
from __future__ import annotations

import os

if os.environ.get("VLLM_PI05_SDPA") == "1":
    import importlib.abc
    import importlib.util
    import json
    import sys

    _TARGET = "vllm_omni.diffusion.models.pi05.modeling_pi05"
    _LOG = os.environ.get("VLLM_PI05_SDPA_LOG")

    def _log(**fields) -> None:
        if _LOG:
            with open(_LOG, "a") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "hook": "vllm_pi05_sdpa", **fields}) + "\n")

    def _apply(module) -> None:
        if not callable(getattr(module, "_attend", None)) or not callable(getattr(module, "_repeat_kv", None)):
            _log(applied=False, refused="modeling_pi05._attend / _repeat_kv not found; the source changed")
            return
        import torch.nn.functional as F

        repeat_kv = module._repeat_kv

        def _attend_sdpa(query_states, key_states, value_states, attention_mask, num_kv_groups, scaling):
            k = repeat_kv(key_states, num_kv_groups)
            v = repeat_kv(value_states, num_kv_groups)
            mask = None if attention_mask is None else attention_mask[:, :, :, : k.shape[-2]].to(query_states.dtype)
            return F.scaled_dot_product_attention(query_states, k, v, attn_mask=mask, scale=scaling)

        module._attend = _attend_sdpa
        _log(applied=True, replaced="_attend")

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
