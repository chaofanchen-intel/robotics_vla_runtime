"""Probe for M*'s BAGEL attention: which kernels run, and what happens without the `flash_attn` pip package.

Why this exists
---------------
M*'s README says to install flash-attn "(recommended for BAGEL)". If M*'s BAGEL lead depended on it, a baseline
without it would be an unfair comparison. Reading the source says it does not: the pip package is imported only by
the ViT encoder (`mstar.model.bagel.components.vit_encoder`, eager path of `BagelViTAttention.attend`); the LLM
attention (prefill, decode and the T2I denoise steps) goes through M*'s own `AttentionSpec(backend=AUTO)`, which
resolves to FlashInfer. A T2I request has no input image, so it never reaches the ViT. This hook turns that reading
into a measurement. It never modifies `external/`; it patches the imported modules in the worker process.

Switches (environment of the M* worker; set with `--env mstar:...`, nothing else is read)
------------------------------------------------------------------------------------------
``MSTAR_ATTN_PROBE``       comma separated, any of

* ``count``             count calls of the ViT's flash-attn varlen kernel and of its SDPA fallback, and log which
                        attention manager class each attention resource resolved to (the LLM backend);
* ``block-flash-attn``  make the ViT behave as if `flash_attn` were not installed (flag and function removed, so
                        `attend` takes the SDPA fallback). The control for "what does the pip package buy".

``MSTAR_ATTN_PROBE_LOG``   path prefix. Each worker process writes ``<prefix>.<pid>.json``, rewritten on every
                           counted call, so a SIGTERM at shutdown loses nothing. The analysis assigns a file to a cell
                           by the time window of its manifest.

Without ``MSTAR_ATTN_PROBE`` set the module does nothing, so an ordinary cell is untouched.
"""
from __future__ import annotations

import os

_MODES = {m.strip() for m in os.environ.get("MSTAR_ATTN_PROBE", "").split(",") if m.strip()}

if _MODES:
    import importlib.abc
    import importlib.util
    import json
    import sys
    import time
    from datetime import datetime, timezone

    _PREFIX = os.environ.get("MSTAR_ATTN_PROBE_LOG")
    _STATE = {
        "pid": os.getpid(),
        "modes": sorted(_MODES),
        "process_start_utc": datetime.now(timezone.utc).isoformat(),
        "vit_module_loaded": False,
        "vit_flash_attn_available": None,
        "vit_flash_attn_calls": 0,
        "vit_sdpa_fallback_calls": 0,
        "first_vit_call_utc": None,
        "last_vit_call_utc": None,
        "attention_managers": [],
    }

    def _flush():
        if not _PREFIX:
            return
        tmp = f"{_PREFIX}.{os.getpid()}.json.tmp"
        with open(tmp, "w") as handle:
            json.dump(_STATE, handle)
        os.replace(tmp, f"{_PREFIX}.{os.getpid()}.json")

    def _stamp(kind):
        now = datetime.now(timezone.utc).isoformat()
        _STATE[kind] += 1
        _STATE["first_vit_call_utc"] = _STATE["first_vit_call_utc"] or now
        _STATE["last_vit_call_utc"] = now
        _flush()

    def _patch_vit(module):
        _STATE["vit_module_loaded"] = True
        if "block-flash-attn" in _MODES:
            module._FLASH_ATTN_AVAILABLE = False
            module.flash_attn_varlen_func = None
        _STATE["vit_flash_attn_available"] = bool(module._FLASH_ATTN_AVAILABLE)
        if "count" in _MODES:
            original_flash = module.flash_attn_varlen_func
            if original_flash is not None:
                def counted_flash(*args, **kwargs):
                    _stamp("vit_flash_attn_calls")
                    return original_flash(*args, **kwargs)
                module.flash_attn_varlen_func = counted_flash
            original_sdpa = module._sdpa_varlen

            def counted_sdpa(*args, **kwargs):
                _stamp("vit_sdpa_fallback_calls")
                return original_sdpa(*args, **kwargs)
            module._sdpa_varlen = counted_sdpa
        _flush()

    def _patch_attention_base(module):
        if "count" not in _MODES:
            return
        build = module.AttentionManager.build.__func__

        def logged_build(cls, spec, *args, **kwargs):
            manager = build(cls, spec, *args, **kwargs)
            _STATE["attention_managers"].append({
                "resource_key": getattr(spec, "resource_key", None),
                "requested_backend": str(getattr(getattr(spec, "config", None), "backend", None)),
                "manager": type(manager).__name__,
            })
            _flush()
            return manager
        module.AttentionManager.build = classmethod(logged_build)

    _TARGETS = {
        "mstar.model.bagel.components.vit_encoder": _patch_vit,
        "mstar.engine.resources.attn.base": _patch_attention_base,
    }

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            patch = _TARGETS.get(name)
            if patch is None:
                return None
            sys.meta_path.remove(self)
            try:
                spec = importlib.util.find_spec(name)
            finally:
                sys.meta_path.insert(0, self)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                patch(module)
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
    _flush()
