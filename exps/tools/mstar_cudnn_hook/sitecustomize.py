"""What-if hook: pin ``torch.backends.cudnn.benchmark`` in M*'s Qwen3-Omni workers (never modifies external/).

M*'s ``qwen3_omni_model.py`` sets ``torch.backends.cudnn.benchmark = True``. With benchmark mode cuDNN times every candidate
algorithm the first time it sees a new input shape. The A2T audio encoder's convolutions see a new shape for each new
audio length, and the engine's own per-request profile shows the audio encoder taking 75-130 ms for the first six requests of
a server and 15-23 ms afterwards (4.1-4.3 s for the first two, which includes compilation); cProfile puts 65 ms of it in the
three ``conv2d`` calls whether torch.compile is on or off. Whether benchmark mode is the cause is what this hook tests.
Read from the environment of the M* worker:

* ``MSTAR_CUDNN_BENCHMARK``  ``0`` or ``1``: the value the flag is pinned to; model code that assigns the other value is
  overridden. Unset = hook inert.
* ``MSTAR_CUDNN_LOG``        optional path; one JSON line when the pin is installed and one per overridden assignment,
  proof the hook ran inside the worker.

Run both arms (``1`` is the control) with this hook loaded, so that the only difference between them is the pinned value.
"""
from __future__ import annotations

import os

_PIN = os.environ.get("MSTAR_CUDNN_BENCHMARK")

if _PIN in ("0", "1"):
    import importlib.abc
    import importlib.util
    import json
    import sys

    _TARGET = "mstar.model.qwen3_omni.qwen3_omni_model"
    _LOG = os.environ.get("MSTAR_CUDNN_LOG")
    _VALUE = _PIN == "1"

    def _log(event: str, **extra) -> None:
        if _LOG:
            with open(_LOG, "a") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "event": event, "pinned": _VALUE, **extra}) + "\n")

    class _Pinned:
        """Descriptor replacing ``CudnnModule.benchmark``: reads the real flag, writes always store the pinned value."""

        def __get__(self, obj, objtype=None):
            import torch
            return torch._C._get_cudnn_benchmark()

        def __set__(self, obj, value):
            import torch
            if bool(value) != _VALUE:
                _log("assignment overridden", requested=bool(value))
            torch._C._set_cudnn_benchmark(_VALUE)

    def _install() -> None:
        import torch
        import torch.backends.cudnn as cudnn
        type(cudnn).benchmark = _Pinned()
        torch._C._set_cudnn_benchmark(_VALUE)
        _log("pin installed", flag_now=bool(torch._C._get_cudnn_benchmark()))

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                _install()

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
