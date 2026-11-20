"""What-if hook for M*'s Talker -> Code2Wav streaming chunk size (never modifies external/).

M* releases codec frames to the vocoder in chunks of ``Code2WavConfig.codec_chunk_frames`` (25 frames = 2 s of audio,
with ``codec_left_context_frames`` = 25 of overlap). vLLM-Omni's async yaml starts with ``initial_codec_chunk_frames: 4``,
so the first-audio latency (TTFA) of the two engines differs by configuration as well as by engine. This hook makes the
chunk size a knob, read from the environment of the M* worker:

* ``MSTAR_QWEN_CODEC_CHUNK_FRAMES``          frames per chunk (int >= 1).
* ``MSTAR_QWEN_CODEC_LEFT_CONTEXT_FRAMES``   optional; default min(25, chunk). ``LeftContextChunkPolicy`` advances the
  first pop by ``chunk - left_context``, so the left context may not exceed the chunk.
* ``MSTAR_QWEN_CODEC_LOG``                   optional path; one JSON line per ``from_dict`` call, proof that the hook ran
  inside the worker (the lesson of the MoE hook: a log line saying "applied" is not proof).

Counterfactual only: a shorter left context also changes the vocoder's warm-up, so audio quality is not asserted.
"""
from __future__ import annotations

import os

_CHUNK = os.environ.get("MSTAR_QWEN_CODEC_CHUNK_FRAMES")

if _CHUNK:
    import importlib.abc
    import importlib.util
    import json
    import sys

    _TARGET = "mstar.model.qwen3_omni.config"
    _LEFT = os.environ.get("MSTAR_QWEN_CODEC_LEFT_CONTEXT_FRAMES")
    _LOG = os.environ.get("MSTAR_QWEN_CODEC_LOG")

    def _resolve() -> tuple[int, int]:
        chunk = int(_CHUNK)
        if chunk < 1:
            raise ValueError(f"MSTAR_QWEN_CODEC_CHUNK_FRAMES must be >= 1, got {chunk}")
        left = int(_LEFT) if _LEFT else min(25, chunk)
        if not 0 <= left <= chunk:
            raise ValueError(f"left context {left} must be within [0, {chunk}]")
        return chunk, left

    def _wrap(original):
        def from_dict(cls, d):
            config = original.__func__(cls, d) if hasattr(original, "__func__") else original(d)
            chunk, left = _resolve()
            config.codec_chunk_frames = chunk
            config.codec_left_context_frames = left
            if _LOG:
                with open(_LOG, "a") as handle:
                    handle.write(json.dumps({"pid": os.getpid(), "codec_chunk_frames": chunk,
                                             "codec_left_context_frames": left}) + "\n")
            return config

        return classmethod(from_dict)

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target=None):
            if name != _TARGET:
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(name)
            execute = spec.loader.exec_module

            def exec_module(module):
                execute(module)
                module.Code2WavConfig.from_dict = _wrap(module.Code2WavConfig.__dict__["from_dict"])

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
