"""Inference-framework adapters. `get("mstar-rust")` returns the backend; `ALL` is the ordered registry."""
from backends.base import Backend
from backends.checkpoints import CHECKPOINTS, Checkpoint
from backends.mstar import MStarPython, MStarRust
from backends.sglang_core import SglangCore, SglangCoreBucketed
from backends.vllm_omni import VllmOmni

ALL: dict[str, Backend] = {b.name: b for b in (MStarRust(), VllmOmni(), SglangCore(), MStarPython(), SglangCoreBucketed())}
LEVEL1 = ("mstar-rust", "vllm-omni", "sglang-core")        # the default Level 1 set
LEVEL1_ALL = LEVEL1 + ("sglang-core-bucketed",)              # plus the SGLang prompt-bucket variant


def get(name: str) -> Backend:
    return ALL[name]


__all__ = ["ALL", "Backend", "CHECKPOINTS", "Checkpoint", "LEVEL1", "LEVEL1_ALL", "get"]
