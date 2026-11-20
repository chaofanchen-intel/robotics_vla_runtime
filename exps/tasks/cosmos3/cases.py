"""Cosmos3-Nano single-image text-to-image workload."""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

STEPS = 50
SIZE = 256
#: Square sizes. 256 is the original recipe case; 512 and 1,024 exist because the GEMMs are weight-bound at 256x256 and compute-bound
#: at larger sizes, where the per-kernel launch gaps that dominate 256x256 should matter less (cosmos3-size). Outside the standard profiles.
SIZES = {"t2i-256": 256, "t2i-512": 512, "t2i-1024": 1024}

TASK = TaskSpec(
    name="cosmos3",
    title="Cosmos3-Nano T2I",
    frameworks=("mstar", "vllm-omni", "sglang-omni"),
    cases=tuple(
        Case(
            name=name, label=f"Five frozen prompts, {size}x{size}, {STEPS} diffusion steps",
            params={"width": size, "height": size, "steps": STEPS},
            profiles=("nightly", "full") if size == SIZE else (),
        )
        for name, size in SIZES.items()
    ),
    runner_module="exps.tasks.cosmos3.cell",
    comparison_label="local_capability_study",
    timing_boundary="POST start through response-body EOF; PNG decode/audit excluded",
    notes="Local recipe study: the M* paper does not benchmark Cosmos3. SGLang-Omni serves its direct-generation pipeline "
          "(Cosmos3PipelineConfig, native multimodal_gen image route), not the interleaved Reasoner pipeline.",
)
