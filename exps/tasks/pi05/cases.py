#!/usr/bin/env python3
"""Pi0.5 vision-language-action inference on a frozen DROID fixture.

Three runtimes, one client
--------------------------
M*, vLLM-Omni and SGLang **core** (`multimodal_gen` with `Pi05Pipeline`, which
is not SGLang-Omni Stage and must never be relabelled as it). All three are
driven by the same client in `transport.py` with the same timing boundary, so
the within-task comparison is not partly a comparison of clients.

Equal work
----------
Each request sends the same three camera images, the same normalised 32-dim
robot state and the same task string from a 46-row fixture rebuilt from a
pinned DROID snapshot. The flow-matching step count is sent explicitly to all
three, because it is the dominant cost and a default mismatch would be
invisible in the response.

The output is an action trajectory of shape [50, 32]. Its shape is asserted
per response: a runtime that returned a shorter horizon would otherwise look
faster.
"""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

#: Flow-matching denoising steps. Sent explicitly to every runtime.
NUM_INFERENCE_STEPS = 10
#: Every step count a cell may run: 10 is the headline; the others are the pi05-steps experiment. M* reads its count from
#: its service YAML (one file per count, exps/configs/mstar/pi05_bfloat16*.yaml); vLLM-Omni and SGLang core read the request.
FLOW_STEP_COUNTS = (1, 5, 10, 20)
ACTION_HORIZON = 50
ACTION_DIM = 32


def _steps_case(steps: int) -> Case:
    # profiles=() keeps these out of the nightly / full sweeps: they run only when an experiment names them.
    return Case(
        name=f"droid-46row-s{steps}",
        label=f"46-row DROID fixture, {steps} flow-matching step{'s' if steps != 1 else ''}",
        params={"num_inference_steps": steps},
        profiles=(),
    )


TASK = TaskSpec(
    name="pi05",
    title="Pi0.5 VLA on DROID",
    frameworks=("mstar", "vllm-omni", "sglang-core"),
    cases=(
        Case(
            name="droid-46row",
            label=f"46-row DROID fixture, {NUM_INFERENCE_STEPS} flow-matching steps",
            params={"num_inference_steps": NUM_INFERENCE_STEPS},
        ),
        _steps_case(1),
        _steps_case(5),
        _steps_case(20),
        # pi05-kv: one fixture row, sent again and again, so a prefix cache can hit (the rows of droid-46row differ).
        Case(
            name="droid-row0-x20",
            label="fixture row 0 repeated (20 identical requests per start), 10 flow-matching steps",
            params={"num_inference_steps": NUM_INFERENCE_STEPS, "fixture_row": 0},
            profiles=(),
        ),
    ),
    runner_module="exps.tasks.pi05.cell",
    comparison_label="work_matched",
    timing_boundary=(
        "M* over HTTP: just before the POST to the end of the response body. vLLM-Omni and "
        "SGLang core over msgpack WebSocket: just before send to the completion of recv. Image "
        "encoding and msgpack packing happen before the timer on every path, decoding after it."
    ),
    notes=(
        "SGLang here is SGLang core (multimodal_gen, Pi05Pipeline), not SGLang-Omni Stage. "
        "Action trajectory shape is asserted per response. vLLM-Omni's OpenPI WebSocket idle "
        "timeout is extended to cover the per-request thermal gates."
    ),
)
