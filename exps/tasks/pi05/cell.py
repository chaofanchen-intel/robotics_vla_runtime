#!/usr/bin/env python3
"""Fresh-service cell plans for the three Pi0.5 DROID runtimes."""
from __future__ import annotations

import json
import shlex
from pathlib import Path

import exps.frameworks as frameworks
from exps.core import overrides, paths
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.lifecycle import ServiceSpec
from exps.core.measure import MeasurementConfig
from exps.core.registry import Case, CellRef
from exps.tasks.pi05.cases import FLOW_STEP_COUNTS, NUM_INFERENCE_STEPS, TASK

SNAPSHOT_REVISION = "22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0"
SNAPSHOT = (
    paths.MODELS_CACHE / "models--lerobot--pi05_base" / "snapshots" / SNAPSHOT_REVISION
)
TOKENIZER = (
    paths.MODELS_CACHE / "models--google--paligemma-3b-pt-224" / "snapshots"
    / "35e4f46485b4d07967e7e9935bc3786aad50687c"
)
CONFIGS = {
    "mstar": paths.CONFIGS / "mstar" / "pi05_bfloat16.yaml",
    "vllm-omni": paths.CONFIGS / "vllm_omni" / "pi05_bfloat16.yaml",
    "sglang-core": paths.CONFIGS / "sglang_core" / "pi05_pipeline.json",
}


def mstar_config(steps: int) -> Path:
    """M* fixes its flow-step count in the service config (`model_kwargs.num_flow_steps`), so each count has its own YAML."""
    if steps == NUM_INFERENCE_STEPS:
        return CONFIGS["mstar"]
    return paths.CONFIGS / "mstar" / f"pi05_bfloat16_steps{steps}.yaml"


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    framework = frameworks.get(ref.framework)
    port = frameworks.port(ref.framework, TASK.name)
    configs = dict(CONFIGS)
    steps = case.params["num_inference_steps"]
    if steps not in FLOW_STEP_COUNTS:
        raise ValueError(f"Pi0.5 cells run {FLOW_STEP_COUNTS} flow steps, got {steps}")
    configs["mstar"] = mstar_config(steps)
    fixture_row = case.params.get("fixture_row")

    if ref.framework == "mstar":
        command = frameworks.mstar_serve(
            model="pi05", config=configs[ref.framework], port_number=port,
            tag=f"pi05-{case.name}-{ref.repeat}",
        )
        base_url = f"http://127.0.0.1:{port}"
    elif ref.framework == "vllm-omni":
        # The recipe runs vLLM-Omni eager. `--opt pi05.vllm_graph=1` turns that off (config + CLI flag) to ask whether the
        # M* / vLLM gap is the eager baseline; whether the diffusion pipeline then compiles anything has to be read from the server log.
        graph = overrides.current().option("pi05", "vllm_graph") == "1"
        if graph:
            configs[ref.framework] = paths.CONFIGS / "vllm_omni" / "pi05_bfloat16_graph.yaml"
        command = frameworks.vllm_omni_serve(
            model_path=SNAPSHOT, deploy_config=configs[ref.framework],
            port_number=port, extra=(
                # The request cooldown can idle this persistent socket for up to 900 seconds.
                ("" if graph else "--enforce-eager ") + "--robot-openpi-idle-timeout 1800 --custom-pipeline-args "
                + shlex.quote(json.dumps({"tokenizer": str(TOKENIZER)}))
            ),
        )
        base_url = f"ws://127.0.0.1:{port}"
    elif ref.framework == "sglang-core":
        command = frameworks.sglang_core_serve(
            model_path=SNAPSHOT, pipeline="Pi05Pipeline",
            pipeline_config=CONFIGS[ref.framework], port_number=port,
        )
        base_url = f"ws://127.0.0.1:{port}"
    else:
        raise KeyError(f"pi05 has no framework {ref.framework!r}")

    return CellPlan(
        ref=ref,
        service=ServiceSpec(
            command=command, health_url=f"http://127.0.0.1:{port}/health",
            ready_timeout_s=1800.0,
        ),
        client_argv=[
            str(framework.python), "-m", "exps.tasks.pi05.client",
            "--framework", ref.framework, "--case", case.name,
            "--base-url", base_url, "--output-dir", str(config.output_dir),
            "--num-inference-steps", str(steps), *protocol_argv(config),
            *(["--fixture-row", str(fixture_row)] if fixture_row is not None else []),
        ],
        context={
            "case_label": case.label,
            "comparison_label": TASK.comparison_label,
            "timing_boundary": TASK.timing_boundary,
            "model": {"id": "lerobot/pi05_base", "revision": SNAPSHOT_REVISION},
            "config": paths.relative_to_repo(configs[ref.framework]),
            "num_inference_steps": steps,
            "fixture_row": fixture_row,
            "step_source": (
                "M*: service config model_kwargs.num_flow_steps (pi05_bfloat16.yaml for 10 steps, "
                "pi05_bfloat16_steps<N>.yaml otherwise); vLLM-Omni and SGLang core: request payload "
                "(the config's default of 10 is overridden per request)"
            ),
            "action_contract": "[50, 32] float32, checked on every response",
            "runtime_label": framework.title,
        },
        provenance_repos=[framework.external_repo],
        client_env={"HF_HOME": str(paths.MODELS_CACHE), "HF_HUB_OFFLINE": "1"},
    )
