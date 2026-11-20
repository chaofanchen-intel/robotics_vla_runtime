#!/usr/bin/env python3
"""Cell plans for Qwen3-Omni audio-to-text."""
from __future__ import annotations

import os
from pathlib import Path

import exps.frameworks as frameworks
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.measure import MeasurementConfig
from exps.core import paths
from exps.core.registry import Case, CellRef
from exps.tasks import qwen_common
from exps.tasks.qwen_a2t.cases import MAX_OUTPUT_TOKENS, TASK

TASK_NAME = TASK.name
VLLM_MOE_WHEEL_DIR = (
    paths.REPO_ROOT / "envs" / "vllm-omni" / "lib" / "python3.12" / "site-packages" / "vllm"
    / "model_executor" / "layers" / "fused_moe" / "configs"
)


def moe_tuned_json() -> Path:
    """vLLM's fused-MoE tile table for the Qwen3-Omni Thinker shapes (E=128, N=768) *on this GPU*.

    M* has no tuned-config lookup of its own (`get_default_config` is fixed), so its "tuned" arm borrows vLLM's table
    through `tools/mstar_moe_tuned_hook`. The table is device-specific: look in `configs/vllm_omni/tuned/` (where a tuner run
    on this machine writes it), then in the vLLM wheel (which ships tables for some GPUs), trying the full device name and
    then the name with trailing words dropped (`NVIDIA_H200_NVL` -> `NVIDIA_H200`). No table for this GPU is an error: using
    another GPU's tiles would not be a tuned config.
    """
    from exps import hardware

    raw = hardware.gpu_name(hardware.current().primary_gpu) or "unknown"
    parts = raw.replace(" ", "_").split("_")
    names = ["_".join(parts[:n]) for n in range(len(parts), 1, -1)]
    searched = []
    for folder in (paths.CONFIGS / "vllm_omni" / "tuned", VLLM_MOE_WHEEL_DIR):
        for name in names:
            candidate = folder / f"E=128,N=768,device_name={name}.json"
            searched.append(str(candidate))
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(
        f"no vLLM fused-MoE tile table for {raw!r} (E=128, N=768). Searched: {searched}. Generate one with vLLM's "
        "benchmarks/kernels/benchmark_moe.py --tune (needs a vLLM source checkout; see exps/GUIDE.md) and put the JSON in "
        "exps/configs/vllm_omni/tuned/")


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    framework = frameworks.get(ref.framework)
    probe_kind = case.params.get("probe_kind")
    stage_timing = probe_kind == "stage-timing"
    service = qwen_common.service(
        ref.framework,
        TASK_NAME,
        f"a2t-{case.name}-{ref.repeat}",
        stage_timing=stage_timing,
    )
    stage_trace_path = config.output_dir / "stage_timings.jsonl" if stage_timing else None
    if stage_trace_path is not None:
        hook_dir = paths.EXPS_ROOT / "tools" / "qwen_a2t_stage_trace_hook"
        python_path_entries = [str(hook_dir), str(paths.REPO_ROOT)]
        inherited_pythonpath = os.environ.get("PYTHONPATH")
        if inherited_pythonpath:
            python_path_entries.append(inherited_pythonpath)
        service.env = {
            **service.env,
            "PYTHONPATH": os.pathsep.join(python_path_entries),
            "QWEN_A2T_STAGE_TRACE_PATH": str(stage_trace_path),
            "QWEN_A2T_STAGE_TRACE_FRAMEWORK": ref.framework,
        }
        if ref.framework == "sglang-omni":
            service.env["SGLANG_OMNI_TRACE_ENCODER_CACHE"] = "1"

    if case.params.get("moe_tuned") or case.params.get("moe_trace"):
        hook_dir = paths.EXPS_ROOT / "tools" / "mstar_moe_tuned_hook"
        hook_env = {
            "PYTHONPATH": os.pathsep.join(
                [str(hook_dir), str(paths.REPO_ROOT)]
                + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
            ),
        }
        if case.params.get("moe_tuned"):
            hook_env["MSTAR_MOE_TUNED_JSON"] = str(moe_tuned_json())
            # proof the hook ran inside the M* worker: one JSON line per distinct token count it re-tiled
            hook_env["MSTAR_MOE_TUNED_LOG"] = str(config.output_dir / "moe_tuned_hook.jsonl")
        if case.params.get("moe_trace"):
            hook_env["MSTAR_MOE_TRACE"] = str(config.output_dir / "moe_trace.jsonl")
        service.env = {**service.env, **hook_env}

    client_argv = [
        str(framework.python),
        "-m", "exps.tasks.qwen_a2t.client",
        "--framework", ref.framework,
        "--case", case.name,
        "--base-url", qwen_common.base_url(ref.framework, TASK_NAME),
        "--output-dir", str(config.output_dir),
        "--language", case.params["language"],
        "--fleurs-config", case.params["fleurs_config"],
        "--max-output-tokens", str(case.params.get("max_output_tokens", MAX_OUTPUT_TOKENS)),
        *protocol_argv(config),
    ]
    precondition_audio_relpaths = case.params.get("precondition_audio_relpaths")
    measured_audio_relpaths = case.params.get("measured_audio_relpaths")
    if measured_audio_relpaths is not None:
        if precondition_audio_relpaths:
            client_argv.extend([
                "--precondition-audio-relpaths",
                *precondition_audio_relpaths,
            ])
        if case.params.get("precondition_match_measured_shape"):
            client_argv.append("--precondition-match-measured-shape")
        client_argv.extend([
            "--measured-audio-relpaths",
            *measured_audio_relpaths,
        ])

    return CellPlan(
        ref=ref,
        service=service,
        client_argv=client_argv,
        context={
            "case_label": case.label,
            "language": case.params["language"],
            "fleurs_config": case.params["fleurs_config"],
            "comparison_label": TASK.comparison_label,
            "timing_boundary": TASK.timing_boundary,
            "probe_kind": probe_kind,
            "moe_tuned_what_if": bool(case.params.get("moe_tuned")),
            "moe_trace": bool(case.params.get("moe_trace")),
            "precondition_audio_relpaths": precondition_audio_relpaths,
            "measured_audio_relpaths": measured_audio_relpaths,
            "precondition_match_measured_shape": bool(
                case.params.get("precondition_match_measured_shape")
            ),
            "stage_trace_path": (
                paths.relative_to_repo(stage_trace_path)
                if stage_trace_path is not None
                else None
            ),
            "stage_timing_profile": (
                "eager_no_compile_no_cuda_graph" if stage_timing else None
            ),
            "stage_timing_hook": (
                paths.relative_to_repo(
                    paths.EXPS_ROOT
                    / "tools"
                    / "qwen_a2t_stage_trace.py"
                )
                if stage_timing
                else None
            ),
            **qwen_common.shared_context(ref.framework, stage_timing=stage_timing),
        },
        provenance_repos=[framework.external_repo, "mstar"],
        client_env=qwen_common.client_env(),
    )
