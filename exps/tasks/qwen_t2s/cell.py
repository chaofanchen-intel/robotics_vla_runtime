#!/usr/bin/env python3
"""Cell plans for Qwen3-Omni text-to-speech."""
from __future__ import annotations

import exps.frameworks as frameworks
from exps.core import paths
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.measure import MeasurementConfig
from exps.core.registry import Case, CellRef
from exps.tasks import qwen_common
from exps.tasks.qwen_t2s.cases import MAX_NEW_TOKENS, TASK

TASK_NAME = TASK.name


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    if ref.framework in case.not_applicable:
        raise KeyError(
            f"{ref}: {case.not_applicable[ref.framework]}"
        )
    arm = case.params["arm"]
    language = case.params["language"]
    framework = frameworks.get(ref.framework)
    # Concurrency cases (batch_size > 1) use the batch-capable vLLM-Omni deploy config.
    batch_config = int(case.params.get("batch_size", 1)) > 1

    return CellPlan(
        ref=ref,
        service=qwen_common.service(
            ref.framework, TASK_NAME, f"t2s-{case.name}-{ref.repeat}", batch_config=batch_config
        ),
        client_argv=[
            str(framework.python),
            "-m", "exps.tasks.qwen_t2s.client",
            "--framework", ref.framework,
            "--case", case.name,
            "--base-url", qwen_common.base_url(ref.framework, TASK_NAME),
            "--output-dir", str(config.output_dir),
            "--arm", arm,
            "--language", language,
            "--batch-size", str(case.params.get("batch_size", 1)),
            *protocol_argv(config),
        ],
        context={
            "batch_size": case.params.get("batch_size", 1),
            "case_label": case.label,
            "arm": arm,
            "language": language,
            "max_new_tokens": MAX_NEW_TOKENS,
            "comparison_label": "work_matched" if arm == "verbatim" else "descriptive",
            "timing_boundary": TASK.timing_boundary,
            **qwen_common.shared_context(ref.framework, batch_config=batch_config, task=TASK_NAME),
        },
        provenance_repos=[framework.external_repo, "mstar"],
        client_env=qwen_common.client_env(),
    )


def canary_argv(ref: CellRef, case: Case, config: MeasurementConfig) -> list[str]:
    """Contract canary for one arm/runtime/language, run before any measurement."""
    framework = frameworks.get(ref.framework)
    return [
        str(framework.python),
        "-m", "exps.tasks.qwen_t2s.client",
        "--framework", ref.framework,
        "--case", case.name,
        "--base-url", qwen_common.base_url(ref.framework, TASK_NAME),
        "--output-dir", str(config.output_dir / "canary"),
        "--arm", case.params["arm"],
        "--language", case.params["language"],
        "--canary",
        *protocol_argv(config),
    ]
