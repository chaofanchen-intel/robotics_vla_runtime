#!/usr/bin/env python3
"""Cell plans for Qwen3-Omni text-to-text."""
from __future__ import annotations

import exps.frameworks as frameworks
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.measure import MeasurementConfig
from exps.core.registry import Case, CellRef
from exps.tasks import qwen_common
from exps.tasks.qwen_txt.cases import TASK

TASK_NAME = TASK.name


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    framework = frameworks.get(ref.framework)
    max_output_tokens = case.params["max_output_tokens"]
    return CellPlan(
        ref=ref,
        service=qwen_common.service(ref.framework, TASK_NAME, f"txt-{case.name}-{ref.repeat}"),
        client_argv=[
            str(framework.python),
            "-m", "exps.tasks.qwen_txt.client",
            "--framework", ref.framework,
            "--case", case.name,
            "--base-url", qwen_common.base_url(ref.framework, TASK_NAME),
            "--output-dir", str(config.output_dir),
            "--max-output-tokens", str(max_output_tokens),
            "--prompt-kind", case.params.get("prompt_kind", "1k"),
            *protocol_argv(config),
        ],
        context={
            "case_label": case.label,
            "max_output_tokens": max_output_tokens,
            "comparison_label": TASK.comparison_label,
            "timing_boundary": TASK.timing_boundary,
            **qwen_common.shared_context(ref.framework),
        },
        provenance_repos=[framework.external_repo, "mstar"],
        client_env=qwen_common.client_env(),
    )
