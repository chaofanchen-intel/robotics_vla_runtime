"""Cross-start summaries and qualified framework comparisons for every task."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import metrics, paths
from .registry import TaskSpec


def analyze(spec: TaskSpec, *, campaign: str | None = None, root: Path | None = None) -> dict[str, Any]:
    base = (root or paths.campaign_root(campaign)) / spec.name.upper()
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    excluded: list[dict[str, str]] = []
    for manifest_path in sorted(base.glob("*/*/*/manifest.json")):
        cell = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_path = manifest_path.with_name("summary.json")
        if cell.get("status") != "completed" or not summary_path.is_file():
            excluded.append({"cell": cell.get("cell", str(manifest_path)), "reason": cell.get("status", "missing_summary")})
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "completed":
            excluded.append({"cell": cell["cell"], "reason": summary.get("status", "missing_status")})
            continue
        start = summary["results"]
        if not start["headline_requests"]:
            excluded.append({"cell": cell["cell"], "reason": "no_valid_requests"})
            continue
        groups[(cell["case"], cell["framework"])].append(start)

    cases: dict[str, dict[str, Any]] = {}
    for case in spec.cases:
        frameworks: dict[str, Any] = {}
        for framework in spec.frameworks:
            starts = groups.get((case.name, framework), [])
            figures = metrics.combine_starts(starts)
            frameworks[framework] = {
                **figures,
                "mean_start_temperature_c": metrics.mean_start_temperature(starts),
            }
        comparisons: dict[str, Any] = {}
        baseline_name = "mstar" if "mstar" in frameworks else spec.frameworks[0]
        baseline = frameworks[baseline_name]
        for framework, candidate in frameworks.items():
            if framework == baseline_name:
                continue
            comparison = metrics.compare(
                baseline, candidate,
                baseline_start_temp_c=baseline["mean_start_temperature_c"],
                candidate_start_temp_c=candidate["mean_start_temperature_c"],
            )
            if comparison["speedup"] is None:
                comparison["labels"] = sorted(set(comparison["labels"] + ["insufficient_valid_starts"]))
            comparison["baseline"] = baseline_name
            comparison["candidate"] = framework
            comparison["task_comparison_label"] = spec.comparison_label
            comparisons[framework] = comparison
        cases[case.name] = {"frameworks": frameworks, "comparisons": comparisons}

    return {
        "task": spec.name,
        "campaign": campaign or paths.campaign_id(),
        "status": "analyzed" if groups else "no_valid_cells",
        "aggregation": (
            "median of per-start nearest-rank p50; never pool requests across starts; "
            "drift and in-request thermal telemetry are labels, not exclusions"
        ),
        "cases": cases,
        "excluded_cells": excluded,
    }
