#!/usr/bin/env python3
"""Per-cell GPU memory footprint from the telemetry the harness already records.

No GPU is touched. For every cell under a campaign this reads:

* ``prestart_gate.csv``                    -> idle baseline before the service starts
                                              (anything another process already held)
* ``loaded_idle_after_ready.csv``          -> resident footprint once the service reports ready
* ``loaded_idle_after_precondition.csv``   -> resident footprint after warmup / graph capture /
                                              allocator growth, i.e. the steady resident state
* ``requests.jsonl`` ``memory_used_mib_max`` -> sampled in-request peak (only present for runs
                                              recorded after the harness began writing it)

Read the numbers with these limits in mind; they are written into the output too.

* ``memory.used`` is device-wide (nvidia-smi), not per-process. On this single-GPU host the cells
  run exclusively, so ``resident - idle_baseline`` is the service's footprint.
* It includes memory the runtime *reserved* up front (KV-cache pool sized by a utilisation knob,
  CUDA graph pools, allocator cache). A larger number can mean a larger configured pool, not a
  larger working set. Compare recipes, not engines.
* The in-request peak is sampled every 0.5 s, so it is a lower bound on the true peak.

Usage::

    envs/mstar/bin/python exps/tools/report_gpu_memory.py \
        --campaign experiments-thermal-isolated-20261016 [--tasks bagel qwen_a2t] [--json out.json]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths


def _gate_memory(path: Path) -> float | None:
    """Median ``memory_used_mib`` of a gate CSV (robust to a transient first sample)."""
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        values = []
        for row in csv.DictReader(handle):
            try:
                values.append(float(row["memory_used_mib"]))
            except (KeyError, TypeError, ValueError):
                continue
    return statistics.median(values) if values else None


def _request_peak(path: Path) -> float | None:
    if not path.is_file():
        return None
    peaks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        value = row.get("memory_used_mib_max")
        if row.get("phase") == "measured" and isinstance(value, (int, float)):
            peaks.append(float(value))
    return max(peaks) if peaks else None


def cell_footprint(cell_dir: Path) -> dict[str, Any] | None:
    idle = _gate_memory(cell_dir / "prestart_gate.csv")
    ready = _gate_memory(cell_dir / "loaded_idle_after_ready.csv")
    settled = _gate_memory(cell_dir / "loaded_idle_after_precondition.csv")
    peak = _request_peak(cell_dir / "requests.jsonl")
    if ready is None and settled is None:
        return None
    resident = settled if settled is not None else ready
    return {
        "idle_baseline_mib": idle,
        "resident_after_ready_mib": ready,
        "resident_after_precondition_mib": settled,
        "in_request_peak_mib": peak,
        "service_resident_mib": (resident - idle) if idle is not None else None,
        "service_peak_mib": (peak - idle) if (peak is not None and idle is not None) else None,
    }


def _median(values: list[float | None]) -> float | None:
    usable = [v for v in values if v is not None]
    return statistics.median(usable) if usable else None


def collect(campaign_root: Path, tasks: set[str] | None) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for manifest in sorted(campaign_root.glob("*/*/*/*/manifest.json")):
        cell_dir = manifest.parent
        case_dir, framework_dir, task_dir = cell_dir.parent, cell_dir.parent.parent, cell_dir.parent.parent.parent
        if task_dir.name.startswith("_"):
            continue
        if tasks and task_dir.name.lower() not in tasks:
            continue
        footprint = cell_footprint(cell_dir)
        if footprint is not None:
            groups[(task_dir.name, framework_dir.name, case_dir.name)].append(footprint)

    rows = []
    for (task, framework, case), starts in sorted(groups.items()):
        rows.append(
            {
                "task": task,
                "framework": framework,
                "case": case,
                "starts": len(starts),
                **{
                    key: _median([s[key] for s in starts])
                    for key in (
                        "idle_baseline_mib",
                        "resident_after_ready_mib",
                        "resident_after_precondition_mib",
                        "service_resident_mib",
                        "in_request_peak_mib",
                        "service_peak_mib",
                    )
                },
                "resident_min_max_mib": [
                    min(s["service_resident_mib"] for s in starts if s["service_resident_mib"] is not None),
                    max(s["service_resident_mib"] for s in starts if s["service_resident_mib"] is not None),
                ]
                if any(s["service_resident_mib"] is not None for s in starts)
                else None,
            }
        )
    return {
        "campaign": campaign_root.name,
        "aggregation": "median across fresh starts of per-cell values; never pooled",
        "caveats": [
            "nvidia-smi device-wide memory.used; includes pre-reserved KV/graph/allocator pools",
            "compare deployment recipes, not engines; configured pool size changes the number",
            "in-request peak is sampled at 0.5 s (lower bound) and is absent for older runs",
        ],
        "rows": rows,
    }


def to_markdown(report: dict[str, Any]) -> str:
    def fmt(value: float | None) -> str:
        return "n/a" if value is None else f"{value:,.0f}"

    lines = [
        f"# GPU memory footprint — {report['campaign']}",
        "",
        "Median across fresh starts. MiB, device-wide `memory.used`.",
        "",
        "| task | framework | case | starts | idle baseline | resident (after warmup) | service resident | in-request peak | service peak |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['task']} | {row['framework']} | {row['case']} | {row['starts']} | "
            f"{fmt(row['idle_baseline_mib'])} | {fmt(row['resident_after_precondition_mib'])} | "
            f"{fmt(row['service_resident_mib'])} | {fmt(row['in_request_peak_mib'])} | "
            f"{fmt(row['service_peak_mib'])} |"
        )
    lines += ["", "Caveats:", *[f"- {c}" for c in report["caveats"]]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=None, help="Campaign id; defaults to the harness default.")
    parser.add_argument("--tasks", nargs="*", default=None, help="Lower-case task names, e.g. bagel qwen_a2t.")
    parser.add_argument("--json", type=Path, default=None, help="Also write the report as JSON here.")
    parser.add_argument("--markdown", type=Path, default=None, help="Also write the markdown table here.")
    args = parser.parse_args()

    root = paths.campaign_root(args.campaign)
    if not root.is_dir():
        print(f"campaign directory not found: {root}", file=sys.stderr)
        return 2
    report = collect(root, {t.lower() for t in args.tasks} if args.tasks else None)
    text = to_markdown(report)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    return 0 if report["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
