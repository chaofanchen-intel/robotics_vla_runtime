#!/usr/bin/env python3
"""Concurrency / throughput curves from batched cells (B concurrent requests per gated unit).

The paper's headline Qwen3-Omni T2S result is B=16 *throughput* and RTF; the C=1 cohort cannot
say anything about it. The batched cases (`verbatim-en-b4/b8/b16`, BAGEL `i2t-256-b2..b16`) send a
wave of B requests at once; the cell's ``e2e_ms`` is the wall time to clear the wave.

Per (task, framework, B), over starts (never pooled), this reports

* wave wall p50 (ms)                 -- median over starts of the per-start p50
* requests/s  = B / wave_wall_s
* audio-seconds/s (T2S only)         -- sum of audio produced per wave / wave wall
* RTF p50 (T2S)                      -- mean per-request RTF within a wave
* throughput scaling vs B=1          -- requests/s(B) / requests/s(1); ideal batching grows with B,
                                        a value near 1.0 means the engine effectively serialised
                                        the wave (check the deploy config's max_num_seqs first).

Every number is a wave-level observation under the thermal protocol of its cell; the card heats
across a wave, and the offered load is exactly B (no steady-state arrival process).

    envs/mstar/bin/python exps/tools/analyze_concurrency.py \
        --campaign CONCURRENCY_01 [--json out.json] [--markdown out.md]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths

# task directory -> (case regex, baseline case that stands for B=1)
TASKS = {
    "QWEN_T2S": (re.compile(r"^verbatim-en(?:-b(\d+))?$"), "verbatim-en"),
    # i2t-512 is the suite's operating point (bagel-concurrency); i2t-256 is the older launch-bound one. One resolution per campaign.
    "BAGEL": (re.compile(r"^(?:i2t-(?:256|512)|(?:i2i|t2i)-512-(?:srv-)?steps5)(?:-b(\d+))?$"), "i2t-512"),
}


def batch_of(match: re.Match[str]) -> int:
    return int(match.group(1)) if match.group(1) else 1


def per_start_values(summary: dict[str, Any], batch: int) -> dict[str, float | None]:
    metrics = (summary.get("results") or {}).get("metrics") or {}
    wall = (metrics.get("e2e_ms") or {}).get("p50")
    audio = (metrics.get("audio_duration_s") or {}).get("p50")
    return {
        "wave_wall_ms": wall,
        "requests_per_s": batch / (wall / 1000.0) if wall else None,
        "audio_s_per_s": (audio / (wall / 1000.0)) if (wall and audio) else None,
        "rtf": (metrics.get("rtf") or {}).get("p50"),
    }


def collect(campaign_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task_dir_name, (pattern, _baseline) in TASKS.items():
        task_root = campaign_root / task_dir_name
        if not task_root.is_dir():
            continue
        groups: dict[tuple[str, int], list[dict[str, float | None]]] = defaultdict(list)
        for manifest in sorted(task_root.glob("*/*/*/manifest.json")):
            cell = json.loads(manifest.read_text(encoding="utf-8"))
            match = pattern.match(cell.get("case", ""))
            summary_path = manifest.with_name("summary.json")
            if not match or cell.get("status") != "completed" or not summary_path.is_file():
                continue
            batch = batch_of(match)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            groups[(cell["framework"], batch)].append(per_start_values(summary, batch))

        frameworks: dict[str, Any] = defaultdict(dict)
        for (framework, batch), starts in sorted(groups.items()):
            def med(key: str) -> float | None:
                vals = [s[key] for s in starts if s[key] is not None]
                return statistics.median(vals) if vals else None
            frameworks[framework][batch] = {
                "starts": len(starts),
                "wave_wall_ms": med("wave_wall_ms"),
                "requests_per_s": med("requests_per_s"),
                "audio_s_per_s": med("audio_s_per_s"),
                "rtf_p50": med("rtf"),
            }
        for framework, curve in frameworks.items():
            base = (curve.get(1) or {}).get("requests_per_s")
            for batch, row in curve.items():
                row["throughput_scaling_vs_b1"] = (
                    row["requests_per_s"] / base if base and row["requests_per_s"] else None
                )
        out[task_dir_name] = {fw: dict(sorted(curve.items())) for fw, curve in frameworks.items()}
    return out


def ratios_vs_reference(task_curves: dict[str, Any], reference: str = "mstar") -> dict[int, dict[str, float]]:
    """requests/s of the reference divided by each other framework's, per B (paper's headline form)."""
    ref = task_curves.get(reference, {})
    result: dict[int, dict[str, float]] = {}
    for batch, row in ref.items():
        for fw, curve in task_curves.items():
            other = curve.get(batch, {}).get("requests_per_s") if fw != reference else None
            if other and row["requests_per_s"]:
                result.setdefault(batch, {})[fw] = row["requests_per_s"] / other
    return result


def to_markdown(report: dict[str, Any]) -> str:
    def f(value: float | None, spec: str = ",.1f") -> str:
        return "n/a" if value is None else format(value, spec)

    lines = ["# Concurrency curves", ""]
    for task, curves in report["tasks"].items():
        lines += [f"## {task}", "",
                  "| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for fw, curve in curves.items():
            for batch, row in curve.items():
                lines.append(
                    f"| {fw} | {batch} | {row['starts']} | {f(row['wave_wall_ms'], ',.0f')} | "
                    f"{f(row['requests_per_s'], '.2f')} | {f(row['audio_s_per_s'], '.2f')} | "
                    f"{f(row['rtf_p50'], '.3f')} | {f(row['throughput_scaling_vs_b1'], '.2f')} |"
                )
        ratios = report["ratios_vs_mstar"].get(task, {})
        if ratios:
            lines += ["", "M* requests/s divided by the other framework's (>1: M* higher throughput):", ""]
            for batch, row in sorted(ratios.items()):
                lines.append(f"- B={batch}: " + ", ".join(f"vs {fw} {v:.2f}×" for fw, v in row.items()))
        lines.append("")
    lines += ["Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; "
              "`starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; "
              "check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all)."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    root = paths.campaign_root(args.campaign)
    tasks = collect(root)
    if not tasks:
        print(f"no batched cells under {root}", file=sys.stderr)
        return 2
    report = {"campaign": root.name, "tasks": tasks,
              "ratios_vs_mstar": {t: ratios_vs_reference(c) for t, c in tasks.items()}}
    text = to_markdown(report)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
