#!/usr/bin/env python3
"""Verify paired V-JEPA source inputs and report same-campaign BF16 measurements."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths

FRAMEWORKS = ("mstar", "meta-native")
CASES = ("h15", "h30")


def _cell(campaign: str, framework: str, case: str, repeat: int, measured: int) -> tuple[float, bool]:
    directory = paths.cell_dir("vjepa_ac", framework, case, f"r{repeat:02d}", campaign=campaign)
    if not (directory / ".cell_done").is_file():
        raise ValueError(f"{directory}: missing completed-cell marker")
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    if manifest["status"] != "completed" or summary["status"] != "completed":
        raise ValueError(f"{directory}: cell is not completed")
    if framework == "meta-native" and (
        summary.get("dtype") != "bfloat16"
        or summary.get("model_parameter_dtype") != "torch.bfloat16"
        or summary.get("precision_proof") != {
            "encoder.patch_embed": "torch.bfloat16",
            "predictor.predictor_embed": "torch.bfloat16",
        }
    ):
        raise ValueError(f"{directory}: native BF16 weights and internal compute are not verified")
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
    measured_rows = [row for row in rows if row["phase"] == "measured"]
    if len(measured_rows) != measured or any(
        not row.get("success") or row.get("cooldown_timed_out")
        or row.get("step_count_verified") is not True
        or row.get("rollout_steps") != int(case[1:])
        or not row.get("source_input_sha256")
        for row in measured_rows
    ):
        raise ValueError(f"{directory}: measured requests lack successful work, cooling, or source-input proof")
    return statistics.median(row["e2e_ms"] for row in measured_rows), (
        manifest.get("result_headline", {}).get("intra_start_drift") == "suspect_intra_start_drift"
    )


def _sources(campaign: str, framework: str, case: str, repeat: int, measured: int) -> dict[int, str]:
    directory = paths.cell_dir("vjepa_ac", framework, case, f"r{repeat:02d}", campaign=campaign)
    rows = [json.loads(line) for line in (directory / "requests.jsonl").read_text().splitlines() if line.strip()]
    measured_rows = [row for row in rows if row["phase"] == "measured"]
    sources = {row["fixture_index"]: row["source_input_sha256"] for row in measured_rows}
    if len(sources) != measured:
        raise ValueError(f"{directory}: duplicate or missing measured fixture indexes")
    return sources


def report(campaign: str, repeats: int = 3, measured: int = 5) -> str:
    lines = [
        f"# V-JEPA BF16 paired cohort: {campaign}",
        "",
        "| horizon | M* HTTP p50 ms | Meta-native in-process p50 ms | Meta/M* | drift starts M* / Meta |",
        "|---|---:|---:|---:|---:|",
    ]
    for case in CASES:
        per_framework: dict[str, list[float]] = {framework: [] for framework in FRAMEWORKS}
        drifts = {framework: 0 for framework in FRAMEWORKS}
        for repeat in range(1, repeats + 1):
            source_rows = {}
            for framework in FRAMEWORKS:
                median, drift = _cell(campaign, framework, case, repeat, measured)
                per_framework[framework].append(median)
                drifts[framework] += drift
                source_rows[framework] = _sources(campaign, framework, case, repeat, measured)
            if source_rows["mstar"] != source_rows["meta-native"]:
                mismatched = sorted(index for index in source_rows["mstar"]
                                    if source_rows["mstar"][index] != source_rows["meta-native"].get(index))
                raise ValueError(
                    f"{campaign}/{case}/r{repeat:02d}: source video/actions/states differ "
                    f"at fixture indexes {mismatched}; run --audit for all mismatches"
                )
        mstar = statistics.median(per_framework["mstar"])
        native = statistics.median(per_framework["meta-native"])
        lines.append(
            f"| {case.upper()} | {mstar:,.0f} | {native:,.0f} | {native / mstar:.2f}x | "
            f"{drifts['mstar']}/{repeats} / {drifts['meta-native']}/{repeats} |"
        )
    lines.extend((
        "",
        "Each cell has the same source video bytes, float32 action/state trajectory, horizon and frame count. "
        "This does not prove equal decoded tensors, output quality, or equal timing boundaries. "
        "M* is measured over HTTP; Meta-native is timed in process. "
        "Native BF16 weights and internal compute are checked; the M* BF16 path is expected from "
        "engine defaults, not re-proven per cell by this client. No residual is assigned to graph + walk.",
        "",
    ))
    return "\n".join(lines)


def audit_report(campaign: str, repeats: int = 3, measured: int = 5) -> str:
    """Keep invalid pairs visible without turning them into a cohort speedup."""
    lines = [
        f"# V-JEPA BF16 source-input audit: {campaign}",
        "",
        "| horizon | start | M* HTTP p50 ms | Meta-native p50 ms | differing fixture indexes | valid pair | drift M* / Meta |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    valid = 0
    for case in CASES:
        for repeat in range(1, repeats + 1):
            medians, drifts, sources = {}, {}, {}
            for framework in FRAMEWORKS:
                medians[framework], drifts[framework] = _cell(campaign, framework, case, repeat, measured)
                sources[framework] = _sources(campaign, framework, case, repeat, measured)
            mismatch = sorted(index for index in sources["mstar"]
                              if sources["mstar"][index] != sources["meta-native"].get(index))
            if not mismatch:
                valid += 1
            lines.append(
                f"| {case.upper()} | r{repeat:02d} | {medians['mstar']:,.0f} | "
                f"{medians['meta-native']:,.0f} | {', '.join(map(str, mismatch)) or 'none'} | "
                f"{'yes' if not mismatch else 'no'} | "
                f"{'suspect' if drifts['mstar'] else 'flat'} / "
                f"{'suspect' if drifts['meta-native'] else 'flat'} |"
            )
    lines.extend((
        "",
        f"Only {valid}/{len(CASES) * repeats} starts have identical recorded source hashes "
        "at every measured fixture. Invalid starts cannot enter the paired cohort; "
        "do not compute an aggregate speedup from this table. The SHA256 combines encoded "
        "video file bytes and float32 actions/states, so a mismatch does not identify which "
        "component changed. Historical videos were overwritten by subsequent dataset builds; "
        "this audit cannot establish whether their decoded frames matched. "
        "M* uses HTTP E2E and Meta-native uses in-process GPU timing; M* BF16 was not re-proven here. "
        "Whole-cohort input identity, task quality, and a graph+walk-only effect are not established.",
        "",
    ))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default="ATTR_VJEPA_BF16_01")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--measured", type=int, default=5)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--audit", action="store_true", help="diagnose invalid pairs without claiming a cohort speedup")
    args = parser.parse_args()
    if args.repeats < 1 or args.measured < 1:
        parser.error("--repeats and --measured must be positive")
    text = (audit_report if args.audit else report)(args.campaign, args.repeats, args.measured)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
