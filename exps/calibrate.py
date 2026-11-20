#!/usr/bin/env python3
"""Measure candidate thermal gates before using a calibrated protocol profile.

Each candidate gets its own fresh-service one-start sweep. This is expensive
on this chassis: inspect --dry-run and select tasks before --execute.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps.core import manifest, metrics, paths, registry


def _measurements(task: str, framework: str, case: str, campaign: str) -> dict:
    root = paths.cell_dir(task, framework, case, "r01", campaign=campaign)
    summary_path = root / "summary.json"
    manifest_path = root / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        return {"status": "missing_cell"}
    record = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if record["status"] != "completed" or summary["status"] != "completed":
        return {"status": record["status"], "summary_status": summary["status"]}
    rows = [
        json.loads(line)
        for line in (root / "requests.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usable = metrics.headline_rows(row for row in rows if row["phase"] == "measured")
    temperatures = [row["temperature_c_before"] for row in usable]
    e2e = [row["e2e_ms"] for row in usable]
    if not usable or any(value is None for value in temperatures + e2e):
        return {"status": "missing_thermal_or_latency_observations"}
    delta = [
        row["temperature_c_max"] - row["temperature_c_before"]
        for row in usable
        if row.get("temperature_c_max") is not None
    ]
    waited = [row["cooldown_waited_s"] for row in usable if row.get("cooldown_waited_s") is not None]
    return {
        "status": "measured",
        "requests": len(usable),
        "start_temperature_c": statistics.fmean(temperatures),
        "delta_temperature_per_request_c": statistics.fmean(delta) if delta else None,
        "cooldown_waited_s": waited,
        "p50_e2e_ms": metrics.nearest_rank(e2e, 0.5),
        "drift": summary["results"]["intra_start_drift"],
    }


def _slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 3 or max(x for x, _ in points) - min(x for x, _ in points) < 3:
        return None
    mx = statistics.fmean(x for x, _ in points)
    my = statistics.fmean(y for _, y in points)
    denominator = sum((x - mx) ** 2 for x, _ in points)
    return sum((x - mx) * (y - my) for x, y in points) / denominator


def recommendation(samples: dict[float, dict], measured: int) -> dict:
    """Choose only an observed, reachable target whose latency stays near the coolest."""
    valid = sorted(
        (target, sample) for target, sample in samples.items()
        if sample["status"] == "measured" and sample["requests"] >= measured
        and sample["delta_temperature_per_request_c"] is not None
        and sample["cooldown_waited_s"] and not sample["drift"]["drifting"]
    )
    points = [(sample["start_temperature_c"], sample["p50_e2e_ms"]) for _, sample in valid]
    slope = _slope(points)
    sensitivity = {
        "slope_ms_per_c": slope,
        "observed_start_temperature_range_c": (
            [min(x for x, _ in points), max(x for x, _ in points)] if points else None
        ),
        "status": "measured" if slope is not None else "insufficient_temperature_range",
    }
    if slope is None or not valid:
        return {"status": "insufficient_evidence", "sensitivity": sensitivity}
    coolest_p50 = valid[0][1]["p50_e2e_ms"]
    acceptable = [
        (target, sample) for target, sample in valid if target <= 70
        and sample["p50_e2e_ms"] <= coolest_p50 * 1.05
    ]
    if not acceptable:
        return {"status": "no_stable_target", "sensitivity": sensitivity}
    target = acceptable[-1][0]
    waits = [
        wait for value in samples.values() if value["status"] == "measured"
        for wait in value["cooldown_waited_s"]
    ]
    return {
        "status": "recommended",
        "cooldown_target_c": target,
        "measured": measured,
        "cooldown_timeout_s": max(900, math.ceil(max(waits) * 1.5)),
        "sensitivity": sensitivity,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", default=list(registry.TASK_ORDER))
    parser.add_argument("--profile", choices=("nightly", "full"), default="nightly")
    parser.add_argument("--targets", nargs="+", type=float, default=[65, 70, 75])
    parser.add_argument("--measured", type=int, default=3)
    parser.add_argument("--output", type=Path, default=paths.PROFILES / "calibrated.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Start real GPU cells; can take days.")
    args = parser.parse_args()
    names = tuple(name for value in args.tasks for name in value.split())
    tasks = registry.load_tasks(names)
    targets = sorted(set(args.targets))
    if not targets or any(not 0 < value < 80 for value in targets) or args.measured < 3:
        parser.error("targets must be in (0, 80) C and --measured must be at least 3")
    if args.execute and args.dry_run:
        parser.error("--execute and --dry-run cannot be combined")
    if not args.execute:
        print(json.dumps({
            "status": "dry_run", "tasks": names, "targets_c": targets,
            "cells": sum(
                len(list(registry.expand(task, profile=args.profile, repeats=1))) for task in tasks
            ) * len(targets),
            "profile": args.profile,
            "note": "Pass --execute to launch fresh-service sweeps; 55 C may be unreachable loaded.",
        }, indent=2))
        return 0

    campaigns: dict[str, dict[float, str]] = {}
    evidence: dict = {}
    profile: dict = {}

    def write_progress() -> None:
        manifest.write_json(args.output, {
            "status": (
                "in_progress" if len(evidence) < len(tasks)
                else "completed" if profile else "no_recommendations"
            ),
            "requested_tasks": [spec.name for spec in tasks],
            "completed_tasks": list(evidence),
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "campaigns": {
                task: {str(target): campaign for target, campaign in by_target.items()}
                for task, by_target in campaigns.items()
            },
            "tasks": profile,
            "evidence": evidence,
        })

    write_progress()
    run_id = datetime.now(timezone.utc).strftime("thermal-calibration-%Y%m%dT%H%M%SZ")
    for spec in tasks:
        campaigns[spec.name] = {}
        for index, target in enumerate(targets):
            campaign = f"{run_id}-{spec.name}-t{index:02d}"
            campaigns[spec.name][target] = campaign
            command = [
                sys.executable, "-m", "exps.run", "sweep", "--tasks", spec.name,
                "--profile", args.profile, "--repeats", "1", "--measured", str(args.measured),
                "--cooldown-target-c", str(target), "--loaded-idle-target-c", str(target),
                "--resume",
            ]
            result = subprocess.run(
                command, cwd=paths.REPO_ROOT,
                env={**os.environ, "EXPS_CAMPAIGN_ID": campaign},
            )
            if result.returncode not in (0, 1):
                raise RuntimeError(
                    f"calibration sweep failed before cells were recorded: {result.returncode}"
                )

        task_evidence: dict = {}
        task_profile: dict = {}
        for case in spec.cases_for_profile(args.profile):
            framework_evidence = {}
            recommendations = []
            for framework in spec.frameworks:
                if framework in case.not_applicable:
                    continue
                samples = {
                    target: _measurements(spec.name, framework, case.name, campaigns[spec.name][target])
                    for target in targets
                }
                selected = recommendation(samples, args.measured)
                framework_evidence[framework] = {
                    "targets": {str(target): value for target, value in samples.items()},
                    "recommendation": selected,
                }
                recommendations.append(selected)
            task_evidence[case.name] = framework_evidence
            if recommendations and all(item["status"] == "recommended" for item in recommendations):
                # One case-wide gate must work for *every* runtime being compared.
                chosen = min(item["cooldown_target_c"] for item in recommendations)
                if all(
                    framework_evidence[fw]["targets"][str(chosen)]["status"] == "measured"
                    for fw in framework_evidence
                ):
                    task_profile[case.name] = {
                        "cooldown_target_c": chosen,
                        "measured": args.measured,
                        "cooldown_timeout_s": max(item["cooldown_timeout_s"] for item in recommendations),
                    }
        evidence[spec.name] = task_evidence
        if task_profile:
            profile[spec.name] = {"cases": task_profile}
        write_progress()
    print(f"Calibrated {sum(len(task['cases']) for task in profile.values())} cases: {args.output}")
    return 0 if profile else 1


if __name__ == "__main__":
    raise SystemExit(main())
