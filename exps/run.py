#!/usr/bin/env python3
"""Unified entry point: run one task, one framework, one case -- or a sweep.

    python -m exps.run list
    python -m exps.run vjepa_ac --framework mstar --case h30 --repeats 3
    python -m exps.run bagel --framework mstar --case t2i-512 --dry-run
    python -m exps.run sweep --profile nightly --resume
    python -m exps.run analyze vjepa_ac

Every protocol flag is optional. Unset values come from
`profiles/calibrated.json` if the thermal calibration has run, otherwise from
`profiles/defaults.json`; whichever supplied them is recorded per cell as
`calibration_source`, so results measured under tested parameters are
distinguishable from results measured under untested ones.

`--dry-run` expands and prints the full plan -- serve commands, client argv,
protocol parameters, time estimate -- without touching the GPU.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps.core import cell as cell_mod
from exps import hardware
from exps.core import analyze as analyzer, cli, manifest, overrides, paths, registry
from exps.doctor import diagnose


def _plan_for(ref: registry.CellRef, spec: registry.TaskSpec, config):
    module = importlib.import_module(spec.runner_module)
    plan = module.plan(ref, spec.case(ref.case), config)
    hardware.check_task_fits(ref.task, plan.service.command if plan.service else None)
    return plan


def _config_for(args: argparse.Namespace, ref: registry.CellRef) -> cli.MeasurementConfig:
    if args.profile == "smoke":
        args = copy.copy(args)
        defaults = registry.PROFILE_PARAMETERS["smoke"]
        if args.measured is None:
            args.measured = defaults["measured"]
        if args.preconditions is None:
            args.preconditions = defaults["preconditions"]
    return cli.measurement_config(
        args,
        output_dir=paths.cell_dir(ref.task, ref.framework, ref.case, ref.repeat),
        task=ref.task,
        case=ref.case,
    )


def _select_campaign(args: argparse.Namespace) -> None:
    """`--campaign` names the run directory; smoke runs get their own so `--resume` never mixes them into a real one."""
    if getattr(args, "campaign", None):
        paths.set_campaign(args.campaign)
    elif args.profile == "smoke":
        paths.set_campaign(f"{paths.DEFAULT_CAMPAIGN_ID}-smoke")
    overrides.set_current(overrides.from_args(args))
    hardware.set_current(getattr(args, "hardware", None), getattr(args, "gpus", None))


def cmd_list(args: argparse.Namespace) -> int:
    """Print every task, framework and case the harness knows about."""
    payload = []
    for spec in registry.load_tasks():
        payload.append(
            {
                "task": spec.name,
                "title": spec.title,
                "frameworks": list(spec.frameworks),
                "comparison_label": spec.comparison_label,
                "cases": [
                    {
                        "name": case.name,
                        "label": case.label,
                        "profiles": list(case.profiles),
                        "not_applicable": case.not_applicable,
                    }
                    for case in spec.cases
                ],
                "notes": spec.notes,
            }
        )
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    for entry in payload:
        print(f"\n{entry['task']}  --  {entry['title']}")
        print(f"  frameworks : {', '.join(entry['frameworks'])}")
        print(f"  comparison : {entry['comparison_label']}")
        for case in entry["cases"]:
            skipped = (
                "  [N/A: " + ", ".join(sorted(case["not_applicable"])) + "]"
                if case["not_applicable"]
                else ""
            )
            print(f"    {case['name']:<18} {case['label']}  ({'/'.join(case['profiles'])}){skipped}")
    if not payload:
        print("no task packages are implemented yet")
    return 0


def _run_cells(refs: list[registry.CellRef], args: argparse.Namespace) -> int:
    """Execute a list of cells, continuing past failures and summarising at the end."""
    specs = {ref.task: registry.load_task(ref.task) for ref in {r.task: r for r in refs}.values()}

    if args.dry_run:
        plans = []
        total_minutes = 0.0
        for ref in refs:
            config = _config_for(args, ref)
            plan = _plan_for(ref, specs[ref.task], config)
            described = cell_mod.describe(plan, config)
            total_minutes += described["estimated_minutes"]
            plans.append(described)
        print(json.dumps({"cells": len(plans), "estimated_hours": round(total_minutes / 60, 1),
                          "plans": plans}, ensure_ascii=False, indent=2))
        return 0

    failed_checks = [check for check in diagnose() if not check["ok"]]
    if failed_checks:
        details = "\n".join(f"  {check['check']}: {check['detail']}" for check in failed_checks)
        print(f"preflight failed; no GPU cell started:\n{details}", file=sys.stderr)
        return 2

    results: list[dict] = []
    started = time.monotonic()
    budget_s = args.max_hours * 3600 if args.max_hours else None
    stopped_on_budget = False
    with cell_mod.gpu_lock(timeout_s=args.lock_wait_s):
        for index, ref in enumerate(refs, start=1):
            if budget_s is not None and (time.monotonic() - started) >= budget_s:
                # Stop between cells, never inside one. A cell killed mid-flight
                # leaves a service holding the GPU and a half-written run
                # directory; stopping here leaves both clean for --resume.
                stopped_on_budget = True
                print(
                    f"\n[budget] {args.max_hours} h reached after {index - 1} cells. "
                    "Stopping between cells; rerun with --resume to continue.",
                    flush=True,
                )
                break
            output_dir = paths.cell_dir(ref.task, ref.framework, ref.case, ref.repeat)
            if args.resume and not args.force and manifest.cell_is_done(output_dir):
                print(f"[{index}/{len(refs)}] SKIP {ref} (already completed)", flush=True)
                results.append({"cell": str(ref), "status": "skipped"})
                continue
            print(f"[{index}/{len(refs)}] START {ref}", flush=True)
            try:
                # shared machine: take the free GPUs this deployment needs, right before the cell starts
                chosen = hardware.select_gpus(ref.task, ref.framework, wait_s=args.gpu_wait_s)
                if chosen.auto_select_gpus:
                    print(f"[{index}/{len(refs)}] GPUs {chosen.visible_devices} (free; {ref.task}/{ref.framework} needs "
                          f"{chosen.gpus_needed(ref.task, ref.framework)})", flush=True)
                config = _config_for(args, ref)
                plan = _plan_for(ref, specs[ref.task], config)
                record = cell_mod.run_cell(plan, config)
            except Exception as exc:  # noqa: BLE001 - a bad cell must not end the sweep
                print(f"[{index}/{len(refs)}] ERROR {ref}: {type(exc).__name__}: {exc}", flush=True)
                results.append({"cell": str(ref), "status": "error", "error": str(exc)})
                continue
            headline = record.get("result_headline", {})
            print(
                f"[{index}/{len(refs)}] {record['status'].upper()} {ref}"
                f"  p50={headline.get('e2e_ms_p50')}  drift={headline.get('intra_start_drift')}",
                flush=True,
            )
            entry = {"cell": str(ref), "status": record["status"], **headline}
            reason = _failure_reason(record.get("failure"))
            if reason:
                print(f"    reason: {reason}", flush=True)
                entry["error"] = reason
            results.append(entry)

    elapsed_h = (time.monotonic() - started) / 3600
    failures = [r for r in results if r["status"] not in ("completed", "skipped")]
    index_path = paths.campaign_logs() / f"sweep_{time.strftime('%Y%m%dT%H%M%S')}.json"
    manifest.write_json(
        index_path,
        {
            "cells": len(refs),
            "completed": sum(1 for r in results if r["status"] == "completed"),
            "skipped": sum(1 for r in results if r["status"] == "skipped"),
            "failed": len(failures),
            "elapsed_hours": round(elapsed_h, 2),
            "remaining": len(refs) - len(results),
            "stopped_on_time_budget": stopped_on_budget,
            "results": results,
        },
    )
    print(f"\n{len(results) - len(failures)}/{len(results)} cells ok in {elapsed_h:.1f} h")
    if len(results) < len(refs):
        print(f"{len(refs) - len(results)} cells not reached; rerun with --resume to continue")
    print(f"sweep index: {paths.relative_to_repo(index_path)}")
    if failures:
        print("failed cells:")
        for failure in failures:
            print(f"  {failure['cell']}: {failure.get('error') or failure['status']}")
    return 1 if failures else 0



def _failure_reason(failure: str | None) -> str | None:
    """One line for a failed cell: the service's last error line, else the failure's first line.

    The full text (with the last 4 kB of server.log) stays in the cell's manifest.json; the sweep
    log and index only need enough to see why without opening it.
    """
    if not failure:
        return None
    lines = [line.strip() for line in failure.splitlines() if line.strip()]
    first = lines[0]
    if "Last 4000 bytes of server.log" in failure:
        tail = [line for line in lines[1:] if not line.startswith(("File ", "^", "~"))]
        errors = [line for line in tail if "Error" in line or "error:" in line]
        if errors:
            return f"{first} | {errors[-1]}"[:600]
    return first[:600]

def cmd_task(args: argparse.Namespace) -> int:
    _select_campaign(args)
    spec = registry.load_task(args.task)
    refs = list(
        registry.expand(
            spec,
            profile=args.profile,
            frameworks=tuple(args.framework) if args.framework else None,
            cases=tuple(args.case) if args.case else None,
            repeats=args.repeats,
        )
    )
    if not refs:
        print(f"no cells selected for task {args.task!r}", file=sys.stderr)
        return 2
    return _run_cells(refs, args)


def cmd_sweep(args: argparse.Namespace) -> int:
    _select_campaign(args)
    task_names = tuple(name for value in args.tasks for name in value.split()) if args.tasks else registry.TASK_ORDER
    refs: list[registry.CellRef] = []
    for spec in registry.load_tasks(task_names):
        refs.extend(
            registry.expand(
                spec,
                profile=args.profile,
                frameworks=tuple(args.framework) if args.framework else None,
                repeats=args.repeats,
            )
        )
    if not refs:
        print("no cells selected", file=sys.stderr)
        return 2
    return _run_cells(refs, args)


def cmd_analyze(args: argparse.Namespace) -> int:
    spec = registry.load_task(args.task)
    report = analyzer.analyze(spec, campaign=args.campaign)
    output = args.output or (paths.analysis_dir(args.campaign) / f"{spec.name}.json")
    manifest.write_json(Path(output), report)
    print(json.dumps(report.get("headline", report), ensure_ascii=False, indent=2))
    print(f"\nwritten to {paths.relative_to_repo(Path(output))}")
    return 0


def _add_selection_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--framework", action="append",
                        help="Restrict to one framework; repeatable.")
    parser.add_argument("--profile", choices=registry.PROFILES, default="nightly")
    parser.add_argument("--repeats", type=int, default=None,
                        help="Fresh service starts per case. Defaults to the profile's value.")
    parser.add_argument("--campaign", default=None,
                        help="Run-directory name under exps/runs/. Use a new one for every protocol, override or "
                             "config change: --resume must never mix cells measured under different conditions.")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cells that already completed.")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even completed cells.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the expanded plan and time estimate; touch nothing.")
    parser.add_argument("--lock-wait-s", type=float, default=float(os.environ.get("EXPS_LOCK_WAIT_S", "0")),
                        help="Seconds to wait for the GPU lock (held by another sweep on this machine) before giving up. "
                             "Default: $EXPS_LOCK_WAIT_S, else 0 (fail at once). exps.suite passes no flag, so the env var "
                             "is how a queued suite waits for the running one.")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="Stop between cells once this much wall clock has passed. The full "
                             "matrix takes far longer than one night under per-request gating, so "
                             "a sweep is meant to be run repeatedly with --resume; repeat is the "
                             "outer loop, so each pass adds a start to every cell rather than "
                             "finishing some cells and never reaching others.")
    cli.add_protocol_arguments(parser)
    overrides.add_arguments(parser)
    machine = parser.add_argument_group("machine", "What GPU(s) this run uses; see exps/hardware.py and exps/profiles/hardware/.")
    machine.add_argument("--hardware", default=None,
                         help="Hardware profile (h200nvl, rtx5880, generic, or your own). Default: detected from the GPU name.")
    machine.add_argument("--gpus", default=None,
                         help="Comma separated physical GPU indices the services use, e.g. 0,1. Default: on a shared machine "
                              "(profile auto_select_gpus) every cell takes the free GPUs it needs, otherwise the profile's.")
    machine.add_argument("--gpu-wait-s", type=float, default=float(os.environ.get("EXPS_GPU_WAIT_S", "0")),
                         help="With automatic GPU selection: seconds to wait for enough free GPUs before a cell is failed "
                              "(default: $EXPS_GPU_WAIT_S or 0 = fail at once; `exps.suite` runs pick the variable up too).")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exps.run",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="Show every task, framework and case.")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(func=cmd_list)

    sweep = sub.add_parser("sweep", help="Run many tasks in one go (what run_all.sh calls).")
    sweep.add_argument("--tasks", nargs="*", help="Task names; defaults to all, in run order.")
    _add_selection_arguments(sweep)
    sweep.set_defaults(func=cmd_sweep)

    analyze = sub.add_parser("analyze", help="Summarise a task's completed cells.")
    analyze.add_argument("task")
    analyze.add_argument("--campaign", default=None)
    analyze.add_argument("--output", type=Path, default=None)
    analyze.set_defaults(func=cmd_analyze)

    for spec_name in registry.TASK_ORDER:
        task_parser = sub.add_parser(spec_name, help=f"Run the {spec_name} task.")
        task_parser.add_argument("--case", action="append",
                                 help="Restrict to one case; repeatable.")
        _add_selection_arguments(task_parser)
        task_parser.set_defaults(func=cmd_task, task=spec_name)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
