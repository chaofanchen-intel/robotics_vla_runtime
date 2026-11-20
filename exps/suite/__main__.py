#!/usr/bin/env python3
"""One entry point for every comparison and ablation.

    python -m exps.suite list [--group baseline|protocol|fair|fairness|advantage|profile|tool]
    python -m exps.suite show   bagel-attn-backend
    python -m exps.suite plan   'baseline-*'                      # cells, hours, the exact `exps.run` command per arm
    python -m exps.suite run    t2s-walk-blind --execute          # no --execute: same as plan
    python -m exps.suite run    protocol-ab-pi05 --execute
    python -m exps.suite run    baseline-bagel --protocol back-to-back --execute   # new campaigns, suffix _B2B
    python -m exps.suite analyze t2s-walk-blind                   # validate every cell, then write exps/results/<name>.md

Names accept shell globs. A fresh `--run-id` gives a fresh set of campaigns (a failed cell needs a new campaign id, not
a hand-edit of the old one).
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys

from exps import hardware, protocols
from exps.core import paths
from exps.suite import analysis, catalogue, runner


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("names", nargs="+", help="experiment names or globs")
    parser.add_argument("--run-id", default="01", help="campaign suffix; use a new one for every re-run")
    parser.add_argument("--protocol", choices=("gated", "back-to-back"), default=None,
                        help="measurement protocol for arms that do not fix their own. Default: the task's decision from "
                             "`decide-protocol` (baselines refuse to run without one), else the hardware profile's default "
                             "(gated on the throttling H200, back-to-back on a machine with stable cooling)")
    parser.add_argument("--repeats", type=int, default=None, help="fresh starts per cell, replacing every arm's default")
    parser.add_argument("--measured", type=int, default=None, help="measured requests per start, replacing every arm's default")
    parser.add_argument("--cases", default=None, help="comma separated cases replacing every arm's cases "
                                                      "(e.g. t2i-512-nocfg,t2i-1024-nocfg)")
    parser.add_argument("--frameworks", default=None, help="comma separated frameworks restricting every arm")
    parser.add_argument("--hardware", default=None, help="hardware profile (default: detected from the GPU name)")
    parser.add_argument("--gpus", default=None, help="physical GPU indices the services use, e.g. 0,1")
    parser.add_argument("--gap-s", type=float, default=0.0, help="with back-to-back: fixed sleep before each request")
    parser.add_argument("--arms", default=None, help="comma separated arm names (default: all, in order)")


def _arms(args) -> list[str] | None:
    return [a for a in args.arms.split(",") if a] if args.arms else None


def _split(value: str | None) -> list[str] | None:
    return [v for v in value.split(",") if v] if value else None


def _selection(args) -> dict:
    """Hardware first: it decides the default protocol and which configs resolve."""
    hardware.set_current(args.hardware, args.gpus)
    return dict(cases=_split(args.cases), frameworks=_split(args.frameworks), hardware_name=args.hardware, gpus=args.gpus,
                repeats=args.repeats, measured=args.measured)


# Display order for the groups we know; any other group the catalogue defines is appended, never dropped.
_GROUP_ORDER = ("baseline", "protocol", "fair", "fairness", "advantage", "profile", "tool")


def _catalogue_groups() -> list[str]:
    present = list(dict.fromkeys(e.group for e in catalogue.ALL.values()))
    return [g for g in _GROUP_ORDER if g in present] + [g for g in present if g not in _GROUP_ORDER]


def cmd_list(args) -> int:
    for group in _catalogue_groups():
        entries = [e for e in catalogue.ALL.values() if e.group == group and args.group in (None, group)]
        if not entries:
            continue
        print(f"\n[{group}]")
        for e in entries:
            arms = "tool" if e.tool else ",".join(a.name for a in e.arms)
            print(f"  {e.name:<28} {e.title}\n  {'':<28} arms: {arms}")
    return 0


def cmd_show(args) -> int:
    for e in runner.select(args.names):
        print(f"# {e.name}: {e.title}\n\nQUESTION: {e.question}\n\nREADING (pre-registered): {e.reading}")
        if e.caveats:
            print(f"\nCAVEATS: {e.caveats}")
        print(f"\ntask={e.task} frameworks={list(e.frameworks)} cases={list(e.cases)} kind={e.kind} "
              f"repeats={e.repeats} measured={e.measured} preconditions={e.preconditions}")
        for a in e.arms:
            print(f"  arm {a.name}: env={list(a.env)} serve_args={list(a.serve_args)} opts={list(a.opts)} "
                  f"protocol={a.protocol} {a.note}")
        print()
    return 0


def cmd_run(args) -> int:
    return runner.main_run(args.names, run_id=args.run_id, protocol=args.protocol, gap_s=args.gap_s, arms=_arms(args),
                           execute_it=args.execute, **_selection(args))


def cmd_plan(args) -> int:
    return runner.main_run(args.names, run_id=args.run_id, protocol=args.protocol, gap_s=args.gap_s, arms=_arms(args),
                           execute_it=False, **_selection(args))


def cmd_analyze(args) -> int:
    status = 0
    selection = _selection(args)
    for experiment in runner.select(args.names):
        if experiment.tool:
            print(f"{experiment.name}: tool entry, nothing to analyze here")
            continue
        try:
            outs = analysis.analyze(experiment, run_id=args.run_id, protocol=args.protocol, gap_s=args.gap_s, arms=_arms(args),
                                    **selection)
        except SystemExit as exc:
            print(exc, file=sys.stderr)
            status = 2
            continue
        for out in outs:
            print(f"{experiment.name}: wrote {paths.relative_to_repo(out)}")
        first = args.protocol or protocols.protocols_for(experiment.task, required=experiment.group == "baseline")[0]
        arm_runs = runner.arm_runs(experiment, run_id=args.run_id, protocol=first, gap_s=args.gap_s, arms=_arms(args),
                                   **selection)
        for template in experiment.post:
            values = {f"campaign:{r.arm.name}": r.campaign for r in arm_runs}
            command = template
            for key, value in values.items():
                command = command.replace("{" + key + "}", value)
            command = runner.expand_text(command, campaign=arm_runs[0].campaign if arm_runs else "")
            if _arms(args) and "{campaign:" in command:
                # --arms restricted this run: a post step for an arm outside it has no campaign to read
                print("  post skipped (arm not in this run):", command)
                continue
            print("  post:", command)
            if subprocess.call(shlex.split(command), cwd=paths.REPO_ROOT):
                print(f"  post command failed: {command}", file=sys.stderr)
                status = 2
    return status


def cmd_decide(args) -> int:
    from exps.suite import decide

    selection = _selection(args)
    selection.pop("repeats", None)
    selection.pop("measured", None)
    _, messages = decide.decide_all(tasks=_split(args.tasks), run_id=args.run_id, tolerance=args.tolerance, **selection)
    print("\n".join(messages))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="exps.suite", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    listing = sub.add_parser("list", help="every experiment")
    listing.add_argument("--group", default=None)
    listing.set_defaults(func=cmd_list)
    show = sub.add_parser("show", help="question, pre-registered reading, arms")
    show.add_argument("names", nargs="+")
    show.set_defaults(func=cmd_show)
    plan = sub.add_parser("plan", help="cells, estimated hours, exact commands (no GPU)")
    _add_run_options(plan)
    plan.set_defaults(func=cmd_plan)
    run = sub.add_parser("run", help="run arms in order (plan unless --execute)")
    _add_run_options(run)
    run.add_argument("--execute", action="store_true", help="actually start services and measure (needs the GPU)")
    run.set_defaults(func=cmd_run)
    decide = sub.add_parser("decide-protocol", help="from finished protocol-ab-* runs, decide which protocol each task's "
                                                    "baselines run under (baseline-* needs this)")
    decide.add_argument("--run-id", default="01")
    decide.add_argument("--tasks", default=None, help="comma separated task names (default: every task with a protocol-ab run)")
    decide.add_argument("--tolerance", type=float, default=0.05, help="relative difference treated as noise (the A1/A2 drift)")
    decide.add_argument("--hardware", default=None)
    decide.add_argument("--gpus", default=None)
    decide.add_argument("--cases", default=None)
    decide.add_argument("--frameworks", default=None)
    decide.add_argument("--protocol", default=None)
    decide.add_argument("--repeats", type=int, default=None)
    decide.add_argument("--measured", type=int, default=None)
    decide.add_argument("--gap-s", type=float, default=0.0)
    decide.add_argument("--arms", default=None)
    decide.set_defaults(func=cmd_decide)
    analyze = sub.add_parser("analyze", help="validate cells and write the results table (CPU)")
    _add_run_options(analyze)
    analyze.set_defaults(func=cmd_analyze)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
