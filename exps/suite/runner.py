#!/usr/bin/env python3
"""Turn an experiment's arms into `exps.run` invocations, and run or plan them.

Every arm is exactly one `python -m exps.run TASK ...` command line (printed by `plan`), so anything the suite does
can be reproduced by hand and the manifests of suite cells and hand-run cells are identical. The suite adds only
the bookkeeping that the earlier shell scripts did by convention: one campaign per arm and protocol, the A/B/A order,
placeholder expansion for hook paths, and refusing to run with instrumentation inherited from a previous shell.
"""
from __future__ import annotations

import contextlib
import fnmatch
import io
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

from exps import hardware, protocols
from exps.core import paths
from exps.suite import catalogue
from exps.suite.spec import Arm, Experiment, campaign_name, effective_protocol

PY = "envs/mstar/bin/python"
#: Environment variables the old harness read; if one is set the operator expects an effect that no longer exists.
STALE_ENV = ("EXPS_MSTAR_EXTRA_ENV", "EXPS_MSTAR_EXTRA_ARGS", "EXPS_VLLM_EXTRA_ENV", "EXPS_NSYS_DIR",
             "EXPERIMENTS_MSTAR_EXTRA_ENV", "EXPERIMENTS_MSTAR_EXTRA_ARGS", "EXPERIMENTS_VLLM_EXTRA_ENV",
             "EXPERIMENTS_NSYS_DIR", "EXPERIMENTS_VJEPA_NATIVE_DTYPE", "EXPERIMENTS_PI05_VLLM_GRAPH",
             "QWEN_VLLM_DEPLOY_CONFIG", "QWEN_SGLANG_CONFIG", "QWEN_VLLM_BATCH_CONFIG")


@dataclass(frozen=True)
class ArmRun:
    experiment: Experiment
    arm: Arm
    campaign: str
    protocol: str
    gap_s: float
    argv: tuple[str, ...]


def select(patterns: list[str]) -> list[Experiment]:
    """Names or shell globs (`protocol-ab-*`, `baseline-*`); an unmatched pattern is an error, never an empty run."""
    chosen: list[Experiment] = []
    for pattern in patterns:
        matched = [e for name, e in catalogue.ALL.items() if fnmatch.fnmatchcase(name, pattern)]
        if not matched:
            raise SystemExit(f"no experiment matches {pattern!r}; `python -m exps.suite list` shows them")
        chosen.extend(e for e in matched if e not in chosen)
    return chosen


def expand_text(text: str, *, campaign: str, extra: dict[str, str] | None = None) -> str:
    """`{hooks}` `{campaign_dir}` `{py}` `{results}` `{repo}` `{bagel_snapshot}`: paths that depend on the arm or machine."""
    values = {
        "hooks": str(paths.EXPS_ROOT / "tools"),
        "campaign_dir": str(paths.campaign_root(campaign)),
        "py": PY,
        "results": str(paths.RESULTS.relative_to(paths.REPO_ROOT)),
        "repo": str(paths.REPO_ROOT),
        "campaign": campaign,
        "hardware": hardware.current().name,
        "thinker_tp": str(hardware.current().qwen_thinker_tp),
        "mstar_thinker_n": str(1536 // hardware.current().qwen_thinker_tp),
        # vLLM / SGLang name a table by (E, N) only. A TP-split Thinker (N=768/TP) that lands on the Talker's N=384 would overwrite
        # the Talker's table, so tuning the Talker first is wasted work: tune the Thinker only, the Talker then reads its table.
        "moe_roles": ("thinker:%d" % hardware.current().qwen_thinker_tp) if 768 // hardware.current().qwen_thinker_tp == 384
                     else "talker:1 thinker:%d" % hardware.current().qwen_thinker_tp,
        **(extra or {}),
    }
    for key, value in values.items():
        text = text.replace("{" + key + "}", value)
    return _expand_hardware(text)


_HW_PLACEHOLDER = re.compile(r"\{(variant|variant_file|variant_abs|hw):([A-Za-z0-9_./-]+)\}")


def _expand_hardware(text: str) -> str:
    """The placeholders that name this machine's own file or text (see `exps.hardware`):

    `{variant:KEY}`       repo-relative path of the machine's config for the hardware-neutral variant KEY (an error if it has none);
    `{variant_file:KEY}`  just its file name (what a `command_has` check looks for);
    `{variant_abs:KEY}`   absolute path (for an environment variable a service reads);
    `{hw:NAME}`           the profile's text token NAME, empty when the profile does not define it.
    """
    def replace(match: re.Match) -> str:
        kind, key = match.groups()
        if kind == "hw":
            return _expand_hardware(hardware.token(key))     # a token may name a variant (the H200 cache-off option does)
        relative = hardware.variant_path(key)
        if kind == "variant_file":
            return Path(relative).name
        return str(paths.REPO_ROOT / relative) if kind == "variant_abs" else relative

    return _HW_PLACEHOLDER.sub(replace, text)


def expand_all(values, *, campaign: str) -> list[str]:
    """`expand_text` over a tuple of arm settings, dropping the ones that expand to nothing (a `{hw:NAME}` this machine does not need)."""
    expanded = (expand_text(value, campaign=campaign) for value in values)
    return [value for value in expanded if value.strip()]


def arm_runs(experiment: Experiment, *, run_id: str = "01", protocol: str | None = None, gap_s: float = 0.0,
             arms: list[str] | None = None, dry_run: bool = False, cases: list[str] | None = None,
             frameworks: list[str] | None = None, hardware_name: str | None = None, gpus: str | None = None,
             repeats: int | None = None, measured: int | None = None) -> list[ArmRun]:
    """One `ArmRun` per (protocol, arm).

    `protocol` None: the protocol(s) decided for the task by `decide-protocol` (a baseline without a decision raises: its
    cross-framework ratio must not come from a protocol chosen by default; a decision of "both" runs every arm under both),
    otherwise the hardware profile's default. `cases` / `frameworks` replace / restrict every arm's selection (an arm left with
    no framework is skipped); `repeats` / `measured` rescale every arm (quick passes, or a longer final one).
    """
    if experiment.task is None:
        return []
    protocol_list = [protocol] if protocol else protocols.protocols_for(experiment.task, required=experiment.group == "baseline")
    selected = [experiment.arm(name) for name in arms] if arms else list(experiment.arms)
    runs: list[ArmRun] = []
    for requested in protocol_list:
        for raw in selected:
            arm = experiment.resolved(raw)
            if cases:
                arm = replace(arm, cases=tuple(cases))
            if repeats:
                arm = replace(arm, repeats=repeats)
            if measured:
                arm = replace(arm, measured=measured)
            if frameworks:
                kept = tuple(f for f in arm.frameworks if f in frameworks)
                if not kept:
                    continue
                arm = replace(arm, frameworks=kept)
            arm_protocol, arm_gap = effective_protocol(arm, requested, gap_s if requested == "back-to-back" else 0.0)
            campaign = campaign_name(experiment, arm, run_id=run_id, protocol=arm_protocol, gap_s=arm_gap)
            argv = ["-m", "exps.run", experiment.task, "--campaign", campaign, "--profile", experiment.profile, "--resume",
                    "--repeats", str(arm.repeats), "--measured", str(arm.measured), "--preconditions", str(arm.preconditions)]
            for framework in arm.frameworks:
                argv += ["--framework", framework]
            for case in arm.cases:
                argv += ["--case", case]
            argv += ["--protocol", arm_protocol]
            if arm_protocol == "back-to-back" and arm_gap > 0:
                argv += ["--inter-request-gap-s", str(arm_gap)]
            for flag, values in (("--env", arm.env), ("--serve-arg", arm.serve_args), ("--opt", arm.opts)):
                for value in expand_all(values, campaign=campaign):
                    argv += [flag, value]
            if arm.nsys:
                flush = expand_text(str(arm.nsys_flush_ms), campaign=campaign).strip() if arm.nsys_flush_ms is not None else ""
                argv += ["--nsys-dir", str(paths.campaign_root(campaign) / "nsys"), "--nsys-flush-ms", flush or "1000"]
                if arm.nsys_graph_trace:
                    argv += ["--nsys-graph-trace", arm.nsys_graph_trace]
            if hardware_name:
                argv += ["--hardware", hardware_name]
            if gpus:
                argv += ["--gpus", gpus]
            if dry_run:
                argv.append("--dry-run")
            run = ArmRun(experiment, arm, campaign, arm_protocol, arm_gap, tuple(argv))
            if run.campaign not in {r.campaign for r in runs}:
                runs.append(run)
    return runs


def shell_line(run: ArmRun) -> str:
    return " ".join([PY, *(shlex.quote(a) for a in run.argv if a != "--dry-run")])


def dry_plan(run: ArmRun) -> dict:
    """Cells, estimated hours and the first cell's serve command, from `exps.run --dry-run` (no GPU)."""
    from exps import run as run_module

    argv = [a for a in run.argv[2:]]  # drop "-m exps.run"
    if "--dry-run" not in argv:
        argv.append("--dry-run")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(io.StringIO()):
        status = run_module.main(argv)
    if status:
        raise SystemExit(f"{run.campaign}: dry-run failed (exit {status})")
    return json.loads(buffer.getvalue())


def check_clean_environment() -> None:
    leftovers = [name for name in STALE_ENV if os.environ.get(name)]
    if leftovers:
        raise SystemExit(
            "these variables were read by the old harness and do nothing now; unset them so nobody expects an effect: "
            + ", ".join(leftovers))


def execute(run: ArmRun) -> int:
    from exps import run as run_module

    print(f"\n=== {run.experiment.name} / arm {run.arm.name}  campaign {run.campaign}  protocol {run.protocol}"
          + (f" gap {run.gap_s:g}s" if run.gap_s else ""), flush=True)
    print("    " + shell_line(run), flush=True)
    return run_module.main(list(run.argv[2:]))


def run_tool(experiment: Experiment, *, run_id: str, execute_it: bool) -> int:
    from exps.tasks.bagel.cell import SNAPSHOT

    command = [expand_text(part, campaign="", extra={"bagel_snapshot": str(SNAPSHOT), "run_id": run_id})
               for part in experiment.tool]
    print("    " + " ".join(shlex.quote(c) for c in command))
    if not execute_it:
        return 0
    return subprocess.call(command, cwd=paths.REPO_ROOT)


def main_run(names: list[str], *, run_id: str, protocol: str | None, gap_s: float, arms: list[str] | None, execute_it: bool,
             cases: list[str] | None = None, frameworks: list[str] | None = None, hardware_name: str | None = None,
             gpus: str | None = None, repeats: int | None = None, measured: int | None = None) -> int:
    experiments = select(names)
    selection = dict(cases=cases, frameworks=frameworks, hardware_name=hardware_name, gpus=gpus, repeats=repeats,
                     measured=measured)
    if execute_it:
        check_clean_environment()
    failures: list[str] = []
    for experiment in experiments:
        print(f"\n##### {experiment.name}: {experiment.title}")
        if experiment.min_gpus > 1:
            machine = hardware.set_current(hardware_name, None)
            if machine.gpu_count < experiment.min_gpus:
                print(f"  skipped: needs {experiment.min_gpus} GPUs, hardware profile {machine.name!r} has {machine.gpu_count}")
                continue
        if experiment.tool:
            if run_tool(experiment, run_id=run_id, execute_it=execute_it):
                failures.append(experiment.name)
            continue
        try:
            runs = arm_runs(experiment, run_id=run_id, protocol=protocol, gap_s=gap_s, arms=arms, **selection)
        except SystemExit as exc:  # e.g. a baseline whose protocol has not been decided yet
            print(f"  not runnable yet: {exc}", file=sys.stderr if execute_it else sys.stdout)
            failures.append(f"{experiment.name} (no protocol decision)")
            continue
        for run in runs:
            if not execute_it:
                try:
                    plan = dry_plan(run)
                except SystemExit as exc:  # e.g. the task does not fit this machine without an overlay config
                    print(f"  arm {run.arm.name:<18} cannot be planned here: {exc}")
                    failures.append(f"{experiment.name}/{run.arm.name} (plan)")
                    continue
                print(f"  arm {run.arm.name:<18} campaign {run.campaign}\n    {plan['cells']} cells, "
                      f"~{plan['estimated_hours']} h by the harness estimate (cooldown waits dominate gated runs)\n    "
                      f"{shell_line(run)}")
                continue
            try:
                status = execute(run)
            except SystemExit as exc:  # a refused plan must not take the remaining experiments down with it
                print(f"  arm {run.arm.name} refused: {exc}", file=sys.stderr)
                status = 2
            if status:
                failures.append(f"{experiment.name}/{run.arm.name}")
                print(f"  arm {run.arm.name} failed; later arms of {experiment.name} are skipped (A/B/A order matters)")
                break
    if failures:
        print("\nFAILED: " + ", ".join(failures), file=sys.stderr)
    return 1 if failures else 0
