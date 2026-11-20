#!/usr/bin/env python3
"""Declarative registry of tasks, frameworks, cases and run profiles.

Everything the CLI and the nightly script can do is derived from here, so
"which cells exist" has exactly one definition. Each task package ships a
lightweight `cases.py` exposing a `TASK` of type `TaskSpec`; those modules must
not import Torch or any runtime SDK, because `run.py list` and `--dry-run` have
to work on a machine with no GPU touched.

Profiles
--------
A profile is a named subset plus protocol parameters:

``smoke``    one case per task, one start, one measured request. Proves the
             chain -- gates, service start, client, manifest -- works.
``nightly``  the default resumable sweep. BAGEL at 256 only.
``full``     adds BAGEL at 512. Both need multiple nights on this host.

1024x1024 is deliberately absent from every profile and from the case tables.
A single 1024 request is roughly four times the work of a 512 one, which on
this chassis means crossing the ~80 C degradation knee mid-request; the cell
would measure the thermal limit rather than the engine.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Iterator

#: Profile -> (repeats, measured requests per start, precondition requests).
PROFILE_PARAMETERS: dict[str, dict[str, int]] = {
    "smoke": {"repeats": 1, "measured": 1, "preconditions": 1},
    "nightly": {"repeats": 5, "measured": 5, "preconditions": 1},
    "full": {"repeats": 5, "measured": 5, "preconditions": 1},
}

PROFILES = tuple(PROFILE_PARAMETERS)

#: Default execution order for sweeps and calibration: protect the priority
#: workloads when a run stops early. Explicit --tasks retains the caller's order.
TASK_ORDER = (
    "bagel",
    "qwen_t2s",
    "vjepa_ac",
    "orpheus",
    "cosmos3",
    "pi05",
    "qwen_txt",
    "qwen_a2t",
)


@dataclass(frozen=True)
class Case:
    """One workload point: a fixed amount of work both engines must perform."""

    name: str
    label: str
    params: dict[str, Any] = field(default_factory=dict)
    #: Profiles this case belongs to. `smoke` is handled separately.
    profiles: tuple[str, ...] = ("nightly", "full")
    #: Frameworks for which this case has no equivalent contract, with the reason.
    not_applicable: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class TaskSpec:
    """One benchmarked workload and everything needed to enumerate its cells."""

    name: str
    title: str
    frameworks: tuple[str, ...]
    cases: tuple[Case, ...]
    #: Dotted module path exposing `run_cell(cell, options)`.
    runner_module: str
    #: What kind of claim this task's ratios can support.
    comparison_label: str
    #: Timing boundaries, in the task's own words, copied into every summary.
    timing_boundary: str
    notes: str = ""

    def case(self, name: str) -> Case:
        for case in self.cases:
            if case.name == name:
                return case
        available = ", ".join(c.name for c in self.cases)
        raise KeyError(f"task {self.name!r} has no case {name!r}; available: {available}")

    def cases_for_profile(self, profile: str) -> tuple[Case, ...]:
        if profile == "smoke":
            return self.cases[:1]
        return tuple(case for case in self.cases if profile in case.profiles)


@dataclass(frozen=True)
class CellRef:
    """A single unit of work: one task, framework, case and repeat."""

    task: str
    framework: str
    case: str
    repeat: str

    def __str__(self) -> str:
        return f"{self.task}/{self.framework}/{self.case}/{self.repeat}"


def load_task(name: str) -> TaskSpec:
    """Import a task's `cases` module and return its `TASK` spec."""
    try:
        module = importlib.import_module(f"exps.tasks.{name}.cases")
    except ModuleNotFoundError as exc:
        raise KeyError(f"unknown task {name!r}: {exc}") from exc
    spec = getattr(module, "TASK", None)
    if not isinstance(spec, TaskSpec):
        raise TypeError(f"exps.tasks.{name}.cases must define TASK: TaskSpec")
    return spec


def load_tasks(names: tuple[str, ...] | None = None) -> list[TaskSpec]:
    """Load all requested tasks; an unknown task must never shorten a sweep silently."""
    return [load_task(name) for name in (names if names is not None else TASK_ORDER)]


def repeat_id(index: int) -> str:
    return f"r{index:02d}"


def rotate(items: tuple[str, ...], offset: int) -> list[str]:
    """Rotate framework order by repeat index.

    Residual thermal drift across a long night should not land preferentially
    on whichever engine always runs last. Rotating the order per repeat spreads
    any leftover bias evenly instead of concentrating it.
    """
    if not items:
        return []
    shift = offset % len(items)
    return list(items[shift:]) + list(items[:shift])


def expand(
    spec: TaskSpec,
    *,
    profile: str = "nightly",
    frameworks: tuple[str, ...] | None = None,
    cases: tuple[str, ...] | None = None,
    repeats: int | None = None,
) -> Iterator[CellRef]:
    """Enumerate the cells of one task, in execution order.

    Repeat is the outer loop so that an interrupted night leaves every cell
    with the same number of starts, rather than three starts for the first
    framework and none for the last.
    """
    if profile not in PROFILE_PARAMETERS:
        raise KeyError(f"unknown profile {profile!r}; available: {', '.join(PROFILES)}")
    selected_cases = (
        tuple(spec.case(name) for name in cases) if cases else spec.cases_for_profile(profile)
    )
    selected_frameworks = frameworks or spec.frameworks
    unknown = set(selected_frameworks) - set(spec.frameworks)
    if unknown:
        raise KeyError(
            f"task {spec.name!r} has no framework(s) {sorted(unknown)}; "
            f"available: {', '.join(spec.frameworks)}"
        )
    total_repeats = repeats if repeats is not None else PROFILE_PARAMETERS[profile]["repeats"]

    for index in range(1, total_repeats + 1):
        for case in selected_cases:
            for framework in rotate(tuple(selected_frameworks), index - 1):
                if framework in case.not_applicable:
                    continue
                yield CellRef(spec.name, framework, case.name, repeat_id(index))


def not_applicable_cells(spec: TaskSpec) -> list[dict[str, str]]:
    """Framework/case pairs skipped by design, with the reason recorded.

    These are written into the campaign index so a gap in the matrix is
    self-explaining rather than looking like a crash.
    """
    return [
        {"framework": framework, "case": case.name, "reason": reason}
        for case in spec.cases
        for framework, reason in case.not_applicable.items()
    ]
