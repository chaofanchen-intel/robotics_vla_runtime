#!/usr/bin/env python3
"""What an experiment is: a question, a set of arms, and how the answer is read.

An *arm* is one configuration of the systems under test: a set of framework / case selections plus the
overrides (`core.overrides`) and the measurement protocol it runs under. An *experiment* is an ordered list of
arms that differ in exactly the thing the question is about, each run in its own campaign (run directory), so
`--resume` can never mix cells measured under different conditions.

Two kinds of experiment
-----------------------
``compare``  Arms are candidate configurations of the *same* framework (or different frameworks, each at its best).
             The headline for a framework is its best candidate arm per case; ratios are best-vs-best. Every
             cross-framework claim (the `baseline-*` experiments) is of this kind: equal work, each side at its best.
``ab``       Arms differ in one switch and the first arm is the reference (A1 / B / A2 orderings expose drift).
             The result is the ratio of each arm to the reference, never a cross-framework claim.

Equal work and best config are properties of the catalogue, not of the analysis: the task packages define only
equal-work headline cases (BAGEL `-nocfg` / `-cfg3`, not the default request that gives M* one more branch than
vLLM-Omni), and their cells default to each framework's best recipe (tuned MoT config for vLLM-Omni BAGEL,
async-chunk + Code2Wav graph for vLLM-Omni T2S, partial start for SGLang-Omni T2S, BF16 for the V-JEPA baseline).
Anything else an arm changes is a deliberate counterfactual and is written down in the arm.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

PROTOCOLS = ("gated", "back-to-back")


@dataclass(frozen=True)
class Expect:
    """What a completed cell of this arm must show, checked by `analyze` before any number is reported.

    A log line or an argv that merely *looks* right is not evidence a switch took effect, so beyond the manifest's
    own record of the overrides, an arm can demand a server-log line, a hook proof file, or a forbidden string.
    """

    command_has: tuple[str, ...] = ()
    command_lacks: tuple[str, ...] = ()
    #: substrings every server log of the arm must contain / must not contain
    log_has: tuple[str, ...] = ()
    log_lacks: tuple[str, ...] = ()
    #: named hook-proof check (see analysis.PROOFS)
    proof: str | None = None


@dataclass(frozen=True)
class Arm:
    name: str
    env: tuple[str, ...] = ()             # --env FAMILY:KEY=VALUE
    serve_args: tuple[str, ...] = ()      # --serve-arg FAMILY:ARG
    opts: tuple[str, ...] = ()            # --opt TASK.KEY=VALUE
    frameworks: tuple[str, ...] | None = None
    cases: tuple[str, ...] | None = None
    protocol: str | None = None           # None: whatever the suite run asks for (default gated)
    gap_s: float = 0.0
    nsys: bool = False
    #: nsys options for this arm (None: the harness defaults, graph trace `node` and a 1,000 ms CUDA flush interval)
    nsys_graph_trace: str | None = None
    #: milliseconds, or a hardware placeholder such as "{hw:NAME}" that expands per machine; an empty expansion keeps 1,000 ms
    nsys_flush_ms: int | str | None = None
    repeats: int | None = None
    measured: int | None = None
    preconditions: int | None = None
    #: in a `compare` experiment, may this arm be a framework's best config? (False: reference / control only)
    candidate: bool = True
    expect: Expect = Expect()
    note: str = ""


@dataclass(frozen=True)
class Experiment:
    name: str
    title: str
    task: str | None
    question: str
    #: pre-registered reading: what each outcome would mean, written before the run
    reading: str
    frameworks: tuple[str, ...] = ()
    cases: tuple[str, ...] = ()
    arms: tuple[Arm, ...] = (Arm("main"),)
    kind: str = "compare"                 # compare | ab
    group: str = "ablation"               # baseline | protocol | ablation | profile | tool
    profile: str = "full"
    repeats: int = 3
    measured: int = 5
    preconditions: int = 1
    #: per-audio-second metric for T2S-style tasks (E2E scales with generated audio length)
    metric: str = "e2e_ms"
    #: command templates (argv, `{py}` `{results}` `{campaign}` `{campaign:ARM}` expanded) run during `analyze`
    post: tuple[str, ...] = ()
    #: for tool-only entries: command to run instead of cells (`run` prints it; `--execute` runs it)
    tool: tuple[str, ...] = ()
    caveats: str = ""
    #: GPUs the experiment needs on one machine; a machine with fewer skips it (not a failure) in `plan` / `run`
    min_gpus: int = 1

    def arm(self, name: str) -> Arm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(f"{self.name} has no arm {name!r}; arms: {', '.join(a.name for a in self.arms)}")

    def resolved(self, arm: Arm) -> Arm:
        """The arm with experiment-level defaults filled in."""
        return replace(
            arm,
            frameworks=arm.frameworks if arm.frameworks is not None else self.frameworks,
            cases=arm.cases if arm.cases is not None else self.cases,
            repeats=arm.repeats if arm.repeats is not None else self.repeats,
            measured=arm.measured if arm.measured is not None else self.measured,
            preconditions=arm.preconditions if arm.preconditions is not None else self.preconditions,
        )


def campaign_name(experiment: Experiment, arm: Arm, *, run_id: str, protocol: str, gap_s: float) -> str:
    """`<EXPERIMENT>_<ARM>_<run id>[_B2B|_GAP30]`: a protocol change is a new campaign, always.

    An arm that fixes its own protocol (protocol A/B) already says so in its name, so it gets no suffix.
    """
    base = experiment.name.upper().replace("-", "_")
    suffix = ""
    if protocol == "back-to-back" and arm.protocol is None:
        suffix = f"_GAP{gap_s:g}" if gap_s > 0 else "_B2B"
    return f"{base}_{arm.name.upper().replace('-', '_')}_{run_id}{suffix}"


def effective_protocol(arm: Arm, requested: str, requested_gap_s: float) -> tuple[str, float]:
    """An arm that fixes its protocol (protocol A/B) wins over the suite-level request."""
    if arm.protocol is not None:
        return arm.protocol, arm.gap_s
    return requested, requested_gap_s


Check = Callable[..., None]
