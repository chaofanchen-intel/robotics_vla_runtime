#!/usr/bin/env python3
"""Which measurement protocol each task runs its baselines under, decided by evidence rather than by habit.

The gated and back-to-back protocols measure different things and, for some tasks, give different cross-framework ratios
(Pi0.5: M*/vLLM 3.48x gated, 2.45x back-to-back). So the order is: `protocol-ab-<task>` (gated vs back-to-back, few starts) first,
`python -m exps.suite decide-protocol` second, `baseline-*` last. The decision is written per hardware profile to
`profiles/protocol/<hardware>.json`; `baseline-*` experiments refuse to plan without it (or an explicit `--protocol`).

Rule, per task, from the pooled p50 ratio r_f = back-to-back / gated of every framework f (tolerance 5%: the A1/A2 drift):

* all |r_f - 1| <= tol                      the protocol does not matter for this task: use back-to-back (far cheaper) -- but only
                                            when the measured SM clock of the back-to-back arm stayed within 3% of the gated arm's,
                                            i.e. it did not throttle; with no clock evidence, or a drop, the hardware's default.
* frameworks shift by different amounts     max(r)/min(r) - 1 > tol: the protocol changes the *ratio between frameworks*:
                                            run the baseline under BOTH protocols and report both; no single number is honest.
* frameworks shift together                 all move by about the same factor: the ratios between them survive; use the
                                            hardware's default and note the absolute shift.

A decision is evidence for one hardware profile; another machine needs its own.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from exps import hardware
from exps.core import paths

DECISION_DIR = paths.PROFILES / "protocol"
TOLERANCE = 0.05


def decision_path(hardware_name: str | None = None) -> Path:
    return DECISION_DIR / f"{hardware_name or hardware.current().name}.json"


def load(hardware_name: str | None = None) -> dict:
    path = decision_path(hardware_name)
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save(decisions: dict, hardware_name: str | None = None) -> Path:
    path = decision_path(hardware_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(decisions, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    return path


CLOCK_FLOOR = 0.97


def decide(ratios: dict[str, float], *, tolerance: float = TOLERANCE, clock_ratios: dict[str, float] | None = None) -> dict:
    """`ratios`: framework -> back-to-back p50 / gated p50 (pooled); `clock_ratios`: framework -> back-to-back mean SM clock / gated's."""
    hw_default = hardware.current().default_protocol
    values = list(ratios.values())
    if not values:
        raise ValueError("no framework ratios to decide on")
    if all(abs(r - 1.0) <= tolerance for r in values):
        if clock_ratios and min(clock_ratios.values()) >= CLOCK_FLOOR:
            return {"protocols": ["back-to-back"], "kind": "insensitive",
                    "reason": f"every framework within {tolerance:.0%} of 1.0 and the back-to-back clock held (>= {CLOCK_FLOOR:.0%} of "
                              "gated): the protocol does not change this task, so the cheap one"}
        return {"protocols": [hw_default], "kind": "insensitive",
                "reason": f"every framework within {tolerance:.0%} of 1.0, but "
                          + ("the back-to-back SM clock dropped (throttling)" if clock_ratios else "there is no clock evidence")
                          + f": hardware default ({hw_default})"}
    if max(values) / min(values) - 1.0 > tolerance:
        return {"protocols": ["gated", "back-to-back"], "kind": "framework-dependent",
                "reason": f"frameworks shift by different amounts (max/min {max(values) / min(values):.2f}): the protocol changes "
                          "the ratio between frameworks; report both"}
    return {"protocols": [hw_default], "kind": "shifts-together",
            "reason": "all frameworks shift by about the same factor: ratios between them survive; hardware default, absolute "
                      "numbers differ by protocol"}


def record(task: str, ratios: dict[str, float], evidence: list[str], *, gap30: dict[str, float] | None = None,
           tolerance: float = TOLERANCE, clock_ratios: dict[str, float] | None = None) -> dict:
    out = decide(ratios, tolerance=tolerance, clock_ratios=clock_ratios)
    if clock_ratios:
        out["clock_ratios_b2b_over_gated"] = {k: round(v, 3) for k, v in sorted(clock_ratios.items())}
    out.update({"ratios_b2b_over_gated": {k: round(v, 3) for k, v in sorted(ratios.items())}, "evidence": evidence,
                "tolerance": tolerance, "hardware": hardware.current().name,
                "decided_at": datetime.now(timezone.utc).date().isoformat()})
    if gap30:
        out["ratios_gap30_over_gated"] = {k: round(v, 3) for k, v in sorted(gap30.items())}
    return out


def protocols_for(task: str, *, required: bool) -> list[str]:
    """Protocols to run `task`'s experiments under when none is given on the command line.

    `required` (baselines): a task without a decision raises; the cross-framework ratio must not be produced under a protocol
    chosen by default. Otherwise: a single-protocol decision wins, else the hardware default.
    """
    decision = load().get(task)
    if decision:
        protocols = decision["protocols"]
        return protocols if required or len(protocols) == 1 else [hardware.current().default_protocol]
    if required:
        raise SystemExit(
            f"no protocol decision for task {task!r} on hardware {hardware.current().name!r}. Run "
            f"`python -m exps.suite run protocol-ab-{task.replace('_', '-')} --execute`, then "
            "`python -m exps.suite decide-protocol`; or pass --protocol explicitly (and say why in the report).")
    return [hardware.current().default_protocol]


def default_for(task: str | None) -> str:
    """Protocol of a hand-run `exps.run` cell: a single-protocol decision for the task, else the hardware default."""
    if task:
        decision = load().get(task)
        if decision and len(decision["protocols"]) == 1:
            return decision["protocols"][0]
    return hardware.current().default_protocol
