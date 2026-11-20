"""Inference Scheduler: when does the runtime ask the framework for a new action chunk.

Three trigger policies, each from an upstream reference:
  period      a timer every `period_s`, independent of the actions (an external Runtime with a fixed rate)
  open_loop   after `replan_steps` actions of the active chunk have run, or when it has none left (OpenPI's reference
              client, open_loop_horizon)
  threshold   when the queued actions cover at most `refill_s` (LeRobot's replay threshold, inference/execution.py:357-384),
              raised to the measured turnaround plus one step when the turnaround is longer

At most one request is in flight (the OpenPI protocol is serial per connection): a trigger that finds one in flight does
nothing (a period trigger counts as skipped). `align` is not a trigger setting but belongs to the cell: it selects how
the executor starts a chunk (action_manager.py).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Trigger:
    kind: str                   # period | open_loop | threshold
    period_s: float = 0.2       # period
    replan_steps: int = 8       # open_loop: request after this many executed actions of the active chunk
    refill_s: float = 0.5       # threshold: request when the queued actions cover at most this long
    align: bool = True          # aligned (skip elapsed actions) or naive (start at action 0 on arrival)

    def label(self) -> str:
        base = {"period": f"period-{int(round(self.period_s * 1000))}ms", "open_loop": f"openloop-{self.replan_steps}",
                "threshold": f"threshold-{self.refill_s:g}s"}[self.kind]
        return base + ("" if self.align else "-naive")


class Scheduler:
    def __init__(self, trigger: Trigger, dt: float):
        self.trigger = trigger
        self.dt = dt

    @property
    def timer_driven(self) -> bool:
        return self.trigger.kind == "period"

    def wants_request(self, active, remaining: int, last_e2e_s: float) -> bool:
        """Evaluated once per control tick for the tick-driven policies. `active` is the active chunk or None."""
        t = self.trigger
        if t.kind == "threshold":
            return active is None or remaining * self.dt <= max(t.refill_s, last_e2e_s + self.dt)
        if t.kind == "open_loop":
            # Also when the chunk has no action left: OpenPI's client requests when the chunk runs out. Without this,
            # aligned execution can stall for good: a slow reply leaves fewer than `replan_steps` actions to run, the
            # count is never reached, and no further request is ever sent.
            return active is None or active.executed >= t.replan_steps or remaining == 0
        return False
