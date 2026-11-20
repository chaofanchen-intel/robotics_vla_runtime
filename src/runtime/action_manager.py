"""Action Manager: chunks with generation numbers, the active chunk, and the execution index.

A chunk is the reply to one request: `horizon` actions planned from the observation taken at `t_snap`. Action i is
planned for time t_snap + i * dt. The manager keeps one active chunk. A chunk with a lower generation than the active
one is dropped; a newer one replaces it from the next control tick (LeRobot's replace_future).

Aligned mode (LeRobot's aligned merge): the executor runs, at time tau, the action planned for that time, so the
actions whose time has already passed when the chunk arrives are skipped. Naive mode (OpenPI's reference client): the
chunk is started at action 0 on arrival, which makes every action run about one request latency late.
When the executor runs past the last action, the tick is an underrun (the robot holds its last setpoint).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from runtime.observation import AGE_STREAMS


@dataclass
class Chunk:
    gen: int
    t_snap: float
    t_send: float
    t_recv: float
    actions: np.ndarray
    capture: dict[str, float]          # stream -> capture time of the sample used
    e2e_ms: float
    infer_ms: float | None
    pack_ms: float
    applied: bool = False
    first_tick: float | None = None    # nominal tick time of the first executed action (naive base)
    skipped: int | None = None         # actions skipped at first use (aligned) = index of the first executed action
    executed: int = 0


class ActionManager:
    def __init__(self, horizon: int, dt: float, tick_dt: float, align: bool):
        self.horizon, self.dt, self.tick_dt, self.align = horizon, dt, tick_dt, align
        self.active: Chunk | None = None
        self.unused_superseded = 0     # chunks replaced before any of their actions ran

    def offer(self, chunk: Chunk) -> bool:
        """A newly arrived chunk. Dropped when older than the active one."""
        if self.active is not None and chunk.gen <= self.active.gen:
            return False
        if self.active is not None and self.active.executed == 0:
            self.unused_superseded += 1
        self.active = chunk
        chunk.applied = True
        return True

    def locate(self, tau: float) -> tuple[Chunk | None, int | None, int]:
        """(active chunk, index of the action due at nominal tick time tau, actions left from that index)."""
        chunk = self.active
        if chunk is None:
            return None, None, 0
        if self.align:
            idx = int(math.ceil((tau - chunk.t_snap) / self.dt - 1e-9))
        else:
            if chunk.first_tick is None:
                chunk.first_tick = tau
            idx = int(round((tau - chunk.first_tick) / self.tick_dt))
        return chunk, idx, max(0, self.horizon - idx)

    def execute(self, chunk: Chunk, idx: int, tau: float, actual: float, remaining: int) -> dict:
        """Run (or fail to run) the action due now. Returns the control-tick record."""
        tick = {"t": tau, "late_ms": (actual - tau) * 1000.0, "gen": chunk.gen, "idx": idx, "remaining": remaining}
        if idx >= self.horizon:
            tick["underrun"] = True
            return tick
        tick["underrun"] = False
        if chunk.skipped is None:
            chunk.skipped = idx
        chunk.executed += 1
        tick["action_age_ms"] = (actual - chunk.t_snap) * 1000.0
        # execution time minus the time this action was planned for (t_snap + idx * dt): about 0 when aligned,
        # about the request latency when the chunk is started at action 0 on arrival (naive)
        tick["intent_lag_ms"] = (actual - (chunk.t_snap + idx * self.dt)) * 1000.0
        for n in AGE_STREAMS:
            tick[f"stale_{n}_ms"] = (actual - chunk.capture[n]) * 1000.0
        return tick
