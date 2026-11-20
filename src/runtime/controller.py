"""Robot Controller: the control clock and the request loop of one Level 1 cell.

One connection, at most one request in flight. The control clock takes one action per tick from the Action Manager;
the Scheduler decides when a request goes out; each request snapshots the Observation Manager, sends it through the
Backend, and hands the reply chunk to the Action Manager. Everything is timed with time.perf_counter; schedule time s is
perf_counter minus the cell's start.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass, field

from backends.base import Backend
from metrics.level1 import summarize
from runtime.action_manager import ActionManager, Chunk
from runtime.observation import AGE_STREAMS, Schedule, snapshot_to_request
from runtime.scheduler import Scheduler, Trigger


@dataclass
class CellResult:
    config: dict
    requests: list = field(default_factory=list)
    ticks: list = field(default_factory=list)
    skipped_triggers: int = 0
    summary: dict = field(default_factory=dict)


async def run_cell(ws, backend: Backend, schedule: Schedule, trigger: Trigger, *, horizon: int, control_hz: float = 15.0,
                   duration_s: float = 40.0, warmup_s: float = 3.0, steps: int = 10, seed: int = 42,
                   session: str = "level1", tick_skew: float = 0.004, phase_s: float = 0.0) -> CellResult:
    # The control clock and the period timer are independent of the camera clocks. With the same nominal 15 Hz they
    # would stay phase-locked and every snapshot would see the same input ages, which no real robot does. The control
    # clock therefore runs `tick_skew` slow (0.4 %: the phase sweeps a camera period in about 17 s) from a start phase
    # of `phase_s`. The action spacing stays 1/control_hz.
    dt = 1.0 / control_hz
    tick_dt = dt * (1.0 + tick_skew)
    t0 = time.perf_counter()

    def now() -> float:
        return time.perf_counter() - t0

    start_s = schedule.first_ready_s() + 0.03 + phase_s
    win_lo, win_hi = start_s + warmup_s, start_s + warmup_s + duration_s
    result = CellResult(config={"framework": backend.name, "trigger": asdict(trigger), "label": trigger.label(),
                                "profile": schedule.profile, "horizon": horizon, "control_hz": control_hz,
                                "duration_s": duration_s, "warmup_s": warmup_s, "steps": steps, "seed": seed,
                                "tick_skew": tick_skew, "phase_s": phase_s})
    actions = ActionManager(horizon, dt, tick_dt, trigger.align)
    scheduler = Scheduler(trigger, dt)
    in_flight = 0
    gen = 0
    last_e2e_s = 0.25                   # LeRobot's turnaround estimate; starts at a typical value, then the measured one
    chunks: list[Chunk] = []
    tasks: list[asyncio.Task] = []

    async def request(my_gen: int) -> None:
        nonlocal in_flight, last_e2e_s
        t_snap = now()
        snap = schedule.snapshot(t_snap)
        example, images = snapshot_to_request(snap)
        t_p = time.perf_counter()
        packed = backend.pack(backend.payload(example, images, session, steps, seed))
        pack_ms = (time.perf_counter() - t_p) * 1000.0
        t_send = now()
        try:
            await ws.send(packed)
            raw = await asyncio.wait_for(ws.recv(), 120.0)
        finally:
            in_flight -= 1
        t_recv = now()
        reply, timing = backend.parse_reply(raw, horizon)
        chunk = Chunk(gen=my_gen, t_snap=t_snap, t_send=t_send, t_recv=t_recv, actions=reply,
                      capture={n: snap[n][1] for n in AGE_STREAMS}, e2e_ms=(t_recv - t_send) * 1000.0,
                      infer_ms=timing.get("infer_ms"), pack_ms=pack_ms)
        last_e2e_s = t_recv - t_send
        chunks.append(chunk)
        actions.offer(chunk)

    def launch() -> bool:
        nonlocal in_flight, gen
        if in_flight >= 1:
            return False
        in_flight += 1
        gen += 1
        tasks.append(asyncio.ensure_future(request(gen)))
        return True

    async def sleep_until(target_s: float) -> None:
        delay = target_s - now()
        if delay > 0:
            await asyncio.sleep(delay)

    async def period_loop() -> None:
        n = 0
        while True:
            t = start_s + n * trigger.period_s * (1.0 + tick_skew)
            if t > win_hi:
                return
            await sleep_until(t)
            if not launch() and t >= win_lo:
                result.skipped_triggers += 1
            n += 1

    async def control_loop() -> None:
        k = 0
        while True:
            tau = start_s + k * tick_dt
            if tau > win_hi:
                return
            await sleep_until(tau)
            actual = now()
            chunk, idx, remaining = actions.locate(tau)
            if scheduler.wants_request(chunk, remaining, last_e2e_s):
                launch()
            k += 1
            if chunk is None:
                continue                       # cold start: nothing has arrived yet
            tick = actions.execute(chunk, idx, tau, actual, remaining)
            if tau >= win_lo:
                result.ticks.append(tick)

    loops = [asyncio.ensure_future(control_loop())]
    if scheduler.timer_driven:
        loops.append(asyncio.ensure_future(period_loop()))
    await asyncio.gather(*loops)
    if tasks:
        await asyncio.wait(tasks, timeout=60.0)          # leave the server idle for the next cell
        for t in tasks:
            if t.done() and t.exception() is not None:
                raise t.exception()

    for c in chunks:
        if c.t_snap < win_lo or c.t_snap > win_hi:
            continue
        result.requests.append({
            "gen": c.gen, "t_snap": c.t_snap, "e2e_ms": c.e2e_ms, "infer_ms": c.infer_ms, "pack_ms": c.pack_ms,
            "snap_to_send_ms": (c.t_send - c.t_snap) * 1000.0, "applied": c.applied, "skipped_at_first_use": c.skipped,
            "executed": c.executed,
            "age_ms": {n: (c.t_snap - c.capture[n]) * 1000.0 for n in AGE_STREAMS}})
    result.summary = summarize(result, dt, actions.unused_superseded)
    return result
