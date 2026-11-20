#!/usr/bin/env python3
"""Level 1 measurement driver: run the runtime (level1_runtime.py) against one running OpenPI WebSocket server.

Usage (server already running; scripts/run_level1.py starts it):
    envs/mstar/bin/python src/scenarios/level1_periodic.py --framework mstar-rust --url ws://127.0.0.1:18104 --horizon 15 --out runs/<dir>/mstar-rust/level1.json

Each cell is one trigger policy on one input profile, run for `--duration` seconds after `--warmup` seconds that are not
counted. Every cell opens a fresh connection. One invocation is one pass over the cells; `--repeat-index R` rotates the
cell order by 3R positions, so that slow drift in the machine does not always land on the same cell. Repeats are separate
invocations (run_level1.py starts a fresh server for each), and the host load average is recorded around every cell.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
import zlib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))            # exps/
sys.path.insert(0, str(REPO / "src"))   # runtime, backends, metrics, scenarios

import numpy as np  # noqa: E402

import backends  # noqa: E402
from runtime import dataset, observation  # noqa: E402
from runtime.controller import run_cell  # noqa: E402
from runtime.scheduler import Trigger  # noqa: E402

T = Trigger
# (cell label, input profile, trigger). The first seven cells use the multi-rate inputs; the rest compare against them.
CELLS: list[tuple[str, str, Trigger]] = [
    ("period-100ms", "multirate", T("period", period_s=0.100)),
    ("period-200ms", "multirate", T("period", period_s=0.200)),
    ("period-300ms", "multirate", T("period", period_s=0.300)),
    ("openloop-8", "multirate", T("open_loop", replan_steps=8)),
    ("threshold-0.3s", "multirate", T("threshold", refill_s=0.3)),
    ("threshold-0.5s", "multirate", T("threshold", refill_s=0.5)),
    ("threshold-0.8s", "multirate", T("threshold", refill_s=0.8)),
    ("openloop-8-naive", "multirate", T("open_loop", replan_steps=8, align=False)),
    ("threshold-0.5s@lockstep", "lockstep", T("threshold", refill_s=0.5)),
    ("period-200ms@lockstep", "lockstep", T("period", period_s=0.200)),
]


def phase_for(repeat: int, label: str, dt: float) -> float:
    """Start phase of the control clock against the camera clocks: the same for every framework in a (repeat, cell)."""
    return float(np.random.default_rng(zlib.crc32(f"{repeat}-{label}".encode())).uniform(0.0, dt))


async def warm_server(backend: backends.Backend, url: str, schedule: observation.Schedule, horizon: int, n: int) -> list[float]:
    ws, _meta = await backend.connect(url)
    times = []
    try:
        for i in range(n):
            example, images = observation.snapshot_to_request(schedule.snapshot(2.0 + 0.5 * i))
            packed = backend.pack(backend.payload(example, images, "level1-warm", 10, 42))
            t = time.perf_counter()
            await ws.send(packed)
            backend.parse_reply(await asyncio.wait_for(ws.recv(), 600.0), horizon)
            times.append((time.perf_counter() - t) * 1000.0)
    finally:
        await ws.close()
    return times


async def main_async(args) -> int:
    backend = backends.get(args.framework)
    episodes = dataset.load_episodes()
    horizon_s = args.warmup + args.duration + 10.0
    schedules = {p: observation.Schedule(episodes, p, args.seed, horizon_s) for p in ("multirate", "lockstep")}
    describe = {p: s.describe() for p, s in schedules.items()}
    warm = await warm_server(backend, args.url, schedules["multirate"], args.horizon, args.warmup_requests)
    print(f"server warm-up {len(warm)} requests: first {warm[0]:.0f} ms, last {warm[-1]:.0f} ms", flush=True)

    labels = [c[0] for c in CELLS]
    selected = [c for c in CELLS if args.cells is None or c[0] in args.cells]
    if not selected:
        raise SystemExit(f"no cell matches {args.cells}; choose from {labels}")
    out = {"framework": args.framework, "repeat_index": args.repeat_index, "checkpoint_horizon": args.horizon, "control_hz": args.control_hz,
           "duration_s": args.duration, "warmup_s": args.warmup, "seed": args.seed, "streams": describe,
           "warmup_request_ms": warm, "cells": []}
    r = args.repeat_index
    shift = (3 * r) % len(selected)
    for label, profile, trigger in selected[shift:] + selected[:shift]:
        ws, _meta = await backend.connect(args.url)
        t = time.time()
        load_before = os.getloadavg()
        try:
            result = await run_cell(ws, backend, schedules[profile], trigger, horizon=args.horizon,
                                       control_hz=args.control_hz, duration_s=args.duration,
                                       warmup_s=args.warmup, seed=42, session=f"level1-{r}-{label}",
                                       phase_s=phase_for(r, label, 1.0 / args.control_hz))
        finally:
            await ws.close()
        load_after = os.getloadavg()
        s = result.summary
        print(f"repeat {r} {label:26s} req={s['requests']['n']:3d} skipped_trig={s['requests']['skipped_triggers']:3d} "
              f"e2e p50={s['e2e_ms'].get('p50', float('nan')):6.1f} p95={s['e2e_ms'].get('p95', float('nan')):6.1f} "
              f"action_age p50={s['action_age_ms'].get('p50', float('nan')):6.1f} p95={s['action_age_ms'].get('p95', float('nan')):6.1f} "
              f"underrun={s['underrun']['ticks']:3d}/{s['ticks']} ({time.time() - t:.0f}s)", flush=True)
        out["cells"].append({"repeat": r, "label": label, "profile": profile, "config": result.config,
                             "host_loadavg_1m": [load_before[0], load_after[0]],
                             "summary": result.summary, "requests": result.requests, "ticks": result.ticks,
                             "skipped_triggers": result.skipped_triggers})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out))
    print(f"wrote {args.out}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", choices=tuple(backends.ALL), required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--horizon", type=int, default=15, help="action chunk length the server returns (pi05_droid 15)")
    p.add_argument("--control-hz", type=float, default=15.0)
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--repeat-index", type=int, default=0, help="rotates the cell order by 3 positions per index")
    p.add_argument("--seed", type=int, default=7, help="seed of the stream jitter schedule")
    p.add_argument("--warmup-requests", type=int, default=10)
    p.add_argument("--cells", nargs="+", default=None, help="cell labels to run (default: all)")
    args = p.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
