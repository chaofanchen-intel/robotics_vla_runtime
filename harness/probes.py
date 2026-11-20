#!/usr/bin/env python3
"""Client-side probes for Pi0.5 over the OpenPI WebSocket (M*, vLLM-Omni, SGLang core).

Run against an already running server; `run_all.py` starts one server per framework and calls this file once per mode.

Modes
-----
e2e          Equal-work comparison. One connection, fixture rows in order, three rounds of 46. For M* the same
             server is also driven over its HTTP /generate path, interleaved block by block, so the transport
             effect is measured inside one service start.
period       Open-loop sends every T ms on one connection (the external Runtime's trigger), no waiting for replies.
             Shows whether the framework keeps up with the period and how the same-connection queue grows.
burst        K requests sent back-to-back on one connection: the same-connection queue, measured directly.
cancel       Does closing a connection stop its in-flight inference? Connection A starts a request, is closed
             after a fraction of the typical latency, and connection B (already open) then sends one request.
             If B's latency is about the control value, A's work was dropped. If it is about control plus the
             remaining time of A, the GPU finished A first.
concurrency  N connections each in a closed loop (send, wait, send). Per-request latency and aggregate throughput.

Timing boundary (same as exps/tasks/pi05/transport.py): from just before `send` to completion of `recv`.
Msgpack packing is done before the timer in e2e, cancel and concurrency; in burst and period it sits inside the timer
(measured at about 0.07 ms per request, so negligible). The server's own `server_timing.infer_ms` is kept when present.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import websockets  # noqa: E402
from websockets.asyncio.client import connect  # noqa: E402

from exps.tasks.pi05 import transport  # noqa: E402

FIXTURE = REPO / "exps" / "inputs" / "PI05" / "fixture_46row" / "fixture.json"
STEPS = 10
SEED = 42


def endpoint(framework: str, base: str) -> str:
    base = base.rstrip("/")
    if framework == "vllm-omni":
        return base + "/v1/realtime/robot/openpi"
    if framework == "sglang-core":
        return base + "/openpi/policy"
    if framework in ("mstar", "mstar-rust"):
        return base + "/"
    raise ValueError(framework)


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "n": len(ordered),
        "p50": statistics.median(ordered),
        "p95": pct(95),
        "p99": pct(99),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


class Fixture:
    def __init__(self, framework: str):
        self.framework = framework
        self.manifest, self.examples = transport._load_fixture(FIXTURE)  # noqa: SLF001
        self.images = [transport._load_images(FIXTURE, ex) for ex in self.examples]  # noqa: SLF001

    def packed(self, row: int, session: str) -> bytes:
        example = self.examples[row % len(self.examples)]
        images = self.images[row % len(self.examples)]
        if self.framework == "sglang-core":
            payload = transport._sglang_payload(example, images, STEPS)  # noqa: SLF001
        else:
            payload = transport._vllm_payload(example, images, session, STEPS, SEED)  # noqa: SLF001
        return transport._pack_msgpack(payload)  # noqa: SLF001


def parse_reply(raw: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    response = transport._unpack_msgpack(raw)  # noqa: SLF001
    if isinstance(response, dict) and response.get("type") == "error":
        raise RuntimeError(str(response.get("message", response)))
    action = np.asarray(transport._extract_action(response), dtype=np.float32)  # noqa: SLF001
    if action.size != 50 * 32:
        raise RuntimeError(f"action has {action.size} values, expected 50x32")
    timing = response.get("server_timing") if isinstance(response, dict) else None
    return action, timing or {}


async def open_ws(framework: str, base: str):
    ws = await connect(
        endpoint(framework, base), max_size=None, compression=None, ping_interval=None,
        open_timeout=300, close_timeout=10,
    )
    meta = transport._unpack_msgpack(await ws.recv())  # noqa: SLF001
    return ws, meta


async def one_request(ws, packed: bytes, timeout: float = 600.0) -> dict[str, Any]:
    t0 = time.perf_counter_ns()
    await ws.send(packed)
    raw = await asyncio.wait_for(ws.recv(), timeout)
    t1 = time.perf_counter_ns()
    _action, timing = parse_reply(raw)
    return {
        "e2e_ms": (t1 - t0) / 1e6,
        "infer_ms": timing.get("infer_ms"),
        "prev_total_ms": timing.get("prev_total_ms"),
    }


# ----------------------------------------------------------------------
# modes
# ----------------------------------------------------------------------

async def mode_e2e(args, fx: Fixture) -> dict[str, Any]:
    ws, meta = await open_ws(args.framework, args.url)
    session = f"probe-{args.framework}-{int(time.time())}"
    rows: list[dict[str, Any]] = []
    # warm-up, not recorded
    for w in range(2):
        await one_request(ws, fx.packed(w, session))

    http_session = None
    if args.framework.startswith("mstar") and args.http:
        import requests

        http_session = requests.Session()
        for w in range(1):
            transport._mstar_request(http_session, args.http, fx.examples[w], fx.images[w], SEED)  # noqa: SLF001

    counter = 0
    for rnd in range(args.rounds):
        for row in range(len(fx.examples)):
            r = await one_request(ws, fx.packed(counter, session))
            rows.append({"transport": "ws", "round": rnd, "row": row, **r})
            counter += 1
        if http_session is not None:
            for row in range(len(fx.examples)):
                example, images = fx.examples[row], fx.images[row]
                _a, e2e, status, _s, _e = await asyncio.to_thread(
                    transport._mstar_request, http_session, args.http, example, images, SEED  # noqa: SLF001
                )
                rows.append({"transport": "http", "round": rnd, "row": row, "e2e_ms": e2e, "status": status})
    await ws.close()
    if http_session is not None:
        http_session.close()

    summary = {}
    for tr in sorted({r["transport"] for r in rows}):
        sel = [r for r in rows if r["transport"] == tr]
        summary[tr] = {"e2e_ms": stats([r["e2e_ms"] for r in sel])}
        infer = [r["infer_ms"] for r in sel if r.get("infer_ms") is not None]
        if infer:
            summary[tr]["server_infer_ms"] = stats(infer)
    return {"metadata": {k: str(v) for k, v in meta.items()}, "rows": rows, "summary": summary}


async def mode_period(args, fx: Fixture) -> dict[str, Any]:
    ws, _meta = await open_ws(args.framework, args.url)
    session = f"probe-period-{int(time.time())}"
    results = []
    for period_ms in args.periods:
        n = args.requests
        sent: list[float] = [0.0] * n
        recv: list[float] = [0.0] * n
        got = 0
        done = asyncio.Event()

        async def receiver():
            nonlocal got
            while got < n:
                raw = await ws.recv()
                recv[got] = time.perf_counter()
                parse_reply(raw)
                got += 1
            done.set()

        rtask = asyncio.create_task(receiver())
        start = time.perf_counter()
        for k in range(n):
            target = start + k * period_ms / 1000.0
            now = time.perf_counter()
            if target > now:
                await asyncio.sleep(target - now)
            sent[k] = time.perf_counter()
            await ws.send(fx.packed(k, session))
        await asyncio.wait_for(done.wait(), 900)
        rtask.cancel()
        lat = [(recv[k] - sent[k]) * 1000 for k in range(n)]
        span = recv[-1] - sent[0]
        results.append({
            "period_ms": period_ms,
            "requests": n,
            "latency_ms": stats(lat),
            "achieved_rps": n / span if span > 0 else None,
            "drain_after_last_send_ms": (recv[-1] - sent[-1]) * 1000,
            "latency_ms_per_request": lat,
        })
        print(f"period {period_ms} ms: p50 {stats(lat)['p50']:.1f} p95 {stats(lat)['p95']:.1f} "
              f"drain {results[-1]['drain_after_last_send_ms']:.1f} ms", flush=True)
    await ws.close()
    return {"results": results}


async def mode_burst(args, fx: Fixture) -> dict[str, Any]:
    ws, _meta = await open_ws(args.framework, args.url)
    session = f"probe-burst-{int(time.time())}"
    for w in range(2):
        await one_request(ws, fx.packed(w, session))
    bursts = []
    for b in range(args.bursts):
        k = args.burst_size
        t0 = time.perf_counter()
        for i in range(k):
            await ws.send(fx.packed(i, session))
        lat = []
        for i in range(k):
            raw = await ws.recv()
            lat.append((time.perf_counter() - t0) * 1000)
            parse_reply(raw)
        bursts.append({"burst": b, "completion_ms": lat})
        print(f"burst {b}: completions ms {[round(x) for x in lat]}", flush=True)
    await ws.close()
    return {"bursts": bursts}


async def mode_cancel(args, fx: Fixture) -> dict[str, Any]:
    session = f"probe-cancel-{int(time.time())}"
    b, _ = await open_ws(args.framework, args.url)
    for w in range(2):
        await one_request(b, fx.packed(w, session))
    control = []
    for i in range(args.trials):
        control.append((await one_request(b, fx.packed(i, session)))["e2e_ms"])
    gap_s = 0.4 * statistics.median(control) / 1000.0
    arms: dict[str, list[float]] = {"control": control, "a_closed": [], "a_open": []}
    b_offset_ms: dict[str, list[float]] = {"a_closed": [], "a_open": []}
    for i in range(args.trials):
        for arm in ("a_closed", "a_open"):
            a, _ = await open_ws(args.framework, args.url)
            await a.send(fx.packed(i, session))
            t_a = time.perf_counter()
            await asyncio.sleep(gap_s)
            close_task = None
            if arm == "a_closed":
                # The close handshake is started but not awaited before B is sent. A server that answers the close
                # frame only after its in-flight work (the Rust frontend's receive loop does) would otherwise hold
                # B back until A had finished, which is not what this probe is for.
                close_task = asyncio.ensure_future(a.close())
            # B is sent at the same offset from A's send in both arms
            b_offset_ms[arm].append((time.perf_counter() - t_a) * 1000)
            r = await one_request(b, fx.packed(i + 1, session))
            arms[arm].append(r["e2e_ms"])
            if close_task is not None:
                await close_task
            else:
                # drain A's reply so the socket is clean before closing it
                try:
                    await asyncio.wait_for(a.recv(), 900)
                except Exception:  # noqa: BLE001
                    pass
                await a.close()
        print(f"trial {i}: control {control[i]:.0f} a_closed {arms['a_closed'][-1]:.0f} "
              f"a_open {arms['a_open'][-1]:.0f} ms", flush=True)
    await b.close()
    return {"gap_after_a_send_ms": gap_s * 1000, "arms_ms": arms,
            "b_send_offset_ms": {k: stats(v) for k, v in b_offset_ms.items()},
            "summary": {k: stats(v) for k, v in arms.items()}}


async def mode_concurrency(args, fx: Fixture) -> dict[str, Any]:
    out = []
    for n_conn in args.connections:
        conns = []
        for _ in range(n_conn):
            conns.append(await open_ws(args.framework, args.url))
        lat_all: list[float] = []

        async def loop(idx: int, ws):
            session = f"probe-conc-{n_conn}-{idx}-{int(time.time())}"
            for k in range(args.requests):
                r = await one_request(ws[0], fx.packed(idx * args.requests + k, session))
                lat_all.append(r["e2e_ms"])

        for ws in conns:
            for w in range(1):
                await one_request(ws[0], fx.packed(w, "warm"))
        lat_all.clear()
        t0 = time.perf_counter()
        await asyncio.gather(*(loop(i, ws) for i, ws in enumerate(conns)))
        wall = time.perf_counter() - t0
        out.append({"connections": n_conn, "requests": len(lat_all), "wall_s": wall,
                    "throughput_rps": len(lat_all) / wall, "latency_ms": stats(lat_all)})
        print(f"connections {n_conn}: throughput {len(lat_all) / wall:.2f} rps, p50 {stats(lat_all)['p50']:.0f} ms",
              flush=True)
        for ws in conns:
            await ws[0].close()
    return {"results": out}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", choices=("mstar", "mstar-rust", "vllm-omni", "sglang-core"), required=True)
    p.add_argument("--url", required=True, help="base URL, ws://127.0.0.1:PORT")
    p.add_argument("--http", default=None, help="M* HTTP base URL, enables the HTTP arm in e2e")
    p.add_argument("--mode", choices=("e2e", "period", "burst", "cancel", "concurrency"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--periods", type=float, nargs="+", default=[50, 75, 100, 150, 200, 300])
    p.add_argument("--requests", type=int, default=40)
    p.add_argument("--bursts", type=int, default=5)
    p.add_argument("--burst-size", type=int, default=8)
    p.add_argument("--trials", type=int, default=15)
    p.add_argument("--connections", type=int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()
    fx = Fixture(args.framework)
    fn = {"e2e": mode_e2e, "period": mode_period, "burst": mode_burst,
          "cancel": mode_cancel, "concurrency": mode_concurrency}[args.mode]
    t0 = time.time()
    result = asyncio.run(fn(args, fx))
    result["meta"] = {"framework": args.framework, "mode": args.mode, "url": args.url,
                      "steps": STEPS, "seed": SEED, "fixture": str(FIXTURE.relative_to(REPO)),
                      "started_unix": t0, "wall_s": time.time() - t0,
                      "websockets": websockets.__version__}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=float))
    print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
