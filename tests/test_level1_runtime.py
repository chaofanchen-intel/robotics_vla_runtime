"""Checks of the Level 1 runtime against a fake OpenPI server with a fixed latency, so every expected value can be
worked out by hand. CPU only. Run:  envs/mstar/bin/python -m pytest tests/test_level1_runtime.py -q
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))            # exps/
sys.path.insert(0, str(REPO / "src"))   # runtime, backends, metrics, scenarios

import backends  # noqa: E402
from exps.tasks.pi05 import transport  # noqa: E402
from runtime import dataset, observation  # noqa: E402
from runtime.action_manager import ActionManager, Chunk  # noqa: E402
from runtime.controller import run_cell  # noqa: E402
from runtime.scheduler import Trigger  # noqa: E402
from websockets.asyncio.server import serve  # noqa: E402

BACKEND = backends.get("mstar-rust")

H = 15
DT_MS = 1000.0 / 15.0


class FakeServer:
    """OpenPI-shaped: metadata frame on connect, then one reply per request after `latency_s`, strictly serial."""

    def __init__(self, latency_s: float):
        self.latency_s = latency_s
        self.payloads: list[dict] = []

    async def handler(self, ws):
        await ws.send(BACKEND.pack({}))
        async for raw in ws:
            self.payloads.append(transport._unpack_msgpack(raw))  # noqa: SLF001
            await asyncio.sleep(self.latency_s)
            await ws.send(BACKEND.pack({
                "actions": np.zeros((H, 32), dtype=np.float32), "server_timing": {"infer_ms": self.latency_s * 1000.0}}))


@pytest.fixture(scope="module")
def schedules():
    episodes = dataset.load_episodes()
    return {p: observation.Schedule(episodes, p, 7, 40.0) for p in ("multirate", "lockstep")}


def run(latency_s: float, trigger: Trigger, schedules, profile: str = "multirate", duration: float = 8.0):
    async def go():
        server = FakeServer(latency_s)
        async with serve(server.handler, "127.0.0.1", 0, max_size=None, compression=None) as srv:
            port = srv.sockets[0].getsockname()[1]
            ws, _ = await BACKEND.connect(f"ws://127.0.0.1:{port}")
            try:
                result = await run_cell(ws, BACKEND, schedules[profile], trigger, horizon=H, duration_s=duration,
                                        warmup_s=1.0)
            finally:
                await ws.close()
        return result, server

    return asyncio.run(go())


def test_period_aligned_no_underrun(schedules):
    result, server = run(0.100, Trigger("period", period_s=0.200), schedules)
    s = result.summary
    assert 38 <= s["requests"]["n"] <= 42                      # 8 s / 0.2 s
    assert 95 <= s["e2e_ms"]["p50"] <= 125
    assert s["underrun"]["ticks"] == 0
    # actions from a chunk are executed between its arrival (about 100 ms after the snapshot) and the next one
    # (about 300 ms after): the executed action's age is 100..300 ms, centred at 200 ms
    assert 170 <= s["action_age_ms"]["p50"] <= 235
    assert s["skipped_actions_at_first_use"]["p50"] in (2.0, 3.0)
    # the request carries what the schedule says: three image slots, a 32-dim state, a prompt
    p = server.payloads[0]
    assert all(f"observation.images.{k}" in p for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))
    assert np.asarray(p["state"]).shape == (32,) and isinstance(p["prompt"], str) and p["prompt"]


def test_period_shorter_than_latency_skips_triggers(schedules):
    result, _ = run(0.150, Trigger("period", period_s=0.100), schedules)
    s = result.summary
    # one in flight at a time: a request can start every other period, 200 ms
    assert 34 <= s["requests"]["n"] <= 42
    assert s["requests"]["skipped_triggers"] >= 30


def test_latency_beyond_chunk_coverage_underruns_always(schedules):
    # the chunk covers 15 steps = 1.0 s; a reply that is 1.2 s old has no action left
    result, _ = run(1.2, Trigger("period", period_s=0.200), schedules, duration=10.0)
    s = result.summary
    assert s["underrun"]["fraction"] > 0.95
    assert s["action_age_ms"].get("n", 0) == 0 or s["underrun"]["fraction"] > 0.95


def test_threshold_replans_when_half_second_is_left(schedules):
    result, _ = run(0.100, Trigger("threshold", refill_s=0.5), schedules)
    s = result.summary
    # a request goes out when 7 actions (<= 0.5 s) are left, i.e. 8 steps after the snapshot, 533 ms; the control clock
    # runs 0.4 % slow against the action spacing, which can bring the "due" action one step earlier (7 ticks, 467 ms)
    assert 440 <= s["replan_interval_ms"]["p50"] <= 620
    assert s["underrun"]["ticks"] == 0


def test_naive_start_is_staler_than_aligned(schedules):
    aligned, _ = run(0.150, Trigger("open_loop", replan_steps=8, align=True), schedules)
    naive, _ = run(0.150, Trigger("open_loop", replan_steps=8, align=False), schedules)
    # The age since the snapshot is the same: each chunk runs from its arrival to the next chunk's arrival either way.
    # What alignment changes is which action runs: naive runs action i about one latency (150 ms) after the time it
    # was planned for, aligned runs it at that time (within one control step).
    a, n = aligned.summary["intent_lag_ms"]["p50"], naive.summary["intent_lag_ms"]["p50"]
    assert -DT_MS <= a <= DT_MS
    assert 100 <= n <= 230


def _chunk(gen: int) -> Chunk:
    return Chunk(gen, 0.0, 0.0, 0.0, np.zeros((H, 32)), {}, 0.0, None, 0.0)


def test_older_generation_is_dropped_and_newer_replaces():
    am = ActionManager(H, 1 / 15, 1 / 15, True)
    first, second = _chunk(2), _chunk(1)
    assert am.offer(first) and am.active is first
    assert not am.offer(second) and am.active is first and not second.applied      # older: dropped
    third = _chunk(3)
    assert am.offer(third) and am.active is third
    assert am.unused_superseded == 1          # chunk 2 was replaced by chunk 3 before it ran any action


def test_aligned_index_skips_elapsed_actions_and_underruns_past_the_horizon():
    am = ActionManager(H, 1 / 15, 1 / 15, True)
    c = _chunk(1)
    c.t_snap = 1.0
    am.offer(c)
    _, idx, remaining = am.locate(1.0 + 3 / 15)             # 3 control steps after the snapshot: action 3 is due
    assert (idx, remaining) == (3, H - 3)
    chunk, idx, remaining = am.locate(1.0 + (H + 2) / 15)   # beyond the last action
    assert idx == H + 2 and remaining == 0
    assert am.execute(chunk, idx, 0.0, 0.0, remaining)["underrun"]


def test_state_interpolation_is_continuous_across_the_angle_wrap():
    eps = dataset.load_episodes()
    st = eps[0].state
    t = np.arange(0.0, (eps[0].n - 1) / 15.0, 0.01)
    out = dataset.interpolate_state(st, t)
    # position dims move by well under a millimetre-scale step per 10 ms; angle dims may only jump by the 2*pi wrap
    assert np.abs(np.diff(out[:, [0, 1, 2]], axis=0)).max() < 0.01
    steps = np.abs(np.diff(out[:, 3:6], axis=0))
    assert ((steps < 0.05) | (np.abs(steps - 2 * np.pi) < 0.05)).all()


# ---- backend adapters --------------------------------------------------------------------------------------------------

def _request():
    snap = observation.Schedule(dataset.load_episodes(), "multirate", 7, 10.0).snapshot(2.0)
    return observation.snapshot_to_request(snap)


def test_endpoints_and_registry():
    assert backends.get("mstar-rust").endpoint("ws://h:1") == "ws://h:1/"
    assert backends.get("vllm-omni").endpoint("ws://h:1/") == "ws://h:1/v1/realtime/robot/openpi"
    assert backends.get("sglang-core").endpoint("ws://h:1") == "ws://h:1/openpi/policy"
    assert set(backends.ALL) == {"mstar", "mstar-rust", "vllm-omni", "sglang-core", "sglang-core-bucketed"}
    assert len({b.port for b in backends.ALL.values()}) == 5


def test_payload_layout_differs_between_sglang_and_the_others():
    example, images = _request()
    for name in ("mstar-rust", "vllm-omni"):
        p = backends.get(name).payload(example, images, "s", 10, 42)
        assert {"state", "prompt", "session_id", "seed", "endpoint"} <= p.keys()
    p = backends.get("sglang-core").payload(example, images, "s", 10, 42)
    assert {"observation.state", "task", "num_inference_steps"} <= p.keys() and "session_id" not in p
    for name in backends.ALL:
        p = backends.get(name).payload(example, images, "s", 10, 42)
        assert all(f"observation.images.{k}" in p for k in ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"))


def test_parse_reply_checks_chunk_length_and_error_frames():
    ok = BACKEND.pack({"actions": np.zeros((15, 32), np.float32), "server_timing": {"infer_ms": 5.0}})
    actions, timing = BACKEND.parse_reply(ok, 15)
    assert actions.shape == (15, 32) and timing["infer_ms"] == 5.0
    with pytest.raises(RuntimeError, match="expected 50x32"):
        BACKEND.parse_reply(ok, 50)
    with pytest.raises(RuntimeError, match="boom"):
        BACKEND.parse_reply(BACKEND.pack({"type": "error", "message": "boom"}), 15)


def test_serve_commands_follow_the_checkpoint():
    for name, b in backends.ALL.items():
        base = b.serve_command(backends.CHECKPOINTS["pi05_base"])
        droid = b.serve_command(backends.CHECKPOINTS["pi05_droid"])
        assert "droid" not in base and "droid" in droid, name
    bucketed = backends.get("sglang-core-bucketed").serve_command(backends.CHECKPOINTS["pi05_droid"])
    assert "sglang_pi05_droid_buckets.json" in bucketed
    import json
    assert json.load(open(REPO / "configs" / "sglang_pi05_droid_buckets.json"))["prompt_token_buckets"] == [200]
    assert json.load(open(REPO / "configs" / "sglang_pi05_base_buckets.json"))["prompt_token_buckets"] == [200]
    assert "--python-frontend" in backends.get("mstar").serve_command(backends.CHECKPOINTS["pi05_droid"])
    assert "--rust-frontend-bin" in backends.get("mstar-rust").serve_command(backends.CHECKPOINTS["pi05_droid"])


def test_open_loop_does_not_stall_when_a_slow_reply_leaves_too_few_actions(schedules):
    # 0.6 s latency = 9 control steps: aligned execution skips 9 of 15, so only 6 actions are left to run and the
    # "after 8 executed" count is never reached. The chunk running out must trigger the next request.
    result, _ = run(0.6, Trigger("open_loop", replan_steps=8), schedules, duration=10.0)
    assert result.summary["requests"]["n"] >= 8        # 10 s / (0.6 s + the remaining 6 steps) is about 8 requests
