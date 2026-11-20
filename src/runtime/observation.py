"""Multi-rate observation streams and the "latest sample per stream" snapshot (the Observation Manager of Level 1).

A real robot delivers its inputs at different rates, so the replay of droid_100 gives each input its own schedule:

    state      100 Hz   joint/pose state, linearly interpolated between the 15 Hz samples (a stand-in for a
                        high-rate proprioception topic; the Euler angles are unwrapped before interpolating)
    ext1       15 Hz    exterior camera 1, native frames
    wrist      15 Hz    wrist camera, native frames, phase-shifted against ext1
    ext2       7.5 Hz   exterior camera 2, every second native frame (a slower camera)
    prompt     event    the language task, one event per episode

Each sample has a capture time (when the scene was true) and an arrival time (capture + a fixed latency + a seeded
jitter, kept monotonic per stream). The latencies and jitters are assumptions of this simulation, not measurements of a
robot. The same schedule (same seed) is used for every framework, so the framework is the only difference between runs.

`Schedule.snapshot(t)` is the "latest sample with arrival <= t, per stream" read of message_filters' Cache, and reports
each stream's age at t. The `lockstep` profile samples all streams from one 15 Hz clock with no latency or jitter,
which is what a dataset row (or a LeRobot observation with one timestamp) looks like.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from runtime.dataset import NATIVE_HZ, Episode, interpolate_state, normalize_state


@dataclass(frozen=True)
class StreamSpec:
    name: str            # state | ext1 | wrist | ext2 | prompt
    kind: str            # state | camera | prompt
    rate_hz: float
    phase_ms: float      # capture-clock offset against the episode start
    latency_ms: float    # capture -> available to the runtime, constant part (assumption)
    jitter_ms: float     # uniform [0, jitter] added to the latency (assumption)
    slot: str | None = None   # Pi0.5 camera slot for camera streams


PROFILES: dict[str, list[StreamSpec]] = {
    "multirate": [
        StreamSpec("state", "state", 100.0, 0.0, 2.0, 1.0),
        StreamSpec("ext1", "camera", 15.0, 0.0, 35.0, 5.0, "base_0_rgb"),
        StreamSpec("wrist", "camera", 15.0, 22.0, 25.0, 5.0, "left_wrist_0_rgb"),
        StreamSpec("ext2", "camera", 7.5, 40.0, 45.0, 8.0, "right_wrist_0_rgb"),
        StreamSpec("prompt", "prompt", 0.0, 0.0, 5.0, 0.0),
    ],
    "lockstep": [
        StreamSpec("state", "state", 15.0, 0.0, 0.0, 0.0),
        StreamSpec("ext1", "camera", 15.0, 0.0, 0.0, 0.0, "base_0_rgb"),
        StreamSpec("wrist", "camera", 15.0, 0.0, 0.0, 0.0, "left_wrist_0_rgb"),
        StreamSpec("ext2", "camera", 15.0, 0.0, 0.0, 0.0, "right_wrist_0_rgb"),
        StreamSpec("prompt", "prompt", 0.0, 0.0, 0.0, 0.0),
    ],
}
STREAM_NAMES = ("state", "ext1", "wrist", "ext2", "prompt")
AGE_STREAMS = ("state", "ext1", "wrist", "ext2")   # the prompt is an event: its age is not a freshness measure


class _Stream:
    """One input stream: capture times, arrival times, and a way to get the value for sample i."""

    def __init__(self, spec: StreamSpec, capture: np.ndarray, arrival: np.ndarray, values):
        self.spec = spec
        self.capture = capture
        self.arrival = arrival
        self.values = values   # callable i -> value


class Schedule:
    """The replayed streams over a long enough timeline (episodes repeat), built once and read at any time t."""

    def __init__(self, episodes: list[Episode], profile: str, seed: int, horizon_s: float):
        self.profile = profile
        self.episodes = episodes
        specs = {s.name: s for s in PROFILES[profile]}
        # episode placement on the timeline, repeating the list until `horizon_s` is covered
        placed: list[tuple[float, Episode]] = []
        t = 0.0
        while t < horizon_s:
            for ep in episodes:
                placed.append((t, ep))
                t += ep.n / NATIVE_HZ
        self.placed = placed
        self.length_s = t
        self.streams: dict[str, _Stream] = {}
        rng = {name: np.random.default_rng([seed, i]) for i, name in enumerate(STREAM_NAMES)}

        def arrivals(spec: StreamSpec, capture: np.ndarray, name: str) -> np.ndarray:
            arr = capture + (spec.latency_ms + rng[name].uniform(0.0, spec.jitter_ms, len(capture))) / 1000.0
            return np.maximum.accumulate(arr + np.arange(len(arr)) * 1e-9)

        # state: 100 Hz (or 15 Hz in lockstep) samples within each episode
        spec = specs["state"]
        cap_list, val_list = [], []
        for t0, ep in placed:
            # up to the episode's end, so there is no gap at the boundary; past the last native sample the value is held
            end = ep.n / NATIVE_HZ - 1e-6
            local = np.arange(0.0, end, 1.0 / spec.rate_hz) + spec.phase_ms / 1000.0
            local = local[local <= end]
            cap_list.append(t0 + local)
            val_list.append(np.stack([normalize_state(r) for r in interpolate_state(ep.state, local)]))
        capture = np.concatenate(cap_list)
        values = np.concatenate(val_list)
        self.streams["state"] = _Stream(spec, capture, arrivals(spec, capture, "state"), lambda i, v=values: v[i])
        # cameras: every `stride`-th native frame
        for name in ("ext1", "wrist", "ext2"):
            spec = specs[name]
            stride = int(round(NATIVE_HZ / spec.rate_hz))
            cap_list, img_refs = [], []
            for t0, ep in placed:
                ks = np.arange(0, ep.n, stride)
                cap_list.append(t0 + ks / NATIVE_HZ + spec.phase_ms / 1000.0)
                img_refs.extend((ep, int(k)) for k in ks)
            capture = np.concatenate(cap_list)
            self.streams[name] = _Stream(spec, capture, arrivals(spec, capture, name),
                                         lambda i, r=img_refs, s=spec.slot: r[i][0].frames[s][r[i][1]])
        # prompt: one event per episode
        spec = specs["prompt"]
        capture = np.asarray([t0 for t0, _ in placed])
        tasks = [ep.task for _, ep in placed]
        self.streams["prompt"] = _Stream(spec, capture, arrivals(spec, capture, "prompt"), lambda i, v=tasks: v[i])

    def first_ready_s(self) -> float:
        """Earliest time at which every stream has delivered at least one sample."""
        return max(float(s.arrival[0]) for s in self.streams.values())

    def snapshot(self, t: float) -> dict:
        """Latest sample per stream with arrival <= t. Returns {name: (value, capture_t, arrival_t, index)}."""
        out = {}
        for name, s in self.streams.items():
            i = int(np.searchsorted(s.arrival, t, side="right")) - 1
            if i < 0:
                raise RuntimeError(f"stream {name} has no sample before t={t:.3f}")
            out[name] = (s.values(i), float(s.capture[i]), float(s.arrival[i]), i)
        return out

    def describe(self) -> dict:
        d = {"profile": self.profile, "timeline_s": self.length_s, "episodes": [e.index for e in self.episodes]}
        for name, s in self.streams.items():
            gaps = np.diff(s.arrival) * 1000.0
            d[name] = {"rate_hz": s.spec.rate_hz, "samples": int(len(s.capture)), "latency_ms": s.spec.latency_ms,
                       "jitter_ms": s.spec.jitter_ms,
                       "arrival_gap_ms_p50": float(np.median(gaps)) if len(gaps) else None}
        return d


def snapshot_to_request(snap: dict) -> tuple[dict, dict]:
    """A snapshot as the framework-independent request content: ({state_normalized_32, task}, {camera slot: HxWx3 uint8})."""
    example = {"state_normalized_32": snap["state"][0], "task": snap["prompt"][0]}
    images = {s.slot: snap[s.name][0] for s in PROFILES["multirate"] if s.kind == "camera"}
    return example, images
