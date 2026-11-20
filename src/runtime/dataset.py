"""lerobot/droid_100 as a Level 1 input: episodes (state, three cameras, task), state normalisation and interpolation.

droid_100 is real DROID teleoperation (Franka arm, three cameras, a language task per episode) stored at one rate, 15 Hz.
The multi-rate replay built on top of it is in observation.py. AV1 videos are decoded once and cached under data/.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
DATASET = REPO / "models" / "cache" / "datasets--lerobot--droid_100" / "snapshots" / "87301a2d2e99340e2010c9ef0f1d8e780b08aaf9"
FIXTURE = REPO / "exps" / "inputs" / "PI05" / "fixture_46row" / "fixture.json"
CACHE = REPO / "data" / "droid_100_cache"
NATIVE_HZ = 15.0
# Pi0.5 camera slot -> droid_100 video key (same mapping as the 46-row fixture)
CAMERA_SOURCE = {
    "base_0_rgb": "observation.images.exterior_image_1_left",
    "left_wrist_0_rgb": "observation.images.wrist_image_left",
    "right_wrist_0_rgb": "observation.images.exterior_image_2_left",
}
# first eight episodes that have a task label (episode_index of the fixture); about 100 seconds of playback
DEFAULT_EPISODES = (0, 1, 2, 3, 4, 6, 10, 11)

@dataclass
class Episode:
    index: int
    task: str
    state: np.ndarray                 # (n, 7) float32, raw
    frames: dict[str, np.ndarray]     # slot -> (n, 180, 320, 3) uint8

    @property
    def n(self) -> int:
        return int(self.state.shape[0])


def _state_normalization() -> tuple[np.ndarray, np.ndarray]:
    manifest = json.loads(FIXTURE.read_text())["state_normalization"]
    return np.asarray(manifest["state_q01"], dtype=np.float32), np.asarray(manifest["state_q99"], dtype=np.float32)


def normalize_state(raw7: np.ndarray) -> np.ndarray:
    """Same mapping as the fixture: per-dimension q01/q99 to [-1, 1], clipped, zero padded to 32."""
    q01, q99 = _state_normalization()
    out = np.zeros(32, dtype=np.float32)
    out[:7] = np.clip(2.0 * (raw7 - q01) / (q99 - q01) - 1.0, -1.0, 1.0)
    return out


def _decode_frames(video_key: str, needed: set[int]) -> dict[int, np.ndarray]:
    import av

    path = DATASET / "videos" / video_key / "chunk-000" / "file-000.mp4"
    last = max(needed)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i in needed:
                frames[i] = frame.to_ndarray(format="rgb24")
            if i >= last:
                break
    missing = needed - frames.keys()
    if missing:
        raise RuntimeError(f"{video_key}: {len(missing)} frames missing, first {min(missing)}")
    return frames


def load_episodes(episode_ids: tuple[int, ...] = DEFAULT_EPISODES) -> list[Episode]:
    import pyarrow.parquet as pq

    table = pq.read_table(DATASET / "data" / "chunk-000" / "file-000.parquet",
                          columns=["episode_index", "index", "observation.state", "task_index"]).to_pandas()
    tasks = pq.read_table(DATASET / "meta" / "tasks.parquet").to_pandas()
    task_by_index = {int(v): str(k) for k, v in zip(tasks.index, tasks["task_index"])}
    CACHE.mkdir(parents=True, exist_ok=True)
    episodes: list[Episode] = []
    missing = [e for e in episode_ids if not (CACHE / f"ep{e:03d}.npz").exists()]
    if missing:
        needed: set[int] = set()
        for e in missing:
            needed.update(int(i) for i in table.loc[table.episode_index == e, "index"])
        decoded = {slot: _decode_frames(key, needed) for slot, key in CAMERA_SOURCE.items()}
        for e in missing:
            idx = table.loc[table.episode_index == e, "index"].to_numpy()
            np.savez(CACHE / f"ep{e:03d}.npz", **{slot: np.stack([decoded[slot][int(i)] for i in idx]) for slot in CAMERA_SOURCE})
    for e in episode_ids:
        rows = table[table.episode_index == e].sort_values("index")
        data = np.load(CACHE / f"ep{e:03d}.npz")
        task = task_by_index[int(rows["task_index"].iloc[0])]
        if not task:
            raise ValueError(f"episode {e} has no task label")
        episodes.append(Episode(e, task, np.stack(rows["observation.state"].to_numpy()).astype(np.float32),
                                {slot: data[slot] for slot in CAMERA_SOURCE}))
    return episodes


def interpolate_state(state: np.ndarray, times: np.ndarray) -> np.ndarray:
    """State at `times` (seconds from episode start) from the 15 Hz samples. Dims 3..5 are Euler angles: unwrapped
    before interpolation and wrapped back to [-pi, pi]; the gripper (dim 6) is interpolated linearly."""
    native_t = np.arange(state.shape[0]) / NATIVE_HZ
    out = np.empty((len(times), state.shape[1]), dtype=np.float32)
    for d in range(state.shape[1]):
        col = np.unwrap(state[:, d]) if d in (3, 4, 5) else state[:, d]
        v = np.interp(times, native_t, col)
        out[:, d] = (v + math.pi) % (2 * math.pi) - math.pi if d in (3, 4, 5) else v
    return out
