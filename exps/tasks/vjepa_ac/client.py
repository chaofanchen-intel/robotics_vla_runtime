#!/usr/bin/env python3
"""Measured client for the M* side of V-JEPA 2-AC.

Requests are built by M*'s own `DROIDDataset`, so the clip selection, frame
count, action and state tensors are exactly what the paper harness sends. The
loop is ours, because the per-request cooldown gate has to run between
requests and `benchmark.runner` exposes no hook for that.

Step-count assertion
--------------------
The server streams rollout latents as the `video` modality. Each step is a
[1, 1, 256, 1408] float32 tensor, so the byte count divides exactly by the
per-step size and recovers how many steps actually ran. A horizon that was
silently clamped shows up here as a mismatch, and the cell is voided rather
than summarised -- which is precisely what round 2 could not do, because it
never looked.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exps.core import adapters, benchmark_client, paths
from exps.core.codecs import ContractViolation
from exps.tasks.vjepa_ac.cases import BYTES_PER_ROLLOUT_STEP, NUM_VIDEO_FRAMES
from exps.tasks.vjepa_ac import frozen_inputs
from exps.tasks.vjepa_ac.input_fingerprint import source_input_sha256

#: DROID episode videos are fetched from the shared HF cache, never re-downloaded.
DROID_LOCAL_CACHE = paths.INPUTS / "VJ-AC" / "droid_local_cache"


def build_requests(horizon: int, count: int, num_video_frames: int) -> list[Any]:
    """Construct DROID V2V requests through M*'s own dataset class.

    `num_requests` is passed as the full count so the dataset selects the same
    leading episodes both engines use; requests are then consumed positionally.
    """
    dataset_module = adapters.load_benchmark_module("benchmark.dataset")
    DROID_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    dataset = dataset_module.DROIDDataset(
        local_file_dir=str(DROID_LOCAL_CACHE),
        num_requests=count,
        task="vjepa2_ac",
        rollout_horizon=horizon,
        cache_dir=str(paths.MODELS_CACHE),
        num_video_frames=num_video_frames,
    )
    requests = list(dataset.get_requests())[:count]
    return frozen_inputs.freeze(requests, horizon=horizon, num_video_frames=num_video_frames)


def make_validator(horizon: int, fingerprints: list[str]):
    """Assert the realised rollout length matches the requested horizon."""

    def validate(result, row: dict[str, Any], index: int) -> dict[str, Any]:
        video_bytes = (result.output_bytes or {}).get("video", 0)
        if not video_bytes:
            raise ContractViolation(
                "the response carried no rollout latents, so no horizon can be verified"
            )
        if video_bytes % BYTES_PER_ROLLOUT_STEP:
            raise ContractViolation(
                f"rollout latents are {video_bytes} bytes, not a whole number of "
                f"{BYTES_PER_ROLLOUT_STEP}-byte steps; the output shape is not what this "
                "assertion assumes and the step count cannot be trusted"
            )
        observed = video_bytes // BYTES_PER_ROLLOUT_STEP
        if observed != horizon:
            raise ContractViolation(
                f"requested horizon {horizon} but the server ran {observed} rollout steps. "
                "This is the max_rollout_horizon clamp: apply "
                "exps/patches/mstar-01-vjepa-max-rollout-horizon-32.diff and rerun. "
                "Round 2 reported clamped cells as H=30 results because nothing checked."
            )
        return {
            "rollout_steps": observed,
            "requested_horizon": horizon,
            "rollout_latent_bytes": video_bytes,
            "step_count_verified": True,
            "source_input_sha256": fingerprints[index],
        }

    return validate


def main() -> int:
    parser = benchmark_client.build_parser(__doc__)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--num-video-frames", type=int, default=NUM_VIDEO_FRAMES)
    args = parser.parse_args()
    if not 1 <= args.horizon < 32:
        parser.error("--horizon must be in [1, 31]: the AC predictor's grid_depth is 32")

    needed = (args.preconditions or 1) + (args.measured or 5)
    request_inputs = build_requests(args.horizon, needed, args.num_video_frames)
    fingerprints = [
        source_input_sha256(request, horizon=args.horizon, num_video_frames=args.num_video_frames)
        for request in request_inputs
    ]

    return benchmark_client.run(
        args=args,
        task="vjepa_ac",
        model_type="vjepa2ac",
        request_inputs=request_inputs,
        validate=make_validator(args.horizon, fingerprints),
        context={
            "horizon": args.horizon,
            "num_video_frames": args.num_video_frames,
            "fixture": "lerobot/droid_100 leading episodes, via M*'s own DROIDDataset",
            "source_input_fingerprint": "SHA256(video file contents, horizon, frame count, float32 actions and states); not a decoded-tensor or quality check",
            "dataset_cache": paths.relative_to_repo(paths.MODELS_CACHE),
            "step_count_assertion": (
                f"rollout latent bytes / {BYTES_PER_ROLLOUT_STEP} must equal the requested "
                "horizon; a mismatch voids the cell"
            ),
            "boundary_note": (
                "M* is timed over HTTP while the Meta FAIR baseline is timed in process. That is "
                "the paper's own pairing, so the ratio is comparable to the published one, but it "
                "is not a byte-identical transport boundary."
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
