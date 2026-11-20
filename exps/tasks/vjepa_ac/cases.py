#!/usr/bin/env python3
"""V-JEPA 2-AC action-conditioned rollout on DROID: M* against Meta FAIR native.

The baseline, and why it changed
--------------------------------
Round 1 compared M* against an *eager PyTorch run of M*'s own component
modules*. That answers "what does M*'s engine add on top of M*'s own
components", which is a useful internal question and the wrong external one:
the paper's baseline is Meta FAIR's upstream implementation. This round uses
`external/vjepa2`'s `vjepa2_ac_vit_giant` with the official AC checkpoint, so
the ratio is comparable to the published one.

The horizon clamp
-----------------
`VJepa2ACModel` ships `max_rollout_horizon = 16`, and a larger requested
horizon is clamped rather than rejected. Round 2's H=30 cells therefore ran 30
steps' worth of request and 16 steps' worth of work, and nothing in the
response said so. Two defences here: the archived patch
`mstar-01-vjepa-max-rollout-horizon-32.diff` raises the cap to 32, and every
repeat asserts that the realised step count equals the requested horizon,
voiding the cell when it does not.

32 is the ceiling, not a round number: the AC predictor's attention mask is
sized by `grid_depth`, which is 32 for this architecture. H={4,15,30} all fit.

Timing boundaries differ, and that is the paper's own pairing
-------------------------------------------------------------
M* is timed over HTTP; the Meta baseline is timed in process. That asymmetry
is inherited from the paper, whose baseline is likewise a synchronous
in-process script -- so the ratio is comparable to the published one, but it is
not a byte-identical transport boundary. The M* side additionally carries
multipart upload, NDJSON base64 framing and loopback transfer. Both boundaries
are written into every summary, and a transport probe bounds the serialisation
term instead of leaving a reader to guess at it.
"""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

#: The paper's horizons. H=30 requires the max_rollout_horizon patch.
HORIZONS = (4, 15, 30)

#: Upstream's published DROID training config uses 8 frames per clip
#: (configs/train/vitg16/droid-256px-8f.yaml: dataset_fpcs: 8). Both sides use
#: the same number, or the encoder workloads would differ.
NUM_VIDEO_FRAMES = 8

#: One rollout step is a [1, 1, 256, 1408] float32 latent. Dividing the
#: returned video-modality byte count by this recovers the step count the
#: server actually ran, which is how the clamp is detected over HTTP.
BYTES_PER_ROLLOUT_STEP = 256 * 1408 * 4


TASK = TaskSpec(
    name="vjepa_ac",
    title="V-JEPA 2-AC rollout on DROID",
    frameworks=("mstar", "meta-native"),
    cases=tuple(
        Case(
            name=f"h{horizon}",
            label=f"action-conditioned rollout, horizon {horizon}",
            params={"horizon": horizon, "num_video_frames": NUM_VIDEO_FRAMES},
            profiles=("nightly", "full"),
        )
        for horizon in HORIZONS
    ),
    runner_module="exps.tasks.vjepa_ac.cell",
    comparison_label="paper_aligned_jct",
    timing_boundary=(
        "M*: client POST through the complete NDJSON response body, including multipart upload, "
        "base64 framing and loopback transfer. Meta FAIR native: in-process encoder forward plus "
        "H predictor steps with explicit CUDA synchronisation. This is the paper's own pairing."
    ),
    notes=(
        "Requires patch mstar-01-vjepa-max-rollout-horizon-32 for H=30. Every repeat asserts the "
        "number of predictor outputs equals the requested horizon; spatial tokens per output are "
        "not additional rollout steps."
    ),
)
