#!/usr/bin/env python3
"""Meta FAIR upstream V-JEPA 2-AC baseline, under the same thermal protocol.

This is the baseline the paper compares against: Facebook Research's own
`vjepa2_ac_vit_giant` with the official AC checkpoint, run in process. Round 1
used an eager run of M*'s own component modules instead, which answers a
different question.

How upstream fidelity is preserved
----------------------------------
The two model-logic blocks below are reproduced from the notebook cells that
upstream publishes, and are fenced with the same BEGIN/END VERBATIM markers
`external/mstar/benchmark/vjepa2_repo_ac.py` uses, citing the same source. The
supporting helpers -- environment check, video decode, upstream commit capture
-- are **imported** from that scaffold rather than copied, so there is one
implementation of each and `external/mstar` stays unmodified.

Why the loop is not upstream's
------------------------------
Upstream's timed loop runs requests back to back with no hook between them. On
this chassis that is exactly the pattern that produces thermal drift, and it
would compare a gated M* against an ungated baseline -- measuring the protocol
rather than the engines. The loop here issues one request per cooldown gate,
identically to the served side. The timed region itself is unchanged: encoder
forward, H predictor steps, explicit CUDA synchronisation, nothing else.

Model load is outside every timed region, and warmups run before the first
gate, so the measured requests see a loaded, compiled, allocator-warm process.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exps.core import adapters, cli, metrics as metrics_mod, paths
from exps.core.codecs import ContractViolation
from exps.core.measure import run_measurement
from exps.tasks.vjepa_ac.cases import NUM_VIDEO_FRAMES
from exps.tasks.vjepa_ac import frozen_inputs
from exps.tasks.vjepa_ac.input_fingerprint import source_input_sha256

UPSTREAM_REPO = paths.EXTERNAL / "vjepa2"
#: Cached official AC checkpoint, linked into TORCH_HOME so torch.hub finds it
#: offline instead of pulling ~12 GB from FAIR S3.
CHECKPOINT = paths.MODELS_CACHE / "vjepa2-ac" / "vjepa2-ac-vitg.pt"
DROID_LOCAL_CACHE = paths.INPUTS / "VJ-AC" / "droid_local_cache"


def _verified_rollout_steps(
    predictions: Sequence[Any], horizon: int, tokens_per_frame: int
) -> int:
    if len(predictions) != horizon:
        raise ContractViolation(
            f"requested horizon {horizon} but the rollout produced {len(predictions)} steps"
        )
    for index, prediction in enumerate(predictions):
        shape = tuple(prediction.shape)
        if len(shape) != 3 or shape[1] != tokens_per_frame:
            raise ContractViolation(
                f"rollout step {index + 1} has shape {shape}; expected "
                f"(batch, {tokens_per_frame} spatial tokens, embedding)"
            )
    return len(predictions)


def link_checkpoint(torch_home: Path) -> Path:
    """Expose the cached checkpoint where `torch.hub` expects to find it."""
    target = torch_home / "hub" / "checkpoints" / CHECKPOINT.name
    if not CHECKPOINT.is_file():
        raise SystemExit(
            f"cached Meta AC checkpoint not found: {CHECKPOINT}\n"
            "This baseline runs offline; the checkpoint must already be in models/cache."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != CHECKPOINT.resolve():
            raise SystemExit(f"unexpected checkpoint link: {target} -> {target.resolve()}")
    elif target.exists():
        raise SystemExit(f"refusing to replace an existing torch hub checkpoint: {target}")
    else:
        target.symlink_to(CHECKPOINT)
    return target


def load_upstream_scaffold():
    """Import M*'s upstream baseline scaffold for its helpers and constants."""
    return adapters.load_benchmark_module("benchmark.vjepa2_repo_ac")


def precision_context(torch, dtype: str, device: str):
    if dtype == "float32":
        return nullcontext()
    return torch.autocast(device_type=device.split(":")[0], dtype=getattr(torch, dtype))


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, required=True)
    parser.add_argument("--num-video-frames", type=int, default=NUM_VIDEO_FRAMES)
    parser.add_argument("--warmups", type=int, default=3,
                        help="Run before the first cooldown gate, so measured requests see a "
                             "warm process without the warmup heat landing on them.")
    # The baseline is BF16 only: M*'s V-JEPA autocasts to bfloat16 (`get_autocast_dtype()` default), so a float32
    # Meta baseline answers a different question (precision) and is deliberately not offered.
    parser.add_argument("--dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--normalize-reps", type=int, default=1)
    parser.add_argument("--save-outputs", action="store_true", default=True)
    parser.add_argument("--no-save-outputs", dest="save_outputs", action="store_false")
    cli.add_client_protocol_arguments(parser)
    args = parser.parse_args()

    scaffold = load_upstream_scaffold()
    if not 1 <= args.horizon < scaffold.MAX_T:
        parser.error(
            f"--horizon must be in [1, {scaffold.MAX_T - 1}]: the AC predictor's attention mask "
            f"is sized for grid_depth={scaffold.MAX_T}"
        )

    config = cli.client_measurement_config(args, args.output_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch_home = output_dir / "torch_home"
    checkpoint_link = link_checkpoint(torch_home)
    import os

    os.environ.setdefault("TORCH_HOME", str(torch_home))

    scaffold._check_env_or_exit(str(UPSTREAM_REPO))  # noqa: SLF001 - upstream's own check
    upstream_commit = scaffold._capture_upstream_commit(str(UPSTREAM_REPO), None)  # noqa: SLF001

    import torch
    import torch.nn.functional as F
    from app.vjepa_droid.transforms import make_transforms
    from notebooks.utils.mpc_utils import compute_new_pose

    dataset_module = adapters.load_benchmark_module("benchmark.dataset")

    load_started = time.perf_counter()
    encoder, predictor = torch.hub.load(str(UPSTREAM_REPO), "vjepa2_ac_vit_giant", source="local")
    model_dtype = getattr(torch, args.dtype)
    encoder = encoder.to(device=args.device, dtype=model_dtype).eval()
    predictor = predictor.to(device=args.device, dtype=model_dtype).eval()
    if next(encoder.parameters()).dtype != model_dtype or next(predictor.parameters()).dtype != model_dtype:
        raise ContractViolation(f"model weights did not convert to {args.dtype}")
    tokens_per_frame = int((scaffold.CROP_SIZE // encoder.patch_size) ** 2)
    model_load_seconds = time.perf_counter() - load_started

    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=scaffold.CROP_SIZE,
    )

    # =====================================================================
    # === BEGIN VERBATIM COPY ===
    # Source: vjepa2/notebooks/energy_landscape_example.ipynb, Cell 5.
    # Reproduced byte-for-byte, matching external/mstar/benchmark/
    # vjepa2_repo_ac.py, so this baseline exercises upstream's published
    # pattern without modification.
    # =====================================================================
    normalize_reps = bool(args.normalize_reps)

    @torch.inference_mode()
    def forward_target(c, normalize_reps=normalize_reps):
        B, C, T, H, W = c.size()
        c = c.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = encoder(c)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        if normalize_reps:
            h = F.layer_norm(h, (h.size(-1),))
        return h

    @torch.inference_mode()
    def step_predictor(_z, _a, _s):
        _z = predictor(_z, _a, _s)[:, -tokens_per_frame:]
        if normalize_reps:
            _z = F.layer_norm(_z, (_z.size(-1),))
        _s = compute_new_pose(_s[:, -1:], _a[:, -1:])
        return _z, _s
    # === END VERBATIM COPY ===

    needed = config.precondition_requests + config.measured_requests
    DROID_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
    dataset = dataset_module.DROIDDataset(
        local_file_dir=str(DROID_LOCAL_CACHE),
        num_requests=needed,
        task="vjepa2_ac",
        rollout_horizon=args.horizon,
        cache_dir=str(paths.MODELS_CACHE),
        num_video_frames=args.num_video_frames,
    )
    requests = frozen_inputs.freeze(
        list(dataset.get_requests())[:needed], horizon=args.horizon, num_video_frames=args.num_video_frames
    )
    if len(requests) < needed:
        raise SystemExit(f"DROIDDataset returned {len(requests)} episodes; need {needed}")
    fingerprints = [
        source_input_sha256(request, horizon=args.horizon, num_video_frames=args.num_video_frames)
        for request in requests
    ]

    def prepare(req):
        clip = scaffold._load_video_clip(req.video_path, num_frames=args.num_video_frames)  # noqa: SLF001
        clip_t = transform(clip).unsqueeze(0).to(device=args.device, dtype=model_dtype, non_blocking=True)
        actions_t = torch.tensor(
            req.model_kwargs["actions"], dtype=torch.float32, device=args.device
        ).unsqueeze(0)
        states_t = torch.tensor(
            req.model_kwargs["states"], dtype=torch.float32, device=args.device
        ).unsqueeze(0)
        if actions_t.size(1) < args.horizon:
            raise ContractViolation(
                f"episode has only {actions_t.size(1)} actions; horizon {args.horizon} needs more"
            )
        return clip_t, actions_t, states_t

    # =====================================================================
    # === BEGIN VERBATIM COPY (driver pattern) ===
    # Source: energy_landscape_example.ipynb Cell 6 (encode) + Cell 5
    # `forward_actions` (rollout body). The only deviation from the notebook
    # is the per-step DROID action slice `actions_t[:, n+1:n+2]` in place of
    # a constant action, because this has a real action trajectory. The
    # growing-history concatenation is required by
    # vjepa2/src/models/ac_predictor.py: that forward derives T from
    # x.size(1) and concatenates [a, s, x] along dim=2.
    # =====================================================================
    def run_rollout(clip_t, actions_t, states_t):
        h = forward_target(clip_t)
        z_hat = h[:, :tokens_per_frame]
        s_hat = states_t[:, :1]
        a_hat = actions_t[:, :1]
        preds = []
        for n in range(args.horizon):
            _z, _s = step_predictor(z_hat, a_hat, s_hat)
            z_hat = torch.cat([z_hat, _z], dim=1)
            s_hat = torch.cat([s_hat, _s], dim=1)
            preds.append(_z)
            if n + 1 < args.horizon:
                a_hat = torch.cat([a_hat, actions_t[:, n + 1 : n + 2]], dim=1)
        return h, preds
    # === END VERBATIM COPY ===

    # Warmups: before the protocol's first gate, so the JIT and allocator cost
    # is paid without its heat landing on a measured request.
    precision_proof: dict[str, str] = {}
    handles = []
    if args.dtype != "float32":
        def capture_dtype(name):
            def record(_module, _inputs, output):
                precision_proof[name] = str(output.dtype)
            return record

        handles = [
            encoder.patch_embed.register_forward_hook(capture_dtype("encoder.patch_embed")),
            predictor.predictor_embed.register_forward_hook(capture_dtype("predictor.predictor_embed")),
        ]
    try:
        for _ in range(args.warmups):
            with precision_context(torch, args.dtype, args.device):
                run_rollout(*prepare(requests[0]))
    finally:
        for handle in handles:
            handle.remove()
    if args.dtype != "float32" and precision_proof != {
        "encoder.patch_embed": str(model_dtype),
        "predictor.predictor_embed": str(model_dtype),
    }:
        raise ContractViolation(f"reduced precision did not reach encoder and predictor matmuls: {precision_proof}")
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    def issue(index: int, phase: str) -> dict[str, Any]:
        offset = (index - 1) if phase == "precondition" else (config.precondition_requests + index - 1)
        clip_t, actions_t, states_t = prepare(requests[offset])

        with precision_context(torch, args.dtype, args.device):
            started = time.perf_counter()
            h = forward_target(clip_t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            encoder_done = time.perf_counter()

            z_hat = h[:, :tokens_per_frame]
            s_hat = states_t[:, :1]
            a_hat = actions_t[:, :1]
            preds = []
            for n in range(args.horizon):
                _z, _s = step_predictor(z_hat, a_hat, s_hat)
                z_hat = torch.cat([z_hat, _z], dim=1)
                s_hat = torch.cat([s_hat, _s], dim=1)
                preds.append(_z)
                if n + 1 < args.horizon:
                    a_hat = torch.cat([a_hat, actions_t[:, n + 1 : n + 2]], dim=1)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            finished = time.perf_counter()

        # Everything below is after the timer, exactly as on the served side.
        steps = _verified_rollout_steps(preds, args.horizon, tokens_per_frame)
        latents = torch.cat(preds, dim=1).float().cpu().numpy()
        if not bool(latents.size) or not bool((latents == latents).all()):
            raise ContractViolation("rollout latents contain NaN; the output is not usable")
        if args.save_outputs:
            import numpy as np

            target = output_dir / "outputs" / f"{phase}_{index:03d}"
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / "latents.npy", latents)

        row = metrics_mod.empty_request_metrics()
        row.update(
            {
                "e2e_ms": (finished - started) * 1000.0,
                "output_bytes": int(latents.nbytes),
                "success": True,
                "encoder_ms": (encoder_done - started) * 1000.0,
                "rollout_ms": (finished - encoder_done) * 1000.0,
                "rollout_steps": steps,
                "requested_horizon": args.horizon,
                "step_count_verified": True,
                "output_shape": list(latents.shape),
                "fixture_index": offset,
                "source_input_sha256": fingerprints[offset],
                "encoder_output_dtype": str(h.dtype),
                "predictor_output_dtype": str(preds[-1].dtype),
            }
        )
        return row

    summary = run_measurement(
        config,
        issue=issue,
        context={
            "task": "vjepa_ac",
            "framework": "meta-native",
            "case": args.case,
            "horizon": args.horizon,
            "num_video_frames": args.num_video_frames,
            "model": "Meta FAIR upstream vjepa2_ac_vit_giant",
            "upstream_repository": paths.relative_to_repo(UPSTREAM_REPO),
            "upstream_commit": upstream_commit,
            "checkpoint": paths.relative_to_repo(CHECKPOINT),
            "checkpoint_link": paths.relative_to_repo(checkpoint_link),
            "dtype": args.dtype,
            "model_parameter_dtype": str(next(encoder.parameters()).dtype),
            "precision_proof": precision_proof,
            "device": args.device,
            "model_load_seconds": model_load_seconds,
            "warmups_before_first_gate": args.warmups,
            "fixture": "lerobot/droid_100 leading episodes, via M*'s own DROIDDataset",
            "source_input_fingerprint": "SHA256(video file contents, horizon, frame count, float32 actions and states); not a decoded-tensor or quality check",
            "request_driver": (
                "In-process upstream rollout. The two verbatim blocks are reproduced from "
                "energy_landscape_example.ipynb Cells 5 and 6; helpers are imported from "
                "external/mstar/benchmark/vjepa2_repo_ac.py rather than copied."
            ),
            "timing_boundary": (
                "Encoder forward plus H predictor steps with explicit CUDA synchronisation. "
                "Excludes video decode, transform, tensor upload, NumPy conversion and disk "
                "writes. No HTTP, multipart or serialisation -- unlike the M* side, which is "
                "timed over the wire. This asymmetry is the paper's own pairing."
            ),
        },
    )
    print(json.dumps({"status": summary["status"], "case": args.case}, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
