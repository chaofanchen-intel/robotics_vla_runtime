#!/usr/bin/env python3
"""Measured client for BAGEL, with all three equal-work assertions enforced.

The assertions are the point of this file. A cell that cannot prove it did the
requested work aborts instead of reporting a number: a missing comparison is
recoverable, a plausible-looking invalid one is not.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exps.core import adapters, benchmark_client, codecs, paths
from exps.core.codecs import ContractViolation
from exps.tasks.bagel.cases import (
    CFG_BY_TASK,
    I2T_MAX_TOKENS,
    SEED,
    SERVED_MODEL_NAME,
    TIMESTEP_POINTS,
)

FIXTURE = paths.INPUTS / "BAGEL" / "bagel_shared_v1.json"
IMAGE_DIR = paths.INPUTS / "BAGEL" / "images"

REQUEST_TYPE_BY_TASK = {"t2i": "text_to_image", "i2i": "image_to_image", "i2t": "image_to_text"}


def load_fixture() -> dict[str, Any]:
    if not FIXTURE.is_file():
        raise SystemExit(f"BAGEL fixture missing: {FIXTURE}")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def source_image(fixture: dict[str, Any], task: str, resolution: int) -> Path:
    """Resolve and hash-verify the pre-resized source image for this cell.

    The manifest path is repository-relative under `exps/inputs/`; resolve it
    from this task's local image directory. The recorded SHA-256 is still
    checked, which is what makes the two engines provably receive identical
    bytes.
    """
    entry = fixture[task]["source_images"][str(resolution)]
    path = IMAGE_DIR / Path(entry["path"]).name
    if not path.is_file():
        raise SystemExit(f"source image missing: {path}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise SystemExit(
            f"{path} does not match the frozen fixture hash "
            f"(recorded {entry['sha256']}, found {digest}). The two engines would no longer be "
            "sent identical bytes."
        )
    width, height = codecs.png_dimensions(path.read_bytes())
    if (width, height) != (resolution, resolution):
        raise SystemExit(
            f"{path} is {width}x{height}, not {resolution}x{resolution}. The source image must "
            "already be the target size; resizing server-side is exactly how the previous round "
            "ended up comparing a 1024 generation against a 512 one."
        )
    return path


def model_kwargs_for(
    task: str,
    resolution: int,
    max_tokens: int,
    timesteps: int = TIMESTEP_POINTS,
    cfg_off: bool = False,
    cfg_img_scale: float | None = None,
    prefix_cache: bool = True,
) -> dict[str, Any]:
    """Per-request kwargs, identical for every engine.

    These carry the three fixes. For the image tasks: explicit width and
    height, so no engine re-derives the output size from the input. For I2T:
    both cap keys plus `ignore_eos`, so every engine decodes exactly the same
    number of tokens rather than stopping at its own EOS.
    """
    kwargs: dict[str, Any] = {
        "seed": SEED,
        "temperature": 0.0,
        # M*'s "default" preprocessor resizes differently from vLLM-Omni's.
        # "vllm" is the value M*'s own harness passes for parity.
        "image_preprocess": "vllm",
    }
    if not prefix_cache:
        # M* only (the caller passes False for no other engine): it opens a KV prefix cache by default and every request
        # repeats the same prompt + image, so the second request onward skips the LLM prefill. The vLLM-Omni deploy configs
        # set enable_prefix_caching: false.
        kwargs["prefix_cache"] = False
    if task in CFG_BY_TASK:
        kwargs.update(CFG_BY_TASK[task])
        if cfg_off:
            # Ablation: both runtimes only build extra CFG branches when a scale exceeds 1.0
            # (M* `_requires_cfg`, vLLM-Omni `do_true_cfg`), so 1.0 collapses to a single branch.
            kwargs.update({"cfg_text_scale": 1.0, "cfg_img_scale": 1.0})
        elif cfg_img_scale is not None:
            # Ablation: a scale above 1.0 makes both engines run (and use) the third CFG branch
            kwargs["cfg_img_scale"] = cfg_img_scale
        kwargs.update(
            {
                "width": resolution,
                "height": resolution,
                "num_inference_steps": timesteps,
                "num_timesteps": timesteps,
            }
        )
    else:
        kwargs.update(
            {
                "max_tokens": max_tokens,
                "max_output_tokens": max_tokens,
                "min_tokens": max_tokens,
                "ignore_eos": True,
            }
        )
        if os.environ.get("EXPS_BAGEL_SAMPLING") == "neutral":
            # --opt bagel.sampling=neutral: vLLM-Omni's yaml default repetition_penalty 1.05 would otherwise apply to it only
            kwargs.update({"repetition_penalty": 1.0, "top_k": 1, "top_p": 1.0})
    return kwargs


def build_requests(
    fixture: dict[str, Any],
    task: str,
    resolution: int,
    count: int,
    max_tokens: int,
    timesteps: int = TIMESTEP_POINTS,
    cfg_off: bool = False,
    cfg_img_scale: float | None = None,
    prefix_cache: bool = True,
) -> list[Any]:
    request_module = adapters.load_benchmark_module("benchmark.request")
    request_type = adapters.request_type(REQUEST_TYPE_BY_TASK[task])
    kwargs = model_kwargs_for(task, resolution, max_tokens, timesteps, cfg_off, cfg_img_scale, prefix_cache)

    if task == "t2i":
        prompt = fixture["t2i"]["text"]
        image_path = None
    elif task == "i2i":
        prompt = fixture["i2i"]["instruction"]
        image_path = str(source_image(fixture, "i2i", resolution))
    else:
        prompt = fixture["i2t"]["prompt"]
        image_path = str(source_image(fixture, "i2t", resolution))

    # One frozen prompt repeated, so per-request work is constant and any
    # variation across a start is the machine, never the input.
    return [
        request_module.RequestInput(
            req_type=request_type,
            prompt=prompt,
            image_path=image_path,
            model_kwargs=dict(kwargs),
        )
        for _ in range(count)
    ]


def make_validator(
    task: str, resolution: int, max_tokens: int, tolerance: int, timesteps: int = TIMESTEP_POINTS
):
    def validate(result, row: dict[str, Any], index: int) -> dict[str, Any]:
        if task in ("t2i", "i2i"):
            image = b"".join(getattr(result, "_image_chunks", None) or [])
            info = codecs.require_image_size(image, resolution, resolution)
            return {
                "returned_image": info,
                "requested_resolution": f"{resolution}x{resolution}",
                "image_size_verified": True,
                "timestep_points": timesteps,
            }

        produced = row.get("output_tokens")
        text = adapters.response_text(result)
        if not text.strip():
            raise ContractViolation("the response carried no caption text")
        if produced is None:
            return {
                "completion_length_verified": False,
                "completion_length_note": (
                    "the server reported no usage.completion_tokens, so the length assertion "
                    "could not run; the caption and its character count are recorded instead"
                ),
                "caption_characters": len(text),
                "caption": text,
            }
        if abs(produced - max_tokens) > tolerance:
            raise ContractViolation(
                f"the completion is {produced} tokens against a requested {max_tokens} "
                f"(tolerance {tolerance}). The runtime did not honor BAGEL's fixed-length "
                "I2T request contract."
            )
        return {
            "completion_length_verified": True,
            "requested_output_tokens": max_tokens,
            "realised_output_tokens": produced,
            "caption_characters": len(text),
            "caption": text,
        }

    return validate


def main() -> int:
    parser = benchmark_client.build_parser(__doc__)
    parser.add_argument("--task", choices=("t2i", "i2i", "i2t"), required=True)
    parser.add_argument("--resolution", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=I2T_MAX_TOKENS)
    parser.add_argument("--length-tolerance", type=int, default=4)
    parser.add_argument("--timesteps", type=int, default=TIMESTEP_POINTS,
                        help="Denoise steps. Ablation only; the baseline cells use 50.")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Concurrent requests per gated unit; the reported e2e is the wave wall time.")
    parser.add_argument("--cfg-off", action="store_true",
                        help="Ablation: set CFG scales to 1.0 so only one branch runs.")
    parser.add_argument("--cfg-img-scale", type=float, default=None,
                        help="Ablation: override cfg_img_scale (2.0 makes both engines run three CFG branches).")
    parser.add_argument("--no-prefix-cache", action="store_true",
                        help="Counterfactual, M* only: send prefix_cache=false so the LLM prefill is recomputed every request.")
    args = parser.parse_args()
    prefix_cache = not (args.no_prefix_cache and args.framework == "mstar")
    if args.resolution not in (256, 512, 1024) and not (args.task == "t2i" and args.resolution in (384, 448, 768)):
        parser.error("--resolution must be 256, 512 or 1024 (T2I sweep cases also allow 384, 448 and 768). 1024 on a "
                     "throttling chassis crosses the thermal knee partway through a request (the H200 NVL); it is meant "
                     "for machines with stable cooling.")

    fixture = load_fixture()
    count = ((args.preconditions or 1) + (args.measured or 5)) * args.batch_size
    kwargs = model_kwargs_for(args.task, args.resolution, args.max_tokens, args.timesteps, args.cfg_off,
                               args.cfg_img_scale, prefix_cache)

    return benchmark_client.run(
        args=args,
        task="bagel",
        model_type="bagel",
        model_instance_kwargs={
            "image_preprocess": "vllm",
            **({"served_model_name": SERVED_MODEL_NAME} if args.framework.startswith("vllm-omni") else {}),
        },
        request_inputs=build_requests(
            fixture, args.task, args.resolution, count, args.max_tokens, args.timesteps, args.cfg_off,
            args.cfg_img_scale, prefix_cache
        ),
        validate=make_validator(
            args.task, args.resolution, args.max_tokens, args.length_tolerance, args.timesteps
        ),
        batch_size=args.batch_size,
        context={
            "batch_size": args.batch_size,
            "bagel_task": args.task,
            "resolution": f"{args.resolution}x{args.resolution}",
            "latent_tokens": (args.resolution // 16) ** 2,
            "timestep_points": args.timesteps,
            "cfg_off": args.cfg_off,
            "cfg_img_scale_override": args.cfg_img_scale,
            "max_tokens": args.max_tokens,
            "fixture": paths.relative_to_repo(FIXTURE),
            "fixture_version": fixture.get("fixture_version"),
            "request_kwargs": kwargs,
            "served_model_name": SERVED_MODEL_NAME if args.framework.startswith("vllm-omni") else None,
            "contract": {
                "t2i": "the returned PNG must be exactly the requested size",
                "i2i": (
                    "the source image is pre-resized to the exact target, the size is sent "
                    "explicitly, and the returned PNG must match it"
                ),
                "i2t": (
                    "both max_tokens and max_output_tokens are sent with ignore_eos, and the "
                    "realised completion length is asserted"
                ),
            }[args.task],
            "cfg_topology": (
                "single GPU: bagel_single_gpu.yaml declares no LLM_cfg_text or LLM_cfg_img node "
                "group, so the three CFG branches are merged into one batched forward rather than "
                "run on separate devices as in the paper's 3xH100 main figure. This corresponds "
                "to the paper's Appendix E single-GPU setting."
            ),
            "timestep_shift_caveat": (
                "M* forces timestep_shift 3.0 while vLLM-Omni reads the checkpoint's 1.0. The "
                "step count is identical, so latency is unaffected, but the two engines produce "
                "systematically different pixels. Any blind quality review must know this."
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
