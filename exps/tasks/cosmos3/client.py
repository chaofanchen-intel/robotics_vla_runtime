"""Measured Cosmos3 T2I client with strict output-size validation."""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import requests

from exps.core import cli, metrics, paths
from exps.core.codecs import ContractViolation, require_image_size
from exps.core.measure import run_measurement
from exps.tasks.cosmos3.cases import SIZE, SIZES, STEPS

FIXTURE = paths.INPUTS / "COS-T2I" / "cosmos3_t2i_prompts_v1.json"
NEGATIVE_PROMPT = "blurry, distorted, low quality"


def build_payload(framework: str, model: str, prompt: str, width: int, height: int, steps: int) -> dict:
    if framework not in ("mstar", "vllm-omni", "sglang-omni") or not model or not prompt.strip():
        raise ValueError("expected a supported framework, model, and nonempty prompt")
    payload = {
        "model": model, "prompt": prompt, "negative_prompt": NEGATIVE_PROMPT,
        "size": f"{width}x{height}", "n": 1, "response_format": "b64_json",
        "num_inference_steps": steps, "guidance_scale": 6.0,
        "flow_shift": 3.0, "seed": 42,
    }
    if framework in ("mstar", "sglang-omni"):
        # SGLang's native image route accepts extra top-level fields (ImageGenerationsRequest allows extras) and parses
        # `guidance_interval` into Cosmos3's SamplingParams; its T2I default is the same (400, 1000).
        payload["guidance_interval"] = [400.0, 1000.0]
    else:
        payload["extra_params"] = {"guidance_interval": [400.0, 1000.0]}
    return payload


def decode_image(body: bytes, width: int, height: int) -> bytes:
    response = json.loads(body)
    entries = response.get("data") if isinstance(response, dict) else None
    if not isinstance(entries, list) or len(entries) != 1:
        raise ContractViolation("expected exactly one image in response.data")
    entry = entries[0]
    encoded = entry.get("b64_json") if isinstance(entry, dict) else None
    if not isinstance(encoded, str) or not encoded:
        raise ContractViolation("missing response.data[0].b64_json")
    try:
        png = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ContractViolation(f"invalid base64 image: {exc}") from exc
    require_image_size(png, width, height)
    return png


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("mstar", "vllm-omni", "sglang-omni"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--width", type=int, default=SIZE)
    parser.add_argument("--height", type=int, default=SIZE)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--request-timeout-s", type=float, default=900)
    parser.add_argument("--no-save-outputs", action="store_false", dest="save_outputs")
    cli.add_client_protocol_arguments(parser)
    args = parser.parse_args()
    size = SIZES.get(args.case)
    if size is None or (args.width, args.height, args.steps) != (size, size, STEPS):
        parser.error(f"the {args.case} case fixes {size}x{size} and {STEPS} steps")
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    prompts = fixture["prompts"]
    if not prompts or any(not item.get("text", "").strip() for item in prompts):
        raise ValueError(f"invalid Cosmos3 prompt fixture: {FIXTURE}")
    config = cli.client_measurement_config(args, args.output_dir)

    with requests.Session() as session:
        def issue(index: int, phase: str) -> dict:
            offset = index - 1 if phase == "precondition" else config.precondition_requests + index - 1
            prompt = prompts[offset % len(prompts)]
            payload = build_payload(
                args.framework, args.model, prompt["text"], args.width, args.height, args.steps
            )
            request = requests.Request(
                "POST", f"{args.base_url.rstrip('/')}/v1/images/generations",
                data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            prepared = session.prepare_request(request)
            start = time.perf_counter_ns()
            with session.send(prepared, timeout=args.request_timeout_s, stream=True) as response:
                body = b"".join(response.iter_content(chunk_size=64 * 1024))
                status = response.status_code
                response.raise_for_status()
            e2e_ms = (time.perf_counter_ns() - start) / 1_000_000
            png = decode_image(body, args.width, args.height)
            if args.save_outputs:
                output = args.output_dir / "outputs"
                output.mkdir(parents=True, exist_ok=True)
                (output / f"{phase}_{index:03d}_{prompt['id']}.png").write_bytes(png)
            row = metrics.empty_request_metrics()
            row.update({
                "success": True, "status_code": status, "e2e_ms": e2e_ms,
                "output_bytes": len(png), "fixture_index": offset,
                "prompt_id": prompt["id"], "image_size_verified": True,
                "num_inference_steps": args.steps,
            })
            return row

        summary = run_measurement(config, issue=issue, context={
            "task": "cosmos3", "framework": args.framework, "case": args.case,
            "fixture": paths.relative_to_repo(FIXTURE),
            "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            "timing_boundary": "POST start through response-body EOF; PNG decode/audit excluded",
            "image_contract": f"{args.width}x{args.height} PNG; one image per response",
        })
    print(json.dumps({"status": summary["status"], "framework": args.framework}))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
