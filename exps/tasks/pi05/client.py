#!/usr/bin/env python3
"""Measured client for Pi0.5, driving all three runtimes through one path."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np

from exps.core import cli, metrics as metrics_mod, paths
from exps.core.codecs import ContractViolation
from exps.core.measure import run_measurement
from exps.tasks.pi05 import transport
from exps.tasks.pi05.cases import ACTION_DIM, ACTION_HORIZON, FLOW_STEP_COUNTS, NUM_INFERENCE_STEPS

FIXTURE = paths.INPUTS / "PI05" / "fixture_46row" / "fixture.json"


def select_examples(examples: list[dict[str, Any]], needed: int, fixture_row: int | None = None) -> list[dict[str, Any]]:
    """The fixture rows one start sends: the rows in order, cycling; or, with `fixture_row`, that one row `needed` times
    (the pi05-kv case, so a prefix cache can hit)."""
    if fixture_row is None:
        return [examples[index % len(examples)] for index in range(needed)]
    if not 0 <= fixture_row < len(examples):
        raise ValueError(f"fixture row {fixture_row} is outside the {len(examples)}-row fixture")
    return [examples[fixture_row]] * needed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--framework", choices=("mstar", "vllm-omni", "sglang-core"), required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-inference-steps", type=int, default=NUM_INFERENCE_STEPS)
    parser.add_argument("--fixture-row", type=int, default=None,
                        help="send this fixture row on every request instead of cycling through the rows (pi05-kv)")
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-outputs", action="store_true", default=True)
    parser.add_argument("--no-save-outputs", dest="save_outputs", action="store_false")
    cli.add_client_protocol_arguments(parser)
    args = parser.parse_args()

    if args.num_inference_steps not in FLOW_STEP_COUNTS:
        parser.error(
            f"Pi0.5 cells run {FLOW_STEP_COUNTS} flow steps; got {args.num_inference_steps}. The M* service config "
            "must name the same count, and the cell plan sets both from one case"
        )
    if not FIXTURE.is_file():
        raise SystemExit(f"Pi0.5 fixture missing: {FIXTURE}")
    config = cli.client_measurement_config(args, args.output_dir)
    manifest, examples = transport._load_fixture(FIXTURE)  # noqa: SLF001 - ported helper
    needed = config.precondition_requests + config.measured_requests
    selected = select_examples(examples, needed, args.fixture_row)
    # Images are decoded once, before any measurement, so no request is charged
    # for PNG decoding that the engine never sees.
    images = [transport._load_images(FIXTURE, example) for example in selected]  # noqa: SLF001

    connection = None
    http_session = None
    metadata: dict[str, Any] = {}
    if args.framework == "mstar":
        import requests

        http_session = requests.Session()
    else:
        connection, metadata = transport._connect_websocket(args.framework, args.base_url)  # noqa: SLF001

    session_id = f"pi05-{args.framework}-{int(time.time())}"

    def issue(index: int, phase: str) -> dict[str, Any]:
        offset = (index - 1) if phase == "precondition" else (config.precondition_requests + index - 1)
        example = selected[offset]
        frame = images[offset]

        if args.framework == "mstar":
            action, elapsed_ms, status, _start, _end = transport._mstar_request(  # noqa: SLF001
                http_session, args.base_url, example, frame, args.seed
            )
        else:
            payload = (
                transport._vllm_payload(example, frame, session_id, args.num_inference_steps, args.seed)  # noqa: SLF001
                if args.framework == "vllm-omni"
                else transport._sglang_payload(example, frame, args.num_inference_steps)  # noqa: SLF001
            )
            action, elapsed_ms, status, _start, _end = transport._websocket_request(  # noqa: SLF001
                connection, payload, args.request_timeout_s
            )

        flat = np.asarray(action, dtype=np.float32).reshape(-1)
        if flat.size != ACTION_HORIZON * ACTION_DIM:
            raise ContractViolation(
                f"the runtime returned {flat.size} action values, not "
                f"{ACTION_HORIZON}x{ACTION_DIM}={ACTION_HORIZON * ACTION_DIM}. A shorter horizon "
                "is less work, not a faster engine."
            )
        trajectory = flat.reshape(ACTION_HORIZON, ACTION_DIM)
        if not np.isfinite(trajectory).all():
            raise ContractViolation("the action trajectory contains non-finite values")

        if args.save_outputs:
            target = Path(args.output_dir) / "outputs"
            target.mkdir(parents=True, exist_ok=True)
            np.save(target / f"{phase}_{index:03d}_actions.npy", trajectory)

        row = metrics_mod.empty_request_metrics()
        row.update(
            {
                "e2e_ms": elapsed_ms,
                "output_bytes": int(trajectory.nbytes),
                "success": True,
                "status_code": status,
                "fixture_index": offset,
                "episode_index": example.get("episode_index"),
                "task_text": example.get("task"),
                "action_shape": list(trajectory.shape),
                "action_shape_verified": True,
                "action_abs_mean": float(np.abs(trajectory).mean()),
                "num_inference_steps": args.num_inference_steps,
            }
        )
        return row

    try:
        summary = run_measurement(
            config,
            issue=issue,
            context={
                "task": "pi05",
                "framework": args.framework,
                "case": args.case,
                "base_url": args.base_url,
                "fixture": paths.relative_to_repo(FIXTURE),
                "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
                "fixture_rows": len(examples),
                "fixture_rows_used": [example.get("episode_index") for example in selected],
                "fixture_row": args.fixture_row,
                "num_inference_steps": args.num_inference_steps,
                "action_contract": f"[{ACTION_HORIZON}, {ACTION_DIM}] float32, asserted per response",
                "camera_slots": list(transport.CAMERA_KEYS),
                "server_metadata": {str(k): str(v) for k, v in (metadata or {}).items()},
                "request_driver": (
                    "one client for all three runtimes, ported from the validated round-3 "
                    "transports. Pi0.5 cannot use the shared benchmark.request adapters because "
                    "only M* serves it over HTTP; vLLM-Omni and SGLang core use msgpack over "
                    "WebSocket in OpenPI's framing."
                ),
                "timing_boundary": (
                    "HTTP: just before POST to end of response body. WebSocket: just before send "
                    "to completion of recv. Image encoding and msgpack packing precede the timer "
                    "on every path; decoding follows it."
                ),
            },
        )
    finally:
        if connection is not None:
            connection.close()
        if http_session is not None:
            http_session.close()

    print(json.dumps({"status": summary["status"], "framework": args.framework}, indent=2))
    return 0 if summary["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
