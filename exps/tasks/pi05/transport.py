#!/usr/bin/env python3
"""Pi0.5 wire transports for M*, vLLM-Omni and SGLang core.

Request construction, msgpack framing and response decoding are implemented
here for the current harness; the per-request cooldown gate and shared metric
schema live in `client.py` and `core/`.

Why Pi0.5 does not use the shared adapter path
----------------------------------------------
Every other served task in this harness goes through M*'s `benchmark.request`
adapters. Pi0.5 cannot: only the M* side speaks HTTP. vLLM-Omni serves it over
`/v1/realtime/robot/openpi` and SGLang core over `/openpi/policy`, both
msgpack-over-WebSocket in OpenPI's own framing. A single client covering all
three is therefore defined here, and -- importantly -- all three runtimes go
through *this* client, so the within-task comparison is still driven by one
code path with one timing boundary.

Timing boundary
---------------
HTTP: from just before the POST to the end of the response body.
WebSocket: from just before `send` to the completion of `recv`.
msgpack packing happens before the timer starts on the WebSocket path and
unpacking after it stops, matching the HTTP path, where image encoding is also
done before the timer. Neither engine is charged for the client's framing.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import msgpack
import numpy as np
import requests
from PIL import Image

ACTION_HORIZON = 50
ACTION_DIM = 32
CAMERA_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def _pack_numpy(value: Any) -> Any:
    if isinstance(value, (np.ndarray, np.generic)) and value.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported numpy dtype for OpenPI msgpack: {value.dtype}")
    if isinstance(value, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_numpy(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    marker = value.get("__ndarray__", value.get(b"__ndarray__"))
    scalar_marker = value.get("__npgeneric__", value.get(b"__npgeneric__"))
    data = value.get("data", value.get(b"data"))
    dtype = value.get("dtype", value.get(b"dtype"))
    shape = value.get("shape", value.get(b"shape"))
    if marker:
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape)
    if scalar_marker:
        return np.dtype(dtype).type(data)
    return value


def _pack_msgpack(payload: dict[str, Any]) -> bytes:
    return msgpack.packb(payload, default=_pack_numpy, use_bin_type=True)


def _unpack_msgpack(payload: bytes | str) -> Any:
    if isinstance(payload, str):
        raise RuntimeError(payload)
    return msgpack.unpackb(payload, object_hook=_unpack_numpy, raw=False)


def _load_fixture(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = json.loads(path.read_text())
    examples = manifest.get("examples")
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Fixture has no examples: {path}")
    if any(set(example.get("camera_images", {})) != set(CAMERA_KEYS) for example in examples):
        raise ValueError("Each Pi0.5 fixture example must contain all three canonical camera slots")
    return manifest, examples


def _load_images(fixture_path: Path, example: dict[str, Any]) -> dict[str, np.ndarray]:
    images = {}
    for key in CAMERA_KEYS:
        image_path = fixture_path.parent / example["camera_images"][key]["path"]
        with Image.open(image_path) as image:
            images[key] = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    return images


def _extract_action(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, dict):
        for key in ("actions", "action"):
            if key in value:
                action = value[key]
                if isinstance(action, dict):
                    action = action.get("values", action.get("data", action))
                return np.asarray(action, dtype=np.float32)
        data = value.get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict):
            candidate = data[0].get("action")
            if isinstance(candidate, dict) and "values" in candidate:
                return np.asarray(candidate["values"], dtype=np.float32)
    raise ValueError(f"Could not extract Pi0.5 action output from {type(value).__name__}")


def _decode_mstar_body(body: bytes) -> np.ndarray:
    action_chunks = []
    try:
        messages = [json.loads(body)]
    except json.JSONDecodeError:
        messages = []
        for line in body.splitlines():
            try:
                messages.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("modality") == "error":
            raise RuntimeError(base64.b64decode(message.get("data", "")).decode("utf-8", "replace"))
        if message.get("modality") == "action" and message.get("data"):
            action_chunks.append(base64.b64decode(message["data"]))
        outputs = message.get("outputs", {}).get("action", [])
        for item in outputs:
            if item.get("data"):
                action_chunks.append(base64.b64decode(item["data"]))
    if not action_chunks:
        raise ValueError("M* response contained no action output")
    raw = b"".join(action_chunks)
    if len(raw) % 4:
        raise ValueError(f"M* action payload is not float32-aligned: {len(raw)} bytes")
    return np.frombuffer(raw, dtype=np.float32).copy()


def _mstar_request(
    session: requests.Session,
    base_url: str,
    example: dict[str, Any],
    images: dict[str, np.ndarray],
    seed: int,
) -> tuple[np.ndarray, float, int, int, int]:
    encoded = json.dumps({"robot_state": example["state_normalized_32"], "seed": seed})
    data = {
        "text": example["task"],
        "input_modalities": "image,text",
        "output_modalities": "action",
        "model_kwargs": encoded,
    }
    png_files = []
    from io import BytesIO

    for key in CAMERA_KEYS:
        buffer = BytesIO()
        Image.fromarray(images[key]).save(buffer, format="PNG")
        png_files.append(
            (
                "files",
                (f"ep{int(example['episode_index']):03d}_{key}.png", buffer.getvalue(), "image/png"),
            )
        )
    start = time.perf_counter_ns()
    with session.post(
        f"{base_url.rstrip('/')}/generate",
        data=data,
        files=png_files,
        stream=True,
        timeout=(30, 300),
    ) as response:
        status = response.status_code
        body = b"".join(response.iter_content(chunk_size=64 * 1024))
        response.raise_for_status()
    response_complete = time.perf_counter_ns()
    elapsed_ms = (response_complete - start) / 1_000_000
    return _decode_mstar_body(body), elapsed_ms, status, start, response_complete


def _vllm_payload(
    example: dict[str, Any], images: dict[str, np.ndarray], session_id: str, steps: int, seed: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        f"observation.images.{key}": images[key] for key in CAMERA_KEYS
    }
    payload.update(
        {
            "state": np.asarray(example["state_normalized_32"], dtype=np.float32),
            "prompt": example["task"],
            "session_id": session_id,
            "sampling_params": {"num_inference_steps": steps},
            "seed": seed,
            "endpoint": "infer",
        }
    )
    return payload


def _sglang_payload(
    example: dict[str, Any], images: dict[str, np.ndarray], steps: int
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": example["task"],
        "observation.state": np.asarray(example["state_normalized_32"], dtype=np.float32),
        "num_inference_steps": steps,
        "output_format": "numpy",
    }
    payload.update({f"observation.images.{key}": images[key] for key in CAMERA_KEYS})
    return payload


def _connect_websocket(runtime: str, base_url: str):
    import websockets.sync.client as ws_client

    if runtime == "vllm-omni":
        endpoint = base_url.rstrip("/") + "/v1/realtime/robot/openpi"
    else:
        endpoint = base_url.rstrip("/") + "/openpi/policy"
    connection = ws_client.connect(
        endpoint,
        compression=None,
        max_size=None,
        ping_interval=300,
        ping_timeout=3600,
        open_timeout=120,
        close_timeout=120,
    )
    metadata = _unpack_msgpack(connection.recv())
    if not isinstance(metadata, dict):
        connection.close()
        raise TypeError(f"Expected OpenPI metadata dictionary, got {type(metadata).__name__}")
    return connection, metadata


def _websocket_request(
    connection, payload: dict[str, Any], timeout_s: float
) -> tuple[np.ndarray, float, int, int, int]:
    packed = _pack_msgpack(payload)
    start = time.perf_counter_ns()
    connection.send(packed)
    raw_response = connection.recv(timeout=timeout_s)
    response_complete = time.perf_counter_ns()
    response = _unpack_msgpack(raw_response)
    elapsed_ms = (response_complete - start) / 1_000_000
    if isinstance(response, dict) and response.get("type") == "error":
        raise RuntimeError(str(response.get("message", response)))
    action = _extract_action(response)
    return action, elapsed_ms, 200, start, response_complete
