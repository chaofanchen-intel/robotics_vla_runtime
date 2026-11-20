#!/usr/bin/env python3
"""Response decoders and payload validators shared by every task client.

Each runtime speaks a different wire format for the same work:

* **M-star** -- ``POST /generate`` with a multipart form, answering with NDJSON
  where every line is ``{"modality": ..., "data": <base64>}``. Audio arrives as
  raw PCM16 chunks (no container), images as PNG bytes, text as UTF-8.
* **vLLM-Omni / SGLang-Omni** -- OpenAI-compatible ``/v1/chat/completions``
  with the payload inside ``choices[0].message``; audio is a complete base64
  WAV, images are base64 PNG, text is the message content.
* **VoxServe** -- ``POST /generate`` with a multipart form, answering with a
  complete WAV body (or a streamed WAV when ``streaming=true``).

Keeping the decoders here rather than in each task means one definition of
"audio duration", one PNG dimension check, and one place where an error
modality is turned into a loud failure instead of an empty result.

Everything in this module runs **after** the latency timer has stopped. Base64
decoding, WAV parsing and hashing are measurement overhead, not service time,
and must never land inside a measured interval.
"""
from __future__ import annotations

import base64
import hashlib
import json
import struct
import wave
from io import BytesIO
from typing import Any

#: Both Orpheus and Qwen3-Omni emit 24 kHz mono PCM16 on this host.
DEFAULT_AUDIO_SAMPLE_RATE_HZ = 24_000
PCM16_BYTES_PER_SAMPLE = 2


class ContractViolation(RuntimeError):
    """A response was structurally valid HTTP but not the work that was asked for.

    Raised rather than returned: a cell that cannot honour its contract must
    stop, not quietly report a number from a different workload. That mistake
    -- measuring a 2048-token completion against a 128-token one, or a 1024px
    image against a 512px one -- is what invalidated the previous round.
    """


# ---------------------------------------------------------------------------
# M* NDJSON
# ---------------------------------------------------------------------------

def iter_ndjson_events(body: bytes) -> list[dict[str, Any]]:
    """Parse an M* NDJSON body into events, skipping blank and malformed lines.

    Malformed lines are skipped rather than fatal because M* occasionally
    flushes a partial line at the tail of a stream; an ``error`` modality,
    which is a real failure, is surfaced by `decode_mstar`.
    """
    events = []
    for line in body.splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def decode_mstar(
    body: bytes,
    *,
    sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ,
) -> dict[str, Any]:
    """Decode an M* NDJSON response into the union of modalities it carried.

    Audio chunks are concatenated raw PCM16 and wrapped in a WAV container here
    so downstream code has one audio representation regardless of runtime.
    """
    pcm_chunks: list[bytes] = []
    image_chunks: list[bytes] = []
    text_parts: list[str] = []
    for event in iter_ndjson_events(body):
        modality = event.get("modality")
        data = event.get("data")
        if modality == "error":
            message = base64.b64decode(data or "").decode("utf-8", "replace")
            raise ContractViolation(f"M* returned an error modality: {message}")
        if not data:
            continue
        payload = base64.b64decode(data)
        if modality == "audio":
            pcm_chunks.append(payload)
        elif modality == "image":
            image_chunks.append(payload)
        elif modality == "text":
            text_parts.append(payload.decode("utf-8", "replace"))

    decoded: dict[str, Any] = {
        "text": "".join(text_parts),
        "image": b"".join(image_chunks) or None,
        "audio": None,
        "audio_duration_s": None,
        "audio_sample_rate_hz": None,
    }
    if pcm_chunks:
        pcm = b"".join(pcm_chunks)
        if len(pcm) % PCM16_BYTES_PER_SAMPLE:
            raise ContractViolation(f"M* returned a truncated PCM16 stream ({len(pcm)} bytes)")
        decoded["audio"] = pcm16_to_wav(pcm, sample_rate_hz)
        decoded["audio_duration_s"] = len(pcm) / PCM16_BYTES_PER_SAMPLE / sample_rate_hz
        decoded["audio_sample_rate_hz"] = sample_rate_hz
    return decoded


# ---------------------------------------------------------------------------
# OpenAI-compatible chat completions (vLLM-Omni, SGLang-Omni)
# ---------------------------------------------------------------------------

def decode_openai_chat(body: bytes) -> dict[str, Any]:
    """Decode an OpenAI-shaped completion into the same dict `decode_mstar` returns."""
    try:
        response = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ContractViolation(f"response body is not JSON: {body[:400]!r}") from exc
    if "choices" not in response:
        raise ContractViolation(f"no choices in response: {json.dumps(response)[:400]}")
    message = response["choices"][0].get("message", {})
    content = message.get("content")
    text = content if isinstance(content, str) else (json.dumps(content, ensure_ascii=False) if content else "")

    decoded: dict[str, Any] = {
        "text": text or "",
        "image": None,
        "audio": None,
        "audio_duration_s": None,
        "audio_sample_rate_hz": None,
        "usage": response.get("usage"),
        "finish_reason": response["choices"][0].get("finish_reason"),
    }

    audio_b64 = (message.get("audio") or {}).get("data")
    if isinstance(audio_b64, str) and audio_b64:
        audio = base64.b64decode(audio_b64, validate=True)
        info = wav_info(audio)
        decoded["audio"] = audio
        decoded["audio_duration_s"] = info["duration_s"]
        decoded["audio_sample_rate_hz"] = info["sample_rate_hz"]

    images = message.get("images") or response["choices"][0].get("images")
    if isinstance(images, list) and images:
        first = images[0]
        payload = first.get("image_url", {}).get("url") if isinstance(first, dict) else first
        if isinstance(payload, str):
            decoded["image"] = base64.b64decode(payload.split(",", 1)[-1])
    return decoded


def completion_tokens(decoded: dict[str, Any]) -> int | None:
    usage = decoded.get("usage") or {}
    value = usage.get("completion_tokens")
    return int(value) if value is not None else None


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

def pcm16_to_wav(pcm: bytes, sample_rate_hz: int = DEFAULT_AUDIO_SAMPLE_RATE_HZ) -> bytes:
    buffer = BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(PCM16_BYTES_PER_SAMPLE)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(pcm)
    return buffer.getvalue()


def wav_info(audio: bytes) -> dict[str, Any]:
    """Parse a WAV container and report whether it is the expected mono PCM16."""
    try:
        with wave.open(BytesIO(audio), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            frames = handle.getnframes()
    except wave.Error as exc:
        raise ContractViolation(f"response is not a readable WAV: {exc}") from exc
    return {
        "channels": channels,
        "sample_width_bytes": sample_width,
        "sample_rate_hz": sample_rate,
        "frames": frames,
        "duration_s": frames / sample_rate if sample_rate else None,
        "valid_mono_pcm16": channels == 1 and sample_width == PCM16_BYTES_PER_SAMPLE and frames > 0,
        "sha256": hashlib.sha256(audio).hexdigest(),
    }


def require_valid_audio(audio: bytes | None, *, expected_sample_rate_hz: int | None = None) -> dict[str, Any]:
    if not audio:
        raise ContractViolation("no audio payload in the response")
    info = wav_info(audio)
    if not info["valid_mono_pcm16"]:
        raise ContractViolation(
            f"expected non-empty mono PCM16 audio, got {info['channels']}ch "
            f"{info['sample_width_bytes'] * 8}-bit, {info['frames']} frames"
        )
    if expected_sample_rate_hz is not None and info["sample_rate_hz"] != expected_sample_rate_hz:
        raise ContractViolation(
            f"audio sample rate {info['sample_rate_hz']} Hz != expected {expected_sample_rate_hz} Hz"
        )
    return info


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def png_dimensions(payload: bytes) -> tuple[int, int]:
    """Width and height straight from the IHDR chunk, without a decode library."""
    if len(payload) < 24 or payload[:8] != b"\x89PNG\r\n\x1a\n" or payload[12:16] != b"IHDR":
        raise ContractViolation("response is not a PNG with an IHDR header")
    return struct.unpack(">II", payload[16:24])


def require_image_size(payload: bytes | None, width: int, height: int) -> dict[str, Any]:
    """Assert the returned image is exactly the size that was requested.

    The previous round compared an M* 1024x1024 generation against a vLLM-Omni
    512x512 one and reported the 3.5x gap as a speed ratio. Nothing in the
    response said the sizes differed, so nothing caught it. This check is the
    reason that cannot recur.
    """
    if not payload:
        raise ContractViolation("no image payload in the response")
    actual_width, actual_height = png_dimensions(payload)
    if (actual_width, actual_height) != (width, height):
        raise ContractViolation(
            f"returned image is {actual_width}x{actual_height}, requested {width}x{height}; "
            "the two engines were not doing equal work"
        )
    return {
        "width": actual_width,
        "height": actual_height,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def sha256(payload: bytes | str) -> str:
    data = payload.encode("utf-8") if isinstance(payload, str) else payload
    return hashlib.sha256(data).hexdigest()
