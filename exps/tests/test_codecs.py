"""Tests for the shared response decoders and contract validators."""
from __future__ import annotations

import base64
import json
import struct
import zlib

import pytest

from exps.core import codecs


def _ndjson(*events: dict) -> bytes:
    return b"\n".join(json.dumps(event).encode() for event in events)


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _png(width: int, height: int) -> bytes:
    """A minimal but structurally valid PNG, enough for the IHDR size check."""
    header = struct.pack(">II", width, height) + bytes([8, 6, 0, 0, 0])
    ihdr = struct.pack(">I", len(header)) + b"IHDR" + header
    ihdr += struct.pack(">I", zlib.crc32(b"IHDR" + header))
    return b"\x89PNG\r\n\x1a\n" + ihdr


def test_decode_mstar_concatenates_pcm_chunks_into_one_wav():
    # 24000 samples of PCM16 == exactly one second at 24 kHz.
    pcm = b"\x00\x01" * 24_000
    body = _ndjson(
        {"modality": "audio", "data": _b64(pcm[: len(pcm) // 2])},
        {"modality": "audio", "data": _b64(pcm[len(pcm) // 2 :])},
    )
    decoded = codecs.decode_mstar(body)
    assert decoded["audio_duration_s"] == pytest.approx(1.0)
    assert codecs.wav_info(decoded["audio"])["valid_mono_pcm16"] is True


def test_decode_mstar_collects_text_and_image_modalities():
    body = _ndjson(
        {"modality": "text", "data": _b64("hello ".encode())},
        {"modality": "text", "data": _b64("world".encode())},
        {"modality": "image", "data": _b64(_png(512, 512))},
    )
    decoded = codecs.decode_mstar(body)
    assert decoded["text"] == "hello world"
    assert codecs.png_dimensions(decoded["image"]) == (512, 512)


def test_decode_mstar_raises_on_the_error_modality():
    body = _ndjson({"modality": "error", "data": _b64(b"talker failed")})
    with pytest.raises(codecs.ContractViolation, match="talker failed"):
        codecs.decode_mstar(body)


def test_decode_mstar_rejects_a_truncated_pcm_stream():
    body = _ndjson({"modality": "audio", "data": _b64(b"\x00\x01\x02")})
    with pytest.raises(codecs.ContractViolation, match="truncated"):
        codecs.decode_mstar(body)


def test_decode_mstar_skips_a_partial_trailing_line():
    body = _ndjson({"modality": "text", "data": _b64(b"ok")}) + b'\n{"modality": "te'
    assert codecs.decode_mstar(body)["text"] == "ok"


def test_decode_openai_chat_extracts_text_and_usage():
    body = json.dumps(
        {
            "choices": [{"message": {"content": "a reply"}, "finish_reason": "length"}],
            "usage": {"completion_tokens": 128},
        }
    ).encode()
    decoded = codecs.decode_openai_chat(body)
    assert decoded["text"] == "a reply"
    assert codecs.completion_tokens(decoded) == 128
    assert decoded["finish_reason"] == "length"


def test_decode_openai_chat_extracts_wav_audio_and_its_duration():
    wav = codecs.pcm16_to_wav(b"\x00\x01" * 12_000)  # half a second
    body = json.dumps(
        {"choices": [{"message": {"content": "", "audio": {"data": _b64(wav)}}}]}
    ).encode()
    decoded = codecs.decode_openai_chat(body)
    assert decoded["audio_duration_s"] == pytest.approx(0.5)
    assert decoded["audio_sample_rate_hz"] == 24_000


def test_decode_openai_chat_rejects_a_non_json_body():
    with pytest.raises(codecs.ContractViolation, match="not JSON"):
        codecs.decode_openai_chat(b"<html>502 Bad Gateway</html>")


def test_require_image_size_rejects_the_round2_mismatch():
    # The failure this check exists for: M* returned 1024 while vLLM returned 512
    # and the 3.5x gap was reported as a speed ratio.
    with pytest.raises(codecs.ContractViolation, match="not doing equal work"):
        codecs.require_image_size(_png(1024, 1024), 512, 512)


def test_require_image_size_accepts_an_exact_match():
    info = codecs.require_image_size(_png(512, 512), 512, 512)
    assert (info["width"], info["height"]) == (512, 512)
    assert len(info["sha256"]) == 64


def test_require_valid_audio_rejects_stereo_and_empty_payloads():
    with pytest.raises(codecs.ContractViolation, match="no audio payload"):
        codecs.require_valid_audio(None)
    with pytest.raises(codecs.ContractViolation):
        codecs.require_valid_audio(b"not a wav at all")


def test_require_valid_audio_enforces_the_expected_sample_rate():
    wav = codecs.pcm16_to_wav(b"\x00\x01" * 16_000, sample_rate_hz=16_000)
    with pytest.raises(codecs.ContractViolation, match="sample rate"):
        codecs.require_valid_audio(wav, expected_sample_rate_hz=24_000)


def test_pcm16_roundtrips_through_wav_with_the_same_duration():
    pcm = b"\x00\x01" * 48_000  # two seconds at 24 kHz
    info = codecs.wav_info(codecs.pcm16_to_wav(pcm))
    assert info["duration_s"] == pytest.approx(2.0)
    assert info["channels"] == 1
    assert info["sample_width_bytes"] == 2
