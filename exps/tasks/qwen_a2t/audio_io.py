#!/usr/bin/env python3
"""Small RIFF/WAVE helpers for the frozen Qwen3-Omni A2T fixtures."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_audio_wav(path: Path) -> tuple[np.ndarray, int]:
    """Read mono 16 kHz PCM16 or IEEE-float32 WAV into normalized float32."""
    contents = path.read_bytes()
    if len(contents) < 12 or contents[:4] != b"RIFF" or contents[8:12] != b"WAVE":
        raise ValueError(f"expected a RIFF/WAVE audio file, got {path}")

    format_info = None
    audio_chunks: list[bytes] = []
    offset = 12
    while offset + 8 <= len(contents):
        chunk_id, chunk_size = struct.unpack_from("<4sI", contents, offset)
        chunk_start = offset + 8
        chunk_end = chunk_start + chunk_size
        if chunk_end > len(contents):
            raise ValueError(f"truncated WAV chunk {chunk_id!r} in {path}")
        chunk_data = contents[chunk_start:chunk_end]
        if chunk_id == b"fmt ":
            if len(chunk_data) < 16:
                raise ValueError(f"invalid WAV format chunk in {path}")
            format_info = struct.unpack_from("<HHIIHH", chunk_data)
        elif chunk_id == b"data":
            audio_chunks.append(chunk_data)
        offset = chunk_end + (chunk_size & 1)

    if format_info is None or not audio_chunks:
        raise ValueError(f"WAV is missing a format or data chunk: {path}")

    format_tag, channels, sample_rate, _byte_rate, block_align, bits_per_sample = format_info
    if channels != 1:
        raise ValueError(f"expected mono FLEURS audio, got {channels} channels")
    if sample_rate != 16_000:
        raise ValueError(f"expected 16 kHz FLEURS audio, got {sample_rate} Hz")
    raw = b"".join(audio_chunks)
    if format_tag == 1 and bits_per_sample == 16:
        bytes_per_sample = 2
        audio = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif format_tag == 3 and bits_per_sample == 32:
        bytes_per_sample = 4
        audio = np.frombuffer(raw, dtype="<f4").astype(np.float32, copy=True)
    else:
        raise ValueError(
            f"unsupported WAV encoding: format_tag={format_tag}, "
            f"bits_per_sample={bits_per_sample}"
        )
    if block_align != bytes_per_sample or len(raw) % block_align:
        raise ValueError(f"invalid WAV block alignment or truncated sample data: {path}")
    if not np.isfinite(audio).all():
        raise ValueError(f"WAV contains non-finite audio samples: {path}")
    return audio, sample_rate


def write_float_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write deterministic mono IEEE-float32 WAV for profile preconditions."""
    samples = np.asarray(audio, dtype="<f4")
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("profile precondition audio must be a non-empty mono vector")
    if sample_rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate}")
    if not np.isfinite(samples).all():
        raise ValueError("profile precondition audio contains non-finite samples")

    raw = samples.tobytes(order="C")
    fmt_payload = struct.pack(
        "<HHIIHH",
        3,
        1,
        sample_rate,
        sample_rate * 4,
        4,
        32,
    )
    fmt_chunk = struct.pack("<4sI", b"fmt ", len(fmt_payload)) + fmt_payload
    data_chunk = struct.pack("<4sI", b"data", len(raw)) + raw
    riff_size = 4 + len(fmt_chunk) + len(data_chunk)
    contents = struct.pack("<4sI4s", b"RIFF", riff_size, b"WAVE") + fmt_chunk + data_chunk
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(contents)
