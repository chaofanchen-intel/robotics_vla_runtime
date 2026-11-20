#!/usr/bin/env python3
"""First-call vs repeat-call latency of the Qwen3-Omni audio-tower conv stem.

Component-only diagnostic, not an E2E benchmark. The audio tower runs three
Conv2d+GELU layers over the mel spectrogram split into 100-frame chunks, so
the conv batch size is ceil(frames / 100) and changes with every new clip
length. If a framework env pays a seconds-long cuDNN plan/NVRTC cost the first
time it sees a new batch size, the first audio request of every new length is
slow in that process and fast afterwards. Run this in each env and compare
the "first" column; the shapes below are the real fixture clips.

    envs/vllm-omni/bin/python -I exps/tools/probe_audio_conv_first_call.py
    envs/mstar/bin/python -I exps/tools/probe_audio_conv_first_call.py
    envs/sglang-omni/bin/python -I exps/tools/probe_audio_conv_first_call.py

Needs a free GPU for a few seconds and under 2 GiB of device memory.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F

# Audio-tower config of Qwen3-Omni-30B-A3B (config.json: audio_config).
NUM_MEL_BINS = 128
CHUNK_FRAMES = 100  # n_window * 2
HIDDEN = 480  # downsample_hidden_size
CONV_CHUNKSIZE = 500
FRAMES_PER_SECOND = 100  # 10 ms hop

# (label, seconds) of the fixture clips used by the A2T smoke / baseline cells.
CLIPS = (
    ("00003_1728 (smoke precondition)", 5.76),
    ("00505_1901 (smoke measured)", 13.48),
    ("00356_1968", 10.8),
    ("00051_1769", 8.0),
)


def batch_for(seconds: float) -> int:
    return math.ceil(seconds * FRAMES_PER_SECOND / CHUNK_FRAMES)


def stem(x, convs):
    outs = []
    for chunk in x.split(CONV_CHUNKSIZE, dim=0):
        for conv in convs:
            chunk = F.gelu(conv(chunk))
        outs.append(chunk)
    return torch.cat(outs, dim=0)


def timed(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    dtype = getattr(torch, args.dtype)
    dev = torch.device("cuda")

    convs = [
        torch.nn.Conv2d(1, HIDDEN, 3, 2, padding=1),
        torch.nn.Conv2d(HIDDEN, HIDDEN, 3, 2, padding=1),
        torch.nn.Conv2d(HIDDEN, HIDDEN, 3, 2, padding=1),
    ]
    convs = [c.to(device=dev, dtype=dtype) for c in convs]

    with torch.no_grad():
        # Library/context initialisation is not the thing under test.
        init = torch.randn(1, 1, NUM_MEL_BINS, CHUNK_FRAMES, device=dev, dtype=dtype)
        init_ms = timed(lambda: stem(init, convs))

        rows = []
        for label, seconds in CLIPS:
            batch = batch_for(seconds)
            x = torch.randn(batch, 1, NUM_MEL_BINS, CHUNK_FRAMES, device=dev, dtype=dtype)
            first = timed(lambda: stem(x, convs))
            repeats = [timed(lambda: stem(x, convs)) for _ in range(5)]
            rows.append(
                {
                    "clip": label,
                    "seconds": seconds,
                    "conv_batch": batch,
                    "first_ms": round(first, 2),
                    "repeat_median_ms": round(sorted(repeats)[len(repeats) // 2], 2),
                }
            )

    report = {
        "torch": torch.__version__,
        "cudnn": torch.backends.cudnn.version(),
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "dtype": args.dtype,
        "cudnn_benchmark": args.cudnn_benchmark,
        "library_init_ms": round(init_ms, 2),
        "shapes": rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
