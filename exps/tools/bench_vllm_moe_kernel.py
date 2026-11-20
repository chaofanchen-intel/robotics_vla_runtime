#!/usr/bin/env python3
"""vLLM's own fused-MoE on the Qwen3-Omni Thinker shapes, to put next to M*'s (bench_mstar_moe_config.py).

M*'s default tile config vs the tuned one is a 1.0x (M <= 128) to 1.4x (M >= 129) effect in isolation, yet in the
A2T nsys profile M*'s MoE kernel time is ~2-3x vLLM's / SGLang's. This times vLLM's `fused_experts_impl`
(its own tuned H200 JSON, its own routing kernels) on the same shapes and the same random inputs, so the
M* / vLLM ratio per token count is measured without the serving stack around it.

    envs/vllm-omni/bin/python exps/tools/bench_vllm_moe_kernel.py --json out.json

Needs an otherwise idle GPU. Compare with: bench_mstar_moe_config.py --vllm-json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

E, HIDDEN, INTER, TOP_K = 128, 2048, 768, 8
DEFAULT_TOKENS = [1, 4, 8, 16, 32, 64, 96, 128, 129, 144, 160, 192, 224, 256, 320, 384, 512, 1024]


def time_ms(fn, iters: int, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tokens", type=int, nargs="+", default=DEFAULT_TOKENS)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl

    torch.manual_seed(0)
    dev = "cuda"
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
    w2 = (torch.randn(E, HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
    rows = []
    with set_current_vllm_config(VllmConfig()):
        for m in args.tokens:
            x = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16).contiguous()
            logits = torch.randn(m, E, device=dev, dtype=torch.float32)
            weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)
            call = lambda: fused_experts_impl(x, w1, w2, weights, ids.to(torch.int32))  # noqa: E731
            call()
            rows.append({"tokens": m, "vllm_ms": time_ms(call, args.iters)})
            print(f"M={m:5d}  vllm fused_experts {rows[-1]['vllm_ms']:.3f} ms", flush=True)
    if args.json:
        args.json.write_text(json.dumps({"device": torch.cuda.get_device_name(), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
