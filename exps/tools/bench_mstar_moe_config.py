#!/usr/bin/env python3
"""Kernel microbenchmark: M*'s fused-MoE tile config vs the tuned H200 config vLLM ships.

Why: in the Qwen3-Omni A2T nsys profile M* spends ~2x the GPU time of vLLM-Omni / SGLang-Omni in the
Thinker's MoE layers. Source reading shows M* always calls `get_default_config` (a fixed tile choice:
BLOCK 16x32x64 for M <= E, no num_warps / num_stages), while vLLM and SGLang look up per-shape tuned
JSON configs for H200. This tool times M*'s own `fused_experts` under both configs on the Thinker's
shapes (E=128, hidden 2048, moe_intermediate 768, top_k 8), so the config effect is isolated from
everything else in the serving path. It also checks the outputs agree (a config must not change maths).

    envs/mstar/bin/python exps/tools/bench_mstar_moe_config.py [--json out.json] [--vllm-json vllm.json]

Needs an otherwise idle GPU. Does not modify external/: the tuned config is injected by replacing
`mstar.utils.fused_moe.runner.get_default_config` inside this process only.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
TUNED_JSON = (
    ROOT / "envs" / "vllm-omni" / "lib" / "python3.12" / "site-packages" / "vllm" / "model_executor"
    / "layers" / "fused_moe" / "configs" / "E=128,N=768,device_name=NVIDIA_H200.json"
)
E, HIDDEN, INTER, TOP_K = 128, 2048, 768, 8


def tuned_config_fn(path: Path):
    table = {int(k): v for k, v in json.loads(path.read_text()).items()}

    def get(M: int, E: int, N: int, K: int, top_k: int) -> dict:  # same nearest-key rule as vLLM
        return dict(table[min(table, key=lambda x: abs(x - M))])

    return get


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
    ap.add_argument("--tuned-json", type=Path, default=TUNED_JSON)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64, 96, 128, 129, 144, 160, 192, 224, 256, 320, 384, 512, 1024])
    ap.add_argument("--json", type=Path)
    ap.add_argument("--vllm-json", type=Path,
                    help="Output of bench_vllm_moe_kernel.py (run under envs/vllm-omni); adds vLLM's own fused_experts column.")
    args = ap.parse_args()
    if not args.tuned_json.is_file():
        print(f"tuned config not found: {args.tuned_json}", file=sys.stderr)
        return 2

    sys.path.insert(0, str(ROOT / "external" / "mstar"))
    from mstar.utils.fused_moe import runner  # noqa: PLC0415

    default_fn = runner.get_default_config
    tuned_fn = tuned_config_fn(args.tuned_json)

    torch.manual_seed(0)
    dev = "cuda"
    w1 = (torch.randn(E, 2 * INTER, HIDDEN, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()
    w2 = (torch.randn(E, HIDDEN, INTER, device=dev, dtype=torch.bfloat16) * 0.02).contiguous()

    rows = []
    for m in args.tokens:
        x = torch.randn(m, HIDDEN, device=dev, dtype=torch.bfloat16).contiguous()
        logits = torch.randn(m, E, device=dev, dtype=torch.float32)
        weights, ids = torch.topk(torch.softmax(logits, dim=-1), TOP_K, dim=-1)
        weights = (weights / weights.sum(-1, keepdim=True)).to(torch.bfloat16)

        results = {}
        for name, fn in (("default", default_fn), ("tuned", tuned_fn)):
            runner.get_default_config = fn
            call = lambda: runner.fused_experts(x, w1, w2, weights, ids)  # noqa: E731
            out = call()
            results[name] = {"ms": time_ms(call, args.iters), "out": out, "config": fn(m, E, 2 * INTER, HIDDEN, TOP_K)}
        runner.get_default_config = default_fn
        diff = (results["default"]["out"].float() - results["tuned"]["out"].float()).abs().max().item()
        rows.append({
            "tokens": m,
            "default_ms": results["default"]["ms"], "tuned_ms": results["tuned"]["ms"],
            "speedup_tuned_over_default": results["default"]["ms"] / results["tuned"]["ms"],
            "max_abs_output_diff": diff,
            "default_config": results["default"]["config"], "tuned_config": results["tuned"]["config"],
        })

    vllm_ms = {}
    if args.vllm_json:
        vllm_ms = {r["tokens"]: r["vllm_ms"] for r in json.loads(args.vllm_json.read_text())["rows"]}
        for r in rows:
            if r["tokens"] in vllm_ms:
                r["vllm_ms"] = vllm_ms[r["tokens"]]
                r["default_over_vllm"] = r["default_ms"] / vllm_ms[r["tokens"]]
    header = "| tokens | default ms | tuned ms | default / tuned | max |Δout| |" + (" vLLM ms | M* default / vLLM |" if vllm_ms else "")
    lines = [header, "|---:|---:|---:|---:|---:|" + ("---:|---:|" if vllm_ms else "")]
    for r in rows:
        extra = f" {r['vllm_ms']:.3f} | {r['default_over_vllm']:.2f} |" if "vllm_ms" in r else (" - | - |" if vllm_ms else "")
        lines.append(f"| {r['tokens']} | {r['default_ms']:.3f} | {r['tuned_ms']:.3f} | "
                     f"{r['speedup_tuned_over_default']:.2f} | {r['max_abs_output_diff']:.3g} |{extra}")
    print("\n".join(lines))
    if args.json:
        args.json.write_text(json.dumps({"tuned_json": str(args.tuned_json), "device": torch.cuda.get_device_name(),
                                         "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
