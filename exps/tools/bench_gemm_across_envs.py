#!/usr/bin/env python3
"""Same bf16 GEMM, two Python environments (GPU, ~1 min each).

Round 5 found, in the t2i-768 nsys reports, the same persistent GEMM (grid 2x66, block 384, 5,488 launches in both)
taking 2,886 us in M* (torch 2.9.1+cu130, kernel `nvjet_tst_192x192...`) and 1,654 us in vLLM-Omni (torch 2.13.0+cu132,
`nvjet_sm90_tst_192x192...`). If a plain `torch.matmul` at BAGEL's shapes shows the same ratio between the two
environments, the gap is the torch / cuBLAS build and not the framework's model code; if the two environments agree,
the engines are calling different shapes (or more rows) or ran at different SM clocks.

Caveats found in round 7: a persistent kernel's grid equals the SM count whatever M is, so "same grid, same launch count"
did NOT show that the two engines ran the same shape; and in round 5 the M* run averaged 1,401 MHz while the vLLM run
averaged 1,636 MHz, so the per-launch times were not taken at the same clock. Each shape here records the SM clock during
the run, and the cell should start from the thermal gate (`exps/core/thermal.py`).

    envs/mstar/bin/python      exps/tools/bench_gemm_across_envs.py run --out exps/runs/GEMM_ENV/mstar.json
    envs/vllm-omni/bin/python  exps/tools/bench_gemm_across_envs.py run --out exps/runs/GEMM_ENV/vllm.json
    envs/mstar/bin/python      exps/tools/bench_gemm_across_envs.py compare exps/runs/GEMM_ENV/mstar.json exps/runs/GEMM_ENV/vllm.json

Shapes are BAGEL-7B-MoT's LLM (hidden 3584, intermediate 18944, 28 heads x 128, 4 KV heads) for M = 2304 latent tokens
(768^2) and M = 6912 (three CFG branches batched), plus 1024 / 3072 (512^2).
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

HIDDEN, INTER, QKV = 3584, 18944, 3584 + 2 * 512
SHAPES = {"qkv": (QKV, HIDDEN), "o": (HIDDEN, HIDDEN), "gate_up": (2 * INTER, HIDDEN), "down": (HIDDEN, INTER)}
ROWS = (1024, 2304, 3072, 6912)


def run(out: Path, warmup: int, iters: int) -> int:
    import torch

    repo = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo))
    try:
        from exps.core.cell import gpu_lock
    except Exception:  # noqa: BLE001 - the benchmark must still run where the harness is not importable
        from contextlib import nullcontext as gpu_lock  # type: ignore[assignment]
    results = {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0), "rows": []}
    with gpu_lock():
        for rows in ROWS:
            for name, (n, k) in SHAPES.items():
                a = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
                w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
                for _ in range(warmup):
                    torch.nn.functional.linear(a, w)
                torch.cuda.synchronize()
                times = []
                for _ in range(iters):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    torch.nn.functional.linear(a, w)
                    e.record()
                    e.synchronize()
                    times.append(s.elapsed_time(e))
                ms = statistics.median(times)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    torch.nn.functional.linear(a, w)
                    torch.cuda.synchronize()
                results["rows"].append({"M": rows, "layer": name, "N": n, "K": k, "ms": ms,
                                        "tflops": 2 * rows * n * k / ms / 1e9, "kernel": heaviest_kernel(prof),
                                        "sm_clock_mhz": sm_clock_mhz()})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1))
    print(f"{results['torch']} cuda {results['cuda']} on {results['device']}: {len(results['rows'])} shapes -> {out}")
    return 0


def heaviest_kernel(prof) -> str:
    """Name of the longest CUDA kernel in the profile. The first CUDA event is often the Memset that zero-fills a split-K
    workspace (round 7 recorded that for every shape), so it says nothing about which GEMM ran."""
    import torch
    events = [ev for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA
              and not ev.name.startswith(("Memset", "Memcpy"))]
    if not events:
        return ""
    best = max(events, key=lambda ev: getattr(ev, "device_time", 0) or getattr(ev, "cuda_time", 0) or 0)
    return best.name[:70]


def sm_clock_mhz() -> float | None:
    """SM clock while the card is busy: a GEMM timed at 1,050 MHz and one at 1,785 MHz are not the same measurement."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5, check=True).stdout
        return float(out.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def known_kernel(name: str) -> str:
    return name if name and not name.startswith("Memset") else "n/a (not recorded)"


def compare(first: Path, second: Path) -> int:
    a, b = json.loads(first.read_text()), json.loads(second.read_text())
    lines = [f"| M | layer | {first.stem} ms ({a['torch']}, cuda {a['cuda']}) | {second.stem} ms ({b['torch']}, cuda {b['cuda']}) | ratio | TFLOPs | SM clock MHz (during run) | kernels |",
             "|---:|---|---:|---:|---:|---:|---|---|"]
    other = {(r["M"], r["layer"]): r for r in b["rows"]}
    for r in a["rows"]:
        o = other[(r["M"], r["layer"])]
        lines.append(f"| {r['M']} | {r['layer']} | {r['ms']:.3f} | {o['ms']:.3f} | {r['ms'] / o['ms']:.2f} | "
                     f"{r['tflops']:.0f} / {o['tflops']:.0f} | {r.get('sm_clock_mhz') or 'n/a'} / {o.get('sm_clock_mhz') or 'n/a'} | "
                     f"`{known_kernel(r['kernel'])[:28]}` / `{known_kernel(o['kernel'])[:28]}` |")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--out", type=Path, required=True)
    r.add_argument("--warmup", type=int, default=20)
    r.add_argument("--iters", type=int, default=50)
    c = sub.add_parser("compare")
    c.add_argument("first", type=Path)
    c.add_argument("second", type=Path)
    args = ap.parse_args()
    return run(args.out, args.warmup, args.iters) if args.cmd == "run" else compare(args.first, args.second)


if __name__ == "__main__":
    raise SystemExit(main())
