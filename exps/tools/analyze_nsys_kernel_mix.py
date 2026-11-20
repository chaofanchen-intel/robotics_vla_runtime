#!/usr/bin/env python3
"""Kernel mix of nsys reports: how many launches / how much GPU time per kernel family (CPU only).

Answers "did a switch change the kernels?" without reading code. Used for the BAGEL compile counterfactual:
`TORCHDYNAMO_DISABLE=1` must take effect, and the proof is zero inductor `triton_*` kernels in the report (the
`torch.compile applied` log line is printed regardless). Also lists the top kernels by launch count, which is where the
launch-count gap between engines sits (a name-heuristic family column is only a reading aid).

    envs/mstar/bin/python exps/tools/analyze_nsys_kernel_mix.py \\
        exps/runs/BAGEL_PROFILE_01/nsys/mstar-t2i-256-r01.sqlite \\
        exps/runs/BAGEL_NOCOMPILE_PROFILE_01/nsys/mstar-t2i-256-r01.sqlite --top 8 --markdown out.md

The report covers the whole process lifetime (warm-up and preconditions included): compare fractions, and use
analyze_nsys_requests.py for per-request totals.
"""
from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path


def family(name: str) -> str:
    low = name.lower()
    if low.startswith("triton"):
        return "inductor / triton fused"
    if any(key in low for key in ("nvjet", "gemm", "cutlass_kernel_flashinf", "cublas", "sm90_xmma", "splitk")):
        return "GEMM-like"
    if any(key in low for key in ("flash", "fmha", "prefillwithkv", "batchdecode", "batchprefill")):
        return "attention"
    if "catarray" in low or "copy" in low or "memcpy" in low:
        return "copy / cat"
    if any(key in low for key in ("index", "gather", "scatter", "embedding")):
        return "index / gather"
    if any(key in low for key in ("reduce", "softmax")):
        return "reduce / softmax"
    if "elementwise" in low or "vectorized" in low or "unrolled" in low:
        return "elementwise"
    return "other"


def mix(db: Path) -> dict:
    con = sqlite3.connect(db)
    rows = con.execute(
        "select s.value, count(*), sum(k.end - k.start) / 1e6 from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName group by 1").fetchall()
    launches = sum(r[1] for r in rows)
    busy = sum(r[2] for r in rows)
    fam_n, fam_ms = Counter(), Counter()
    for name, n, ms in rows:
        fam_n[family(name)] += n
        fam_ms[family(name)] += ms
    triton = [r for r in rows if r[0].startswith("triton")]
    return {"launches": launches, "busy_ms": busy, "families": (fam_n, fam_ms), "rows": sorted(rows, key=lambda r: -r[1]),
            "triton": (sum(r[1] for r in triton), sum(r[2] for r in triton), len(triton))}


def markdown(dbs: list[Path], top: int) -> str:
    mixes = {f"{db.parent.parent.name}/{db.stem}": mix(db) for db in dbs}
    lines = ["| report | kernels | summed kernel ms | inductor `triton_*` kernels (ms, distinct) |", "|---|---:|---:|---|"]
    for name, m in mixes.items():
        n, ms, distinct = m["triton"]
        lines.append(f"| {name} | {m['launches']:,} | {m['busy_ms']:,.0f} | {n:,} ({ms:,.0f}, {distinct}) |")
    families = sorted({f for m in mixes.values() for f in m["families"][0]})
    lines += ["", "Share of launches / of summed kernel time by name-heuristic family:", "",
              "| family | " + " | ".join(mixes) + " |", "|---|" + "---:|" * len(mixes)]
    for fam in families:
        cells = []
        for m in mixes.values():
            n, ms = m["families"]
            cells.append(f"{100 * n[fam] / m['launches']:.0f}% / {100 * ms[fam] / m['busy_ms']:.0f}%")
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    for name, m in mixes.items():
        lines += ["", f"Top {top} kernels by launches, {name}:", "", "| share | mean us | kernel |", "|---:|---:|---|"]
        for kernel, n, ms in m["rows"][:top]:
            lines.append(f"| {100 * n / m['launches']:.1f}% | {1000 * ms / n:.1f} | `{kernel[:90]}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    text = markdown(args.db, args.top)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
