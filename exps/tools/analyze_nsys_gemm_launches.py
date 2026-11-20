#!/usr/bin/env python3
"""Per-launch time of the heaviest GEMM-like kernels in nsys reports, side by side (CPU only).

The question it answers: when two engines run "the same" layer, is one launch slower, or does it just run on more rows? Kernel
names differ between cuBLAS builds (`nvjet_tst_...` vs `nvjet_sm90_tst_...`); the tile shape in the name and the launch count are what
pair them (equal launches per request means the same call sites), not their rank. A persistent GEMM's grid equals the SM count
whatever the shape, so the grid column is not evidence of shape; only the per-launch time against known rows is.

    envs/mstar/bin/python exps/tools/analyze_nsys_gemm_launches.py \\
        exps/runs/BAGEL_PROFILE_07/nsys/mstar-t2i-768-r01.sqlite \\
        exps/runs/BAGEL_PROFILE_07/nsys/vllm-omni-single-stage-t2i-768-r01.sqlite --top 3 --markdown out.md

Per-launch means mix every layer that uses the kernel, and the report covers the whole process (warm-up included): compare
between reports of the same case, not with a microbenchmark. The SM clock during the profile is in each request's
`thermal_monitor` (requests.jsonl); a launch-time ratio is only the engine's after dividing out the clock ratio.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

GEMM_KEYS = ("nvjet", "gemm", "cutlass", "cublas", "sm90_xmma", "splitk")


def top_gemms(db: Path, top: int) -> tuple[list[dict], float]:
    con = sqlite3.connect(db)
    rows = con.execute(
        "select s.value, count(*), sum(k.end - k.start) / 1e6, avg(k.end - k.start) / 1e3, "
        "min(k.end - k.start) / 1e3, max(k.end - k.start) / 1e3 "
        "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName group by 1").fetchall()
    busy = sum(r[2] for r in rows)
    gemms = [r for r in rows if any(key in r[0].lower() for key in GEMM_KEYS) and "memset" not in r[0].lower()]
    gemms.sort(key=lambda r: -r[2])
    return [{"name": n, "launches": c, "total_ms": t, "mean_us": m, "min_us": lo, "max_us": hi} for n, c, t, m, lo, hi in gemms[:top]], busy


def markdown(dbs: list[Path], top: int) -> str:
    reports = {f"{db.parent.parent.name}/{db.stem}": top_gemms(db, top) for db in dbs}
    lines = ["| report | rank | kernel | launches | mean us / launch | min - max us | total ms | share of kernel time |",
             "|---|---:|---|---:|---:|---|---:|---:|"]
    for name, (rows, busy) in reports.items():
        for rank, row in enumerate(rows, 1):
            lines.append(f"| {name} | {rank} | `{row['name'][:44]}` | {row['launches']:,} | {row['mean_us']:,.0f} | "
                         f"{row['min_us']:,.0f} - {row['max_us']:,.0f} | {row['total_ms']:,.0f} | {row['total_ms'] / busy:.0%} |")
    lines += ["", "Pair kernels by tile in the name and by launch count, not by rank: a different tile choice or a split of one layer over two kernels "
              "makes the ranks of two engines refer to different layers. Means cover the whole process lifetime."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sqlite", nargs="+", type=Path)
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    text = markdown(args.sqlite, args.top)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
