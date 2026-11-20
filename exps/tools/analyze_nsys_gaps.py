#!/usr/bin/env python3
"""GPU idle gaps inside one request (gap-clustered nsys reports): micro-gaps vs scheduling-sized gaps.

The BAGEL step loop is either launch-bound (many ~10 us gaps between small kernels) or interrupted by
scheduling-sized gaps (>= 0.3 ms: a conductor round trip, an edge hand-off, a CPU sync). This counts
both for the last request cluster of each report, so the share of the request that sits in scheduling-sized
gaps is a measurement instead of an impression.

    envs/mstar/bin/python exps/tools/analyze_nsys_gaps.py exps/runs/BAGEL_PROFILE_01/nsys/*t2i-256-r01.sqlite
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.tools.analyze_nsys_requests import NS, clusters, load


def gap_summary(cluster) -> dict:
    cluster = sorted(cluster)
    gaps, end = [], cluster[0][1]
    for start, stop, _ in cluster[1:]:
        if start > end:
            gaps.append((start - end) / NS)
        end = max(end, stop)
    span = (max(e for _, e, _ in cluster) - cluster[0][0]) / NS
    mid = [g for g in gaps if 0.3 <= g < 2.0]
    big = [g for g in gaps if g >= 2.0]
    return {"span_ms": span, "gaps": len(gaps), "median_gap_us": statistics.median(gaps) * 1e3 if gaps else 0.0,
            "mid_n": len(mid), "mid_ms": sum(mid), "big_n": len(big), "big_ms": sum(big),
            "idle_ms": sum(gaps)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", nargs="+", type=Path)
    ap.add_argument("--min-launches", type=int, default=200)
    ap.add_argument("--last", type=int, default=3, help="Use only the last N request clusters (the measured ones).")
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    lines = ["| report | span ms | idle ms | median gap us | gaps 0.3-2 ms (n / ms) | gaps >= 2 ms (n / ms) |",
             "|---|---:|---:|---:|---:|---:|"]
    for db in args.db:
        rows, _ = load(db)
        found = [c for c in clusters(rows, int(2e9)) if len(c) >= args.min_launches]
        if not found:
            lines.append(f"| {db.parent.parent.name}/{db.stem} | no request cluster | | | | |")
            continue
        # warm-up / compile / precondition clusters come first and are far longer; the measured requests are the
        # last `--last` clusters (3 per nsys cell). Medians over those, one start per report.
        per = [gap_summary(c) for c in found[-args.last:]]
        med = lambda key: statistics.median(g[key] for g in per)  # noqa: E731
        label = f"{db.parent.parent.name}/{db.stem}"
        lines.append(f"| {label} ({len(per)} requests) | {med('span_ms'):.0f} | {med('idle_ms'):.0f} | {med('median_gap_us'):.1f} | "
                     f"{med('mid_n'):.0f} / {med('mid_ms'):.0f} | {med('big_n'):.0f} / {med('big_ms'):.0f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
