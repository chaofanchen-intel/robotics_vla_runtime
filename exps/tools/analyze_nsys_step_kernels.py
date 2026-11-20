#!/usr/bin/env python3
"""Per-denoise-step kernel time of BAGEL T2I nsys reports, by kernel name (CPU only).

Why: the equal-work nsys table says M* does less GPU work per step than vLLM-Omni two-stage, not *which* kernels.
This tool takes the last burst of GPU activity, counts its requests as span / measured E2E, divides kernel time and launches by 50 steps per request, and lists the top kernels. Kernel names that exist
in only one engine are the localisation: vLLM-Omni's `mot_unified_gemm_kernel` (a Triton MoT GEMM whose config is
tuned per device; without a tuned config it logs "No tuned MoT config" and uses a conservative default).

    envs/mstar/bin/python exps/tools/analyze_nsys_step_kernels.py \\
        exps/runs/ATTR_BAGEL_EQ_NSYS_01/nsys --markdown out.md

Only nsys reports exported to sqlite are read (analyze_nsys_requests.py exports them on demand). Kernel-time sums
double count overlapping streams, so compare with the busy column of analyze_nsys_requests.py, not instead of it.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path

from exps.tools.analyze_nsys_requests import clusters, load, units_for

NS = 1e6


def normalise(name: str) -> str:
    """Make cuBLAS kernel names comparable across engines (`nvjet_sm90_tst_*` vs `nvjet_tst_*`)."""
    return re.sub(r"\(.*", "", name.replace("sm90_", "")).strip()[:90]


def request_e2e_ms(nsys_dir: Path, fw: str, case: str) -> float | None:
    """Median measured E2E of the matching cell (nsys-run), to count how many back-to-back requests a burst holds."""
    for path in nsys_dir.parent.glob(f"*/{fw}/{case}/r*/requests.jsonl"):
        values = []
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("phase") == "measured" and row.get("success"):
                values.append(row["e2e_ms"])
        if values:
            return statistics.median(values)
    return None


def union_ms(intervals: list[tuple[float, float]]) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for start, end in sorted(intervals):
        if cur_s is None or start > cur_e:
            if cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = start, end
        else:
            cur_e = max(cur_e, end)
    return total + (cur_e - cur_s if cur_s is not None else 0.0)


def per_step(db: Path, case: str, e2e_ms: float | None, gap_s: float, min_launches: int) -> dict:
    """Kernel time and launches per denoise step over the LAST burst of GPU activity.

    Back-to-back requests (no cool-down) leave no idle gap, so one burst holds the precondition and every measured
    request. The request count is span / measured E2E, and steps = 50 x that count; counting `measured` clusters, as
    this tool once did, divided a three-request burst by 50 and overstated every per-step number 2-3x.
    """
    data, _ = load(db)
    found = [c for c in clusters(data, int(gap_s * 1e9)) if len(c) >= min_launches]
    if not found or not e2e_ms:
        return {}
    cluster = found[-1]
    lo, hi = cluster[0][0], max(end for _, end, _ in cluster)
    requests = max(1, round((hi - lo) / NS / e2e_ms))
    con = sqlite3.connect(db)
    kernels = con.execute(
        "select k.start, k.end, coalesce(s.value, '') from CUPTI_ACTIVITY_KIND_KERNEL k "
        "left join StringIds s on s.id = k.demangledName where k.start >= ? and k.start <= ?", (lo, hi)).fetchall()
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for start, end, name in kernels:
        row = agg[normalise(name)]
        row[0] += (end - start) / NS
        row[1] += 1
    steps = units_for(case) * requests
    out = {name: (ms / steps, count / steps) for name, (ms, count) in agg.items()}
    out["__busy_union__"] = (union_ms([(s / NS, e / NS) for s, e, _ in kernels]) / steps, 0)
    out["__span__"] = ((hi - lo) / NS / steps, requests)
    return out


def report(nsys_dir: Path, gap_s: float, min_launches: int, top: int) -> str:
    lines = ["| report | requests in burst | span ms/step | GPU busy (union) ms/step | kernel-time sum ms/step | launches/step "
             "| `mot_*` ms/step | top kernels (ms/step × launches/step) |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    for db in sorted(nsys_dir.glob("*.sqlite")):
        m = re.match(r"(?P<fw>.+?)-(?P<case>t2i-\d+-(?:nocfg|cfg3))-r\d+$", db.stem)
        if not m:
            continue
        rows = per_step(db, m["case"], request_e2e_ms(nsys_dir, m["fw"], m["case"]), gap_s, min_launches)
        if not rows:
            continue
        busy = rows.pop("__busy_union__")[0]
        span, requests = rows.pop("__span__")
        total_ms = sum(v[0] for v in rows.values())
        total_n = sum(v[1] for v in rows.values())
        mot_ms = sum(v[0] for k, v in rows.items() if k.startswith(("mot_", "_mot_")))
        ranked = sorted(rows.items(), key=lambda kv: -kv[1][0])[:top]
        names = "<br>".join(f"{ms:.2f} × {n:.0f} `{name}`" for name, (ms, n) in ranked)
        lines.append(f"| {m['fw']} {m['case']} | {requests:.0f} | {span:.1f} | {busy:.1f} | {total_ms:.1f} | {total_n:.0f} "
                     f"| {mot_ms:.2f} | {names} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("nsys_dir", type=Path)
    parser.add_argument("--gap-s", type=float, default=0.5)
    parser.add_argument("--min-launches", type=int, default=2000)
    parser.add_argument("--top", type=int, default=6)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    text = report(args.nsys_dir, args.gap_s, args.min_launches, args.top)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)


if __name__ == "__main__":
    main()
