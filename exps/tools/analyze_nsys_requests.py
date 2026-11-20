#!/usr/bin/env python3
"""Per-request GPU activity from an Nsight Systems report (C=1 request cells).

Why: the BAGEL ablation localises the M* vs vLLM-Omni gap to the per-step / per-token cost. This
tool asks the next question for one cell: is that cost GPU kernel time, or GPU idle (CPU-side
launch / sync overhead)?

Method. At C=1 with per-request thermal gating, requests are separated by long idle gaps
(cool-down >= 30 s). Kernels / memcpys are clustered by gap (default 2 s); each cluster with enough
GPU work is one request (or the start-up warm-up). For each cluster:

  span_ms     first activity start -> last activity end
  busy_ms     union of kernel+memcpy intervals (streams may overlap, so not a plain sum)
  idle_pct    1 - busy/span
  kernels     number of kernel launches
  top kernels by summed duration

Only the last `--last` clusters (default: precondition + measured requests) are reported. This is
a coarse view: it separates "GPU-bound" from "launch/CPU-bound", it does not attribute the gap to
the graph abstraction. Output is JSON + markdown.

    nsys export --type sqlite <report>.nsys-rep >/dev/null
    envs/mstar/bin/python exps/tools/analyze_nsys_requests.py <report>.sqlite --last 4
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

NS = 1e6


def load(db: Path) -> tuple[list[tuple[int, int, str]], dict[str, int]]:
    con = sqlite3.connect(db)
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    rows: list[tuple[int, int, str]] = []
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        q = (
            "select k.start, k.end, coalesce(s.value, k.demangledName) from CUPTI_ACTIVITY_KIND_KERNEL k "
            "left join StringIds s on s.id = k.demangledName"
        )
        rows += [(a, b, str(n)) for a, b, n in con.execute(q)]
    n_kernels = len(rows)
    for table, label in (("CUPTI_ACTIVITY_KIND_MEMCPY", "[memcpy]"), ("CUPTI_ACTIVITY_KIND_MEMSET", "[memset]")):
        if table in tables:
            rows += [(a, b, label) for a, b in con.execute(f"select start, end from {table}")]
    rows.sort()
    return rows, {"kernels": n_kernels, "total": len(rows)}


def clusters(rows, gap_ns: int):
    out, cur, last_end = [], [], None
    for r in rows:
        if last_end is not None and r[0] - last_end > gap_ns:
            out.append(cur)
            cur = []
        cur.append(r)
        last_end = r[1] if last_end is None else max(last_end, r[1])
    if cur:
        out.append(cur)
    return out


def union_ns(rows) -> int:
    total, cur_s, cur_e = 0, None, None
    for s, e, _ in rows:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def summarize(cluster, top: int) -> dict:
    span = max(e for _, e, _ in cluster) - cluster[0][0]
    busy = union_ns(cluster)
    by_name: dict[str, list[int]] = {}
    for s, e, n in cluster:
        agg = by_name.setdefault(n, [0, 0])
        agg[0] += e - s
        agg[1] += 1
    ranked = sorted(by_name.items(), key=lambda kv: -kv[1][0])[:top]
    return {
        "span_ms": span / NS,
        "busy_ms": busy / NS,
        "idle_pct": 100.0 * (1 - busy / span) if span else 0.0,
        "launches": len(cluster),
        "kernel_sum_ms": sum(e - s for s, e, _ in cluster) / NS,
        "mean_kernel_us": sum(e - s for s, e, _ in cluster) / len(cluster) / 1e3,
        "top": [
            {"name": n[:110], "ms": v[0] / NS, "count": v[1], "share_pct": 100.0 * v[0] / max(1, sum(e - s for s, e, _ in cluster))}
            for n, v in ranked
        ],
    }


def units_for(case: str) -> int:
    """Work units a case does per request: denoise steps (image tasks) or completion tokens (I2T)."""
    import re

    m = re.search(r"tokens(\d+)", case)
    if m:
        return int(m.group(1))
    if case.startswith("i2t"):
        return 128
    m = re.search(r"srv-steps(\d+)|steps(\d+)", case)
    return int(next(g for g in m.groups() if g)) if m else 50


def campaign_table(nsys_dir: Path, measured: int, gap_s: float, min_launches: int) -> str:
    """One row per report: median over the last `measured` request clusters."""
    import statistics
    import subprocess

    rows = []
    for rep in sorted(nsys_dir.glob("*.nsys-rep")):
        db = rep.with_suffix(".sqlite")
        if not db.exists() or db.stat().st_size == 0:
            subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o", str(db), str(rep)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        data, _ = load(db)
        stem0 = rep.stem
        # CUDA-graph decode paths issue few launches per token: a short I2T request can fall under
        # the default threshold, so the I2T cases use a low one (start-up clusters are dropped by `[-measured:]`).
        floor = 20 if "-i2t-" in stem0 else min_launches
        cl = [c for c in clusters(data, int(gap_s * 1e9)) if len(c) >= floor][-measured:]
        sums = [summarize(c, 3) for c in cl]
        if not sums:
            continue
        stem = rep.stem
        if "_service_lifecycles-" in stem:  # name written before lifecycle.nsys_wrap normalised it
            stem = "-".join(stem.split("_service_lifecycles-", 1)[1].split("_")[-3:])
        fw, case = None, stem
        for prefix in ("vllm-omni-single-stage", "vllm-omni-two-stage", "mstar", "sglang-omni", "vllm-omni"):
            if stem.startswith(prefix + "-"):
                fw, case = prefix, stem[len(prefix) + 1:].rsplit("-r", 1)[0]
                break
        unit = units_for(case) if case.startswith(("t2i", "i2i", "i2t")) else 0
        med = lambda k: statistics.median(s[k] for s in sums)  # noqa: E731
        rows.append((fw, case, len(sums), unit, med("span_ms"), med("busy_ms"), med("idle_pct"),
                     med("launches"), med("mean_kernel_us")))
    out = ["| framework | case | n | span ms | GPU busy ms | idle % | launches | mean kernel us |"
           " busy ms/unit | launches/unit |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fw, case, n, unit, span, busy, idle, launches, mk in rows:
        per = f"{busy / unit:.2f} | {launches / unit:.0f}" if unit else "- | -"
        out.append(f"| {fw} | {case} | {n} | {span:.0f} | {busy:.0f} | {idle:.1f} | {launches:.0f} | {mk:.1f} | {per} |")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("db", type=Path, help="A .sqlite export, or a directory of .nsys-rep files (table mode).")
    ap.add_argument("--measured", type=int, default=3, help="Table mode: request clusters per report.")
    ap.add_argument("--last", type=int, default=4, help="Report the last N clusters (precondition + measured).")
    ap.add_argument("--gap-s", type=float, default=2.0)
    ap.add_argument("--min-launches", type=int, default=200)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--steps", type=int, default=0, help="Divide per-request numbers by this many steps/tokens.")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()

    if args.db.is_dir():
        text = campaign_table(args.db, args.measured, args.gap_s, args.min_launches)
        print(text)
        if args.markdown:
            args.markdown.write_text(text)
        return 0

    rows, counts = load(args.db)
    cl = [c for c in clusters(rows, int(args.gap_s * 1e9)) if len(c) >= args.min_launches]
    chosen = cl[-args.last:]
    summaries = [summarize(c, args.top) for c in chosen]
    result = {"db": str(args.db), "counts": counts, "clusters_found": len(cl), "requests": summaries}
    if args.steps:
        for s in summaries:
            s["per_unit_busy_ms"] = s["busy_ms"] / args.steps
            s["per_unit_span_ms"] = s["span_ms"] / args.steps
            s["per_unit_launches"] = s["launches"] / args.steps

    lines = [f"# {args.db.name}", "", f"clusters with >= {args.min_launches} launches: {len(cl)}; reporting last {len(chosen)}", ""]
    lines.append("| # | span ms | busy ms | idle % | launches | mean kernel us |" + (" busy ms/unit | launches/unit |" if args.steps else ""))
    lines.append("|---|---:|---:|---:|---:|---:|" + ("---:|---:|" if args.steps else ""))
    for i, s in enumerate(summaries):
        extra = f" {s['per_unit_busy_ms']:.2f} | {s['per_unit_launches']:.0f} |" if args.steps else ""
        lines.append(
            f"| {i} | {s['span_ms']:.0f} | {s['busy_ms']:.0f} | {s['idle_pct']:.1f} | {s['launches']} | {s['mean_kernel_us']:.1f} |{extra}"
        )
    if summaries:
        lines += ["", "Top kernels (last request):", "", "| kernel | ms | count | share % |", "|---|---:|---:|---:|"]
        for t in summaries[-1]["top"]:
            lines.append(f"| `{t['name']}` | {t['ms']:.1f} | {t['count']} | {t['share_pct']:.1f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
