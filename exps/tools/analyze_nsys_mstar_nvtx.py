#!/usr/bin/env python3
"""Where does an M* request spend host time: inside node forwards, or between them (scheduling)?

Needs an nsys report of an M* cell started with `mstar serve --enable-nvtx` (the engine wraps each
node forward in `engine.<node>.<walk>.bs<N>`). For every measured request (client UTC window against the
nsys session start) and every (node, walk) it reports

  calls        number of forwards
  forward ms   host time inside the forward ranges (kernel *launch* time under nsys, not GPU time)
  gap ms       idle host time between consecutive forwards on the engine thread, i.e. the cost of
               everything outside the forward: conductor round trip, plan/admit, edge hand-off
  GPU busy ms  kernel time launched from inside the node's forwards

and the request totals: share of the span inside forwards, median gap, p90 gap. This is the direct
measurement of "does graph + edge scheduling cost anything per step?": if the gap is small next to the
forward, per-step time is forward-bound and the scheduling layer is not what makes M* fast or slow.

    envs/mstar/bin/python exps/tools/analyze_nsys_mstar_nvtx.py --campaign BAGEL_NVTX_02 \\
        --task-dir BAGEL --case t2i-256
    envs/mstar/bin/python exps/tools/analyze_nsys_mstar_nvtx.py --campaign QWEN_A2T_NVTX_01 \\
        --task-dir QWEN_A2T --case a2t-en --match fused_moe      # MoE kernel launches per Thinker forward
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths
from exps.tools.analyze_nsys_t2s_stages import NS, NVTX_NODE, union

FRAMEWORK = "mstar"


def node_ranges(con: sqlite3.Connection, t0: float, t1: float) -> list[tuple[int, int, str, str, int]]:
    """(start, end, node, walk, globalTid) of node-forward ranges inside [t0, t1], time ordered."""
    out = []
    for start, end, text, gtid in con.execute(
        "select start, end, text, globalTid from NVTX_EVENTS "
        "where text like 'engine.%' and end is not null and start between ? and ? order by start", (t0, t1)
    ):
        m = NVTX_NODE.match(text or "")
        if m:
            out.append((start, end, m.group(1), m.group(2), gtid))
    return out


def _kernels_by_node(con: sqlite3.Connection, ranges, t0: float, t1: float):
    """Yield ((node, walk), kernel_start, kernel_end, correlationId) for kernels launched from inside a node range."""
    import bisect

    per_thread: dict[int, tuple[list[int], list[tuple[int, str, str]]]] = {}
    grouped: dict[int, list[tuple[int, int, str, str]]] = defaultdict(list)
    for start, end, node, walk, gtid in ranges:
        grouped[gtid].append((start, end, node, walk))
    for gtid, rows in grouped.items():
        rows.sort()
        per_thread[gtid] = ([r[0] for r in rows], [(r[1], r[2], r[3]) for r in rows])
    kernels = con.execute(
        "select k.correlationId, k.start, k.end from CUPTI_ACTIVITY_KIND_KERNEL k where k.start between ? and ?",
        (t0, t1)).fetchall()
    for i in range(0, len(kernels), 500):
        chunk = kernels[i:i + 500]
        by_cid: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
        for kernel in chunk:  # a CUDA-graph launch is one correlationId for many kernels
            by_cid[kernel[0]].append(kernel)
        marks = ",".join("?" * len(by_cid))
        for cid, gtid, launched in con.execute(
            f"select correlationId, globalTid, start from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId in ({marks})",
            list(by_cid)):
            starts, ends = per_thread.get(gtid, ([], []))
            k = bisect.bisect_right(starts, launched) - 1
            if k >= 0 and ends[k][0] >= launched:
                for _cid, kstart, kend in by_cid[cid]:
                    yield (ends[k][1], ends[k][2]), kstart, kend, cid


def gpu_busy_by_node(con: sqlite3.Connection, ranges, t0: float, t1: float) -> dict[tuple[str, str], float]:
    """Kernel time (ms, merged) launched from inside each node's ranges, via the launching runtime call."""
    intervals: dict[tuple[str, str], list[tuple[int, int]]] = defaultdict(list)
    for key, start, end, _cid in _kernels_by_node(con, ranges, t0, t1):
        intervals[key].append((start, end))
    return {key: sum(e - s for s, e in union(v)) / NS for key, v in intervals.items()}


def launches_by_node(con: sqlite3.Connection, ranges, t0: float, t1: float, pattern: str) -> dict[tuple[str, str], int]:
    """Number of kernels whose name matches `pattern` (regex, case-insensitive) launched inside each node's ranges."""
    import re

    rx = re.compile(pattern, re.I)
    names = {sid: value for sid, value in con.execute("select id, value from StringIds")}
    short = {(cid, start): name for cid, start, name in con.execute(
        "select correlationId, start, shortName from CUPTI_ACTIVITY_KIND_KERNEL where start between ? and ?", (t0, t1))}
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for key, start, _end, cid in _kernels_by_node(con, ranges, t0, t1):
        if rx.search(names.get(short.get((cid, start)), "") or ""):
            counts[key] += 1
    return dict(counts)


def batch_sizes(con: sqlite3.Connection, t0: float, t1: float) -> dict[tuple[str, str], dict[int, int]]:
    """How many forwards ran at each batch size, per (node, walk): the `.bs<N>` suffix of the NVTX range."""
    out: dict[tuple[str, str], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for text, in con.execute("select text from NVTX_EVENTS where text like 'engine.%' and end is not null "
                             "and start between ? and ?", (t0, t1)):
        m = NVTX_NODE.match(text or "")
        if m:
            out[(m.group(1), m.group(2))][int(text.rsplit(".bs", 1)[1])] += 1
    return {key: dict(v) for key, v in out.items()}


def per_request(db: Path, requests_file: Path, pad_s: float = 0.05, match: str | None = None) -> list[dict]:
    con = sqlite3.connect(db)
    origin = con.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    measured = [json.loads(line) for line in requests_file.read_text().splitlines()]
    measured = [r for r in measured if r["phase"] == "measured" and r.get("success")]
    out = []
    for req in measured:
        t0 = datetime.fromisoformat(req["started_at_utc"]).timestamp() * 1e9 - origin
        t1 = t0 + req["e2e_ms"] * 1e6
        ranges = node_ranges(con, t0 - pad_s * 1e9, t1 + pad_s * 1e9)
        if not ranges:
            continue
        by_thread: dict[int, list] = defaultdict(list)
        for r in ranges:
            by_thread[r[4]].append(r)
        gaps, covered = [], 0
        for rows in by_thread.values():
            rows.sort()
            end_prev = None
            for start, end, *_ in rows:
                covered += end - start
                if end_prev is not None and start > end_prev:
                    gaps.append((start - end_prev) / NS)
                end_prev = max(end_prev or end, end)
        by_node: dict[tuple[str, str], list[float]] = defaultdict(list)
        for start, end, node, walk, _ in ranges:
            by_node[(node, walk)].append((end - start) / NS)
        busy = gpu_busy_by_node(con, ranges, t0 - pad_s * 1e9, t1 + pad_s * 1e9)
        sizes = batch_sizes(con, t0 - pad_s * 1e9, t1 + pad_s * 1e9)
        matched = launches_by_node(con, ranges, t0 - pad_s * 1e9, t1 + pad_s * 1e9, match) if match else {}
        span = (ranges[-1][1] - ranges[0][0]) / NS
        out.append({
            "e2e_ms": req["e2e_ms"], "span_ms": span, "forward_share": covered / NS / span if span else 0.0,
            "gaps": {"n": len(gaps), "median_ms": statistics.median(gaps) if gaps else 0.0,
                     "p90_ms": sorted(gaps)[int(0.9 * (len(gaps) - 1))] if gaps else 0.0, "total_ms": sum(gaps)},
            "nodes": {f"{n}.{w}": {"calls": len(v), "forward_ms": sum(v), "median_forward_ms": statistics.median(v),
                                    "gpu_busy_ms": busy.get((n, w), 0.0), "matched": matched.get((n, w), 0),
                                    "batch_sizes": sizes.get((n, w), {})}
                      for (n, w), v in by_node.items()},
        })
    return out


def markdown(name: str, rows: list[dict], match: str | None = None) -> str:
    if not rows:
        return (f"# {name}\n\nNo `engine.<node>.<walk>.bs<N>` NVTX ranges inside any measured request. "
                "Run the M* cell with EXPS_MSTAR_EXTRA_ARGS=--enable-nvtx (carried by tools/mstar_serve_nvtx.py; \n"
                "`mstar serve` itself rejects the flag).\n")
    med = statistics.median
    nodes = sorted({n for r in rows for n in r["nodes"]})
    lines = [f"# {name}: M* host time inside vs between node forwards (medians over {len(rows)} requests)", "",
             f"client E2E (nsys-run) {med(r['e2e_ms'] for r in rows):.0f} ms; first-to-last forward span "
             f"{med(r['span_ms'] for r in rows):.0f} ms; inside forwards {100 * med(r['forward_share'] for r in rows):.0f}%; "
             f"gap between forwards: median {med(r['gaps']['median_ms'] for r in rows):.2f} ms, "
             f"p90 {med(r['gaps']['p90_ms'] for r in rows):.2f} ms, total {med(r['gaps']['total_ms'] for r in rows):.0f} ms "
             f"({med(r['gaps']['n'] for r in rows):.0f} gaps)", "",
             "| node.walk | calls | forward ms (host) | median forward ms | GPU busy ms | batch sizes (forwards) |"
             + (f" `{match}` launches | per call |" if match else ""), "|---|---:|---:|---:|---:|---|" + ("---:|---:|" if match else "")]
    for n in sorted(nodes, key=lambda x: -med(r["nodes"][x]["forward_ms"] for r in rows if x in r["nodes"])):
        pick = lambda key: med(r["nodes"][n][key] for r in rows if n in r["nodes"])  # noqa: E731
        merged: dict[int, int] = defaultdict(int)
        for r in rows:
            for bs, count in r["nodes"].get(n, {}).get("batch_sizes", {}).items():
                merged[bs] += count
        shown = ", ".join(f"bs{bs}: {round(count / len(rows))}" for bs, count in sorted(merged.items()))
        tail = f" {pick('matched'):.0f} | {pick('matched') / max(pick('calls'), 1):.1f} |" if match else ""
        lines.append(f"| {n} | {pick('calls'):.0f} | {pick('forward_ms'):.0f} | {pick('median_forward_ms'):.2f} | {pick('gpu_busy_ms'):.0f} | {shown} |{tail}")
    lines += ["", "Limits: one start; nsys inflates host time (BAGEL +6-18%, A2T +34%, T2S x1.6-2.2 and most for M*, the engine with the most launches; gaps and overlap fractions are structural, not absolute); ranges are host-side (launch), GPU busy is by launching range."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--task-dir", default="BAGEL", help="BAGEL or QWEN_T2S")
    ap.add_argument("--case", required=True)
    ap.add_argument("--match", help="Regex on kernel names; adds launches per node.walk (e.g. 'fused_moe' for MoE passes).")
    ap.add_argument("--markdown", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    root = paths.campaign_root(args.campaign)
    cell = next(iter(sorted((root / args.task_dir / FRAMEWORK / args.case).glob("r*"))), None)
    if cell is None:
        print(f"no cell under {root / args.task_dir / FRAMEWORK / args.case}", file=sys.stderr)
        return 2
    if not (cell / "requests.jsonl").exists():
        print(f"{cell} has no requests.jsonl: the cell failed before any request ran (see its server.log)", file=sys.stderr)
        return 2
    db = root / "nsys" / f"{FRAMEWORK}-{args.case}-{cell.name}.sqlite"
    rep = db.with_suffix(".nsys-rep")
    if (not db.exists() or db.stat().st_size == 0) and rep.exists():
        import subprocess
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o", str(db), str(rep)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rows = per_request(db, cell / "requests.jsonl", match=args.match)
    text = markdown(f"{args.campaign}/{args.case}", rows, args.match)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
