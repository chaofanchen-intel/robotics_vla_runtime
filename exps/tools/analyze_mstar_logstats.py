#!/usr/bin/env python3
"""Summarise M*'s own per-request profile (`mstar serve --log-stats --log-stats-file F`), CPU only.

M* stamps each request with perf_counter checkpoints (recv -> preprocess -> conductor ingest -> first chunk -> last chunk
-> conductor done -> finish) and, per node.walk, the CPU time of the engine step split into forward / pre / post. No nsys:
the cost is a few perf_counter calls per step, so unlike nsys (T2S runs 1.6-2.2x slower under it, most for M*, the engine
with the most launches) the numbers are close to the real run. Two readings it makes possible:

* graph / edge scheduling cost: `pre` (prepare inputs, readiness) and `post` (output routing) against `fwd`, per node.walk;
* overlap without a profiler: if the sum of the nodes' busy time exceeds the request's wall time, nodes ran concurrently.

    envs/mstar/bin/python exps/tools/analyze_mstar_logstats.py F --last 10 [--markdown out.md]

`--last N` keeps the N most recent requests in the file (the measured ones; warm-up requests come first; 0 keeps all).
`--series NODE.WALK` adds the per-request `all` time of that step in request order, which shows a cost that is paid once per
new input shape (A2T `audio_encoder.prefill_audio`: 130 ms for the first six requests of a server, 15-23 ms afterwards).
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

RID = re.compile(r"^ Request profile: (\S+)")
SEGMENT = re.compile(r"^   (.+?)\s{2,}([\d.]+) ms\s*$")
NODE = re.compile(r"^   (\S+)\s*$")
WALK = re.compile(r"^     (\S+)\s+n=(\d+)\s+(.*)$")
CELL = re.compile(r"([\d.]+)\s+\(\s*([\d.]+)\)")


def parse(text: str) -> list[dict]:
    """One dict per request: {rid, segments: {label: ms}, walks: {node.walk: {n, all, fwd, pre, post}} (totals in ms)}."""
    requests, current, section, node = [], None, None, None
    for line in text.splitlines():
        m = RID.match(line)
        if m:
            current = {"rid": m.group(1), "segments": {}, "walks": {}}
            requests.append(current)
            section, node = None, None
            continue
        if current is None:
            continue
        if line.startswith(" Timeline"):
            section = "timeline"
            continue
        if line.startswith(" Graph timings"):
            section, node = "graph", None
            continue
        if line.startswith(" Tensor transfer") or line.startswith("====") or line.startswith("----"):
            section = None if not line.startswith("----") else section
            if line.startswith(("====", " Tensor")):
                section = None
            continue
        if section == "timeline":
            s = SEGMENT.match(line)
            if s:
                current["segments"][s.group(1).strip()] = float(s.group(2))
        elif section == "graph":
            w = WALK.match(line)
            if w and node:
                cells = CELL.findall(w.group(3))
                if len(cells) == 4:
                    current["walks"][f"{node}.{w.group(1)}"] = {
                        "n": int(w.group(2)), **{k: float(c[0]) for k, c in zip(("all", "fwd", "pre", "post"), cells)}}
                continue
            n = NODE.match(line)
            if n and "n=" not in line and not line.strip().startswith("*"):
                node = n.group(1)
    return requests


def series(requests: list[dict], key: str) -> list[str]:
    """`all` / `fwd` of one node.walk per request, in file order (request 1 is the first one the server ever saw)."""
    lines = [f"`{key}` per request, in order:", "", "| request | all ms | fwd ms |", "|---:|---:|---:|"]
    for i, r in enumerate(requests, 1):
        w = r["walks"].get(key)
        lines.append(f"| {i} | {w['all']:.1f} | {w['fwd']:.1f} |" if w else f"| {i} | n/a | n/a |")
    return lines + [""]


def markdown(name: str, requests: list[dict], series_key: str | None = None) -> str:
    if not requests:
        return f"# {name}\n\nNo `Request profile` blocks found (was `--log-stats-file` passed to `mstar serve`?).\n"
    med = statistics.median
    lines = [f"# {name}: M* per-request profile (median of {len(requests)} requests; ms)", ""]
    labels = sorted({k for r in requests for k in r["segments"]})
    totals = [r["segments"].get("total", 0.0) for r in requests]
    if labels:
        lines += ["| timeline segment | median ms |", "|---|---:|"] + [
            f"| {k} | {med(r['segments'][k] for r in requests if k in r['segments']):.1f} |" for k in labels] + [""]
    walks = sorted({k for r in requests for k in r["walks"]},
                   key=lambda k: -med(r["walks"][k]["all"] for r in requests if k in r["walks"]))
    lines += ["| node.walk | execs | all | fwd | pre | post | (pre+post) / all |", "|---|---:|---:|---:|---:|---:|---:|"]
    busy = 0.0
    for k in walks:
        rows = [r["walks"][k] for r in requests if k in r["walks"]]
        a, f, p, q = (med(x[key] for x in rows) for key in ("all", "fwd", "pre", "post"))
        busy += a
        lines.append(f"| {k} | {med(x['n'] for x in rows):.0f} | {a:.1f} | {f:.1f} | {p:.1f} | {q:.1f} | {100 * (p + q) / a if a else 0:.0f}% |")
    wall = med(totals) if any(totals) else 0.0
    lines += [""] + (series(requests, series_key) if series_key else [])
    lines += [f"sum of node.walk `all` = {busy:.0f} ms; request wall (total) = {wall:.0f} ms"
              + (f"; ratio {busy / wall:.2f} (> 1 means nodes ran concurrently)" if wall else ""),
              "", "Limits: CPU perf_counter times of the engine step; `post` overlaps the next step under speculative scheduling, "
              "so `all` is not the sum of the columns; one start per cell."]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", type=Path)
    ap.add_argument("--last", type=int, default=10)
    ap.add_argument("--series", help="node.walk whose per-request `all` time is listed in order")
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    if not args.file.exists():
        print(f"no such file: {args.file}", file=sys.stderr)
        return 2
    requests = parse(args.file.read_text(errors="replace"))
    requests = requests[-args.last:] if args.last > 0 else requests
    text = markdown(args.file.parent.name + "/" + args.file.name, requests, args.series)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
