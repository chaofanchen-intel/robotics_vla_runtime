#!/usr/bin/env python3
"""Summarise the pstats dumps written by tools/mstar_cprofile_hook (CPU only).

    envs/mstar/bin/python exps/tools/analyze_cprofile.py \
        compile-on=exps/runs/BAGEL_CPROF_PREFILL_ON/cprofile compile-off=exps/runs/BAGEL_CPROF_PREFILL_OFF/cprofile \
        --top 15 --markdown out.md

Per group: the profiled calls' total wall time (from hook_proof.jsonl) and the functions with the largest cumulative
time per call. With two groups, also the functions whose per-call cumulative time differs most (first group minus the
second), which names the frames that the switch added. cProfile inflates Python-heavy code: read ratios, not
milliseconds, and only between profiles taken the same way.
"""
from __future__ import annotations

import argparse
import json
import pstats
import statistics
import sys
from pathlib import Path


def load(directory: Path) -> tuple[pstats.Stats | None, int, float]:
    files = sorted(directory.glob("*.pstats"))
    if not files:
        return None, 0, 0.0
    stats = pstats.Stats(str(files[0]))
    for extra in files[1:]:
        stats.add(str(extra))
    proof = directory / "hook_proof.jsonl"
    walls = [json.loads(line)["wall_ms"] for line in proof.read_text().splitlines()] if proof.exists() else []
    return stats, len(files), statistics.median(walls) if walls else 0.0


def per_call(stats: pstats.Stats, calls: int) -> dict[tuple, tuple[float, float, int]]:
    out = {}
    for func, (cc, nc, tt, ct, _callers) in stats.stats.items():
        out[func] = (ct * 1e3 / calls, tt * 1e3 / calls, nc / calls)
    return out


def label(func: tuple) -> str:
    path, line, name = func
    short = "/".join(Path(path).parts[-3:]) if path and not path.startswith("~") else path
    return f"{name} ({short}:{line})" if line else name


def self_time(stats: pstats.Stats, calls: int, top: int) -> list[str]:
    """Functions with the most time *in themselves* (callees excluded): the table that names a blocking call, which the
    cumulative table hides under its callers."""
    rows = sorted(stats.stats.items(), key=lambda kv: -kv[1][2])[:top]
    lines = ["| self ms / call | calls / call | function |", "|---:|---:|---|"]
    lines += [f"| {tt * 1e3 / calls:.1f} | {nc / calls:.1f} | `{label(func)}` |" for func, (_cc, nc, tt, _ct, _c) in rows]
    return lines


def markdown(groups: dict[str, Path], top: int) -> str:
    lines = []
    data = {}
    for name, directory in groups.items():
        stats, calls, wall = load(directory)
        if stats is None:
            lines += [f"## {name}", "", f"No .pstats under `{directory}` (hook did not write: check hook_proof.jsonl).", ""]
            continue
        rows = per_call(stats, calls)
        data[name] = rows
        lines += [f"## {name}: {calls} profiled calls, median wall {wall:.0f} ms per call (profiler on)", "",
                  "| cumulative ms / call | self ms / call | calls / call | function |", "|---:|---:|---:|---|"]
        for func, (ct, tt, n) in sorted(rows.items(), key=lambda kv: -kv[1][0])[:top]:
            lines.append(f"| {ct:.1f} | {tt:.1f} | {n:.1f} | `{label(func)}` |")
        lines += ["", f"{name}: largest self time per call (total self time {sum(v[2] for v in stats.stats.values()) * 1e3 / calls:.0f} ms)",
                  ""] + self_time(stats, calls, 8) + [""]
    if len(data) >= 2:
        (a_name, a), (b_name, b) = list(data.items())[:2]
        diff = {f: a.get(f, (0, 0, 0))[0] - b.get(f, (0, 0, 0))[0] for f in set(a) | set(b)}
        lines += [f"## {a_name} minus {b_name}: largest cumulative-time differences per call", "",
                  f"| delta ms / call | {a_name} | {b_name} | function |", "|---:|---:|---:|---|"]
        for func, d in sorted(diff.items(), key=lambda kv: -abs(kv[1]))[:top]:
            lines.append(f"| {d:+.1f} | {a.get(func, (0,))[0]:.1f} | {b.get(func, (0,))[0]:.1f} | `{label(func)}` |")
        lines.append("")
    lines.append("Limits: cProfile covers the engine's GPU thread only and inflates Python-heavy code; compare profiles taken the same way.")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", help="name=directory with .pstats files")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    groups = {}
    for item in args.groups:
        name, _, path = item.partition("=")
        if not path:
            print(f"expected name=directory, got {item!r}", file=sys.stderr)
            return 2
        groups[name] = Path(path)
    text = markdown(groups, args.top)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
