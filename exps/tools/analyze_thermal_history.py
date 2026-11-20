#!/usr/bin/env python3
"""Audit GPU thermal state across every measured request in exps/runs (CPU only, read-only).

The harness (core/thermal.py) gates each request on <= 70 C and the full nominal SM clock, and records what the card did
*during* the request: `thermal_monitor` holds min/mean/max SM clock, max temperature, mean power and the clock-event
reasons that were active. This tool turns those fields into the answers to two questions:

* Did the card throttle or drift while we measured?  (share of thermally clean requests, lowest clock, hottest request,
  clock-event reasons, cooldown waits that timed out)
* Does latency move with temperature inside one cell?  (E2E divided by the cell median, against the temperature at the
  start of the request; reported as % per degree with the number of requests behind it)

    envs/mstar/bin/python exps/tools/analyze_thermal_history.py                  # everything under exps/runs
    envs/mstar/bin/python exps/tools/analyze_thermal_history.py \\
        --split 2026-10-27T05:48:00+00:00 --markdown out.md                             # before / after a cooling change
    envs/mstar/bin/python exps/tools/analyze_thermal_history.py --campaign CONCURRENCY_04 --campaign GEMM_ENV_01

``--split`` takes the UTC instant of a change (reboot, fan policy); requests are grouped by their ``started_at_utc``.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

RUNS = Path(__file__).resolve().parents[1] / "runs"


def load(runs: Path, campaigns: list[str] | None) -> list[dict]:
    """One row per measured request that has a thermal record, with its cell identity."""
    rows = []
    for path in sorted(runs.glob("*/*/*/*/*/requests.jsonl")):
        campaign = path.relative_to(runs).parts[0]
        if campaigns and campaign not in campaigns:
            continue
        parts = path.relative_to(runs).parts
        cell = "/".join(parts[:-1])
        framework, case = parts[2], parts[3]
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("phase") != "measured" or not r.get("success") or not r.get("started_at_utc"):
                continue
            monitor = r.get("thermal_monitor") or {}
            if not monitor and r.get("temperature_c_before") is None:
                continue
            rows.append({
                "campaign": campaign, "cell": cell, "framework": framework, "case": case,
                "when": datetime.fromisoformat(r["started_at_utc"]).astimezone(timezone.utc),
                "e2e_ms": r.get("e2e_ms"),
                "t_before": r.get("temperature_c_before"),
                "t_max": monitor.get("temperature_c_max", r.get("temperature_c_max")),
                "clock_min": monitor.get("sm_clock_mhz_min"),
                "clock_mean": monitor.get("sm_clock_mhz_mean"),
                "nominal": monitor.get("nominal_sm_mhz", r.get("sm_clock_mhz_nominal")),
                "power_w": monitor.get("power_w_mean"),
                "clean": r.get("thermally_clean"),
                "reasons": r.get("throttle_reasons_active") or monitor.get("event_reasons_seen") or [],
                "waited_s": r.get("cooldown_waited_s"),
                "timed_out": bool(r.get("cooldown_timed_out")),
            })
    return rows


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def summarise(rows: list[dict]) -> dict:
    clocks = [r["clock_min"] for r in rows if r["clock_min"] is not None]
    t_max = [r["t_max"] for r in rows if r["t_max"] is not None]
    t0 = [r["t_before"] for r in rows if r["t_before"] is not None]
    waited = [r["waited_s"] for r in rows if r["waited_s"] is not None]
    reasons = Counter(x for r in rows for x in r["reasons"])
    clean = [r["clean"] for r in rows if r["clean"] is not None]
    nominal = next((r["nominal"] for r in rows if r["nominal"]), None)
    return {
        "n": len(rows),
        "clean": f"{sum(1 for c in clean if c)}/{len(clean)}" if clean else "n/a",
        "clock_min": min(clocks) if clocks else None,
        "below_nominal": sum(1 for c in clocks if nominal and c < nominal) if clocks else None,
        "t_start_med": statistics.median(t0) if t0 else None,
        "t_max_p95": quantile(t_max, 0.95) if t_max else None,
        "t_max_max": max(t_max) if t_max else None,
        "power_w": statistics.median([r["power_w"] for r in rows if r["power_w"] is not None]) if any(
            r["power_w"] is not None for r in rows) else None,
        "wait_med": statistics.median(waited) if waited else None,
        "timeouts": sum(1 for r in rows if r["timed_out"]),
        "reasons": dict(reasons),
    }


def sensitivity(rows: list[dict], min_n: int = 6) -> tuple[float | None, int, float | None]:
    """Least-squares slope of (E2E / cell median - 1) against start temperature, pooled over cells with a real spread.

    Cells that start every request at the same temperature say nothing about sensitivity, so only cells whose start
    temperature varies by >= 3 C and that have at least ``min_n`` requests contribute. Returns (% per C, requests used,
    temperature span covered)."""
    cells = defaultdict(list)
    for r in rows:
        if r["e2e_ms"] and r["t_before"] is not None:
            cells[r["cell"]].append(r)
    xs, ys = [], []
    for members in cells.values():
        temps = [m["t_before"] for m in members]
        if len(members) < min_n or max(temps) - min(temps) < 3:
            continue
        base = statistics.median(m["e2e_ms"] for m in members)
        mean_t = statistics.fmean(temps)
        xs += [m["t_before"] - mean_t for m in members]       # centre per cell: the slope is within-cell only
        ys += [m["e2e_ms"] / base - 1 for m in members]
    if len(xs) < min_n:
        return None, len(xs), None
    denominator = sum(x * x for x in xs)
    if denominator == 0:
        return None, len(xs), None
    slope = sum(x * y for x, y in zip(xs, ys)) / denominator
    return slope * 100, len(xs), max(xs) - min(xs)


def fmt(value, spec="{:.0f}") -> str:
    return "n/a" if value is None else spec.format(value)


def table(groups: dict[str, list[dict]]) -> list[str]:
    lines = ["| group | measured requests | thermally clean | lowest in-request SM clock (MHz) | requests below nominal clock | "
             "start temp median (C) | max temp p95 / max (C) | median power (W) | cooldown wait median (s) | cooldown timeouts | "
             "clock-event reasons seen |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for name, rows in groups.items():
        if not rows:
            lines.append(f"| {name} | 0 | | | | | | | | | |")
            continue
        s = summarise(rows)
        lines.append(f"| {name} | {s['n']} | {s['clean']} | {fmt(s['clock_min'])} | {fmt(s['below_nominal'])} | "
                     f"{fmt(s['t_start_med'])} | {fmt(s['t_max_p95'])} / {fmt(s['t_max_max'])} | {fmt(s['power_w'])} | "
                     f"{fmt(s['wait_med'])} | {s['timeouts']} | {s['reasons'] or 'none'} |")
    return lines


def by_case(rows: list[dict]) -> list[str]:
    """Per (campaign, framework, case): how far the clock fell and what the latency was, so engines can be compared at equal
    clocks. ``clock mean`` is the mean over requests of the in-request mean SM clock."""
    groups = defaultdict(list)
    for r in rows:
        groups[(r["campaign"], r["framework"], r["case"])].append(r)
    lines = ["| campaign | framework | case | n | E2E p50 (ms) | clock mean (MHz) | clock min (MHz) | clean | start / max temp (C) | "
             "power (W) | reasons |", "|---|---|---|---:|---:|---:|---:|---:|---|---:|---|"]
    for (campaign, framework, case), members in sorted(groups.items()):
        s = summarise(members)
        e2e = [m["e2e_ms"] for m in members if m["e2e_ms"]]
        means = [m["clock_mean"] for m in members if m["clock_mean"] is not None]
        lines.append(f"| {campaign} | {framework} | {case} | {s['n']} | {fmt(statistics.median(e2e) if e2e else None)} | "
                     f"{fmt(statistics.fmean(means) if means else None)} | {fmt(s['clock_min'])} | {s['clean']} | "
                     f"{fmt(s['t_start_med'])} / {fmt(s['t_max_max'])} | {fmt(s['power_w'])} | {s['reasons'] or 'none'} |")
    return lines


def cooldowns(runs: Path, campaigns: list[str] | None = None, hot_c: float = 76.0, target_c: float = 70.0) -> list[dict]:
    """Per measured request that ended hot: how long the gate took to bring the card back to ``target_c``.

    Reads the gate traces (``cooldowns/measured_*.csv``). Only traces that start at or above ``hot_c`` and reach ``target_c``
    count, so the numbers are comparable between cells with different peak temperatures: ``s_per_c`` normalises by the drop.
    """
    rows = []
    for path in sorted(runs.glob("*/*/*/*/*/cooldowns/measured_*.csv")):
        parts = path.relative_to(runs).parts
        if campaigns and parts[0] not in campaigns:
            continue
        with path.open() as handle:
            trace = list(csv.DictReader(handle))
        if len(trace) < 8:
            continue
        times = [datetime.fromisoformat(r["wall_clock"]) for r in trace]
        temps = [float(r["temperature_c"]) for r in trace]
        if temps[0] < hot_c:
            continue
        reached = next(((t - times[0]).total_seconds() for t, c in zip(times, temps) if c <= target_c), None)
        if reached is None:
            continue
        stamp = times[0] if times[0].tzinfo else times[0].astimezone()      # the gate stamps local wall-clock time
        rows.append({"campaign": parts[0], "framework": parts[2], "case": parts[3], "when": stamp.astimezone(timezone.utc),
                     "t_start": temps[0], "seconds": reached, "s_per_c": reached / max(temps[0] - target_c, 1.0)})
    return rows


def cooldown_table(rows: list[dict], split: datetime | None, matched: tuple[str, ...] = ()) -> list[str]:
    """Median gate time per (framework, case), before and after ``split``, for cells present on both sides."""
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: {"before": [], "after": []})
    for r in rows:
        if matched and r["campaign"] not in matched:
            continue
        groups[(r["framework"], r["case"])]["after" if split and r["when"] >= split else "before"].append(r)
    lines = ["| framework | case | n before / after | start temp C before -> after | seconds to <= 70 C before -> after | "
             "seconds per C before -> after |", "|---|---|---:|---|---|---|"]
    for (framework, case), g in sorted(groups.items()):
        if split and not (g["before"] and g["after"]):
            continue
        pair = lambda key: " -> ".join(  # noqa: E731
            f"{statistics.median(r[key] for r in g[w]):.0f}" if g[w] else "n/a" for w in ("before", "after"))
        lines.append(f"| {framework} | {case} | {len(g['before'])} / {len(g['after'])} | {pair('t_start')} | {pair('seconds')} | "
                     f"{pair('s_per_c')} |")
    return lines


def same_cell(rows: list[dict], split: datetime, exclude: tuple[str, ...] = (), matched: tuple[str, ...] = ()) -> list[str]:
    """Same (framework, case) measured both before and after ``split``: did the cooling change move clock, power or latency?

    ``exclude`` drops campaigns that changed something else (a hook, extra logging); ``matched``, when given, keeps only those
    campaigns. A case name can hide different server configs across campaigns (compile off, nsys, cProfile all reuse `t2i-256`), so
    a whole-history table is not like-for-like: pass ``matched`` with the default-config campaigns of both periods.
    """
    groups: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: {"before": [], "after": []})
    for r in rows:
        if r["campaign"] in exclude or r["e2e_ms"] is None or (matched and r["campaign"] not in matched):
            continue
        groups[(r["framework"], r["case"])]["after" if r["when"] >= split else "before"].append(r)
    lines = ["| framework | case | n before / after | E2E p50 ms before -> after | SM clock mean MHz before -> after | "
             "power W before -> after | max temp C before -> after |", "|---|---|---:|---|---|---|---|"]
    for (framework, case), g in sorted(groups.items()):
        if not g["before"] or not g["after"]:
            continue

        def stat(key, which, spec="{:.0f}", agg=statistics.median):
            values = [r[key] for r in g[which] if r[key] is not None]
            return spec.format(agg(values)) if values else "n/a"
        pair = lambda key, **kw: f"{stat(key, 'before', **kw)} -> {stat(key, 'after', **kw)}"  # noqa: E731
        lines.append(f"| {framework} | {case} | {len(g['before'])} / {len(g['after'])} | {pair('e2e_ms')} | "
                     f"{pair('clock_mean', agg=statistics.mean)} | {pair('power_w', agg=statistics.mean)} | {pair('t_max', agg=max)} |")
    return lines


def report(rows: list[dict], split: datetime | None, cells: bool = False, exclude: tuple[str, ...] = (),
           cooling: list[dict] | None = None, matched: tuple[str, ...] = ()) -> str:
    out = ["# GPU thermal audit of measured requests", "",
           f"{len(rows)} measured requests with thermal telemetry from {len({r['campaign'] for r in rows})} campaigns; "
           "per request the harness records the SM clock and temperature *during* the request "
           "(core/thermal.py RequestThermalMonitor), and the gate before it requires <= 70 C and the full clock.", ""]
    if split:
        before = [r for r in rows if r["when"] < split]
        after = [r for r in rows if r["when"] >= split]
        out += [f"## Before / after {split.isoformat()}", ""] + table({"before": before, "after": after}) + [""]
        out += ["## Same framework and case before / after", "",
                "Campaigns excluded because they changed something other than the room: " + (", ".join(exclude) or "none") + ". "
                + ("Only these campaigns are compared (default-config cells of both periods): " + ", ".join(matched) + "."
                   if matched else "No whitelist: a case name can hide different server configs, so read this as a screen, not a measurement."), ""]
        out += same_cell(rows, split, exclude, matched) + [""]
        if cooling:
            out += ["## Cooling after a hot request, same framework and case", "",
                    "From the gate traces of requests that ended at >= 76 C: time until the card read <= 70 C. Faster cooling after "
                    "the change would show as smaller numbers in the `after` side. Heat soak (a chassis that has been loaded for "
                    "hours cools slower) is not separated from the room here.", ""] + cooldown_table(cooling, split, matched) + [""]
        slopes = {"before": sensitivity(before), "after": sensitivity(after)}
    else:
        slopes = {"all": sensitivity(rows)}
    out += ["## Latency against start temperature, within a cell", "",
            "| group | slope (% E2E per C) | requests used | temperature span (C) |", "|---|---:|---:|---:|"]
    for name, (slope, n, span) in slopes.items():
        out.append(f"| {name} | {fmt(slope, '{:+.2f}')} | {n} | {fmt(span, '{:.0f}')} |")
    out += ["", "Only cells whose start temperature varied by >= 3 C and that hold >= 6 requests contribute; a cell that always "
            "starts at the gate temperature cannot show a slope, so `n/a` or a tiny `n` means *not measurable here*, not *zero*.", ""]
    by_campaign = defaultdict(list)
    for r in rows:
        by_campaign[r["campaign"]].append(r)
    if cells:
        out += ["## Per campaign, framework and case", ""] + by_case(rows) + [""]
    first = {c: min(r["when"] for r in members) for c, members in by_campaign.items()}
    ordered = {c: by_campaign[c] for c in sorted(by_campaign, key=first.get)}
    out += ["## Per campaign (in time order)", ""] + table(ordered) + [""]
    flagged = [c for c, members in ordered.items() if any(not r["clean"] for r in members if r["clean"] is not None)]
    out += ["Campaigns with at least one request that was *not* thermally clean: " + (", ".join(flagged) if flagged else "none"),
            "", "Limits: the monitor samples the card every few hundred ms, so a throttle episode shorter than that is missed; "
            "`start temp` is the gate result (<= 70 C by construction), so it shows the card was allowed to start, not how hot "
            "the chassis was. CPU clocks and CPU thermal throttling are not recorded per request; see `cpu_thermal_snapshot` "
            "in exps/run_round8_experiments.sh for the host side."]
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=Path, default=RUNS)
    ap.add_argument("--campaign", action="append", help="restrict to these campaigns (repeatable)")
    ap.add_argument("--split", help="ISO instant with offset; group before / after (e.g. 2026-10-27T05:48:00+00:00)")
    ap.add_argument("--exclude", action="append", help="campaign to leave out of the before/after same-cell table (repeatable)")
    ap.add_argument("--matched", action="append", help="campaign to keep in the before/after same-cell and cooling tables (repeatable); "
                    "give the default-config campaigns of both periods")
    ap.add_argument("--cooldowns", action="store_true", help="add the gate-trace cooling-time table (needs --split)")
    ap.add_argument("--cells", action="store_true", help="add the per (campaign, framework, case) table with mean clock")
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    rows = load(args.runs, args.campaign)
    if not rows:
        print(f"no measured requests with thermal telemetry under {args.runs}", file=sys.stderr)
        return 2
    split = datetime.fromisoformat(args.split).astimezone(timezone.utc) if args.split else None
    cooling = cooldowns(args.runs, args.campaign) if args.cooldowns else None
    text = report(rows, split, args.cells, tuple(args.exclude or ()), cooling, tuple(args.matched or ()))
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
