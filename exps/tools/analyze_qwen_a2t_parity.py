#!/usr/bin/env python3
"""A2T cold-path vs steady-state parity across the three runtimes.

Hypothesis under test (from QWEN_A2T_CAUSE_01, M* only): M*'s higher A2T p50 in the main cohort is
driven by a one-off cold path on the first long audio request (new feature shape / graph / allocator
growth), not by its steady-state per-request cost. The main cohort has only 3 measured requests per
start, so one cold request dominates a median-of-three and hides what the other two show.

For every (framework, case) this reports, with starts kept separate:

* ``p50_all``       median E2E over all measured requests of a start
* ``first_ms``      E2E of the first measured request
* ``p50_warm``      median E2E excluding spike requests (E2E > ``SPIKE_FACTOR`` x the start's median
                    of the remaining requests) -- the steady-state estimate
* ``spikes``        number of requests excluded as cold spikes

Pre-registered decision rule (fix before looking at the new data):

* **parity**   -- M* ``p50_warm`` is within ±``PARITY_BAND`` of *each* other framework's ``p50_warm``
                  for the same case, over >= 3 starts.
* **residual gap** -- M* ``p50_warm`` exceeds another framework's by more than the band: the cold
                  path does not explain the whole gap; localise with stage traces.
* **cold-dominated** -- ``p50_all`` / ``p50_warm`` > ``COLD_INFLATION`` for M* but not for the others.

Run on a cohort with enough steady requests, e.g.::

    bash exps/run_qwen_a2t_cause_probe.sh --execute parity
    envs/mstar/bin/python exps/tools/analyze_qwen_a2t_parity.py --campaign QWEN_A2T_PARITY_01

Caveats: warm-up covers only the preconditioning clips; a measured clip longer than all of them can
still be cold (it shows up as a spike and is reported, not hidden). E2E only -- no stage attribution.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths

SPIKE_FACTOR = 3.0
PARITY_BAND = 0.10
COLD_INFLATION = 1.20
MIN_STARTS = 3
REFERENCE = "mstar"


def split_spikes(values: list[float]) -> tuple[list[float], int]:
    """Drop requests slower than SPIKE_FACTOR x the median of the others; return (warm, n_dropped)."""
    if len(values) < 3:
        return list(values), 0
    warm = []
    for index, value in enumerate(values):
        others = values[:index] + values[index + 1:]
        if value > SPIKE_FACTOR * statistics.median(others):
            continue
        warm.append(value)
    return warm, len(values) - len(warm)


def start_stats(requests_path: Path) -> dict[str, Any] | None:
    rows = [json.loads(line) for line in requests_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    measured = [r["e2e_ms"] for r in rows if r.get("phase") == "measured" and r.get("success") and r.get("e2e_ms")]
    if len(measured) < 2:
        return None
    warm, spikes = split_spikes(measured)
    return {
        "p50_all": statistics.median(measured),
        "first_ms": measured[0],
        "p50_warm": statistics.median(warm) if warm else None,
        "spikes": spikes,
        "n": len(measured),
    }


def collect(campaign_root: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    cases: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for manifest in sorted((campaign_root / "QWEN_A2T").glob("*/*/*/manifest.json")):
        cell = json.loads(manifest.read_text(encoding="utf-8"))
        requests = manifest.with_name("requests.jsonl")
        if cell.get("status") != "completed" or not requests.is_file():
            continue
        stats = start_stats(requests)
        if stats:
            cases[cell["case"]][cell["framework"]].append(stats)
    return cases


def judge(per_framework: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ref = per_framework.get(REFERENCE)
    if not ref or ref["starts"] < MIN_STARTS:
        return {"verdict": "insufficient_reference_starts"}
    comparisons, verdict_parts = {}, []
    for fw, row in per_framework.items():
        if fw == REFERENCE:
            continue
        if row["starts"] < MIN_STARTS:
            comparisons[fw] = {"verdict": "insufficient_starts"}
            continue
        ratio = ref["p50_warm"] / row["p50_warm"]
        gap = ratio - 1.0
        verdict = "parity" if abs(gap) <= PARITY_BAND else ("residual_gap" if gap > 0 else "mstar_faster")
        cold = (ref["p50_all"] / ref["p50_warm"] > COLD_INFLATION) and (row["p50_all"] / row["p50_warm"] <= COLD_INFLATION)
        comparisons[fw] = {"mstar_over_other_warm": ratio, "verdict": verdict, "mstar_cold_dominated": cold}
        verdict_parts.append(verdict)
    return {"comparisons": comparisons,
            "verdict": "parity" if verdict_parts and all(v in ("parity", "mstar_faster") for v in verdict_parts)
            else "residual_gap" if verdict_parts else "no_comparison"}


def analyze(campaign_root: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"campaign": campaign_root.name, "cases": {},
                           "rule": {"spike_factor": SPIKE_FACTOR, "parity_band": PARITY_BAND,
                                    "cold_inflation": COLD_INFLATION, "min_starts": MIN_STARTS}}
    for case, by_fw in collect(campaign_root).items():
        summary: dict[str, dict[str, Any]] = {}
        for fw, starts in by_fw.items():
            def med(key: str) -> float | None:
                vals = [s[key] for s in starts if s[key] is not None]
                return statistics.median(vals) if vals else None
            summary[fw] = {"starts": len(starts), "p50_all": med("p50_all"), "first_ms": med("first_ms"),
                           "p50_warm": med("p50_warm"), "spikes_per_start": statistics.fmean(s["spikes"] for s in starts),
                           "requests_per_start": starts[0]["n"]}
        out["cases"][case] = {"frameworks": summary, "judgement": judge(summary)}
    return out


def to_markdown(report: dict[str, Any]) -> str:
    def f(v: float | None) -> str:
        return "n/a" if v is None else f"{v:,.0f}"
    lines = ["# Qwen A2T cold vs steady-state parity", "",
             f"Rule: spike = E2E > {SPIKE_FACTOR}x median of the rest; parity = M* warm p50 within "
             f"±{PARITY_BAND:.0%}; needs >= {MIN_STARTS} starts.", ""]
    for case, row in report["cases"].items():
        lines += [f"## {case}", "", "| framework | starts | req/start | first (ms) | p50 all | p50 warm | spikes/start |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for fw, s in row["frameworks"].items():
            lines.append(f"| {fw} | {s['starts']} | {s['requests_per_start']} | {f(s['first_ms'])} | "
                         f"{f(s['p50_all'])} | {f(s['p50_warm'])} | {s['spikes_per_start']:.1f} |")
        j = row["judgement"]
        lines.append("")
        lines.append(f"Verdict: **{j['verdict']}**")
        for fw, c in j.get("comparisons", {}).items():
            if "mstar_over_other_warm" in c:
                lines.append(f"- vs {fw}: M*/other (warm) = {c['mstar_over_other_warm']:.2f} → {c['verdict']}"
                             f"{' (M* cold-dominated)' if c['mstar_cold_dominated'] else ''}")
            else:
                lines.append(f"- vs {fw}: {c['verdict']}")
        lines.append("")
    lines.append("Limits: E2E only; warm = spike-excluded estimate; an un-warmed clip shape can still spike.")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--cases", nargs="*", default=None, help="Restrict to these case names.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    root = paths.campaign_root(args.campaign)
    report = analyze(root)
    if args.cases:
        report["cases"] = {k: v for k, v in report["cases"].items() if k in set(args.cases)}
    if not report["cases"]:
        print(f"no completed QWEN_A2T cells under {root}", file=sys.stderr)
        return 2
    text = to_markdown(report)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
