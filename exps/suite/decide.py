"""Turn finished `protocol-ab-<task>` experiments into the per-task protocol decision `baseline-*` needs."""
from __future__ import annotations

import statistics
from pathlib import Path

from exps import hardware, protocols
from exps.core import paths
from exps.suite import analysis, catalogue


def ratios_for(datas: list[analysis.ArmData], metric: str):
    """(b2b / gated p50, gap30 / gated p50, b2b / gated SM clock, campaigns, table rows), each per framework. Pooled p50 per cell,
    median over cases."""
    grouped = analysis.group_cells(datas)
    arms = {d.run.arm.name: d.run.campaign for d in datas}
    frameworks = list(dict.fromkeys(fw for (_a, fw, _c) in grouped))
    cases = list(dict.fromkeys(c for (_a, _f, c) in grouped))
    b2b: dict[str, float] = {}
    gap: dict[str, float] = {}
    clocks: dict[str, float] = {}
    rows: list[str] = []
    for fw in frameworks:
        per_b2b, per_gap, per_clock = [], [], []
        for case in cases:
            stat = {a: analysis.cell_stat(grouped.get((a, fw, case), []), metric) for a in ("gated", "b2b", "gap30")}
            if stat["gated"] and stat["b2b"]:
                per_b2b.append(stat["b2b"].p50 / stat["gated"].p50)
                if stat["gated"].clock_mhz and stat["b2b"].clock_mhz:
                    per_clock.append(stat["b2b"].clock_mhz / stat["gated"].clock_mhz)
                if stat["gap30"]:
                    per_gap.append(stat["gap30"].p50 / stat["gated"].p50)
                rows.append(f"| {fw} | {case} | {stat['gated'].p50:,.1f} | {stat['b2b'].p50:,.1f} | {per_b2b[-1]:.2f}x | "
                            + (f"{per_gap[-1]:.2f}x" if stat["gap30"] else "-") + " |")
        if per_b2b:
            b2b[fw] = statistics.median(per_b2b)
        if per_gap:
            gap[fw] = statistics.median(per_gap)
        if per_clock:
            clocks[fw] = statistics.median(per_clock)
    return b2b, gap, clocks, sorted(set(arms.values())), rows


def decide_all(*, tasks: list[str] | None, run_id: str, tolerance: float, **selection) -> tuple[dict, list[str]]:
    decisions = protocols.load()
    report = ["# Protocol decisions", "",
              f"Hardware profile: {hardware.current().name}. Rule and tolerance ({tolerance:.0%}) in `exps/protocols.py`. "
              "Ratio = back-to-back p50 / gated p50 (pooled), < 1: the engine is faster without the gate.", "",
              "| framework | case | gated p50 | back-to-back p50 | b2b / gated | gap30 / gated |", "|---|---|---:|---:|---:|---:|"]
    messages: list[str] = []
    summary: list[str] = ["", "| task | decision | kind | why |", "|---|---|---|---|"]
    for experiment in catalogue.PROTOCOL_AB:
        if tasks and experiment.task not in tasks:
            continue
        datas = analysis.collect(experiment, run_id=run_id, protocol=None, gap_s=0.0, arms=None, **selection)
        problems = [p for d in datas for p in d.problems]
        if problems:
            messages.append(f"{experiment.task}: not decided, {len(problems)} problem(s): " + "; ".join(problems[:3]))
            continue
        b2b, gap, clocks, evidence, rows = ratios_for(datas, experiment.metric)
        decisions[experiment.task] = protocols.record(experiment.task, b2b, evidence, gap30=gap or None, tolerance=tolerance,
                                                      clock_ratios=clocks or None)
        report += rows
        d = decisions[experiment.task]
        summary.append(f"| {experiment.task} | {' + '.join(d['protocols'])} | {d['kind']} | {d['reason']} |")
        messages.append(f"{experiment.task}: {' + '.join(d['protocols'])} ({d['kind']})")
    path = protocols.save(decisions)
    out = paths.RESULTS / f"protocol-decision_{hardware.current().name}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(report + summary) + "\n", encoding="utf-8")
    messages.append(f"wrote {paths.relative_to_repo(path)} and {paths.relative_to_repo(out)}")
    return decisions, messages
