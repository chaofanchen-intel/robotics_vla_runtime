#!/usr/bin/env python3
"""Validate an experiment's cells, then summarise them. CPU only, read only.

Numbers are reported only when every cell the experiment promised exists, completed, ran under the declared protocol and
overrides, and shows its switch took effect. A cell that fails any check turns the whole analysis into an error listing
every problem; a table with a hole in it is not published as if it were whole.

Statistics
----------
* **headline p50** = median over *all* measured requests of all starts of the cell (pooled). The old harness reported the
  median of per-start medians, which differs by a few percent on small cells and is not comparable with a pooled number.
* the **start range** beside it is [min-max] of the per-start medians: how far fresh starts of the same cell disagree.
* a ratio is `other / reference`; for cross-framework tables `S = baseline / M*` (> 1: M* faster).
* thermal columns come from the request-time telemetry (mean temperature before the request, mean SM clock) and the share
  of requests that started clean; they are never used to drop requests, only to qualify a ratio.
"""
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

from exps import hardware, protocols
from exps.core import manifest as manifest_mod
from exps.core import overrides, paths, registry
from exps.suite import runner
from exps.suite.spec import Arm, Experiment, Expect


class AnalysisError(SystemExit):
    pass


@dataclass
class CellData:
    framework: str
    case: str
    repeat: str
    path: Path
    manifest: dict[str, Any]
    rows: list[dict[str, Any]]


@dataclass
class ArmData:
    run: runner.ArmRun
    cells: list[CellData] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_cells(task: str, arm: Arm) -> list[tuple[str, str, str]]:
    spec = registry.load_task(task)
    cells = []
    for index in range(1, int(arm.repeats) + 1):
        for case in arm.cases:
            na = spec.case(case).not_applicable
            for framework in arm.frameworks:
                if framework not in na:
                    cells.append((framework, case, registry.repeat_id(index)))
    return cells


def metric_value(row: dict[str, Any], metric: str) -> float | None:
    if metric == "ms_per_audio_s":
        audio = row.get("audio_duration_s")
        return row["e2e_ms"] / audio if audio and row.get("e2e_ms") is not None else None
    value = row.get(metric)
    return float(value) if value is not None else None


def headline_rows(cell: CellData) -> list[dict[str, Any]]:
    return [r for r in cell.rows if r.get("phase") == "measured" and r.get("success") and not r.get("cooldown_timed_out")]


# --- validation ------------------------------------------------------------------------------------------------------------
def _within(started: str | None, finished: str | None, instant: str) -> bool:
    if not (started and finished and instant):
        return False
    t = datetime.fromisoformat(instant)
    return datetime.fromisoformat(started) <= t <= datetime.fromisoformat(finished)


def probe_files(campaign: str) -> list[dict[str, Any]]:
    root = paths.campaign_root(campaign)
    return [json.loads(p.read_text()) for p in sorted(root.glob("attn_probe.*.json"))]


def probe_for_cell(cell: CellData, campaign: str) -> dict[str, Any]:
    """Counters of the worker process(es) that started inside this cell's time window and loaded the BAGEL ViT module."""
    started, finished = cell.manifest.get("started_at_utc"), cell.manifest.get("finished_at_utc")
    mine = [p for p in probe_files(campaign) if _within(started, finished, p["process_start_utc"]) and p["vit_module_loaded"]]
    return {
        "processes": len(mine),
        "flash_calls": sum(p["vit_flash_attn_calls"] for p in mine),
        "sdpa_calls": sum(p["vit_sdpa_fallback_calls"] for p in mine),
        "flash_available": sorted({p["vit_flash_attn_available"] for p in mine}),
        "managers": sorted({m["manager"] for p in mine for m in p["attention_managers"]}),
    }


def _uses_vit(case: str) -> bool:
    return case.startswith(("i2t", "i2i"))


def check_proof(kind: str, run: runner.ArmRun, cell: CellData) -> list[str]:
    where = f"{cell.framework}/{cell.case}/{cell.repeat}"
    if kind in ("sampler_hook", "hook_applied"):
        proof = paths.campaign_root(run.campaign) / "hook_proof.jsonl"
        lines = _jsonl(proof) if proof.is_file() else []
        if not lines:
            return [f"{run.campaign}: sampler hook proof missing or empty ({proof}); the hook was not loaded in any worker"]
        if not all(line.get("applied") for line in lines):
            reasons = {line.get("refused") for line in lines if not line.get("applied")}
            return [f"{run.campaign}: sampler hook loaded but NOT applied: {reasons}"]
        return []
    if kind in ("attn_probe", "attn_probe_blocked"):
        probe = probe_for_cell(cell, run.campaign)
        problems = []
        if probe["processes"] == 0:
            return [f"{where}: no attn-probe process matched this cell's time window; the probe was not loaded"]
        if kind == "attn_probe":
            if "FlashInferManager" not in probe["managers"]:
                problems.append(f"{where}: LLM attention managers {probe['managers']}, expected FlashInferManager")
            if probe["flash_available"] != [True]:
                problems.append(f"{where}: pip flash_attn not available in the control arm: {probe['flash_available']}")
            if _uses_vit(cell.case) and probe["flash_calls"] == 0:
                problems.append(f"{where}: an image-input case made 0 flash-attn ViT calls")
            if not _uses_vit(cell.case) and (probe["flash_calls"] or probe["sdpa_calls"]):
                problems.append(f"{where}: a text-only case ran the ViT ({probe})")
        else:
            if probe["flash_available"] != [False] or probe["flash_calls"]:
                problems.append(f"{where}: flash_attn was not blocked: {probe}")
            if _uses_vit(cell.case) and probe["sdpa_calls"] == 0:
                problems.append(f"{where}: ViT never used the SDPA fallback although it ran")
        return problems
    return [f"unknown proof kind {kind!r}"]


def validate_arm(arm_data: ArmData) -> None:
    run = arm_data.run
    arm = run.experiment.resolved(run.arm)
    task = run.experiment.task
    problems = arm_data.problems
    want = overrides.build(
        env=runner.expand_all(arm.env, campaign=run.campaign), serve_args=runner.expand_all(arm.serve_args, campaign=run.campaign),
        options=runner.expand_all(arm.opts, campaign=run.campaign)).as_manifest()
    expect = _expanded_expect(arm.expect, run)
    proofs_done: set[str] = set()
    for framework, case, repeat in expected_cells(task, arm):
        cell_dir = paths.cell_dir(task, framework, case, repeat, campaign=run.campaign)
        where = f"{run.campaign}/{framework}/{case}/{repeat}"
        if not manifest_mod.cell_is_done(cell_dir):
            problems.append(f"{where}: no .cell_done (missing, failed or void)")
            continue
        record = json.loads((cell_dir / "manifest.json").read_text(encoding="utf-8"))
        rows = _jsonl(cell_dir / "requests.jsonl") if (cell_dir / "requests.jsonl").is_file() else []
        cell = CellData(framework, case, repeat, cell_dir, record, rows)
        arm_data.cells.append(cell)
        if record.get("status") != "completed":
            problems.append(f"{where}: status {record.get('status')!r}")
        recorded = record.get("overrides") or {}
        for key in ("env", "serve_args", "options"):
            if recorded.get(key, {}) != want[key]:
                problems.append(f"{where}: recorded overrides.{key} {recorded.get(key)} != arm's {want[key]}")
        protocol = record.get("protocol") or {}
        if bool(protocol.get("back_to_back")) != (run.protocol == "back-to-back"):
            problems.append(f"{where}: ran with back_to_back={protocol.get('back_to_back')}, arm wants {run.protocol}")
        if float(protocol.get("inter_request_gap_s") or 0.0) != float(run.gap_s):
            problems.append(f"{where}: inter-request gap {protocol.get('inter_request_gap_s')} != {run.gap_s}")
        measured = [r for r in rows if r.get("phase") == "measured"]
        if len(measured) != int(arm.measured):
            problems.append(f"{where}: {len(measured)} measured requests, expected {arm.measured}")
        failed = [r for r in measured if not r.get("success")]
        if failed:
            problems.append(f"{where}: {len(failed)} failed measured requests")
        if run.protocol == "gated" and any(r.get("cooldown_timed_out") for r in measured):
            problems.append(f"{where}: a cooldown gate timed out (those requests are excluded from the headline)")
        command = record.get("service_command")
        for needle in expect.command_has:
            if not command or needle not in command:
                problems.append(f"{where}: serve command lacks {needle!r}")
        for needle in expect.command_lacks:
            if command and needle in command:
                problems.append(f"{where}: serve command contains forbidden {needle!r}")
        if expect.log_has or expect.log_lacks:
            log = paths.lifecycle_dir(task, framework, case, repeat, campaign=run.campaign) / "server.log"
            text = log.read_text(errors="replace") if log.is_file() else ""
            for needle in (n for n in map(_scope, expect.log_has) if n[0] is None or framework.startswith(n[0])):
                if not _in_log(needle[1], text):
                    problems.append(f"{where}: server log lacks {needle[1]!r}")
            for needle in (n for n in map(_scope, expect.log_lacks) if n[0] is None or framework.startswith(n[0])):
                if _in_log(needle[1], text):
                    problems.append(f"{where}: server log contains {needle[1]!r}")
        if arm.expect.proof and framework == "mstar":
            if arm.expect.proof in ("sampler_hook", "hook_applied"):  # one proof file per campaign, not per cell
                if arm.expect.proof not in proofs_done:
                    proofs_done.add(arm.expect.proof)
                    problems.extend(check_proof(arm.expect.proof, run, cell))
            else:
                problems.extend(check_proof(arm.expect.proof, run, cell))


def _in_log(needle: str, text: str) -> bool:
    """A plain substring, or a regular expression when the entry starts with `re:` (for evidence a substring cannot express, e.g. a non-zero counter)."""
    return re.search(needle[3:], text) is not None if needle.startswith("re:") else needle in text


def _scope(entry: str) -> tuple[str | None, str]:
    """`mstar@kernel table: no moe_gemm` applies to M* logs only; an entry without `@` applies to every framework's log."""
    head, sep, rest = entry.partition("@")
    return (head, rest) if sep and head in ("mstar", "vllm-omni", "sglang-omni", "sglang-core", "voxserve") else (None, entry)


def _expand(value: str, run: runner.ArmRun) -> str:
    return runner.expand_text(value, campaign=run.campaign)


def _expanded_expect(expect: Expect, run: runner.ArmRun) -> Expect:
    """The arm's checks with `{variant_file:KEY}` / `{hw:NAME}` filled in for this machine; a check that expands to nothing is
    one only another machine needs (the M* TP=2 'prefix cache off at world size' line has no counterpart on one rank) and drops out."""
    return replace(expect, **{name: tuple(runner.expand_all(getattr(expect, name), campaign=run.campaign))
                              for name in ("command_has", "command_lacks", "log_has", "log_lacks")})


# --- statistics ------------------------------------------------------------------------------------------------------------
@dataclass
class Stat:
    starts: int
    n: int
    p50: float
    start_min: float
    start_max: float
    clean_pct: float | None
    temp_c: float | None
    clock_mhz: float | None


def cell_stat(cells: list[CellData], metric: str) -> Stat | None:
    pooled: list[float] = []
    per_start: list[float] = []
    clean: list[bool] = []
    temps: list[float] = []
    clocks: list[float] = []
    for cell in cells:
        values = []
        for row in headline_rows(cell):
            value = metric_value(row, metric)
            if value is None:
                continue
            values.append(value)
            clean.append(bool(row.get("thermally_clean")))
            if row.get("temperature_c_before") is not None:
                temps.append(row["temperature_c_before"])
            monitor = row.get("thermal_monitor") or {}
            if monitor.get("sm_clock_mhz_mean"):
                clocks.append(monitor["sm_clock_mhz_mean"])
            elif row.get("sm_clock_mhz_before"):
                clocks.append(row["sm_clock_mhz_before"])
        if values:
            pooled += values
            per_start.append(statistics.median(values))
    if not pooled:
        return None
    return Stat(
        starts=len(per_start), n=len(pooled), p50=statistics.median(pooled), start_min=min(per_start),
        start_max=max(per_start), clean_pct=100.0 * sum(clean) / len(clean) if clean else None,
        temp_c=statistics.mean(temps) if temps else None, clock_mhz=statistics.mean(clocks) if clocks else None)


def _fmt(stat: Stat | None, metric: str) -> str:
    if stat is None:
        return "-"
    digits = 1 if stat.p50 < 100 else 0
    return f"{stat.p50:,.{digits}f} [{stat.start_min:,.{digits}f}–{stat.start_max:,.{digits}f}]"


def group_cells(arm_datas: list[ArmData]) -> dict[tuple[str, str, str], list[CellData]]:
    grouped: dict[tuple[str, str, str], list[CellData]] = defaultdict(list)
    for data in arm_datas:
        for cell in data.cells:
            grouped[(data.run.arm.name, cell.framework, cell.case)].append(cell)
    return grouped


def render(experiment: Experiment, arm_datas: list[ArmData]) -> str:
    metric = experiment.metric
    unit = "ms per audio-second" if metric == "ms_per_audio_s" else "E2E ms"
    grouped = group_cells(arm_datas)
    arms = [d.run.arm.name for d in arm_datas]
    frameworks = list(dict.fromkeys(fw for (_a, fw, _c) in grouped))
    cases = list(dict.fromkeys(c for (_a, _f, c) in grouped))
    lines = [f"# {experiment.title}", "", f"**Question.** {experiment.question}", "", f"**Pre-registered reading.** {experiment.reading}", ""]
    if experiment.caveats:
        lines += [f"**Caveats.** {experiment.caveats}", ""]
    lines += ["Campaigns: " + ", ".join(f"`{d.run.campaign}` ({d.run.arm.name}, {d.run.protocol}"
                                         + (f", gap {d.run.gap_s:g} s" if d.run.gap_s else "") + ")" for d in arm_datas), "",
              f"Headline = pooled p50 of {unit} over every measured request of every start; [min–max] = per-start medians. "
              "`clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread "
              "<= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, "
              "SM MHz = mean clock during the request: read both beside `clean`.", ""]
    ref_arm = arms[0]
    lines += [f"| framework | case | arm | starts | n | {unit} p50 [start range] | clean | T °C | SM MHz | vs {ref_arm} |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    stats: dict[tuple[str, str, str], Stat | None] = {}
    for fw in frameworks:
        for case in cases:
            for arm in arms:
                cells = grouped.get((arm, fw, case))
                if not cells:
                    continue
                stat = stats[(arm, fw, case)] = cell_stat(cells, metric)
                ref = stats.get((ref_arm, fw, case)) or cell_stat(grouped.get((ref_arm, fw, case), []), metric)
                ratio = f"{stat.p50 / ref.p50:.2f}x" if stat and ref and arm != ref_arm else "-"
                drift = _a1_a2_drift(stats, grouped, ref_arm, fw, case, metric)
                if stat and ref and arm not in (ref_arm, "A2") and drift is not None and abs(stat.p50 / ref.p50 - 1) <= drift:
                    ratio += " †"  # inside the A1/A2 spread: not distinguishable from drift
                if stat is None:
                    lines.append(f"| {fw} | {case} | {arm} | 0 | 0 | - | - | - | - | - |")
                    continue
                lines.append(
                    f"| {fw} | {case} | {arm} | {stat.starts} | {stat.n} | {_fmt(stat, metric)} | "
                    f"{'-' if stat.clean_pct is None else f'{stat.clean_pct:.0f}%'} | "
                    f"{'-' if stat.temp_c is None else f'{stat.temp_c:.0f}'} | "
                    f"{'-' if stat.clock_mhz is None else f'{stat.clock_mhz:,.0f}'} | {ratio} |")
    drifts = {(fw, case): _a1_a2_drift(stats, grouped, ref_arm, fw, case, metric) for fw in frameworks for case in cases}
    drifts = {key: value for key, value in drifts.items() if value is not None}
    if drifts:
        lines += ["", "**A1/A2 drift** (the reference arm measured before and after the others): "
                  + "; ".join(f"{fw} {case} {value * 100:.0f}%" for (fw, case), value in drifts.items())
                  + ". `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot "
                    "be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive "
                    "for that case and must be re-measured, not interpreted."]
    if experiment.kind == "compare":
        lines += ["", *_best_table(experiment, arm_datas, stats, frameworks, cases, metric)]
    return "\n".join(lines) + "\n"


def _a1_a2_drift(stats, grouped, ref_arm: str, fw: str, case: str, metric: str) -> float | None:
    """|A2 / A1 - 1| for an A1 ... A2 ablation; None when the experiment has no A2 arm for this framework and case."""
    if ref_arm != "A1" or not grouped.get(("A2", fw, case)):
        return None
    first = stats.get(("A1", fw, case)) or cell_stat(grouped.get(("A1", fw, case), []), metric)
    last = stats.get(("A2", fw, case)) or cell_stat(grouped["A2", fw, case], metric)
    return abs(last.p50 / first.p50 - 1) if first and last else None


def _best_table(experiment: Experiment, arm_datas: list[ArmData], stats, frameworks, cases, metric: str) -> list[str]:
    """Each framework at its best candidate arm per case, then every baseline against M*."""
    candidates = {d.run.arm.name for d in arm_datas if d.run.arm.candidate}
    out = ["## Each framework at its best configuration, per case", "",
           f"| case | framework | best arm | p50 | S = framework / M* |", "|---|---|---|---:|---:|"]
    for case in cases:
        best: dict[str, tuple[str, Stat]] = {}
        for fw in frameworks:
            options = [(arm, stats[(arm, fw, case)]) for arm in candidates if stats.get((arm, fw, case))]
            if options:
                best[fw] = min(options, key=lambda item: item[1].p50)
        mstar = best.get("mstar")
        for fw, (arm, stat) in best.items():
            ratio = "1.00x" if fw == "mstar" else (f"{stat.p50 / mstar[1].p50:.2f}x" if mstar else "-")
            out.append(f"| {case} | {fw} | {arm} | {_fmt(stat, metric)} | {ratio} |")
        others = [(fw, s) for fw, (a, s) in best.items() if fw != "mstar"]
        if mstar and len(others) > 1:
            fw, s = min(others, key=lambda item: item[1].p50)
            out.append(f"| {case} | **strongest baseline: {fw}** | | {_fmt(s, metric)} | **{s.p50 / mstar[1].p50:.2f}x** |")
    out += ["", "S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are "
            "per-framework context. All rows share the protocol named in the campaigns."]
    return out


# --- driver ----------------------------------------------------------------------------------------------------------------
def collect(experiment: Experiment, *, run_id: str, protocol: str | None, gap_s: float, arms: list[str] | None,
            **selection) -> list[ArmData]:
    datas = [ArmData(run) for run in runner.arm_runs(experiment, run_id=run_id, protocol=protocol, gap_s=gap_s, arms=arms,
                                                      **selection)]
    for data in datas:
        validate_arm(data)
    return datas


def analyze(experiment: Experiment, *, run_id: str, protocol: str | None, gap_s: float, arms: list[str] | None,
            output: Path | None = None, **selection) -> list[Path]:
    """Validate and write one results file per protocol the experiment ran under (two for a task whose protocol decision is "both")."""
    protocol_list = [protocol] if protocol else protocols.protocols_for(experiment.task, required=experiment.group == "baseline")
    outputs: list[Path] = []
    for proto in protocol_list:
        datas = collect(experiment, run_id=run_id, protocol=proto, gap_s=gap_s if proto == "back-to-back" else 0.0, arms=arms,
                        **selection)
        problems = [p for d in datas for p in d.problems]
        if problems:
            raise AnalysisError(f"{experiment.name} [{proto}]: {len(problems)} problem(s), no numbers reported:\n  "
                                + "\n  ".join(problems))
        text = render(experiment, datas)
        suffix = "" if proto == "gated" else ("_b2b" if not gap_s else f"_gap{gap_s:g}")
        # the file name carries the hardware profile: one results folder serves several machines without collisions
        out = output if (output and len(protocol_list) == 1) else paths.RESULTS / f"{experiment.name}{suffix}_{hardware.current().name}.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        outputs.append(out)
    return outputs
