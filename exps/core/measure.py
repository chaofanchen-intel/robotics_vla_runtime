#!/usr/bin/env python3
"""The measured phase: preconditioning, per-request gating, recording.

This runs inside the **task client**, under the runtime's own virtualenv, and
is the piece that actually enforces the thermal protocol. It is shared by every
task so that "one measured request" means the same thing for BAGEL as it does
for Orpheus.

Order of operations, and why
----------------------------
1. **loaded-idle gate** once the service reports ready. Loading 30B of weights
   heats the card; measuring from that state would charge the engine for the
   loader's heat.
2. **preconditioning requests**, unscored. The first request through any of
   these runtimes pays JIT compilation, CUDA graph capture and allocator
   growth. Round 2 folded that cost into its first measured request.
3. **loaded-idle gate** again, to shed the preconditioning heat.
4. **per-request cooldown gate, then one measured request**, repeated. The gate
   requires the card at or below target temperature, at full nominal SM clock,
   with no throttle reason active -- see `thermal.cooldown_gate`.
5. optional **throughput phase**, clearly separated.

Why throughput is a separate phase
----------------------------------
Under per-request gating there is no sustained throughput to measure: the wall
clock is dominated by cooling. Reporting requests-per-second from a gated loop
would be arithmetic on waiting. So throughput, when asked for, is measured as
its own back-to-back batch after a full cooldown, and is labelled
`thermally_drifting_throughput` because on this chassis that is exactly what it
is. It is never mixed into the latency aggregate.
"""
from __future__ import annotations

import json
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

from . import metrics as metrics_mod
from . import thermal
from .codecs import ContractViolation
from .manifest import write_json


class RequestIssuer(Protocol):
    """A task client's per-request callable.

    Must return a dict carrying at least ``success`` plus whatever subset of
    `metrics.REQUEST_METRIC_FIELDS` the task can measure, and must raise
    `ContractViolation` when the response does not represent the work that was
    requested -- a wrong image size, a wrong completion length, a spoken answer
    instead of a verbatim reading. Returning a number for mismatched work is
    the failure mode this whole harness exists to prevent.
    """

    def __call__(self, index: int, phase: str) -> dict[str, Any]: ...


@dataclass
class MeasurementConfig:
    """Protocol parameters for one cell. Defaults are the pre-calibration values."""

    output_dir: Path
    measured_requests: int = 5
    precondition_requests: int = 1
    gpu_index: int = 0
    cooldown_target_c: float = thermal.DEFAULT_COOLDOWN_TARGET_C
    cooldown_continuous_s: float = thermal.DEFAULT_COOLDOWN_CONTINUOUS_S
    cooldown_timeout_s: float = 900.0
    #: False on a profile whose card cannot hold full clock while idle (rtx5880): the cooldown then waits for temperature
    #: only, and the clock and event reasons are recorded per request instead of gating the run
    cooldown_gates_frequency: bool = True
    loaded_idle_target_c: float = thermal.DEFAULT_COOLDOWN_TARGET_C
    loaded_idle_continuous_s: float = thermal.DEFAULT_SERVICE_GATE_CONTINUOUS_S
    loaded_idle_timeout_s: float = 1800.0
    monitor_period_s: float = 0.5
    monitor_min_clock_fraction: float = 0.98
    monitor_max_clock_spread_mhz: float = 60.0
    throughput_requests: int = 0
    throughput_concurrency: int = 1
    #: Skip the loaded-idle gate after preconditioning and every per-request cooldown gate: measured requests go out
    #: consecutively, the GPU never idles between them. Which protocol an engine's latency was measured under
    #: matters (Pi0.5 on vLLM-Omni: 333 ms gated vs 222 ms back-to-back, temperature and clocks identical), so this
    #: is an explicit per-experiment choice, available for every task; thermal cleanliness is still recorded per request.
    back_to_back: bool = False
    #: With `back_to_back`: sleep this long before each measured request, with no thermal condition attached. Separates
    #: "the card sat idle for a while" from "the gate demanded a cool, full-clock card" (the two differ only in gated runs).
    inter_request_gap_s: float = 0.0
    calibration_source: str = "default_not_measured"
    abort_on_contract_violation: bool = True

    def as_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["output_dir"] = str(self.output_dir)
        return record


@dataclass
class _Collector:
    rows: list[dict[str, Any]] = field(default_factory=list)
    gate_verdicts: list[dict[str, Any]] = field(default_factory=list)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _blank_row(index: int, phase: str) -> dict[str, Any]:
    """A row with every schema field present, so absent never means zero."""
    row: dict[str, Any] = {
        "phase": phase,
        "sequence": index,
        "started_at_utc": None,
        "success": False,
        "error": None,
    }
    row.update(metrics_mod.empty_request_metrics())
    row.update({field_name: None for field_name in metrics_mod.REQUEST_THERMAL_FIELDS})
    return row


def _thermal_context(before: dict, verdict: dict, cooldown: dict | None) -> dict[str, Any]:
    return {
        "cooldown_waited_s": cooldown.get("waited_seconds") if cooldown else None,
        "cooldown_timed_out": bool(cooldown.get("timed_out")) if cooldown else None,
        "temperature_c_before": before.get("temperature_c"),
        "memory_temperature_c_before": before.get("memory_temperature_c"),
        "sm_clock_mhz_before": before.get("clocks_sm_mhz"),
        "sm_clock_mhz_nominal": before.get("clocks_max_sm_mhz"),
        "sm_clock_spread_mhz": verdict.get("sm_clock_spread_mhz"),
        "temperature_c_max": verdict.get("temperature_c_max"),
        "memory_used_mib_before": before.get("memory_used_mib"),
        "memory_used_mib_max": verdict.get("memory_used_mib_max"),
        "throttle_reasons_active": verdict.get("event_reasons_seen"),
        "thermally_clean": verdict.get("thermally_clean"),
        "thermal_monitor": verdict,
    }


def _issue_monitored(
    issue: RequestIssuer,
    index: int,
    phase: str,
    config: MeasurementConfig,
    cooldown: dict | None,
) -> dict[str, Any]:
    """One request with telemetry wrapped around it, never raising upward.

    A failed request becomes a recorded row rather than an exception, so a
    single timeout does not discard the four good measurements beside it. The
    one exception is a contract violation, which the caller may escalate.
    """
    from . import gpu  # local import keeps module import cost off the client path

    row = _blank_row(index, phase)
    row["started_at_utc"] = _now_utc()
    before = gpu.sample(config.gpu_index)
    monitor = thermal.RequestThermalMonitor(config.gpu_index, period_seconds=config.monitor_period_s)
    violation: ContractViolation | None = None
    with monitor:
        try:
            result = issue(index, phase)
            row.update(result)
            row.setdefault("success", True)
        except ContractViolation as exc:
            violation = exc
            row["error"] = f"ContractViolation: {exc}"
            row["success"] = False
        except Exception as exc:  # noqa: BLE001 - one bad request must not kill the cell
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["error_traceback"] = traceback.format_exc(limit=5)
            row["success"] = False
    verdict = monitor.verdict(
        min_clock_fraction=config.monitor_min_clock_fraction,
        max_clock_spread_mhz=config.monitor_max_clock_spread_mhz,
    )
    row.update(_thermal_context(before, verdict, cooldown))
    row["finished_at_utc"] = _now_utc()
    if row.get("rtf") is None:
        row["rtf"] = metrics_mod.compute_rtf(row.get("e2e_ms"), row.get("audio_duration_s"))
    if violation is not None and config.abort_on_contract_violation:
        row["_violation"] = violation
    return row


def run_measurement(
    config: MeasurementConfig,
    *,
    issue: RequestIssuer,
    context: dict[str, Any],
    throughput_batch: Callable[[int, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute the full measured phase and write `requests.jsonl` + `summary.json`.

    `context` is merged into the summary verbatim: it is where a task records
    its own identity, endpoint, fixture, contract settings and timing-boundary
    statement. Returns the summary dict; also writes a `VOID.json` beside it if
    a contract invariant failed.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requests_path = output_dir / "requests.jsonl"
    if requests_path.exists():
        requests_path.unlink()  # a re-run replaces, never appends to, a previous attempt

    collector = _Collector()
    phases: dict[str, Any] = {}
    violation: ContractViolation | None = None

    def record(row: dict[str, Any]) -> None:
        pending = row.pop("_violation", None)
        serialisable = {key: value for key, value in row.items() if not isinstance(value, BaseException)}
        collector.rows.append(serialisable)
        with requests_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(serialisable, ensure_ascii=False, default=str) + "\n")
        if pending is not None:
            raise pending

    try:
        # 1. shed the heat the model loader produced
        phases["loaded_idle_after_ready"] = thermal.loaded_idle_gate(
            output_dir / "loaded_idle_after_ready.csv",
            gpu_index=config.gpu_index,
            max_temperature_celsius=config.loaded_idle_target_c,
            continuous_seconds=config.loaded_idle_continuous_s,
            timeout_seconds=config.loaded_idle_timeout_s,
        )

        # 2. preconditioning: JIT, CUDA graphs, allocator growth -- never scored
        for index in range(1, config.precondition_requests + 1):
            record(_issue_monitored(issue, index, "precondition", config, None))

        # 3. shed the preconditioning heat (not in back-to-back mode: that gate would idle the card)
        if config.precondition_requests and not config.back_to_back:
            phases["loaded_idle_after_precondition"] = thermal.loaded_idle_gate(
                output_dir / "loaded_idle_after_precondition.csv",
                gpu_index=config.gpu_index,
                max_temperature_celsius=config.loaded_idle_target_c,
                continuous_seconds=config.loaded_idle_continuous_s,
                timeout_seconds=config.loaded_idle_timeout_s,
            )

        # 4. the measured loop: gate, measure, repeat
        for index in range(1, config.measured_requests + 1):
            cooldown = None
            if config.back_to_back and config.inter_request_gap_s > 0:
                time.sleep(config.inter_request_gap_s)
            if not config.back_to_back:
                cooldown = thermal.cooldown_gate(
                    gpu_index=config.gpu_index,
                    target_celsius=config.cooldown_target_c,
                    continuous_seconds=config.cooldown_continuous_s,
                    timeout_seconds=config.cooldown_timeout_s,
                    output=output_dir / "cooldowns" / f"measured_{index:03d}.csv",
                    require_frequency_gate=config.cooldown_gates_frequency,
                )
                collector.gate_verdicts.append(cooldown)
            record(_issue_monitored(issue, index, "measured", config, cooldown))

        # 5. optional throughput phase, from a fully cooled card, kept separate
        if config.throughput_requests and throughput_batch is not None:
            phases["cooldown_before_throughput"] = thermal.cooldown_gate(
                gpu_index=config.gpu_index,
                target_celsius=config.cooldown_target_c,
                continuous_seconds=config.cooldown_continuous_s,
                timeout_seconds=config.cooldown_timeout_s,
                output=output_dir / "cooldowns" / "throughput.csv",
                require_frequency_gate=config.cooldown_gates_frequency,
            )
            started = time.perf_counter()
            with thermal.RequestThermalMonitor(config.gpu_index, config.monitor_period_s) as monitor:
                batch = throughput_batch(config.throughput_requests, config.throughput_concurrency)
            elapsed = time.perf_counter() - started
            completed = int(batch.get("completed", 0))
            phases["throughput"] = {
                "label": "thermally_drifting_throughput",
                "note": (
                    "Back-to-back requests from a cooled card. On this chassis the card heats "
                    "through the batch, so this is an observed rate under drift, not a steady-state "
                    "engine throughput. Never combine it with the gated latency figures."
                ),
                "requests": config.throughput_requests,
                "concurrency": config.throughput_concurrency,
                "completed": completed,
                "wall_seconds": elapsed,
                "requests_per_second": completed / elapsed if elapsed else None,
                "thermal_monitor": monitor.verdict(
                    min_clock_fraction=config.monitor_min_clock_fraction,
                    max_clock_spread_mhz=config.monitor_max_clock_spread_mhz,
                ),
                "detail": batch,
            }

    except ContractViolation as exc:
        violation = exc

    summary = _build_summary(config, context, collector, phases, violation)
    write_json(output_dir / "summary.json", summary)
    if violation is not None:
        write_json(
            output_dir / "VOID.json",
            {
                "status": "void_contract_violation",
                "reason": str(violation),
                "note": (
                    "The engine did not do the work that was requested, so no latency from this "
                    "cell is comparable. A missing comparison is recoverable; a plausible-looking "
                    "invalid one is not."
                ),
            },
        )
    return summary


def _build_summary(
    config: MeasurementConfig,
    context: dict[str, Any],
    collector: _Collector,
    phases: dict[str, Any],
    violation: ContractViolation | None,
) -> dict[str, Any]:
    measured = [row for row in collector.rows if row["phase"] == "measured"]
    start_summary = metrics_mod.summarize_start(collector.rows)
    gate_failures = [v for v in collector.gate_verdicts if v.get("status") != "passed"]
    loaded_gate_failures = [
        value for name, value in phases.items()
        if name.startswith("loaded_idle_") and value.get("status") != "passed"
    ]
    status = (
        "void" if violation
        else "failed_loaded_idle_gate" if loaded_gate_failures
        else "no_measured_requests" if not measured
        else "no_valid_requests" if not start_summary["headline_requests"]
        else "completed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "finished_at_utc": _now_utc(),
        **context,
        "protocol": {
            "name": (
                "fixed-gap" if config.back_to_back and config.inter_request_gap_s > 0
                else "back-to-back" if config.back_to_back else "per-request-cooldown-gated"
            ),
            "description": (
                "Fresh service per start. One loaded-idle gate after readiness, unscored "
                f"preconditioning requests, then measured requests each preceded by a fixed {config.inter_request_gap_s:g} s "
                "sleep and no thermal gate (the card idles for that long; thermal state is recorded, not enforced)."
                if config.back_to_back and config.inter_request_gap_s > 0 else
                "Fresh service per start. One loaded-idle gate after readiness, unscored "
                "preconditioning requests, then measured requests issued consecutively with no gate "
                "between them (the card never idles; thermal state is recorded, not enforced)."
                if config.back_to_back else
                "Fresh service per start. One loaded-idle gate after readiness, unscored "
                "preconditioning requests, a second loaded-idle gate, then every measured request "
                "preceded by a cooldown gate requiring the card at or below the target temperature"
                + (", at full nominal SM clock, with no throttle reason active." if config.cooldown_gates_frequency
                   else "; SM clock and throttle reasons are recorded per request, not gated on this profile.")
            ),
            **config.as_record(),
        },
        "gates": {
            **phases,
            "per_request_cooldowns": thermal.summarize_gate_series(collector.gate_verdicts),
            "per_request_cooldown_failures": len(gate_failures),
        },
        "results": start_summary,
        "contract_violation": str(violation) if violation else None,
        "outputs": {
            "requests_jsonl": "requests.jsonl",
            "cooldown_csvs": "cooldowns/",
            "void_marker": "VOID.json (present only when a contract invariant failed)",
        },
    }
