#!/usr/bin/env python3
"""Thermal gates and per-request thermal monitoring.

Why the gates look the way they do
----------------------------------
This H200 NVL is a passively cooled board in a chassis that does not feed it
air: it idles at ~47 C on ~71 W, and under sustained load it pins at 345 MHz of
1785 MHz, 99 C, 185 W of a 600 W TDP, with hardware thermal slowdown
permanently engaged. Locking clocks cannot help -- `-lgc` is only a ceiling --
and power capping cannot either, because the minimum settable limit (200 W) is
above what the throttled card draws.

So the only way to get comparable numbers is to make every measured request
start from the same un-throttled state, and to prove it did rather than assume
it. Two historical telemetry observations are preserved below and set the
thresholds. Their source logs are not loaded by the current harness:

* **The knee is around 80 C.** Cells that stayed under it held 1785 MHz and
  produced flat latency across ten consecutive requests (I2T-256: 879 -> 870 ms
  while the card walked 65 -> 72 C). The one cell that crossed it degraded
  immediately (I2I-256 vLLM two-stage: 72 -> 85 C, SM clock fell to 1710 MHz on
  request 9, latency 6485 -> 7639 ms, +18%).
* **Cooling costs roughly 10 s per degree.** Measured from empty-GPU cooldown
  traces, including the 60 s continuous window: 85 C -> 385 s, 80 C -> 350 s,
  75 C -> 292 s, 65 C -> 180 s, 54 C -> 57 s.

Together those say: gate below the knee with margin, not at room temperature. A
55 C target costs about five minutes per request and buys nothing a 70 C target
does not already guarantee, because the card is equally un-throttled at both.

The three gate conditions
-------------------------
`cooldown_gate` passes only when all three hold continuously:

1. core temperature <= target (default 70 C, i.e. 10 C below the knee)
2. ``clocks_sm_mhz == clocks_max_sm_mhz`` -- the card is at full nominal clock.
   This is the condition that actually matters; temperature is a proxy for it.
3. no clock event / throttle reason is active

Condition 2 is why this module exists rather than reusing a temperature-only
wait: a card can read 68 C and still be clocked down coming out of a throttle
event, and such a request is not comparable to one that started at full clock.
"""
from __future__ import annotations

import csv
import os
import statistics
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, TextIO

from exps import hardware

from . import gpu

#: An empty GPU on this host settles at ~14 MiB. 17 leaves room for rounding.
EMPTY_GPU_MAX_MEMORY_MIB = 17.0
MPS_EMPTY_GPU_MAX_MEMORY_MIB = 128.0
#: A loaded Qwen3-Omni / BAGEL service holds tens of GiB; the loaded-idle gate
#: must not wait for memory to drop, only for heat to.
LOADED_GPU_MAX_MEMORY_MIB = 140_000.0

#: Default per-request cooldown target: 10 C below the ~80 C degradation knee.
DEFAULT_COOLDOWN_TARGET_C = 70.0
#: The gate wants a *settled* state, not a single lucky sample.
DEFAULT_COOLDOWN_CONTINUOUS_S = 30.0
#: Service-level gates run on an empty GPU and can afford the longer window.
DEFAULT_SERVICE_GATE_CONTINUOUS_S = 60.0


class GateTimeout(RuntimeError):
    """A gate did not reach its target within its timeout."""


def _open_csv(path: Path) -> tuple[TextIO, csv.DictWriter]:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=list(gpu.SAMPLE_CSV_FIELDS))
    writer.writeheader()
    return handle, writer


def _write_row(writer: csv.DictWriter, handle: TextIO, row: dict, passes: bool) -> None:
    writer.writerow({field: row.get(field, "") for field in gpu.SAMPLE_CSV_FIELDS} | {"passes": passes})
    handle.flush()


def _public(row: dict) -> dict:
    """A sample without the nested reason dict, safe to embed in JSON verdicts."""
    return {key: value for key, value in row.items() if key not in ("event_reasons", "monotonic")}


def _at_full_clock(row: dict) -> bool:
    current = row.get("clocks_sm_mhz")
    nominal = row.get("clocks_max_sm_mhz")
    if current is None or not nominal:
        return False
    return current >= nominal


def _conditions(
    row: dict,
    *,
    max_utilization_percent: float | None,
    max_memory_used_mib: float | None,
    max_temperature_celsius: float | None,
    max_memory_temperature_celsius: float | None,
    require_full_clock: bool,
    require_event_free: bool = True,
) -> dict[str, bool]:
    """Each gate condition evaluated separately, so failures name themselves.

    `require_full_clock` and `require_event_free` are the frequency conditions. A profile can switch them off
    (`cooldown_gates_frequency`): the clock and event reasons are then still recorded, but a card that runs slower
    cannot hold up a run.
    """
    checks: dict[str, bool] = {}
    if max_utilization_percent is not None:
        value = row["utilization_percent"]
        checks["utilization"] = value is not None and value <= max_utilization_percent
    if max_memory_used_mib is not None:
        value = row["memory_used_mib"]
        checks["memory_free"] = value is not None and value <= max_memory_used_mib
    if max_temperature_celsius is not None:
        value = row["temperature_c"]
        checks["temperature"] = value is not None and value <= max_temperature_celsius
    if max_memory_temperature_celsius is not None:
        value = row["memory_temperature_c"]
        checks["memory_temperature"] = value is not None and value <= max_memory_temperature_celsius
    if require_event_free:
        checks["no_event_reason"] = not row["any_event_reason_active"]
    if require_full_clock:
        checks["full_sm_clock"] = _at_full_clock(row)
    return checks


def _blocking_reason(row: dict, checks: dict[str, bool], targets: dict) -> str:
    """The first failing condition, phrased so a log reader can act on it."""
    if not checks.get("memory_free", True):
        if row["memory_used_mib"] is None:
            return "GPU memory telemetry unavailable"
        return (
            f"memory {row['memory_used_mib']} MiB > {targets['max_memory_used_mib']} MiB "
            "(another process still holds the GPU)"
        )
    if not checks.get("utilization", True):
        if row["utilization_percent"] is None:
            return "GPU utilization telemetry unavailable"
        return f"utilization {row['utilization_percent']}% > {targets['max_utilization_percent']}%"
    if not checks.get("temperature", True):
        if row["temperature_c"] is None:
            return "GPU temperature telemetry unavailable"
        return (
            f"temperature {row['temperature_c']} C > {targets['max_temperature_celsius']} C "
            "(the chassis has not cooled the card)"
        )
    if not checks.get("memory_temperature", True):
        if row["memory_temperature_c"] is None:
            return "GPU memory temperature telemetry unavailable"
        return (
            f"memory junction {row['memory_temperature_c']} C > "
            f"{targets['max_memory_temperature_celsius']} C"
        )
    if not checks.get("no_event_reason", True):
        return f"clock event reasons still active: {row['event_reasons']}"
    if not checks.get("full_sm_clock", True):
        return (
            f"SM clock {row['clocks_sm_mhz']} MHz is below nominal "
            f"{row['clocks_max_sm_mhz']} MHz; the card has not returned to full clock"
        )
    return "unknown"


def wait_for_state(
    *,
    name: str,
    gpu_index: int = 0,
    continuous_seconds: float,
    timeout_seconds: float,
    period_seconds: float = 1.0,
    max_utilization_percent: float | None = None,
    max_memory_used_mib: float | None = None,
    max_temperature_celsius: float | None = None,
    max_memory_temperature_celsius: float | None = None,
    require_full_clock: bool = False,
    require_event_free: bool = True,
    output: Path | None = None,
    on_tick: Callable[[dict], None] | None = None,
) -> dict:
    """Block until every enabled condition has held continuously, or time out.

    Returns a verdict dict in both cases; it never raises on timeout, because
    the caller decides whether a timed-out gate voids a cell or merely flags a
    request. `output`, when given, receives every raw sample as CSV.
    """
    targets = {
        "max_utilization_percent": max_utilization_percent,
        "max_memory_used_mib": max_memory_used_mib,
        "max_temperature_celsius": max_temperature_celsius,
        "max_memory_temperature_celsius": max_memory_temperature_celsius,
        "require_full_clock": require_full_clock,
        "require_event_free": require_event_free,
        "continuous_seconds": continuous_seconds,
    }
    handle = writer = None
    if output is not None:
        handle, writer = _open_csv(output)
    started = time.monotonic()
    deadline = started + timeout_seconds
    pass_start: float | None = None
    first_pass_at: float | None = None
    try:
        while True:
            now = time.monotonic()
            row = gpu.sample(gpu_index)
            checks = _conditions(
                row,
                max_utilization_percent=max_utilization_percent,
                max_memory_used_mib=max_memory_used_mib,
                max_temperature_celsius=max_temperature_celsius,
                max_memory_temperature_celsius=max_memory_temperature_celsius,
                require_full_clock=require_full_clock,
                require_event_free=require_event_free,
            )
            passes = all(checks.values())
            if writer is not None and handle is not None:
                _write_row(writer, handle, row, passes)
            if passes:
                if pass_start is None:
                    pass_start = now
                    first_pass_at = now
                if now - pass_start >= continuous_seconds:
                    return {
                        "gate": name,
                        "status": "passed",
                        "waited_seconds": now - started,
                        "seconds_until_first_pass": (first_pass_at - started) if first_pass_at else 0.0,
                        "continuous_seconds": now - pass_start,
                        "targets": targets,
                        "last_sample": _public(row),
                        "samples_csv": str(output) if output else None,
                    }
            else:
                pass_start = None
            if now >= deadline:
                return {
                    "gate": name,
                    "status": "failed_timeout",
                    "waited_seconds": now - started,
                    "targets": targets,
                    "conditions": checks,
                    "blocking_reason": _blocking_reason(row, checks, targets),
                    "last_sample": _public(row),
                    "samples_csv": str(output) if output else None,
                }
            if on_tick is not None:
                on_tick(row)
            else:
                time.sleep(period_seconds)
    finally:
        if handle is not None:
            handle.close()


def idle_gate(
    output: Path,
    *,
    gpu_index: int = 0,
    max_temperature_celsius: float = DEFAULT_COOLDOWN_TARGET_C,
    continuous_seconds: float = DEFAULT_SERVICE_GATE_CONTINUOUS_S,
    timeout_seconds: float = 1800.0,
) -> dict:
    """Service-level gate on an *empty* GPU: before start and after shutdown.

    Outside MPS, memory must be back to ~17 MiB. MPS keeps a small server
    context resident, so MPS runs allow up to 128 MiB while still requiring
    zero GPU utilization.

    Full clock is deliberately *not* required here. An idle GPU downclocks by
    design -- this board sits at 345 MHz with no client attached, numerically
    the same value it reaches under thermal slowdown -- so the idle clock
    carries no health signal. The full-clock condition is enforced where it is
    meaningful: in `cooldown_gate`, immediately before a measured request.
    """
    memory_limit_mib = (
        MPS_EMPTY_GPU_MAX_MEMORY_MIB
        if os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        else hardware.current().empty_gpu_max_memory_mib
    )
    return wait_for_state(
        name="idle",
        gpu_index=gpu_index,
        continuous_seconds=continuous_seconds,
        timeout_seconds=timeout_seconds,
        max_utilization_percent=0.0,
        max_memory_used_mib=memory_limit_mib,
        max_temperature_celsius=max_temperature_celsius,
        require_full_clock=False,
        output=output,
    )


def loaded_idle_gate(
    output: Path,
    *,
    gpu_index: int = 0,
    max_temperature_celsius: float = DEFAULT_COOLDOWN_TARGET_C,
    continuous_seconds: float = DEFAULT_SERVICE_GATE_CONTINUOUS_S,
    timeout_seconds: float = 1800.0,
) -> dict:
    """Gate with the service resident: weights stay on the card, heat must go.

    Used after the service reports ready and after preconditioning, so the
    measured phase begins from a known state without paying the model load
    again.
    """
    return wait_for_state(
        name="loaded_idle",
        gpu_index=gpu_index,
        continuous_seconds=continuous_seconds,
        timeout_seconds=timeout_seconds,
        max_utilization_percent=0.0,
        max_memory_used_mib=hardware.current().loaded_limit_mib(),
        max_temperature_celsius=max_temperature_celsius,
        require_full_clock=False,
        output=output,
    )


def cooldown_gate(
    *,
    gpu_index: int = 0,
    target_celsius: float = DEFAULT_COOLDOWN_TARGET_C,
    memory_target_celsius: float | None = None,
    continuous_seconds: float = DEFAULT_COOLDOWN_CONTINUOUS_S,
    timeout_seconds: float = 900.0,
    period_seconds: float = 2.0,
    output: Path | None = None,
    require_frequency_gate: bool = True,
) -> dict:
    """The per-request gate: run immediately before every measured request.

    Temperature at or below target must hold continuously. With
    `require_frequency_gate` (the default) the SM clock must also sit at full nominal and no event reason may be
    active. A profile that cannot hold full clock while idle (rtx5880: idle 210 MHz, nominal 3,105 MHz) turns that
    off: the clock and the event reasons are still recorded in the gate CSV and on every request, not waited for. A timeout
    does **not** void the cell: the caller records `cooldown_timed_out` on that
    request, keeps it in `requests.jsonl`, and excludes it from the headline
    aggregate. Losing one request to a slow chassis is recoverable; silently
    measuring a throttled one is not.
    """
    verdict = wait_for_state(
        name="cooldown",
        gpu_index=gpu_index,
        continuous_seconds=continuous_seconds,
        timeout_seconds=timeout_seconds,
        period_seconds=period_seconds,
        max_utilization_percent=None,
        max_memory_used_mib=None,
        max_temperature_celsius=target_celsius,
        max_memory_temperature_celsius=memory_target_celsius,
        require_full_clock=require_frequency_gate,
        require_event_free=require_frequency_gate,
        output=output,
    )
    verdict["timed_out"] = verdict["status"] != "passed"
    return verdict


class RequestThermalMonitor:
    """Sample the GPU in the background for the duration of one request.

    Checking clocks only at the edges of a request would miss a throttle that
    starts and ends inside it. On a long request this host does exactly that:
    it crosses the thermal limit partway through and recovers before the
    request returns. Sampling throughout is what makes the per-request validity
    flag evidence rather than decoration.
    """

    def __init__(self, gpu_index: int = 0, period_seconds: float = 0.5) -> None:
        self.gpu_index = gpu_index
        self.period_seconds = period_seconds
        self._samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._samples.append(gpu.sample(self.gpu_index))
            except Exception:  # noqa: BLE001 - telemetry must never fail a request
                pass
            self._stop.wait(self.period_seconds)

    def __enter__(self) -> "RequestThermalMonitor":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def samples(self) -> list[dict]:
        return list(self._samples)

    def verdict(self, *, min_clock_fraction: float = 0.98, max_clock_spread_mhz: float = 60.0) -> dict:
        """Summarise what the card did during the request.

        `thermally_clean` requires no event reason, the clock never dropping
        below `min_clock_fraction` of nominal, and a spread no wider than
        `max_clock_spread_mhz`. The default fraction is deliberately tight
        (0.98): the gate guarantees the request *started* at full clock, so any
        meaningful drop during it is the thing we are trying to detect.
        """
        if not self._samples:
            return {
                "samples": 0,
                "thermally_clean": False,
                "reason": "no telemetry captured for this request",
            }
        clocks = [s["clocks_sm_mhz"] for s in self._samples if s["clocks_sm_mhz"] is not None]
        temperatures = [s["temperature_c"] for s in self._samples if s["temperature_c"] is not None]
        memory_temps = [
            s["memory_temperature_c"] for s in self._samples if s["memory_temperature_c"] is not None
        ]
        powers = [s["power_draw_w"] for s in self._samples if s["power_draw_w"] is not None]
        memory_used = [s["memory_used_mib"] for s in self._samples if s.get("memory_used_mib") is not None]
        nominal = next(
            (s["clocks_max_sm_mhz"] for s in self._samples if s["clocks_max_sm_mhz"]),
            None,
        )
        throttled = any(s["any_event_reason_active"] for s in self._samples)
        reasons_seen = sorted(
            {
                reason
                for s in self._samples
                for reason, active in (s.get("event_reasons") or {}).items()
                if active
            }
        )
        if not clocks or not nominal:
            return {
                "samples": len(self._samples),
                "thermally_clean": False,
                "reason": "clock telemetry unavailable during this request",
            }
        spread = max(clocks) - min(clocks)
        floor = nominal * min_clock_fraction
        held_clock = min(clocks) >= floor
        narrow = spread <= max_clock_spread_mhz
        return {
            "samples": len(self._samples),
            "sm_clock_mhz_min": min(clocks),
            "sm_clock_mhz_mean": statistics.fmean(clocks),
            "sm_clock_mhz_max": max(clocks),
            "sm_clock_spread_mhz": spread,
            "nominal_sm_mhz": nominal,
            "clock_floor_mhz": floor,
            "temperature_c_start": temperatures[0] if temperatures else None,
            "temperature_c_max": max(temperatures) if temperatures else None,
            "memory_temperature_c_max": max(memory_temps) if memory_temps else None,
            # Device-wide `memory.used` sampled every `period_seconds`: a lower bound on the true
            # in-request peak (a spike shorter than the period is missed) and it includes
            # pre-reserved pools (KV cache, CUDA graphs, allocator cache), not live tensors only.
            "memory_used_mib_start": memory_used[0] if memory_used else None,
            "memory_used_mib_min": min(memory_used) if memory_used else None,
            "memory_used_mib_max": max(memory_used) if memory_used else None,
            "power_w_mean": statistics.fmean(powers) if powers else None,
            "event_reason_active_during_request": throttled,
            "event_reasons_seen": reasons_seen,
            "held_clock_floor": held_clock,
            "clock_spread_within_tolerance": narrow,
            "thermally_clean": (not throttled) and held_clock and narrow,
        }


def wait_for_empty_gpu(*, gpu_index: int = 0, timeout_seconds: float = 120.0) -> bool:
    """Poll until no process holds a CUDA context, so the next cell starts clean."""
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if not gpu.compute_apps(gpu_index):
            return True
        time.sleep(2.0)
    return not gpu.compute_apps(gpu_index)


def summarize_gate_series(verdicts: Iterable[dict]) -> dict:
    """Aggregate gate verdicts so a cell summary can report gate cost at a glance."""
    waits = [v["waited_seconds"] for v in verdicts if v.get("waited_seconds") is not None]
    timeouts = sum(1 for v in verdicts if v.get("status") != "passed")
    return {
        "count": len(waits),
        "timed_out": timeouts,
        "waited_seconds_total": sum(waits) if waits else 0.0,
        "waited_seconds_mean": statistics.fmean(waits) if waits else None,
        "waited_seconds_max": max(waits) if waits else None,
    }
