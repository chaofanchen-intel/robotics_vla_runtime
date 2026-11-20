#!/usr/bin/env python3
"""nvidia-smi telemetry and SM-clock control for the experiments harness.

This is the single place the harness talks to the driver. Everything above it
-- gates, per-request monitors, calibration -- consumes `sample()`.

Two things this host makes non-negotiable, and which shaped this module:

1. `clocks.max.sm` must be read on every sample, not cached. The whole point of
   the per-request gate is to assert `clocks_sm_mhz == clocks_max_sm_mhz`, and
   that comparison is meaningless against a stale nominal.
2. Memory-junction temperature is sampled alongside the core temperature. The
   core falls faster than the memory junction does, so a gate that watches only
   `temperature.gpu` can pass while the card is still heat-soaked. Recording
   both lets the calibration decide which one actually predicts latency here.

Clock locking is kept because the API is cheap to carry, but on this chassis it
does nothing: `find_sustainable_clock.py` showed every locked value from
1785 MHz down to 600 MHz settling at the same 345 MHz / 99 C / 185 W, because
`-lgc` is only a ceiling and HW thermal slowdown owns the floor. The harness
therefore defaults to no locking and records that fact rather than pretending.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time

QUERY_FIELDS = (
    "name",
    "utilization.gpu",
    "memory.used",
    "temperature.gpu",
    "temperature.memory",
    "clocks.sm",
    "clocks.max.sm",
    "power.draw",
    "enforced.power.limit",
)

EVENT_REASON_SUFFIXES = (
    "sw_power_cap",
    "hw_slowdown",
    "hw_thermal_slowdown",
    "sw_thermal_slowdown",
)

#: Fields written to every gate CSV, in order.
SAMPLE_CSV_FIELDS = (
    "wall_clock",
    "utilization_percent",
    "memory_used_mib",
    "temperature_c",
    "memory_temperature_c",
    "clocks_sm_mhz",
    "clocks_max_sm_mhz",
    "power_draw_w",
    "any_event_reason_active",
    "passes",
)


class NvidiaSmiUnavailable(RuntimeError):
    """nvidia-smi is missing or refused a query the harness cannot work without."""


def _run(args: list[str], *, privileged: bool = False) -> subprocess.CompletedProcess[str]:
    """Run nvidia-smi, elevating only the calls that genuinely need privilege.

    Only `-lgc` / `-rgc` need root. Elevating just those -- rather than running
    the whole campaign under sudo -- keeps every run directory and __pycache__
    entry owned by the invoking user. A campaign launched with sudo leaves
    root-owned artefacts that the next ordinary run cannot overwrite.
    """
    if shutil.which("nvidia-smi") is None:
        raise NvidiaSmiUnavailable("nvidia-smi is not on PATH")
    command = ["nvidia-smi", *args]
    if privileged and os.geteuid() != 0 and _sudo_available():
        command = ["sudo", "-n", *command]
    return subprocess.run(command, capture_output=True, text=True)


def _sudo_available() -> bool:
    """True when `sudo -n` runs without prompting, so a matrix never blocks."""
    if shutil.which("sudo") is None:
        return False
    return subprocess.run(["sudo", "-n", "true"], capture_output=True, text=True).returncode == 0


def _as_float(raw: str) -> float | None:
    """nvidia-smi prints '[N/A]' for fields a given board does not expose."""
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def read_state(gpu_index: int = 0) -> dict[str, str]:
    result = _run(
        [f"--id={gpu_index}", f"--query-gpu={','.join(QUERY_FIELDS)}", "--format=csv,noheader,nounits"]
    )
    if result.returncode != 0:
        raise NvidiaSmiUnavailable(f"nvidia-smi query failed: {result.stderr.strip()}")
    values = [part.strip() for part in result.stdout.strip().split(",")]
    return dict(zip(QUERY_FIELDS, values, strict=True))


def read_event_reasons(gpu_index: int = 0) -> dict[str, bool]:
    """Active clock event reasons as booleans keyed by reason name.

    Driver versions disagree on the prefix (`clocks_event_reasons` replaced
    `clocks_throttle_reasons`), so both are tried before giving up. An empty
    dict means the driver exposes neither, which the gates treat as "no reason
    active" -- that is the only non-blocking interpretation available, and the
    gates still have temperature and clock conditions to fall back on.
    """
    for prefix in ("clocks_event_reasons", "clocks_throttle_reasons"):
        fields = [f"{prefix}.{suffix}" for suffix in EVENT_REASON_SUFFIXES]
        result = _run([f"--id={gpu_index}", f"--query-gpu={','.join(fields)}", "--format=csv,noheader"])
        if result.returncode == 0:
            active = [part.strip().lower() == "active" for part in result.stdout.strip().split(",")]
            return dict(zip(EVENT_REASON_SUFFIXES, active, strict=True))
    return {}


def sample(gpu_index: int = 0) -> dict:
    """One telemetry snapshot. The unit every gate and monitor is built from."""
    state = read_state(gpu_index)
    reasons = read_event_reasons(gpu_index)
    return {
        "wall_clock": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "monotonic": time.monotonic(),
        "utilization_percent": _as_float(state["utilization.gpu"]),
        "memory_used_mib": _as_float(state["memory.used"]),
        "temperature_c": _as_float(state["temperature.gpu"]),
        "memory_temperature_c": _as_float(state["temperature.memory"]),
        "clocks_sm_mhz": _as_float(state["clocks.sm"]),
        "clocks_max_sm_mhz": _as_float(state["clocks.max.sm"]),
        "power_draw_w": _as_float(state["power.draw"]),
        "enforced_power_limit_w": _as_float(state["enforced.power.limit"]),
        "any_event_reason_active": any(reasons.values()),
        "event_reasons": reasons,
    }


def compute_apps(gpu_index: int = 0) -> list[int]:
    """PIDs currently holding a CUDA context on the GPU."""
    result = _run([f"--id={gpu_index}", "--query-compute-apps=pid", "--format=csv,noheader"])
    if result.returncode != 0:
        return []
    return [int(line.strip()) for line in result.stdout.split() if line.strip().isdigit()]


def persistence_mode_enabled(gpu_index: int = 0) -> bool | None:
    result = _run([f"--id={gpu_index}", "--query-gpu=persistence_mode", "--format=csv,noheader"])
    if result.returncode != 0:
        return None
    return result.stdout.strip().lower() == "enabled"


def probe(gpu_index: int = 0) -> dict:
    """Human-readable state plus whether this process could lock clocks at all."""
    row = sample(gpu_index)
    current = row["clocks_sm_mhz"]
    nominal = row["clocks_max_sm_mhz"]
    return {
        "status": "ok",
        "gpu_index": gpu_index,
        "gpu_name": read_state(gpu_index)["name"],
        "clocks_sm_mhz": current,
        "clocks_max_sm_mhz": nominal,
        "clock_fraction_of_max": round(current / nominal, 4) if current and nominal else None,
        "temperature_c": row["temperature_c"],
        "memory_temperature_c": row["memory_temperature_c"],
        "power_draw_w": row["power_draw_w"],
        "enforced_power_limit_w": row["enforced_power_limit_w"],
        "utilization_percent": row["utilization_percent"],
        "memory_used_mib": row["memory_used_mib"],
        "compute_app_pids": compute_apps(gpu_index),
        "persistence_mode": persistence_mode_enabled(gpu_index),
        "active_event_reasons": row["event_reasons"],
        "thermally_limited_now": bool(
            row["event_reasons"].get("hw_thermal_slowdown")
            or row["event_reasons"].get("sw_thermal_slowdown")
        ),
        "clock_lock_privileged": os.geteuid() == 0 or _sudo_available(),
    }


def lock_clock(gpu_index: int, min_mhz: int, max_mhz: int) -> dict:
    result = _run([f"--id={gpu_index}", "-lgc", f"{min_mhz},{max_mhz}"], privileged=True)
    combined = (result.stdout + result.stderr).lower()
    if result.returncode == 0:
        status = "locked"
    elif "permission" in combined or "privile" in combined:
        status = "refused_insufficient_privilege"
    else:
        status = "failed"
    return {
        "action": "lock",
        "status": status,
        "gpu_index": gpu_index,
        "requested_min_mhz": min_mhz,
        "requested_max_mhz": max_mhz,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
        "state_after": probe(gpu_index) if result.returncode == 0 else None,
        "note": (
            "On this chassis -lgc is only a ceiling and HW thermal slowdown owns the floor, "
            "so a successful lock does not imply the clock is held. See PROTOCOL.md."
        ),
    }


def release_clock(gpu_index: int) -> dict:
    result = _run([f"--id={gpu_index}", "-rgc"], privileged=True)
    return {
        "action": "release",
        "status": "released" if result.returncode == 0 else "failed",
        "gpu_index": gpu_index,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


if __name__ == "__main__":  # pragma: no cover - operator convenience
    print(json.dumps(probe(0), indent=2))
