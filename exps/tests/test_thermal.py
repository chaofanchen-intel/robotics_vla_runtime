"""Tests for the thermal gates and the per-request monitor.

The driver is replaced by a scripted sample sequence, so the gate logic is
tested without a GPU and without waiting for real cooling.
"""
from __future__ import annotations

import itertools

import pytest

from exps.core import gpu, thermal


def make_sample(
    *,
    temperature=60.0,
    memory_temperature=55.0,
    sm=1785.0,
    nominal=1785.0,
    memory_used=14.0,
    utilization=0.0,
    reasons=None,
):
    reasons = reasons or {}
    return {
        "wall_clock": "2026-10-16T00:00:00",
        "monotonic": 0.0,
        "utilization_percent": utilization,
        "memory_used_mib": memory_used,
        "temperature_c": temperature,
        "memory_temperature_c": memory_temperature,
        "clocks_sm_mhz": sm,
        "clocks_max_sm_mhz": nominal,
        "power_draw_w": 90.0,
        "enforced_power_limit_w": 600.0,
        "any_event_reason_active": any(reasons.values()),
        "event_reasons": reasons,
    }


@pytest.fixture
def scripted(monkeypatch):
    """Feed the gates a fixed sample sequence and a clock that never sleeps."""

    def install(samples, *, step=1.0):
        sequence = iter(samples)
        last = {"value": samples[-1]}

        def fake_sample(_gpu_index=0):
            try:
                last["value"] = next(sequence)
            except StopIteration:
                pass
            return last["value"]

        ticks = itertools.count(0.0, step)
        monkeypatch.setattr(gpu, "sample", fake_sample)
        monkeypatch.setattr(thermal.time, "monotonic", lambda: next(ticks))
        monkeypatch.setattr(thermal.time, "sleep", lambda _s: None)

    return install


def test_cooldown_gate_passes_once_all_three_conditions_hold(scripted, tmp_path):
    scripted([make_sample(temperature=68.0)] * 200)
    verdict = thermal.cooldown_gate(
        target_celsius=70.0, continuous_seconds=5.0, timeout_seconds=100.0,
        output=tmp_path / "cooldown.csv",
    )
    assert verdict["status"] == "passed"
    assert verdict["timed_out"] is False
    assert (tmp_path / "cooldown.csv").exists()


def test_cooldown_gate_rejects_a_cool_card_that_is_still_downclocked(scripted):
    # 68 C is under target, but the card is at 1710 of 1785 MHz coming out of a
    # throttle. Measuring here would not be comparable to a full-clock start.
    scripted([make_sample(temperature=68.0, sm=1710.0)] * 100)
    verdict = thermal.cooldown_gate(
        target_celsius=70.0, continuous_seconds=5.0, timeout_seconds=30.0
    )
    assert verdict["status"] == "failed_timeout"
    assert "below nominal" in verdict["blocking_reason"]


def test_cooldown_gate_rejects_an_active_throttle_reason(scripted):
    scripted([make_sample(temperature=60.0, reasons={"hw_thermal_slowdown": True})] * 100)
    verdict = thermal.cooldown_gate(
        target_celsius=70.0, continuous_seconds=5.0, timeout_seconds=30.0
    )
    assert verdict["status"] == "failed_timeout"
    assert "event reasons" in verdict["blocking_reason"]


@pytest.mark.parametrize("field", ["temperature", "sm", "nominal"])
def test_cooldown_gate_rejects_missing_start_telemetry(scripted, field):
    scripted([make_sample(**{field: None})] * 100)
    verdict = thermal.cooldown_gate(
        target_celsius=70.0, continuous_seconds=5.0, timeout_seconds=20.0
    )
    assert verdict["status"] == "failed_timeout"


def test_idle_gate_rejects_missing_memory_telemetry(scripted, tmp_path):
    scripted([make_sample(memory_used=None)] * 100)
    verdict = thermal.idle_gate(
        tmp_path / "idle.csv", max_temperature_celsius=70.0,
        continuous_seconds=5.0, timeout_seconds=20.0,
    )
    assert verdict["status"] == "failed_timeout"
    assert verdict["blocking_reason"] == "GPU memory telemetry unavailable"


def test_cooldown_gate_requires_the_window_to_be_continuous(scripted):
    # Cool, cool, then a throttle spike: the continuous window must restart.
    samples = (
        [make_sample(temperature=68.0)] * 3
        + [make_sample(temperature=68.0, reasons={"sw_thermal_slowdown": True})]
        + [make_sample(temperature=68.0)] * 100
    )
    scripted(samples)
    verdict = thermal.cooldown_gate(
        target_celsius=70.0, continuous_seconds=5.0, timeout_seconds=200.0
    )
    assert verdict["status"] == "passed"
    # The spike at t=3 forced a restart, so the pass cannot have happened at t=5.
    assert verdict["waited_seconds"] > 5.0


def test_cooldown_gate_times_out_without_raising(scripted):
    scripted([make_sample(temperature=90.0)] * 100)
    verdict = thermal.cooldown_gate(
        target_celsius=70.0, continuous_seconds=5.0, timeout_seconds=20.0
    )
    assert verdict["timed_out"] is True
    assert "has not cooled the card" in verdict["blocking_reason"]


def test_cooldown_gate_can_also_watch_the_memory_junction(scripted):
    scripted([make_sample(temperature=65.0, memory_temperature=85.0)] * 100)
    verdict = thermal.cooldown_gate(
        target_celsius=70.0, memory_target_celsius=70.0,
        continuous_seconds=5.0, timeout_seconds=30.0,
    )
    assert verdict["status"] == "failed_timeout"
    assert "memory junction" in verdict["blocking_reason"]


@pytest.mark.parametrize(
    ("mps_enabled", "memory_used", "expected_status"),
    [
        (False, 88.0, "failed_timeout"),
        (False, 41_000.0, "failed_timeout"),
        (True, 88.0, "passed"),
        (True, 41_000.0, "failed_timeout"),
    ],
)
def test_idle_gate_uses_mps_specific_memory_budget(
    scripted, tmp_path, monkeypatch, mps_enabled, memory_used, expected_status
):
    if mps_enabled:
        monkeypatch.setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/mps")
    else:
        monkeypatch.delenv("CUDA_MPS_PIPE_DIRECTORY", raising=False)
    scripted([make_sample(temperature=50.0, memory_used=memory_used)] * 100)
    verdict = thermal.idle_gate(
        tmp_path / "idle.csv", max_temperature_celsius=70.0,
        continuous_seconds=5.0, timeout_seconds=20.0,
    )
    assert verdict["status"] == expected_status
    if expected_status == "failed_timeout":
        assert "another process still holds the GPU" in verdict["blocking_reason"]


def test_idle_gate_does_not_require_full_clock(scripted, tmp_path):
    # An idle GPU sits at 345 MHz by design; gating on clock here would never pass.
    scripted([make_sample(temperature=48.0, sm=345.0)] * 100)
    verdict = thermal.idle_gate(
        tmp_path / "idle.csv", max_temperature_celsius=70.0,
        continuous_seconds=5.0, timeout_seconds=50.0,
    )
    assert verdict["status"] == "passed"


def test_loaded_idle_gate_tolerates_resident_model_weights(scripted, tmp_path):
    scripted([make_sample(temperature=62.0, memory_used=78_000.0)] * 100)
    verdict = thermal.loaded_idle_gate(
        tmp_path / "loaded.csv", max_temperature_celsius=70.0,
        continuous_seconds=5.0, timeout_seconds=50.0,
    )
    assert verdict["status"] == "passed"


def _monitor_with(samples):
    monitor = thermal.RequestThermalMonitor()
    monitor._samples = list(samples)  # noqa: SLF001 - exercising verdict() directly
    return monitor


def test_monitor_verdict_is_clean_at_a_steady_full_clock():
    verdict = _monitor_with([make_sample()] * 10).verdict()
    assert verdict["thermally_clean"] is True
    assert verdict["sm_clock_spread_mhz"] == 0.0


def test_monitor_verdict_catches_a_throttle_that_begins_and_ends_mid_request():
    samples = (
        [make_sample()] * 3
        + [make_sample(sm=345.0, reasons={"hw_thermal_slowdown": True})] * 2
        + [make_sample()] * 3
    )
    verdict = _monitor_with(samples).verdict()
    assert verdict["thermally_clean"] is False
    assert verdict["event_reason_active_during_request"] is True
    assert "hw_thermal_slowdown" in verdict["event_reasons_seen"]


def test_monitor_verdict_catches_a_silent_downclock_with_no_reason_flag():
    # The archived I2I-256 cell dropped to 1710 MHz without a reason bit set.
    verdict = _monitor_with([make_sample()] * 5 + [make_sample(sm=1710.0)] * 5).verdict()
    assert verdict["thermally_clean"] is False
    assert verdict["held_clock_floor"] is False


def test_monitor_verdict_without_telemetry_is_unclean_not_silently_clean():
    verdict = thermal.RequestThermalMonitor().verdict()
    assert verdict["thermally_clean"] is False
    assert verdict["samples"] == 0


def test_summarize_gate_series_counts_timeouts():
    summary = thermal.summarize_gate_series(
        [
            {"status": "passed", "waited_seconds": 90.0},
            {"status": "passed", "waited_seconds": 110.0},
            {"status": "failed_timeout", "waited_seconds": 900.0},
        ]
    )
    assert summary["count"] == 3
    assert summary["timed_out"] == 1
    assert summary["waited_seconds_max"] == 900.0
