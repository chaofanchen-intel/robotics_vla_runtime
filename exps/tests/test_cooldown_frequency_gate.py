"""The cooldown gate can wait for temperature only, so a card that idles below nominal clock cannot block a run.

Decision of 2026-11-15: rtx5880 records the SM clock and the event reasons per request and in the gate CSV, but does not
wait for them (its idle clock is 210 MHz against a nominal 3,105 MHz, so the old gate timed out on every request).
The H200 profile and the other profiles keep the full gate.
"""

import argparse

from exps import hardware
from exps.core import cli, thermal
from exps.core.measure import MeasurementConfig

IDLE_COOL = {
    "wall_clock": "2026-11-15T00:00:00",
    "monotonic": 0.0,
    "utilization_percent": 0.0,
    "memory_used_mib": 15280.0,
    "temperature_c": 30.0,
    "memory_temperature_c": None,
    "clocks_sm_mhz": 210.0,
    "clocks_max_sm_mhz": 3105.0,
    "power_draw_w": 28.0,
    "enforced_power_limit_w": 285.0,
    "any_event_reason_active": False,
    "event_reasons": {},
}

TEMPERATURE_ONLY = dict(
    max_utilization_percent=None,
    max_memory_used_mib=None,
    max_temperature_celsius=70.0,
    max_memory_temperature_celsius=None,
)


def test_rtx5880_does_not_gate_on_frequency_and_the_other_profiles_still_do():
    profiles = hardware.profiles()
    assert profiles["rtx5880"].cooldown_gates_frequency is False
    assert profiles["h200nvl"].cooldown_gates_frequency is True
    assert profiles["generic"].cooldown_gates_frequency is True


def test_frequency_checks_are_dropped_only_when_asked():
    gated = thermal._conditions(IDLE_COOL, **TEMPERATURE_ONLY, require_full_clock=True)
    assert gated["full_sm_clock"] is False and gated["no_event_reason"] is True

    relaxed = thermal._conditions(IDLE_COOL, **TEMPERATURE_ONLY, require_full_clock=False, require_event_free=False)
    assert "full_sm_clock" not in relaxed and "no_event_reason" not in relaxed
    assert all(relaxed.values())

    throttled = {**IDLE_COOL, "any_event_reason_active": True}
    assert thermal._conditions(throttled, **TEMPERATURE_ONLY, require_full_clock=False,
                               require_event_free=True)["no_event_reason"] is False
    assert all(thermal._conditions(throttled, **TEMPERATURE_ONLY, require_full_clock=False,
                                   require_event_free=False).values())


def test_cooldown_gate_passes_an_idle_card_when_frequency_is_recorded_not_gated(monkeypatch):
    monkeypatch.setattr(thermal.gpu, "sample", lambda gpu_index=0: dict(IDLE_COOL))

    relaxed = thermal.cooldown_gate(target_celsius=70.0, continuous_seconds=0.05, timeout_seconds=2.0,
                                    period_seconds=0.01, require_frequency_gate=False)
    assert relaxed["status"] == "passed"
    assert relaxed["targets"]["require_full_clock"] is False
    assert relaxed["targets"]["require_event_free"] is False
    assert relaxed["last_sample"]["clocks_sm_mhz"] == 210.0  # the clock is still in the verdict

    strict = thermal.cooldown_gate(target_celsius=70.0, continuous_seconds=0.05, timeout_seconds=0.3,
                                   period_seconds=0.01)
    assert strict["status"] != "passed"
    assert strict["targets"]["require_full_clock"] is True


def test_the_flag_reaches_the_client_through_the_argv(tmp_path):
    parser = argparse.ArgumentParser()
    cli.add_client_protocol_arguments(parser)
    for flag in (False, True):
        config = MeasurementConfig(output_dir=tmp_path, cooldown_gates_frequency=flag)
        args = parser.parse_args(cli.protocol_argv(config))
        assert cli.client_measurement_config(args, tmp_path).cooldown_gates_frequency is flag


def test_the_measurement_config_defaults_to_the_full_gate_and_records_it(tmp_path):
    config = MeasurementConfig(output_dir=tmp_path)
    assert config.cooldown_gates_frequency is True
    assert config.as_record()["cooldown_gates_frequency"] is True
