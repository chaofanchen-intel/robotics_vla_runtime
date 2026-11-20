"""Tests for the shared metric schema and aggregation rules."""
from __future__ import annotations

import pytest

from exps.core import metrics


def test_nearest_rank_returns_observed_values_only():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert metrics.nearest_rank(values, 0.50) == 30.0
    assert metrics.nearest_rank(values, 0.95) == 50.0
    # Never interpolates: every result is a value that was actually measured.
    assert all(metrics.nearest_rank(values, q) in values for q in (0.1, 0.5, 0.9, 0.99))


def test_nearest_rank_on_empty_input_is_none_not_zero():
    assert metrics.nearest_rank([], 0.5) is None
    assert metrics.nearest_rank([None, None], 0.5) is None


def test_summarize_values_reports_count_zero_rather_than_omitting_keys():
    summary = metrics.summarize_values([])
    assert summary["count"] == 0
    for key in ("mean", "min", "max", "p50", "p95", "p99"):
        assert summary[key] is None


def test_empty_request_metrics_has_every_schema_field():
    row = metrics.empty_request_metrics()
    assert set(row) == set(metrics.REQUEST_METRIC_FIELDS)
    assert all(value is None for value in row.values())


def test_compute_rtf_guards_against_zero_duration():
    assert metrics.compute_rtf(1000.0, 2.0) == pytest.approx(0.5)
    assert metrics.compute_rtf(1000.0, 0.0) is None
    assert metrics.compute_rtf(None, 2.0) is None


def test_inter_arrival_ms():
    assert metrics.inter_arrival_ms([10.0, 15.0, 25.0]) == [5.0, 10.0]
    assert metrics.inter_arrival_ms([10.0]) == []


def test_intra_start_drift_flags_the_archived_degrading_cell():
    # The real I2I-256 vLLM two-stage trace: ten requests, +18% by the end.
    observed = [6485, 6502, 6411, 6417, 6391, 6697, 6372, 6570, 6619, 7639]
    verdict = metrics.intra_start_drift(observed)
    assert verdict["drifting"] is True
    assert verdict["status"] == "suspect_intra_start_drift"


def test_intra_start_drift_accepts_the_archived_flat_cell():
    # The real I2T-256 M* trace: flat across ten requests despite 65 -> 72 C.
    observed = [879, 880, 885, 881, 879, 880, 850, 872, 889, 870]
    verdict = metrics.intra_start_drift(observed)
    assert verdict["drifting"] is False
    assert verdict["status"] == "flat"


def test_intra_start_drift_reports_insufficient_data_rather_than_passing():
    verdict = metrics.intra_start_drift([100.0, 101.0])
    assert verdict["status"] == "insufficient_requests"
    assert verdict["drifting"] is False


def test_intra_start_drift_does_not_compare_different_workload_fixtures():
    verdict = metrics.intra_start_drift(
        [100.0, 150.0, 200.0],
        workload_keys=["fixture-a", "fixture-b", "fixture-c"],
    )

    assert verdict["status"] == "not_assessed_variable_workload"
    assert verdict["workload_count"] == 3
    assert verdict["drifting"] is False


def test_intra_start_drift_still_compares_repeated_same_workload():
    verdict = metrics.intra_start_drift(
        [100.0, 150.0, 200.0],
        workload_keys=["fixture-a", "fixture-a", "fixture-a"],
    )

    assert verdict["status"] == "suspect_intra_start_drift"
    assert verdict["drifting"] is True


def test_intra_start_drift_does_not_assess_incomplete_workload_keys():
    verdict = metrics.intra_start_drift(
        [100.0, 150.0, 200.0],
        workload_keys=["fixture-a", None, "fixture-c"],
    )

    assert verdict["status"] == "not_assessed_missing_workload_key"
    assert verdict["drifting"] is False


def _row(e2e, *, success=True, clean=True, timed_out=False, temp=68.0, phase="measured"):
    return {
        "phase": phase,
        "success": success,
        "e2e_ms": e2e,
        "thermally_clean": clean,
        "cooldown_timed_out": timed_out,
        "temperature_c_before": temp,
        "sm_clock_mhz_before": 1785.0,
        "sm_clock_mhz_nominal": 1785.0,
        "itl_ms": [],
    }


def test_headline_rows_excludes_failed_and_timed_out_but_keeps_thermal_diagnostics():
    rows = [
        _row(100.0),
        _row(200.0, success=False),
        _row(300.0, timed_out=True),
        _row(400.0, clean=False),
    ]
    assert [row["e2e_ms"] for row in metrics.headline_rows(rows)] == [100.0, 400.0]


def test_summarize_start_counts_exclusions_separately_from_the_headline():
    rows = [
        _row(50.0, phase="precondition"),
        _row(100.0),
        _row(110.0),
        _row(120.0),
        _row(900.0, timed_out=True),
        _row(950.0, clean=False),
    ]
    summary = metrics.summarize_start(rows)
    assert summary["measured_requests"] == 5
    assert summary["headline_requests"] == 4
    assert summary["excluded_cooldown_timeout"] == 1
    assert summary["requests_not_thermally_clean"] == 1
    # The preconditioning request must never reach the aggregate.
    assert summary["metrics"]["e2e_ms"]["min"] == 100.0
    assert summary["metrics"]["e2e_ms"]["max"] == 950.0


def test_summarize_start_flags_a_request_that_did_not_start_at_full_clock():
    rows = [_row(100.0), _row(110.0)]
    rows[1]["sm_clock_mhz_before"] = 1710.0
    summary = metrics.summarize_start(rows)
    assert summary["thermal_start_state"]["all_requests_started_at_full_clock"] is False


def _start_summary(p50_values, *, drifting=False, temp=68.0):
    return {
        "metrics": {"e2e_ms": {"p50": p50_values}},
        "intra_start_drift": {"drifting": drifting},
        "thermal_start_state": {"temperature_c": {"mean": temp}},
    }


def test_combine_starts_takes_the_median_of_per_start_p50s():
    combined = metrics.combine_starts(
        [_start_summary(100.0), _start_summary(120.0), _start_summary(110.0)]
    )
    assert combined["headline_p50"] == 110.0
    assert combined["valid_starts"] == 3
    assert combined["label"] == "ok"


def test_combine_starts_labels_a_single_start_as_preliminary():
    combined = metrics.combine_starts([_start_summary(100.0)])
    assert "preliminary_single_start" in combined["labels"]


def test_combine_starts_surfaces_the_round1_start_variance():
    # Round 1's E07: three starts of an identical workload, 1.39x apart.
    combined = metrics.combine_starts(
        [_start_summary(94.75), _start_summary(118.66), _start_summary(132.08)]
    )
    assert combined["between_start_spread"] > 1.3


def test_combine_starts_propagates_drift_labels():
    combined = metrics.combine_starts(
        [_start_summary(100.0, drifting=True), _start_summary(110.0)]
    )
    assert "suspect_intra_start_drift" in combined["labels"]


def test_combine_starts_labels_drift_not_assessed_for_variable_workloads():
    start = _start_summary(100.0)
    start["intra_start_drift"] = {
        "status": "not_assessed_variable_workload",
        "drifting": False,
    }

    combined = metrics.combine_starts([start, _start_summary(110.0)])

    assert combined["headline_p50"] == 105.0
    assert combined["starts_with_unassessed_drift"] == 1
    assert "intra_start_drift_not_assessed_variable_workload" in combined["labels"]


def test_compare_computes_speedup_in_the_documented_direction():
    baseline = metrics.combine_starts([_start_summary(200.0), _start_summary(200.0)])
    candidate = metrics.combine_starts([_start_summary(100.0), _start_summary(100.0)])
    result = metrics.compare(baseline, candidate)
    assert result["speedup"] == pytest.approx(2.0)
    assert result["comparison_label"] == "work_matched"


def test_compare_disqualifies_engines_that_started_from_different_temperatures():
    baseline = metrics.combine_starts([_start_summary(200.0), _start_summary(200.0)])
    candidate = metrics.combine_starts([_start_summary(100.0), _start_summary(100.0)])
    result = metrics.compare(
        baseline, candidate, baseline_start_temp_c=55.0, candidate_start_temp_c=72.0
    )
    assert "unequal_thermal_start" in result["labels"]
    assert result["comparison_label"] == "qualified"
    # The ratio is still reported, just labelled -- never silently dropped.
    assert result["speedup"] == pytest.approx(2.0)


def test_compare_accepts_a_small_temperature_difference():
    baseline = metrics.combine_starts([_start_summary(200.0), _start_summary(200.0)])
    candidate = metrics.combine_starts([_start_summary(100.0), _start_summary(100.0)])
    result = metrics.compare(
        baseline, candidate, baseline_start_temp_c=68.0, candidate_start_temp_c=69.5
    )
    assert result["labels"] == []
