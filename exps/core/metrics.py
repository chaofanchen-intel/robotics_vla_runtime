#!/usr/bin/env python3
"""One metric schema for every task, and the aggregation rules around it.

The problem this solves
-----------------------
In the previous rounds each client invented its own fields: `e2e_ms` here,
`client_e2e_ms` there, `p50` in one summary and `p50_nearest_rank` in another,
TTFB present for audio-to-text and absent everywhere else. Cross-task reading
was manual and error-prone, and a missing field was indistinguishable from a
field that was measured as zero.

Here every task emits the same per-request keys and the same summary shape. A
metric a task cannot produce is written as ``None``, never omitted -- "not
applicable" and "not measured" must not look like "absent".

Aggregation rules
-----------------
* **Percentiles are nearest-rank.** With 5 measured requests per start, linear
  interpolation invents values between samples that were never observed.
* **Starts are never pooled.** Each fresh start is summarised on its own, then
  the per-start p50s are combined by median. Round 1's E07 measured 94.75 /
  118.66 / 132.08 ms for three starts of an identical workload -- a 1.39x
  spread attributable to nothing but service startup. Pooling requests across
  starts hides that inside a single distribution.
* **A ratio backed by one start is labelled, not reported.** `preliminary_single_start`.
* **Only successful, non-timed-out requests enter the headline.** In-request
  thermal telemetry remains diagnostic; only the pre-request cooldown gate
  determines whether a successful request is eligible.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Iterable, Sequence

#: Per-request metric keys. Every task writes all of them; unsupported ones are None.
REQUEST_METRIC_FIELDS = (
    "e2e_ms",            # POST start -> response body EOF
    "ttfb_ms",           # first response byte of any kind
    "ttft_ms",           # first text token
    "ttfa_ms",           # first audio sample
    "itl_ms",            # inter-token/chunk latencies, list
    "output_tokens",
    "output_bytes",
    "audio_duration_s",
    "rtf",               # e2e_ms / (audio_duration_s * 1000)
)

#: Per-request thermal context keys, written by the cell runner, not the client.
REQUEST_THERMAL_FIELDS = (
    "cooldown_waited_s",
    "cooldown_timed_out",
    "temperature_c_before",
    "memory_temperature_c_before",
    "sm_clock_mhz_before",
    "sm_clock_mhz_nominal",
    "sm_clock_spread_mhz",
    "temperature_c_max",
    "throttle_reasons_active",
    "thermally_clean",
    # Sampled device memory around the request; see thermal.RequestThermalMonitor.verdict.
    "memory_used_mib_before",
    "memory_used_mib_max",
)

#: Summary quantiles, nearest-rank.
QUANTILES = (("p50", 0.50), ("p95", 0.95), ("p99", 0.99))

#: Intra-start drift thresholds. A cell breaching either is `suspect_intra_start_drift`.
DRIFT_MAX_MIN_RATIO = 1.15
DRIFT_RELATIVE_SLOPE_PER_REQUEST = 0.02  # 2% of the mean per request index

#: Two engines whose mean starting temperature differs by more than this are not
#: compared as equal work; the ratio is labelled `unequal_thermal_start`.
MAX_CROSS_FRAMEWORK_START_TEMP_SPREAD_C = 3.0


def nearest_rank(values: Sequence[float], quantile: float) -> float | None:
    """Nearest-rank percentile: always returns a value that was actually observed."""
    usable = sorted(v for v in values if v is not None)
    if not usable:
        return None
    index = max(0, math.ceil(quantile * len(usable)) - 1)
    return usable[index]


def summarize_values(values: Sequence[float | None]) -> dict[str, Any]:
    """The standard distribution block used for every metric, everywhere."""
    usable = [float(v) for v in values if v is not None]
    if not usable:
        return {"count": 0, "mean": None, "min": None, "max": None, **{name: None for name, _ in QUANTILES}}
    return {
        "count": len(usable),
        "mean": statistics.fmean(usable),
        "min": min(usable),
        "max": max(usable),
        "stdev": statistics.stdev(usable) if len(usable) > 1 else 0.0,
        **{name: nearest_rank(usable, q) for name, q in QUANTILES},
    }


def empty_request_metrics() -> dict[str, Any]:
    """A fully-populated metric dict with every value None.

    Clients start from this and fill in what they can measure, which is what
    keeps "this task has no TTFT" visible instead of silently absent.
    """
    return {field: None for field in REQUEST_METRIC_FIELDS}


def compute_rtf(e2e_ms: float | None, audio_duration_s: float | None) -> float | None:
    """Real-time factor: wall time per second of generated audio. Lower is faster."""
    if e2e_ms is None or not audio_duration_s:
        return None
    return e2e_ms / (audio_duration_s * 1000.0)


def inter_arrival_ms(arrival_ms: Sequence[float]) -> list[float]:
    """Gaps between successive streamed chunks, for ITL."""
    return [b - a for a, b in zip(arrival_ms, arrival_ms[1:], strict=False)]


def _linear_slope(values: Sequence[float]) -> float | None:
    """Ordinary least-squares slope of value against index (0, 1, 2, ...)."""
    n = len(values)
    if n < 3:
        return None
    xs = list(range(n))
    mean_x = statistics.fmean(xs)
    mean_y = statistics.fmean(values)
    denominator = sum((x - mean_x) ** 2 for x in xs)
    if denominator == 0:
        return None
    return sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values, strict=True)) / denominator


def intra_start_drift(
    e2e_values: Sequence[float | None],
    *,
    workload_keys: Sequence[str | None] | None = None,
) -> dict[str, Any]:
    """Detect latency creeping upward across the measured requests of one start.

    This is the direct test of "did heat accumulate despite the per-request
    gate?". The gate should make the answer no; this says so with evidence
    rather than by assumption. When requests used different workload fixtures,
    latency changes are not evidence of thermal drift and are not assessed.
    Fewer than three comparable requests cannot support a slope.
    """
    if workload_keys is not None and len(workload_keys) != len(e2e_values):
        raise ValueError("workload_keys must have one entry per e2e value")

    usable_pairs = [
        (
            float(value),
            workload_keys[index] if workload_keys is not None else None,
        )
        for index, value in enumerate(e2e_values)
        if value is not None
    ]
    usable = [value for value, _ in usable_pairs]

    if workload_keys is not None and usable_pairs:
        keys = [key for _, key in usable_pairs]
        if any(not key for key in keys):
            return {
                "status": "not_assessed_missing_workload_key",
                "requests": len(usable),
                "drifting": False,
                "note": "one or more measured requests had no workload key",
            }
        unique_keys = set(keys)
        if len(unique_keys) > 1:
            return {
                "status": "not_assessed_variable_workload",
                "requests": len(usable),
                "workload_count": len(unique_keys),
                "drifting": False,
                "note": "requests used different workload fixtures; latency changes are not comparable",
            }

    if len(usable) < 3:
        return {
            "status": "insufficient_requests",
            "requests": len(usable),
            "drifting": False,
            "note": "at least 3 measured requests are needed to fit a slope",
        }
    mean_value = statistics.fmean(usable)
    slope = _linear_slope(usable)
    relative_slope = (slope / mean_value) if slope is not None and mean_value else None
    ratio = max(usable) / min(usable) if min(usable) > 0 else None
    drifting = bool(
        (relative_slope is not None and relative_slope > DRIFT_RELATIVE_SLOPE_PER_REQUEST)
        or (ratio is not None and ratio > DRIFT_MAX_MIN_RATIO)
    )
    return {
        "status": "suspect_intra_start_drift" if drifting else "flat",
        "requests": len(usable),
        "drifting": drifting,
        "slope_ms_per_request": slope,
        "relative_slope_per_request": relative_slope,
        "max_over_min": ratio,
        "thresholds": {
            "relative_slope_per_request": DRIFT_RELATIVE_SLOPE_PER_REQUEST,
            "max_over_min": DRIFT_MAX_MIN_RATIO,
        },
    }


def headline_rows(rows: Iterable[dict]) -> list[dict]:
    """Requests eligible for the headline aggregate.

    Excluded, but kept in `requests.jsonl`: failures, requests whose cooldown
    gate timed out (they did not start from the agreed state). In-request
    thermal and frequency telemetry remains diagnostic.
    """
    return [
        row
        for row in rows
        if row.get("success")
        and not row.get("cooldown_timed_out")
    ]


def summarize_start(rows: Sequence[dict]) -> dict[str, Any]:
    """Summarise the measured requests of a single fresh start."""
    measured = [row for row in rows if row.get("phase") == "measured"]
    eligible = headline_rows(measured)
    e2e = [row.get("e2e_ms") for row in eligible]
    start_temps = [row.get("temperature_c_before") for row in measured if row.get("temperature_c_before")]
    workload_keys = [row.get("drift_workload_key") for row in eligible]
    drift = intra_start_drift(
        e2e,
        workload_keys=workload_keys if any(key is not None for key in workload_keys) else None,
    )
    summary: dict[str, Any] = {
        "measured_requests": len(measured),
        "successful_requests": sum(1 for row in measured if row.get("success")),
        "headline_requests": len(eligible),
        "excluded_cooldown_timeout": sum(1 for row in measured if row.get("cooldown_timed_out")),
        "requests_not_thermally_clean": sum(
            1 for row in measured if row.get("thermally_clean") is False
        ),
        "metrics": {
            field: summarize_values([row.get(field) for row in eligible])
            for field in REQUEST_METRIC_FIELDS
            if field != "itl_ms"
        },
        "intra_start_drift": drift,
        "gpu_memory_mib": {
            "before_request": summarize_values([row.get("memory_used_mib_before") for row in eligible]),
            "peak_during_request": summarize_values([row.get("memory_used_mib_max") for row in eligible]),
            "note": (
                "nvidia-smi device-wide memory.used sampled during each request; a lower bound on "
                "the in-request peak that includes pre-reserved pools. None for runs recorded "
                "before this field existed."
            ),
        },
        "thermal_start_state": {
            "temperature_c": summarize_values(start_temps),
            "all_requests_started_at_full_clock": all(
                row.get("sm_clock_mhz_before") is not None
                and row.get("sm_clock_mhz_nominal") is not None
                and row["sm_clock_mhz_before"] >= row["sm_clock_mhz_nominal"]
                for row in measured
            )
            if measured
            else None,
        },
    }
    itl_all = [value for row in eligible for value in (row.get("itl_ms") or [])]
    summary["metrics"]["itl_ms"] = summarize_values(itl_all)
    return summary


def combine_starts(start_summaries: Sequence[dict], *, metric: str = "e2e_ms") -> dict[str, Any]:
    """Combine per-start summaries into one cell-level figure.

    The headline is the **median of the per-start p50s**, not a pooled
    percentile. Between-start spread is reported alongside it, because on this
    host it has been as large as 1.39x on an identical workload.
    """
    per_start_p50 = [
        s["metrics"].get(metric, {}).get("p50")
        for s in start_summaries
        if s["metrics"].get(metric, {}).get("p50") is not None
    ]
    valid_starts = len(per_start_p50)
    if not per_start_p50:
        return {
            "metric": metric,
            "valid_starts": 0,
            "headline_p50": None,
            "label": "no_valid_starts",
        }
    spread = max(per_start_p50) / min(per_start_p50) if min(per_start_p50) > 0 else None
    drifting_starts = sum(1 for s in start_summaries if s["intra_start_drift"].get("drifting"))
    drift_statuses = [
        (s.get("intra_start_drift") or {}).get("status")
        for s in start_summaries
    ]
    unassessed_variable_workload_starts = sum(
        1 for status in drift_statuses if status == "not_assessed_variable_workload"
    )
    unassessed_missing_workload_key_starts = sum(
        1 for status in drift_statuses if status == "not_assessed_missing_workload_key"
    )
    unassessed_drift_starts = sum(
        1 for status in drift_statuses if status and status.startswith("not_assessed_")
    )
    labels: list[str] = []
    if valid_starts < 2:
        labels.append("preliminary_single_start")
    if drifting_starts:
        labels.append("suspect_intra_start_drift")
    if unassessed_variable_workload_starts:
        labels.append("intra_start_drift_not_assessed_variable_workload")
    if unassessed_missing_workload_key_starts:
        labels.append("intra_start_drift_not_assessed_missing_workload_key")
    return {
        "metric": metric,
        "valid_starts": valid_starts,
        "per_start_p50": per_start_p50,
        "headline_p50": statistics.median(per_start_p50),
        "between_start_spread": spread,
        "starts_with_drift": drifting_starts,
        "starts_with_unassessed_drift": unassessed_drift_starts,
        "label": "ok" if not labels else "+".join(labels),
        "labels": labels,
    }


def mean_start_temperature(start_summaries: Sequence[dict]) -> float | None:
    means = [
        s["thermal_start_state"]["temperature_c"]["mean"]
        for s in start_summaries
        if s["thermal_start_state"]["temperature_c"]["mean"] is not None
    ]
    return statistics.fmean(means) if means else None


def compare(
    baseline: dict,
    candidate: dict,
    *,
    baseline_start_temp_c: float | None = None,
    candidate_start_temp_c: float | None = None,
    metric: str = "e2e_ms",
) -> dict[str, Any]:
    """Speedup of `candidate` over `baseline`, with every disqualifier attached.

    `speedup = baseline_p50 / candidate_p50`; above 1.0 means the candidate is
    faster. The ratio is emitted together with the labels that say how far it
    can be trusted -- a single-start ratio, a drifting cell, or two engines
    that ran from materially different starting temperatures are all reported
    and labelled rather than suppressed or silently ranked.
    """
    labels = sorted(set(baseline.get("labels", []) + candidate.get("labels", [])))
    temp_delta = None
    if baseline_start_temp_c is not None and candidate_start_temp_c is not None:
        temp_delta = abs(baseline_start_temp_c - candidate_start_temp_c)
        if temp_delta > MAX_CROSS_FRAMEWORK_START_TEMP_SPREAD_C:
            labels.append("unequal_thermal_start")
    base_p50 = baseline.get("headline_p50")
    cand_p50 = candidate.get("headline_p50")
    speedup = (base_p50 / cand_p50) if base_p50 and cand_p50 else None
    return {
        "metric": metric,
        "baseline_p50": base_p50,
        "candidate_p50": cand_p50,
        "speedup": speedup,
        "baseline_mean_start_temperature_c": baseline_start_temp_c,
        "candidate_mean_start_temperature_c": candidate_start_temp_c,
        "start_temperature_delta_c": temp_delta,
        "max_allowed_start_temperature_delta_c": MAX_CROSS_FRAMEWORK_START_TEMP_SPREAD_C,
        "labels": sorted(set(labels)),
        "comparison_label": "work_matched" if not labels else "qualified",
    }
