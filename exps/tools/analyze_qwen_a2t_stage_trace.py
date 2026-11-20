#!/usr/bin/env python3
"""Join optional Qwen3-Omni server stage spans to A2T request intervals."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # direct script invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths


def _unix_ns(value: str | None) -> int | None:
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1e9)


def _within(event: dict[str, Any], request: dict[str, Any]) -> bool:
    request_start = _unix_ns(request.get("started_at_utc"))
    request_end = _unix_ns(request.get("finished_at_utc"))
    event_start = event.get("started_unix_ns")
    event_end = event.get("finished_unix_ns")
    if None in (request_start, request_end, event_start, event_end):
        return False
    return int(event_start) < int(request_end) and int(event_end) > int(request_start)


def _sum(events: list[dict[str, Any]], key: str) -> float | None:
    values = [event[key] for event in events if event.get(key) is not None]
    return float(sum(values)) if values else None


def _stage_rollup(events: list[dict[str, Any]]) -> dict[str, Any]:
    stage_counts: dict[str, int] = {}
    for event in events:
        stage = event.get("stage")
        if isinstance(stage, str):
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
    preprocessor = [
        event for event in events if event.get("stage") == "request_preprocessor_total"
    ]
    feature_extractor = [event for event in events if event.get("stage") == "audio_feature_extractor"]
    encoder = [event for event in events if event.get("stage") == "audio_encoder"]
    thinker = [
        event for event in events if event.get("stage") == "thinker_transformer_core"
    ]
    thinker_prefill = []
    thinker_decode = []
    thinker_other = []
    for event in thinker:
        phase = str(event.get("phase") or "").lower()
        if "prefill" in phase or "extend" in phase:
            thinker_prefill.append(event)
        elif "decode" in phase:
            thinker_decode.append(event)
        else:
            thinker_other.append(event)
    thinker_runner = [
        event
        for event in events
        if event.get("stage") in ("thinker_submodule_step", "thinker_runner_step")
    ]
    preprocessor_wall_ms = _sum(preprocessor, "wall_ms")
    encoder_wall_ms = _sum(encoder, "wall_ms")
    thinker_prefill_wall_ms = _sum(thinker_prefill, "wall_ms")
    thinker_decode_wall_ms = _sum(thinker_decode, "wall_ms")
    thinker_core_wall_ms = _sum(thinker, "wall_ms")
    thinker_core_cuda_ms = _sum(thinker, "cuda_event_ms")
    return {
        "event_count": len(events),
        "stage_counts": stage_counts,
        "request_preprocessor_inclusive_wall_ms": preprocessor_wall_ms,
        "audio_feature_extractor_wall_ms": _sum(feature_extractor, "wall_ms"),
        "audio_encoder_wall_ms": encoder_wall_ms,
        "audio_encoder_cuda_event_ms": _sum(encoder, "cuda_event_ms"),
        "preprocessor_plus_audio_encoder_wall_ms": (
            preprocessor_wall_ms + encoder_wall_ms
            if preprocessor_wall_ms is not None and encoder_wall_ms is not None
            else None
        ),
        "thinker_prefill_wall_ms": thinker_prefill_wall_ms,
        "thinker_prefill_cuda_event_ms": _sum(thinker_prefill, "cuda_event_ms"),
        "thinker_decode_wall_ms": thinker_decode_wall_ms,
        "thinker_decode_cuda_event_ms": _sum(thinker_decode, "cuda_event_ms"),
        "thinker_transformer_core_wall_ms": thinker_core_wall_ms,
        "thinker_transformer_core_cuda_event_ms": thinker_core_cuda_ms,
        "thinker_other_phase_wall_ms": _sum(thinker_other, "wall_ms"),
        "thinker_transformer_forward_calls": len(thinker),
        "thinker_runner_step_wall_ms": _sum(thinker_runner, "wall_ms"),
    }


def analyze_cell(cell_dir: Path) -> dict[str, Any]:
    manifest_path = cell_dir / "manifest.json"
    request_path = cell_dir / "requests.jsonl"
    trace_path = cell_dir / "stage_timings.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    request_log_exists = request_path.is_file()
    requests = (
        [
            json.loads(line)
            for line in request_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if request_log_exists
        else []
    )
    events = (
        [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if trace_path.is_file()
        else []
    )

    phase_results: dict[str, list[dict[str, Any]]] = {
        "precondition": [],
        "measured": [],
    }
    for request in requests:
        phase = request.get("phase")
        if phase not in phase_results:
            continue
        joined = [
            event
            for event in events
            if event.get("stage") != "trace_module_patched"
            and _within(event, request)
        ]
        phase_results[phase].append(
            {
                "phase": phase,
                "sequence": request.get("sequence"),
                "success": request.get("success"),
                "e2e_ms": request.get("e2e_ms"),
                "ttft_ms": request.get("ttft_ms"),
                "output_tokens": request.get("output_tokens"),
                "fixture_id": request.get("fixture_id"),
                "audio_relpath": request.get("audio_relpath"),
                "temperature_c_before": request.get("temperature_c_before"),
                "sm_clock_mhz_before": request.get("sm_clock_mhz_before"),
                "thermally_clean": request.get("thermally_clean"),
                "cooldown_timed_out": request.get("cooldown_timed_out"),
                "matched_stage_events": len(joined),
                "stage_spans": _stage_rollup(joined),
            }
        )

    initialization_events = [
        event for event in events if event.get("stage") == "trace_module_patched"
    ]
    measured_stage_names = {
        stage
        for row in phase_results["measured"]
        for stage in row["stage_spans"]["stage_counts"]
    }
    required_stage_names = {
        "request_preprocessor_total",
        "audio_encoder",
        "thinker_transformer_core",
    }
    missing_stage_names = sorted(required_stage_names - measured_stage_names)
    matched_stage_event_count = sum(
        row["matched_stage_events"] for row in phase_results["measured"]
    )
    failure = manifest.get("failure")
    return {
        "cell": manifest.get("cell"),
        "status": manifest.get("status"),
        "failure_summary": (
            str(failure).splitlines()[0]
            if isinstance(failure, str) and failure.strip()
            else None
        ),
        "framework": manifest.get("framework"),
        "case": manifest.get("case"),
        "probe_kind": manifest.get("probe_kind"),
        "repeat": manifest.get("repeat"),
        "config": manifest.get("config"),
        "request_log_status": "present" if request_log_exists else "missing",
        "stage_trace_file": (
            paths.relative_to_repo(trace_path) if trace_path.is_file() else None
        ),
        "trace_status": (
            "missing_request_file"
            if not request_log_exists
            else "missing_file"
            if not trace_path.is_file()
            else "no_events"
            if not events
            else "no_stage_events_for_measured_requests"
            if matched_stage_event_count == 0
            else "partial"
            if missing_stage_names
            else "captured"
        ),
        "missing_required_stages": missing_stage_names,
        "hook_modules_seen": sorted(
            {
                str(event.get("module"))
                for event in initialization_events
                if event.get("module")
            }
        ),
        "precondition_requests": phase_results["precondition"],
        "measured_requests": phase_results["measured"],
    }


def analyze_campaign(campaign: str) -> dict[str, Any]:
    campaign_root = paths.campaign_root(campaign)
    cells = []
    for manifest_path in campaign_root.glob("QWEN_A2T/*/*/*/manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("probe_kind") == "stage-timing":
            cells.append(manifest_path.parent)
    cells.sort()
    results = [analyze_cell(cell) for cell in cells]
    return {
        "schema_version": 1,
        "campaign": campaign,
        "profile_only": True,
        "interpretation": (
            "CUDA-event and wall spans are diagnostic measurements from eager, "
            "uncompiled stage-timing runs. request_preprocessor_inclusive_wall_ms "
            "may contain the nested feature-extractor span; do not add both. "
            "Thinker transformer spans exclude the framework's outer step/logit "
            "handling; thinker_runner_step_wall_ms is reported separately when "
            "instrumented. Do not treat E2E minus component spans as exact "
            "exclusive Thinker time."
        ),
        "cell_count": len(results),
        "captured_cells": sum(row["trace_status"] == "captured" for row in results),
        "partial_cells": sum(row["trace_status"] == "partial" for row in results),
        "failed_cells": sum(row["status"] == "failed" for row in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        default=paths.campaign_id(),
        help="Campaign under exps/runs/ (default: current campaign ID).",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    report = analyze_campaign(args.campaign)
    output = args.output or (
        paths.analysis_dir(args.campaign) / "qwen_a2t_stage_timing.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary = {
        key: report[key]
        for key in (
            "campaign",
            "cell_count",
            "captured_cells",
            "partial_cells",
            "failed_cells",
        )
    }
    summary["trace_statuses"] = {
        status: sum(row["trace_status"] == status for row in report["results"])
        for status in sorted({row["trace_status"] for row in report["results"]})
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nwritten to {paths.relative_to_repo(output)}")
    return 0 if report["cell_count"] and report["captured_cells"] == report["cell_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
