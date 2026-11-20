#!/usr/bin/env python3
"""Shared entry point for every served-runtime task client.

A task client's job reduces to three things: say which requests to send,
say what a valid response looks like, and describe itself for the manifest.
Everything else -- argument parsing, the gated measured loop, metric
translation, artefact writing, summary assembly -- is identical across tasks
and lives here.

Keeping it here is not only about code volume. It means the thermal protocol
cannot drift apart between tasks: there is one implementation of "precondition,
gate, measure, gate, measure", so a BAGEL request and an Orpheus request are
gated by the same code under the same conditions. In the previous round each
client grew its own warmup and timing conventions, and the resulting numbers
were not safely comparable across workloads.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

from . import adapters, cli, paths
from .codecs import ContractViolation
from . import metrics as metrics_mod
from .measure import MeasurementConfig, run_measurement

#: A validator receives (raw RequestMetrics, our metric row, request index) and
#: returns audit fields to merge into the row. It raises ContractViolation when
#: the response is not the work that was requested.
Validator = Callable[[Any, dict[str, Any], int], dict[str, Any]]


def build_parser(description: str) -> argparse.ArgumentParser:
    """The argument surface every task client shares."""
    parser = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--framework", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--request-timeout-s", type=float, default=1800.0)
    parser.add_argument("--save-outputs", action="store_true", default=True,
                        help="Persist generated artefacts after the timer stops, for auditing.")
    parser.add_argument("--no-save-outputs", dest="save_outputs", action="store_false")
    cli.add_client_protocol_arguments(parser)
    return parser


def run(
    *,
    args: argparse.Namespace,
    task: str,
    model_type: str,
    request_inputs: Sequence[Any],
    context: dict[str, Any],
    validate: Validator | None = None,
    model_kwargs: dict[str, Any] | None = None,
    model_instance_kwargs: dict[str, Any] | None = None,
    batch_size: int = 1,
) -> int:
    """Run one cell's measured phase. Returns a process exit code.

    `request_inputs` must hold at least `preconditions + measured` entries; it
    is indexed positionally so both engines in a case send the same inputs in
    the same order. Where a task has fewer fixture rows than requests it may
    repeat them, but the repetition has to be deterministic and identical on
    both sides -- `prepare_request_inputs` helpers do that cycling rather than
    leaving it to chance here.
    """
    config = cli.client_measurement_config(args, args.output_dir)
    if batch_size < 1:
        raise SystemExit(f"{task}: batch_size must be >= 1, got {batch_size}")
    units = config.precondition_requests + config.measured_requests
    needed = units * batch_size
    if len(request_inputs) < needed:
        raise SystemExit(
            f"{task}: need {needed} request inputs "
            f"({units} gated units x batch size {batch_size}) "
            f"but the fixture provided {len(request_inputs)}"
        )

    system = adapters.inference_system(args.framework)
    model = adapters.model_instance(model_type, **(model_instance_kwargs or {}))

    def _unit_offset(index: int, phase: str) -> int:
        """Where this gated unit's inputs start.

        Preconditioning units come first, then measured ones, so no fixture row
        is reused between the two phases within a start.
        """
        unit = (index - 1) if phase == "precondition" else (config.precondition_requests + index - 1)
        return unit * batch_size

    def issue(index: int, phase: str) -> dict[str, Any]:
        offset = _unit_offset(index, phase)
        if batch_size == 1:
            result = adapters.send_sequential(
                system,
                request_inputs[offset],
                base_url=args.base_url,
                request_id=offset,
                model=model,
                model_kwargs=model_kwargs,
                timeout_s=args.request_timeout_s,
            )
            row = adapters.to_metric_row(result)
            if not row["success"]:
                raise RuntimeError(row.get("error") or f"request failed: {row.get('status')}")
            if validate is not None:
                row.update(validate(result, row, offset))
            if args.save_outputs:
                adapters.write_request_outputs(
                    result, args.output_dir / "outputs", f"{phase}_{index:03d}"
                )
            row["fixture_index"] = offset
            row["batch_size"] = 1
            return row

        # A batched unit: `batch_size` requests offered at once, gated as one.
        # The whole wave is the measured object, because that is what a batch
        # point means -- how fast the engine clears B concurrent requests.
        wave = list(request_inputs[offset : offset + batch_size])
        wall_started = time.perf_counter()
        results = adapters.send_batch(
            system,
            wave,
            base_url=args.base_url,
            first_request_id=offset,
            model=model,
            model_kwargs=model_kwargs,
            timeout_s=args.request_timeout_s,
        )
        wall_ms = (time.perf_counter() - wall_started) * 1000.0
        rows = [adapters.to_metric_row(result) for result in results]
        failures = [row.get("error") for row in rows if not row["success"]]
        if failures:
            raise RuntimeError(f"{len(failures)}/{len(rows)} requests in the wave failed: {failures[0]}")
        # Save before validating: a wave voided by one bad response is exactly the case where the audio and
        # text of that response are needed to see what the engine did.
        if args.save_outputs:
            for position, result in enumerate(results):
                adapters.write_request_outputs(
                    result, args.output_dir / "outputs", f"{phase}_{index:03d}_{position:02d}"
                )
        if validate is not None:
            # Each request is validated against *its own* fixture row (offset + position); using
            # the wave's first offset for all of them would check B-1 responses against the wrong
            # target text.
            for position, (result, row) in enumerate(zip(results, rows, strict=True)):
                try:
                    row.update(validate(result, row, offset + position))
                except ContractViolation as exc:
                    raise ContractViolation(
                        f"wave request {position} of {batch_size} (fixture {offset + position}, outputs/"
                        f"{phase}_{index:03d}_{position:02d}): {exc}"
                    ) from exc
        return _fold_wave(rows, wall_ms, batch_size, offset)

    def throughput_batch(count: int, concurrency: int) -> dict[str, Any]:
        """Back-to-back waves of `concurrency`, reported separately from latency."""
        completed = 0
        failures = 0
        latencies: list[float] = []
        for start in range(0, count, concurrency):
            wave = [
                request_inputs[(start + offset) % len(request_inputs)]
                for offset in range(min(concurrency, count - start))
            ]
            results = adapters.send_batch(
                system, wave, base_url=args.base_url, first_request_id=10_000 + start,
                model=model, model_kwargs=model_kwargs, timeout_s=args.request_timeout_s,
            )
            for result in results:
                row = adapters.to_metric_row(result)
                if row["success"]:
                    completed += 1
                    if row["e2e_ms"] is not None:
                        latencies.append(row["e2e_ms"])
                else:
                    failures += 1
        return {"completed": completed, "failed": failures, "e2e_ms": latencies}

    summary = run_measurement(
        config,
        issue=issue,
        context={
            "task": task,
            "framework": args.framework,
            "case": args.case,
            "base_url": args.base_url,
            "model_type": model_type,
            "model_kwargs": model_kwargs or {},
            "request_driver": (
                "M*'s own benchmark.request adapter for this runtime, imported read-only from "
                "external/mstar. Every runtime in this harness is driven through the same "
                "adapter family, so no engine is measured by a client written around it."
            ),
            "timing_boundary": (
                "Anchored immediately before the HTTP send and closed at the end of the response "
                "body. Decoding, validation, hashing and disk writes happen after the timer."
            ),
            **context,
        },
        throughput_batch=throughput_batch if config.throughput_requests else None,
    )

    print(json.dumps(_headline(summary), indent=2))
    return 0 if summary["status"] == "completed" else 1


def _fold_wave(rows: list[dict[str, Any]], wall_ms: float, batch_size: int, offset: int) -> dict[str, Any]:
    """Collapse one concurrent wave into a single schema row.

    Two different latencies matter at a batch point and both are kept:
    `e2e_ms` is the wall time to clear the whole wave -- the number a batch
    point is about -- while `per_request_e2e_ms` preserves the individual
    requests underneath it. RTF is the mean of the per-request real-time
    factors, which is how VoxServe's own harness reports it, so the two
    engines' RTF figures mean the same thing.
    """
    def mean_of(key: str) -> float | None:
        values = [row[key] for row in rows if row.get(key) is not None]
        return statistics.fmean(values) if values else None

    durations = [row["audio_duration_s"] for row in rows if row.get("audio_duration_s") is not None]
    folded = metrics_mod.empty_request_metrics()
    folded.update(
        {
            "e2e_ms": wall_ms,
            "ttfb_ms": mean_of("ttfb_ms"),
            "ttft_ms": mean_of("ttft_ms"),
            "ttfa_ms": mean_of("ttfa_ms"),
            "itl_ms": [value for row in rows for value in (row.get("itl_ms") or [])],
            "output_tokens": sum(row["output_tokens"] or 0 for row in rows) or None,
            "output_bytes": sum(row["output_bytes"] or 0 for row in rows) or None,
            "audio_duration_s": sum(durations) if durations else None,
            "rtf": mean_of("rtf"),
            "success": True,
            "batch_size": batch_size,
            "fixture_index": offset,
            "wave_wall_ms": wall_ms,
            "wave_requests_per_second": (batch_size / (wall_ms / 1000.0)) if wall_ms else None,
            "per_request_e2e_ms": [row["e2e_ms"] for row in rows],
            "per_request_rtf": [row["rtf"] for row in rows],
            "per_request_audio_duration_s": durations,
            "mean_request_e2e_ms": mean_of("e2e_ms"),
            "max_request_e2e_ms": max(
                (row["e2e_ms"] for row in rows if row.get("e2e_ms") is not None), default=None
            ),
        }
    )
    return folded


def _headline(summary: dict[str, Any]) -> dict[str, Any]:
    results = summary.get("results") or {}
    metric_block = results.get("metrics") or {}
    return {
        "status": summary.get("status"),
        "task": summary.get("task"),
        "framework": summary.get("framework"),
        "case": summary.get("case"),
        "measured_requests": results.get("measured_requests"),
        "headline_requests": results.get("headline_requests"),
        "e2e_ms": {key: metric_block.get("e2e_ms", {}).get(key) for key in ("p50", "p95", "mean")},
        "rtf": {key: metric_block.get("rtf", {}).get(key) for key in ("p50", "mean")},
        "intra_start_drift": (results.get("intra_start_drift") or {}).get("status"),
        "contract_violation": summary.get("contract_violation"),
    }


def bootstrap_path() -> None:
    """Allow `python exps/tasks/<x>/client.py` as well as `-m`."""
    if __package__ in (None, ""):  # pragma: no cover - direct invocation only
        sys.path.insert(0, str(paths.REPO_ROOT))
