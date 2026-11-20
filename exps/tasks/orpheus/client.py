#!/usr/bin/env python3
"""Measured client for Orpheus TTS, identical for M* and VoxServe.

One code path drives both runtimes: the same frozen Seed-TTS EN text list in
the same order, the same adapter family from M*'s `benchmark.request`, the
same timing boundary, and the same metric set -- including the real-time factor
that the previous round's M* path never produced.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exps.core import adapters, benchmark_client, paths
from exps.core.codecs import ContractViolation
from exps.tasks.orpheus.cases import SAMPLE_RATE_HZ, VOICE

FIXTURE = paths.INPUTS / "ORPHEUS" / "seedtts_en_targets_v1.jsonl"

#: Shortest plausible utterance. A response far below this is a truncated or
#: empty synthesis being passed off as a fast one.
MIN_AUDIO_SECONDS = 0.25


def load_fixture() -> list[dict[str, Any]]:
    if not FIXTURE.is_file():
        raise SystemExit(
            f"Orpheus fixture missing: {FIXTURE}\n"
            "Restore exps/inputs from the frozen copy (the fixtures are not regenerated; see exps/inputs/README.md)"
        )
    return [
        json.loads(line)
        for line in FIXTURE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_requests(count: int) -> tuple[list[Any], list[dict[str, Any]]]:
    """Build `count` T2S requests from the frozen list, in file order.

    If more requests are needed than the fixture holds, rows are cycled -- and
    cycled identically on both sides, because the index arithmetic is here
    rather than in each engine's client.
    """
    request_module = adapters.load_benchmark_module("benchmark.request")
    rows = load_fixture()
    selected = [rows[index % len(rows)] for index in range(count)]
    return (
        [
            request_module.RequestInput(
                req_type=adapters.request_type("text_to_speech"),
                prompt=row["text"],
            )
            for row in selected
        ],
        selected,
    )


def validate(result, row: dict[str, Any], index: int) -> dict[str, Any]:
    """Reject a response that is not a usable spoken utterance.

    A synthesis that returns nothing, or a fraction of a second of audio for a
    full sentence, would otherwise be recorded as an extremely fast request.
    """
    duration = row.get("audio_duration_s")
    if not duration:
        raise ContractViolation(
            "the response carried no audio; a text-to-speech latency cannot be read from it"
        )
    if duration < MIN_AUDIO_SECONDS:
        raise ContractViolation(
            f"synthesised only {duration:.3f} s of audio for a full sentence; this is a "
            "truncated response, not a fast one"
        )
    return {
        "audio_sample_rate_hz": SAMPLE_RATE_HZ,
        "voice": VOICE,
        "audio_bytes": (result.output_bytes or {}).get("audio"),
    }


def main() -> int:
    parser = benchmark_client.build_parser(__doc__)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    units = (args.preconditions or 1) + (args.measured or 5)
    request_inputs, selected = build_requests(units * args.batch_size)

    return benchmark_client.run(
        args=args,
        task="orpheus",
        model_type="orpheus",
        request_inputs=request_inputs,
        validate=validate,
        batch_size=args.batch_size,
        context={
            "batch_size": args.batch_size,
            "voice": VOICE,
            "audio_format": "mono PCM16 at 24 kHz on both runtimes",
            "fixture": paths.relative_to_repo(FIXTURE),
            "fixture_rows_used": [row["fixture_id"] for row in selected],
            "fixture_order": "file order, never shuffled; both runtimes read the same sequence",
            "rtf_definition": (
                "e2e_ms / (audio_duration_s * 1000). Lower is faster. Reported for both runtimes; "
                "the previous round's M* path produced no RTF at all."
            ),
            "protocol_asymmetry": (
                "M* streams NDJSON with base64-framed PCM chunks; VoxServe returns a WAV body. "
                "Base64 costs roughly 33% more bytes plus per-chunk JSON parsing. This is a real "
                "difference between the systems, but a transport one rather than a synthesis one."
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
