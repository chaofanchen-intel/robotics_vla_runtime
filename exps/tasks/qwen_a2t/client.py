#!/usr/bin/env python3
"""Measured client for Qwen3-Omni audio-to-text on FLEURS."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exps.core import adapters, benchmark_client, paths
from exps.core.codecs import ContractViolation
from exps.tasks.qwen_a2t.audio_io import read_audio_wav, write_float_wav
from exps.tasks.qwen_common import client_model_instance_kwargs
from exps.tasks.qwen_a2t.cases import MAX_OUTPUT_TOKENS

MANIFEST = paths.INPUTS / "QW-A2T" / "fleurs_test_100_manifest.json"
AUDIO_ROOT = paths.INPUTS / "QW-A2T" / "audio"

TRANSCRIBE_INSTRUCTION = {
    "en": "Transcribe the audio exactly as spoken. Output only the transcription.",
    "zh": "请逐字转写这段音频，只输出转写文本。",
}


def load_samples(fleurs_config: str) -> list[dict[str, Any]]:
    if not MANIFEST.is_file():
        raise SystemExit(f"FLEURS manifest missing: {MANIFEST}")
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    try:
        return data["dataset"]["configs"][fleurs_config]["samples"]
    except KeyError:
        available = ", ".join(data["dataset"]["configs"])
        raise SystemExit(f"no FLEURS config {fleurs_config!r}; available: {available}") from None


def verify_audio(sample: dict[str, Any]) -> Path:
    """Check the clip against the hash recorded when the fixture was frozen.

    A silently re-encoded or truncated clip would change the workload without
    changing anything visible in a latency number.
    """
    path = AUDIO_ROOT / sample["audio_relpath"]
    if not path.is_file():
        raise SystemExit(f"FLEURS audio missing: {path}")
    expected = sample.get("audio_sha256")
    if expected:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise SystemExit(
                f"{path} does not match the frozen fixture hash "
                f"(recorded {expected}, found {digest})"
            )
    return path


def sampling_kwargs(max_output_tokens: int) -> dict[str, Any]:
    if os.environ.get("EXPS_QWEN_SAMPLING") == "lookahead-eligible":
        # Counterfactual (--opt qwen.sampling=lookahead-eligible): no seed, so SGLang-Omni's lookahead decode stays eligible.
        return {
            "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "max_tokens": max_output_tokens, "max_output_tokens": max_output_tokens,
        }
    return {
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": max_output_tokens,
        "max_output_tokens": max_output_tokens,
    }


def build_requests(
    fleurs_config: str, language: str, count: int, max_output_tokens: int, cycle: int | None = None
) -> tuple[list[Any], list[dict[str, Any]]]:
    request_module = adapters.load_benchmark_module("benchmark.request")
    samples = load_samples(fleurs_config)
    # cycle: row i % cycle, so a precondition phase of `cycle` requests has served every clip the measured phase sends
    modulus = min(cycle, len(samples)) if cycle else len(samples)
    selected = [samples[index % modulus] for index in range(count)]
    request_type = adapters.request_type("audio_to_text")
    inputs = [
        request_module.RequestInput(
            req_type=request_type,
            prompt=TRANSCRIBE_INSTRUCTION[language],
            audio_path=str(verify_audio(sample)),
            model_kwargs=sampling_kwargs(max_output_tokens),
        )
        for sample in selected
    ]
    return inputs, selected


def select_probe_samples(
    samples: Sequence[dict[str, Any]],
    *,
    precondition_count: int,
    measured_count: int,
    precondition_audio_relpaths: Sequence[str],
    measured_audio_relpaths: Sequence[str],
) -> list[dict[str, Any]]:
    """Select exact fixture paths for the unscored and measured phases."""
    if precondition_count < 0:
        raise ValueError("precondition_count must be >= 0")
    if measured_count < 1:
        raise ValueError("measured_count must be >= 1")

    by_relpath: dict[str, dict[str, Any]] = {}
    for sample in samples:
        relpath = sample.get("audio_relpath")
        if not isinstance(relpath, str):
            raise ValueError("every FLEURS fixture row must have an audio_relpath")
        if relpath in by_relpath:
            raise ValueError(f"duplicate audio_relpath in FLEURS manifest: {relpath}")
        by_relpath[relpath] = sample

    def repeat_paths(
        relpaths: Sequence[str],
        count: int,
        phase: str,
    ) -> list[dict[str, Any]]:
        if count == 0:
            return []
        if not relpaths:
            raise ValueError(f"{phase} audio paths are required when count is {count}")
        missing = [relpath for relpath in relpaths if relpath not in by_relpath]
        if missing:
            raise ValueError(
                f"{phase} audio path(s) are not in the selected FLEURS config: "
                + ", ".join(missing)
            )
        return [by_relpath[relpaths[index % len(relpaths)]] for index in range(count)]

    return [
        *repeat_paths(precondition_audio_relpaths, precondition_count, "precondition"),
        *repeat_paths(measured_audio_relpaths, measured_count, "measured"),
    ]


def materialize_shape_matched_precondition(
    sample: dict[str, Any],
    *,
    target_num_samples: int,
    target_sample_rate: int,
    output_dir: Path,
    ordinal: int,
) -> tuple[dict[str, Any], Path]:
    """Trim/pad a different fixture utterance to the measured raw-audio shape."""
    if target_num_samples < 1 or target_sample_rate < 1:
        raise ValueError("target sample count and sample rate must be positive")
    source_path = verify_audio(sample)
    waveform, sample_rate = read_audio_wav(source_path)
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"precondition sample rate {sample_rate} does not match measured "
            f"sample rate {target_sample_rate}"
        )
    recorded_num_samples = sample.get("num_samples")
    if recorded_num_samples is not None and recorded_num_samples != len(waveform):
        raise ValueError(
            f"{sample['audio_relpath']} records {recorded_num_samples} samples, "
            f"but contains {len(waveform)}"
        )

    if len(waveform) > target_num_samples:
        adjusted = waveform[:target_num_samples]
        transform = "trim_right"
    elif len(waveform) < target_num_samples:
        import numpy as np

        adjusted = np.pad(
            waveform,
            (0, target_num_samples - len(waveform)),
            mode="constant",
        )
        transform = "zero_pad_right"
    else:
        adjusted = waveform
        transform = "none"

    generated_path = (
        output_dir / "precondition_audio" / f"shape_matched_{ordinal:03d}.wav"
    )
    write_float_wav(generated_path, adjusted, sample_rate)
    generated_sha256 = hashlib.sha256(generated_path.read_bytes()).hexdigest()
    return (
        {
            **sample,
            "duration_seconds": target_num_samples / sample_rate,
            "num_samples": target_num_samples,
            "probe_source_audio_relpath": sample["audio_relpath"],
            "probe_source_audio_sha256": sample.get("audio_sha256"),
            "probe_precondition_audio_sha256": generated_sha256,
            "probe_audio_transform": transform,
            "probe_audio_relpath": str(generated_path.relative_to(output_dir)),
        },
        generated_path,
    )


def build_probe_requests(
    fleurs_config: str,
    language: str,
    precondition_count: int,
    measured_count: int,
    precondition_audio_relpaths: Sequence[str],
    measured_audio_relpaths: Sequence[str],
    max_output_tokens: int,
    *,
    precondition_match_measured_shape: bool = False,
    output_dir: Path | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    request_module = adapters.load_benchmark_module("benchmark.request")
    samples = select_probe_samples(
        load_samples(fleurs_config),
        precondition_count=precondition_count,
        measured_count=measured_count,
        precondition_audio_relpaths=precondition_audio_relpaths,
        measured_audio_relpaths=measured_audio_relpaths,
    )
    request_audio_paths: list[Path] = []
    if precondition_match_measured_shape:
        if precondition_count < 1:
            raise ValueError("shape-matched preconditioning requires at least one precondition")
        if output_dir is None:
            raise ValueError("output_dir is required to materialize matched-shape audio")
        target_sample = samples[precondition_count]
        target_num_samples = target_sample.get("num_samples")
        if not isinstance(target_num_samples, int) or target_num_samples < 1:
            raise ValueError(
                "the measured fixture row must provide a positive num_samples value "
                "for shape-matched preconditioning"
            )

        target_path = verify_audio(target_sample)
        target_audio, target_rate = read_audio_wav(target_path)
        if len(target_audio) != target_num_samples:
            raise ValueError(
                "measured fixture num_samples does not match its WAV payload: "
                f"{target_sample['audio_relpath']} records {target_num_samples}, "
                f"contains {len(target_audio)}"
            )

        for index, sample in enumerate(samples):
            source_path = verify_audio(sample)
            if index >= precondition_count:
                request_audio_paths.append(source_path)
                continue
            adjusted_sample, generated_path = materialize_shape_matched_precondition(
                sample,
                target_num_samples=target_num_samples,
                target_sample_rate=target_rate,
                output_dir=output_dir,
                ordinal=index + 1,
            )
            samples[index] = adjusted_sample
            request_audio_paths.append(generated_path)
    else:
        request_audio_paths = [verify_audio(sample) for sample in samples]

    request_type = adapters.request_type("audio_to_text")
    inputs = [
        request_module.RequestInput(
            req_type=request_type,
            prompt=TRANSCRIBE_INSTRUCTION[language],
            audio_path=str(audio_path),
            model_kwargs=sampling_kwargs(max_output_tokens),
        )
        for sample, audio_path in zip(samples, request_audio_paths, strict=True)
    ]
    return inputs, samples


def make_validator(samples: list[dict[str, Any]]):
    def validate(result, row: dict[str, Any], index: int) -> dict[str, Any]:
        sample = samples[index % len(samples)]
        text = adapters.response_text(result)
        if not text.strip():
            raise ContractViolation(
                "the response carried no transcription text; a transcription latency cannot be "
                "read from an empty result"
            )
        duration = sample.get("duration_seconds")
        return {
            "fixture_id": sample.get("id"),
            "audio_relpath": sample.get("audio_relpath"),
            "audio_duration_s_input": duration,
            "audio_sha256": sample.get("audio_sha256"),
            "probe_source_audio_relpath": sample.get("probe_source_audio_relpath"),
            "probe_source_audio_sha256": sample.get("probe_source_audio_sha256"),
            "probe_precondition_audio_sha256": sample.get(
                "probe_precondition_audio_sha256"
            ),
            "probe_audio_transform": sample.get("probe_audio_transform"),
            "probe_audio_relpath": sample.get("probe_audio_relpath"),
            "transcription": text,
            "transcription_characters": len(text),
            # Latency per second of input audio: the natural work-normalised
            # figure for transcription, and the one that stays meaningful when
            # clip lengths differ.
            "latency_per_audio_second": (
                row["e2e_ms"] / (duration * 1000.0) if row.get("e2e_ms") and duration else None
            ),
            "reference_transcription": sample.get("transcription") or sample.get("raw_transcription"),
        }

    return validate


def main() -> int:
    parser = benchmark_client.build_parser(__doc__)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--fleurs-config", required=True)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--precondition-audio-relpaths", nargs="*", default=None)
    parser.add_argument("--measured-audio-relpaths", nargs="+", default=None)
    parser.add_argument("--precondition-match-measured-shape", action="store_true")
    args = parser.parse_args()

    precondition_count = args.preconditions if args.preconditions is not None else 1
    measured_count = args.measured if args.measured is not None else 5
    if args.measured_audio_relpaths is not None:
        request_inputs, samples = build_probe_requests(
            args.fleurs_config,
            args.language,
            precondition_count,
            measured_count,
            args.precondition_audio_relpaths or (),
            args.measured_audio_relpaths,
            args.max_output_tokens,
            precondition_match_measured_shape=args.precondition_match_measured_shape,
            output_dir=args.output_dir,
        )
    else:
        if args.precondition_audio_relpaths is not None or args.precondition_match_measured_shape:
            raise SystemExit(
                "explicit precondition audio options require --measured-audio-relpaths"
            )
        count = precondition_count + measured_count
        cycle = measured_count if os.environ.get("EXPS_QWEN_ROWS") == "cycle" else None
        request_inputs, samples = build_requests(
            args.fleurs_config, args.language, count, args.max_output_tokens, cycle
        )
    durations = [s.get("duration_seconds") for s in samples if s.get("duration_seconds")]

    return benchmark_client.run(
        args=args,
        task="qwen_a2t",
        model_type="qwen3omni",
        model_instance_kwargs=client_model_instance_kwargs(args.framework),
        request_inputs=request_inputs,
        validate=make_validator(samples),
        context={
            "language": args.language,
            "fleurs_config": args.fleurs_config,
            "fixture": paths.relative_to_repo(MANIFEST),
            "audio_root": paths.relative_to_repo(AUDIO_ROOT),
            "fixture_rows_used": [sample.get("audio_relpath") for sample in samples],
            "probe_selection": (
                {
                    "precondition_audio_relpaths": args.precondition_audio_relpaths or [],
                    "measured_audio_relpaths": args.measured_audio_relpaths,
                    "precondition_match_measured_shape": (
                        args.precondition_match_measured_shape
                    ),
                }
                if args.measured_audio_relpaths is not None
                else None
            ),
            "input_audio_seconds_total": sum(durations) if durations else None,
            "input_audio_format": (
                "Frozen 16 kHz mono IEEE-float32 WAV fixture; no experiment-side resampling"
            ),
            "sampling": sampling_kwargs(args.max_output_tokens),
            "rows_mode": "cycle" if os.environ.get("EXPS_QWEN_ROWS") == "cycle" else "sequential",
            "contract": (
                "each clip's SHA-256 is verified against the frozen fixture before it is sent, "
                "and a response with no transcription text is a violation"
            ),
            "work_normalised_metric": (
                "latency_per_audio_second = e2e_ms / (input duration * 1000); comparable across "
                "clips of different lengths"
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
