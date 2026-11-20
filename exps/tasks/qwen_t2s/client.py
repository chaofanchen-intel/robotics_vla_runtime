#!/usr/bin/env python3
"""Measured client for Qwen3-Omni text-to-speech, with the verbatim contract.

Supports a `--canary` mode that issues three ungated requests and checks the
contract without producing a measurement. Running the full sweep on a broken
contract only multiplies the defect, which is what the previous round did.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exps.core import adapters, benchmark_client, paths
from exps.core.codecs import ContractViolation
from exps.core.manifest import write_json
from exps.tasks.qwen_common import client_model_instance_kwargs
from exps.tasks.qwen_t2s import contract
from exps.tasks.qwen_t2s.cases import MAX_NEW_TOKENS, READ_ALOUD_INSTRUCTION, SEED

FIXTURES = {
    "en": paths.INPUTS / "SEED-TTS" / "seedtts_en_local_v1.jsonl",
    "zh": paths.INPUTS / "SEED-TTS" / "seedtts_zh_local_v1.jsonl",
}
CANARY_REQUESTS = 3


def load_rows(language: str) -> list[dict[str, Any]]:
    fixture = FIXTURES[language]
    if not fixture.is_file():
        raise SystemExit(
            f"fixture missing: {fixture}\n"
            "Restore exps/inputs from the frozen copy (the fixtures are not regenerated; see exps/inputs/README.md)"
        )
    return [
        json.loads(line)
        for line in fixture.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def user_text(arm: str, row: dict[str, Any], language: str) -> str:
    target = row["synthesis_text"]
    if arm == "verbatim":
        return READ_ALOUD_INSTRUCTION[language].format(target=target)
    if language == "en":
        return (
            f'Listen to the audio above. The speaker is reading: "{row["prompt_text"]}". '
            f"Now read the following text aloud in the same voice and style, exactly as "
            f"written: {target}"
        )
    return (
        f'听上面的音频，说话人正在朗读："{row["prompt_text"]}"。'
        f"现在请用同样的声音和风格，逐字朗读以下文本：{target}"
    )


def sampling_kwargs() -> dict[str, Any]:
    """Greedy on Thinker, Talker and code predictor, with an explicit seed.

    M*'s adapter defaults the seed to the request id, which would make two
    engines decode from different seeds for the same fixture row. Pinning it
    removes that as a source of difference.
    """
    return {
        "seed": SEED,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        # `max_tokens` alone is ignored by M*; `max_output_tokens` is the key
        # mstar/model/base.py actually reads. Both are sent so the cap applies
        # whichever the server honours.
        "max_tokens": MAX_NEW_TOKENS,
        "max_output_tokens": MAX_NEW_TOKENS,
        "repetition_penalty": 1.0,
        "talker_temperature": 0.0,
        "talker_top_k": 1,
        "talker_top_p": 1.0,
        "talker_repetition_penalty": 1.0,
        "talker_max_new_tokens": MAX_NEW_TOKENS,
        "talker_max_output_tokens": MAX_NEW_TOKENS,
        "code_predictor_temperature": 0.0,
        "code_predictor_top_k": 0,  # M* rejects negatives; 0 is "off" (and temperature 0 is greedy anyway)
        "code_predictor_top_p": 1.0,
    }


def voice_clone_kwargs(row: dict[str, Any], language: str) -> dict[str, Any]:
    """Reference-audio fields for the OpenAI-shaped runtimes.

    Carried per request rather than per cell, because each fixture row has its
    own reference utterance.
    """
    reference = paths.REPO_ROOT / row["prompt_audio_path"]
    return {
        "ref_audio": "data:audio/wav;base64," + base64.b64encode(reference.read_bytes()).decode("ascii"),
        "ref_text": row["prompt_text"],
        "task_type": "Base",
        "language": "English" if language == "en" else "Chinese",
    }


SHUFFLE_SEED = 52


def shuffled_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """`count` rows, one fresh seeded permutation of the fixture per pass (each row once per pass)."""
    import random

    selected: list[dict[str, Any]] = []
    pass_no = 0
    while len(selected) < count:
        order = list(range(len(rows)))
        random.Random(SHUFFLE_SEED + pass_no).shuffle(order)
        selected.extend(rows[i] for i in order)
        pass_no += 1
    return selected[:count]


def probe_rows(rows: list[dict[str, Any]], count: int, batch: int, probe_id: str, fill: str) -> list[dict[str, Any]]:
    """Waves of `batch` rows: the probe sentence once per wave at position `wave % batch`, the other slots per `fill`."""
    import random

    probe = next(r for r in rows if r.get("fixture_id") == probe_id)
    others = [r for r in rows if r is not probe]
    if fill == "short":
        others.sort(key=lambda r: len(r["synthesis_text"]))
    elif fill == "long":
        others.sort(key=lambda r: -len(r["synthesis_text"]))
    elif fill == "random":
        random.Random(SHUFFLE_SEED).shuffle(others)
    selected: list[dict[str, Any]] = []
    for wave in range(-(-count // batch)):
        if fill == "same":
            neighbours = [probe] * (batch - 1)
        else:
            neighbours = [others[(wave * (batch - 1) + k) % len(others)] for k in range(batch - 1)]
        position = wave % batch
        selected.extend(neighbours[:position] + [probe] + neighbours[position:])
    return selected[:count]


def build_requests(
    arm: str, language: str, count: int, cycle: int | None = None, shuffle: bool = False,
    probe: tuple[str, str, int] | None = None,
) -> tuple[list[Any], list[dict[str, Any]]]:
    request_module = adapters.load_benchmark_module("benchmark.request")
    rows = load_rows(language)
    # cycle: row i % cycle, so a precondition phase of `cycle` requests has served every row the measured phase sends
    modulus = min(cycle, len(rows)) if cycle else len(rows)
    if probe:
        selected = probe_rows(rows, count, probe[2], probe[0], probe[1])
    elif shuffle:
        selected = shuffled_rows(rows, count)
    else:
        selected = [rows[index % modulus] for index in range(count)]
    request_type = adapters.request_type("text_to_speech")
    inputs = []
    for row in selected:
        per_request = dict(sampling_kwargs())
        if arm == "voice_clone":
            per_request.update(voice_clone_kwargs(row, language))
        inputs.append(
            request_module.RequestInput(
                req_type=request_type,
                prompt=user_text(arm, row, language),
                model_kwargs=per_request,
            )
        )
    return inputs, selected


def make_validator(arm: str, language: str, rows: list[dict[str, Any]]):
    def validate(result, row: dict[str, Any], index: int) -> dict[str, Any]:
        fixture_row = rows[index % len(rows)]  # rows is the per-request list (len == count), index is the request number
        duration = row.get("audio_duration_s")
        try:
            if arm == "verbatim":
                audit = contract.verify_verbatim(
                    result, fixture_row, language=language, audio_duration_s=duration
                )
            else:
                audit = contract.verify_voice_clone(
                    result, fixture_row, language=language, audio_duration_s=duration
                )
        except ContractViolation as exc:
            if os.environ.get("EXPS_QWEN_CONTRACT") != "record":
                raise
            # diagnostic mode: keep the response, flag it, do not void the cell
            audit = {"contract_honoured": False, "violation": str(exc), "target_characters": len(fixture_row["synthesis_text"])}
        audit["drift_workload_key"] = fixture_row["fixture_id"]
        return audit

    return validate


def run_canary(args, request_inputs, rows, validate) -> int:
    """Three ungated requests, checked against the contract. No measurement."""
    system = adapters.inference_system(args.framework)
    model = adapters.model_instance(
        "qwen3omni", **client_model_instance_kwargs(args.framework)
    )
    checks: list[dict[str, Any]] = []
    violations: list[str] = []
    for index in range(min(CANARY_REQUESTS, len(request_inputs))):
        try:
            result = adapters.send_sequential(
                system, request_inputs[index], base_url=args.base_url,
                request_id=index, model=model, timeout_s=args.request_timeout_s,
            )
            row = adapters.to_metric_row(result)
            if not row["success"]:
                raise ContractViolation(row.get("error") or "request failed")
            checks.append({"index": index, **validate(result, row, index)})
        except ContractViolation as exc:
            violations.append(str(exc))
            checks.append({"index": index, "contract_honoured": False, "violation": str(exc)})

    report = {
        "canary": True,
        "framework": args.framework,
        "case": args.case,
        "requests": len(checks),
        "honoured": sum(1 for check in checks if check.get("contract_honoured")),
        "violations": violations,
        "status": "passed" if not violations and checks else "failed",
        "note": (
            "A canary proves the request contract before any measurement. Round 2 ran a full "
            "sweep on a contract that made the Thinker answer the sentence instead of reading it."
        ),
        "checks": checks,
    }
    write_json(Path(args.output_dir) / "canary.json", report)
    print(json.dumps({k: report[k] for k in ("status", "requests", "honoured", "violations")}, indent=2))
    return 0 if report["status"] == "passed" else 1


def main() -> int:
    parser = benchmark_client.build_parser(__doc__)
    parser.add_argument("--arm", choices=("verbatim", "voice_clone"), required=True)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Concurrent requests per gated unit; the reported e2e is the wave wall time.")
    parser.add_argument("--canary", action="store_true",
                        help="Check the contract on three ungated requests and stop.")
    args = parser.parse_args()

    if args.arm == "voice_clone" and args.framework == "mstar":
        raise SystemExit(
            "M* exposes no reference-audio conditioning for Qwen3-Omni. This cell is recorded as "
            "not-applicable in the campaign index rather than measured."
        )

    count = (
        CANARY_REQUESTS
        if args.canary
        else ((args.preconditions or 1) + (args.measured or 5)) * args.batch_size
    )
    cycle = (args.measured or 5) if os.environ.get("EXPS_QWEN_ROWS") == "cycle" and args.batch_size == 1 else None
    shuffle = os.environ.get("EXPS_QWEN_ROWS") == "shuffle"
    probe = None
    if os.environ.get("EXPS_QWEN_ROWS") == "probe":
        probe = (os.environ["EXPS_QWEN_PROBE_ROW"], os.environ.get("EXPS_QWEN_PROBE_FILL", "random"), max(args.batch_size, 1))
    request_inputs, rows = build_requests(args.arm, args.language, count, cycle, shuffle, probe)
    validate = make_validator(args.arm, args.language, rows)

    if args.canary:
        return run_canary(args, request_inputs, rows, validate)

    return benchmark_client.run(
        args=args,
        task="qwen_t2s",
        model_type="qwen3omni",
        model_instance_kwargs=client_model_instance_kwargs(args.framework),
        request_inputs=request_inputs,
        validate=validate,
        batch_size=args.batch_size,
        context={
            "batch_size": args.batch_size,
            "arm": args.arm,
            "language": args.language,
            "fixture": paths.relative_to_repo(FIXTURES[args.language]),
            "fixture_rows_used": [row["fixture_id"] for row in rows],
            "rows_mode": (f"probe {probe[0]}, neighbours {probe[1]}" if probe else "shuffle (seeded permutation per pass)") if (shuffle or probe) else (f"cycle over {cycle} rows" if cycle else "sequential"),
            "sampling": sampling_kwargs(),
            "contract": (
                "verbatim: returned text must resemble the target and audio duration must fall "
                "inside a speech-rate band for its character count"
                if args.arm == "verbatim"
                else "voice_clone: audio present and plausible; descriptive only, never ranked"
            ),
            "comparison_label": "work_matched" if args.arm == "verbatim" else "descriptive",
            "topology": (
                "single H200 with Thinker, Talker and Code2Wav colocated. The paper uses 2xH200 "
                "with Thinker and Talker on separate devices; that is not reproduced here."
            ),
            "paper_scale_fraction": 0.1,
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
