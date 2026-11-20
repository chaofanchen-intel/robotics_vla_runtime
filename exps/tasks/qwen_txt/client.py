#!/usr/bin/env python3
"""Measured client for Qwen3-Omni text-to-text on the fixed 1K-token prompt."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from exps.core import adapters, benchmark_client, paths
from exps.core.codecs import ContractViolation
from exps.tasks.qwen_common import (
    MODEL_REVISION,
    SNAPSHOT,
    client_model_instance_kwargs,
)
from exps.tasks.qwen_txt.cases import MAX_OUTPUT_TOKENS, PROMPT_LENGTHS, TINY_PROMPT

PAYLOAD = Path(__file__).with_name("qwen3_omni_text_1k.json")


def load_tiny_prompt() -> tuple[str, int, str]:
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True).tokenizer
    return (
        TINY_PROMPT,
        len(tokenizer.encode(TINY_PROMPT, add_special_tokens=False)),
        hashlib.sha256(TINY_PROMPT.encode("utf-8")).hexdigest(),
    )


#: Tokens the Omni servers add around a user message (system prompt and chat template). Read off SGLang-Omni's server log:
#: the fixed 1,024-token body arrives as `#new-token: 1072`, the tiny 8-token prompt as 56.
SERVER_TEMPLATE_TOKENS = 48


def load_length_prompt(total_tokens: int) -> tuple[str, int, str]:
    """The fixed prompt's text cut (or cycled, past 1,024 tokens) to a body that the server sees as ``total_tokens``.

    Used by the prompt-length sweep: the same words at a different length, so only the prefill length moves. The body is re-encoded to
    check its length; a cut that re-encodes one token short is lengthened until it is exact.
    """
    content, _, _ = load_prompt()
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(SNAPSHOT, local_files_only=True).tokenizer
    ids = tokenizer.encode(content, add_special_tokens=False)
    want = total_tokens - SERVER_TEMPLATE_TOKENS
    for cut in range(want, want + 8):
        text = tokenizer.decode((ids * (cut // len(ids) + 2))[:cut])
        if len(tokenizer.encode(text, add_special_tokens=False)) == want:
            return text, want, hashlib.sha256(text.encode("utf-8")).hexdigest()
    raise SystemExit(f"could not build a {want}-token body for a server-side total of {total_tokens}")


def load_prompt() -> tuple[str, int, str]:
    if not PAYLOAD.is_file():
        raise SystemExit(f"fixture missing: {PAYLOAD}")
    payload = json.loads(PAYLOAD.read_text(encoding="utf-8"))
    if payload.get("tokenizer_revision") != MODEL_REVISION:
        raise SystemExit(
            f"fixture tokenizer revision {payload.get('tokenizer_revision')!r} "
            f"does not match the served model revision {MODEL_REVISION}"
        )
    if payload.get("target_output_tokens") != MAX_OUTPUT_TOKENS:
        raise SystemExit(
            f"fixture target_output_tokens must be {MAX_OUTPUT_TOKENS}"
        )
    content = payload["messages"][0]["content"]
    if not isinstance(content, str):
        raise SystemExit("Qwen TXT fixture message content must be a string")

    expected_input_tokens = payload.get("expected_input_tokens")
    if type(expected_input_tokens) is not int:
        raise SystemExit("Qwen TXT fixture expected_input_tokens must be an integer")
    from transformers import AutoProcessor

    tokenizer = AutoProcessor.from_pretrained(
        SNAPSHOT, local_files_only=True
    ).tokenizer
    input_tokens = len(tokenizer.encode(content, add_special_tokens=False))
    if input_tokens != expected_input_tokens:
        raise SystemExit(
            f"Qwen TXT fixture has {input_tokens} input tokens, expected "
            f"{expected_input_tokens}"
        )

    prompt_sha256 = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if prompt_sha256 != payload.get("prompt_sha256"):
        raise SystemExit("Qwen TXT fixture prompt SHA-256 does not match its recorded hash")
    return content, input_tokens, prompt_sha256


def sampling_kwargs(max_output_tokens: int, framework: str) -> dict[str, Any]:
    # M*'s conductor counts the token sampled at the end of prefill toward
    # `max_output_tokens`, so the cap is the total completion length for every
    # runtime. (Asking M* for 255 returned 255 in SMOKE_01.)
    runtime_max_output_tokens = max_output_tokens
    if os.environ.get("EXPS_QWEN_SAMPLING") == "lookahead-eligible":
        # Counterfactual (--opt qwen.sampling=lookahead-eligible): no seed, no min_tokens, so SGLang-Omni's one-step
        # lookahead decode stays eligible. ignore_eos + the cap still pin the length; the validator asserts it.
        return {
            "temperature": 0.0, "top_p": 1.0, "top_k": 1,
            "max_tokens": runtime_max_output_tokens, "max_output_tokens": runtime_max_output_tokens,
            "ignore_eos": True,
        }
    return {
        "seed": 42,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 1,
        # Both keys: `max_tokens` is the OpenAI name the Omni servers read,
        # `max_output_tokens` is the one M* reads. Sending only the first lets
        # M* decode to natural EOS -- measured at ~361 tokens against the
        # others' 256 in a previous round.
        "max_tokens": runtime_max_output_tokens,
        "max_output_tokens": runtime_max_output_tokens,
        # Decode exactly the cap rather than stopping at EOS, so every request
        # has the requested total completion length.
        "ignore_eos": True,
        "min_tokens": runtime_max_output_tokens,
    }


def build_requests(
    prompt: str, count: int, max_output_tokens: int, framework: str
) -> list[Any]:
    request_module = adapters.load_benchmark_module("benchmark.request")
    params = sampling_kwargs(max_output_tokens, framework)
    return [
        request_module.RequestInput(
            req_type=adapters.request_type("text_to_text"),
            prompt=prompt,
            model_kwargs=params,
        )
        for _ in range(count)
    ]


def make_validator(max_output_tokens: int):
    """Assert the completion length, so latency is per equal decode work.

    The request asks each runtime to pin the length where its API supports it.
    Since a server may ignore unsupported sampling fields, reported usage is
    authoritative; missing or out-of-range lengths invalidate the cell.
    """

    def validate(_result, row: dict[str, Any], _index: int) -> dict[str, Any]:
        produced = row.get("output_tokens")
        if produced is None:
            raise ContractViolation(
                "the server did not report usage.completion_tokens, so the realized completion "
                "length cannot be verified for this work-matched comparison"
            )
        # A 1-token probe measures the first token only (TTFT). M* returns 2 tokens for max_output_tokens=1 (measured: the sampled-at-prefill token
        # plus one decode step; its 256-token cap is exact), so the cap=1 cases accept cap+1 and are read by TTFT, never E2E.
        allowed = {max_output_tokens, max_output_tokens + 1} if max_output_tokens == 1 else {max_output_tokens}
        if produced not in allowed:
            raise ContractViolation(
                f"the completion is {produced} tokens against a requested {max_output_tokens}. "
                "This cell is not a comparable equal-work measurement."
            )
        return {
            "completion_length_verified": True,
            "requested_output_tokens": max_output_tokens,
            "realised_output_tokens": produced,
        }

    return validate


def main() -> int:
    parser = benchmark_client.build_parser(__doc__)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    parser.add_argument("--prompt-kind", choices=("1k", "tiny", *map(str, PROMPT_LENGTHS)), default="1k")
    args = parser.parse_args()

    if args.max_output_tokens not in (MAX_OUTPUT_TOKENS, 1):
        raise SystemExit(
            f"this fixed workload requires --max-output-tokens {MAX_OUTPUT_TOKENS} (or 1 for the first-token cases)"
        )
    if args.prompt_kind == "tiny":
        prompt, prompt_tokens, prompt_sha256 = load_tiny_prompt()
    elif args.prompt_kind in map(str, PROMPT_LENGTHS):
        prompt, prompt_tokens, prompt_sha256 = load_length_prompt(int(args.prompt_kind))
    else:
        prompt, prompt_tokens, prompt_sha256 = load_prompt()
    count = (args.preconditions or 1) + (args.measured or 5)

    return benchmark_client.run(
        args=args,
        task="qwen_txt",
        model_type="qwen3omni",
        model_instance_kwargs=client_model_instance_kwargs(args.framework),
        request_inputs=build_requests(
            prompt, count, args.max_output_tokens, args.framework
        ),
        validate=make_validator(args.max_output_tokens),
        context={
            "fixture": paths.relative_to_repo(PAYLOAD),
            "prompt_characters": len(prompt),
            "input_tokens": prompt_tokens,
            "prompt_sha256": prompt_sha256,
            "tokenizer_revision": MODEL_REVISION,
            "sampling": sampling_kwargs(args.max_output_tokens, args.framework),
            "contract": (
                "every request decodes exactly the requested number of tokens; the realised "
                "completion length is asserted per response"
            ),
            "output_modality": "text only -- the Talker is not exercised",
            "constant_work_note": (
                "One frozen prompt and a pinned completion length make per-request work constant, "
                "so any drift across a start is the machine or the engine, never the input."
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(main())
