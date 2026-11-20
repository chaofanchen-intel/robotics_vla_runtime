#!/usr/bin/env python3
"""Qwen3-Omni text-to-text with a fixed ~1K-token prompt and 256-token output.

The frozen prompt asks for a detailed queue-service explanation rather than a
short acknowledgement. All runtimes must return exactly 256 total completion
tokens; SGLang-Omni requires the managed min_tokens/ignore_eos support for
this. Greedy decoding and a fixed prompt keep per-request work constant.

That property is what makes this task the clearest read on whether the thermal
protocol is working. If latency drifts upward across a start here, the drift
cannot be blamed on a longer prompt or an unlucky sentence.

Text-only output
----------------
`modalities: ["text"]` is sent explicitly. Omitting it makes the Omni servers
fall back to text+audio, because the Talker runs unconditionally -- which would
add speech synthesis to what is supposed to be a text decode measurement. M*'s
benchmark adapters already set this; it is restated here because it is a
silent, large workload difference when it is wrong.
"""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

#: Server-side prompt lengths of the length-sweep cases (txt-<n>-1tok). 1,100 / 1,300 / 1,536 / 1,800 sit between M*'s 1,024 chunk budget and the
#: 2,048 rung, where its prefill graph ladder has no exact fit (qwen-prefill-len-dense).
PROMPT_LENGTHS = (512, 1024, 1100, 1300, 1536, 1800, 2048)

#: Fixed output length makes decode work comparable rather than measuring when
#: each runtime naturally decides to stop.
MAX_OUTPUT_TOKENS = 256

#: Prompt for the first-token decomposition cases: 8 body tokens, 56 at the server, so E2E at one output token is the per-request fixed cost
#: (HTTP, template, tokenisation, scheduling, one tiny forward, sampling, first chunk) with almost no prefill.
TINY_PROMPT = "Reply with exactly one word: ready."


TASK = TaskSpec(
    name="qwen_txt",
    title="Qwen3-Omni text-to-text (fixed 1K prompt)",
    frameworks=("mstar", "sglang-omni", "vllm-omni"),
    cases=(
        Case(
            name="txt-1k",
            label="fixed ~1K-token prompt, 256-token greedy completion",
            params={"max_output_tokens": MAX_OUTPUT_TOKENS},
        ),
        Case(
            name="txt-1k-1tok",
            label="fixed ~1K-token prompt, completion capped at 1 token (E2E ~ first-token latency of a 1K prefill)",
            params={"max_output_tokens": 1, "prompt_kind": "1k"},
            profiles=(),
        ),
        # Prompt-length sweep (qwen-prefill-len): the same words cut or cycled so the SERVER sees 512 / 1,024 / 2,048 prompt tokens
        # (48 of them are the system prompt and template; the fixed 1k case is 1,072). Each sits at a different place against M*'s
        # 1,024-token chunk budget and its 128...2048 prefill graph ladder.
        *(
            Case(
                name=f"txt-{n}-1tok",
                label=f"~{n}-token prompt at the server (same words as the 1k prompt), completion capped at 1 token",
                params={"max_output_tokens": 1, "prompt_kind": str(n)},
                profiles=(),
            )
            for n in PROMPT_LENGTHS
        ),
        Case(
            name="txt-tiny-1tok",
            label="56-token (8 + the server's 48 template) prompt, completion capped at 1 token (the per-request fixed cost, no prefill)",
            params={"max_output_tokens": 1, "prompt_kind": "tiny"},
            profiles=(),
        ),
    ),
    runner_module="exps.tasks.qwen_txt.cell",
    comparison_label="work_matched",
    timing_boundary=(
        "Anchored immediately before the HTTP send and closed at the end of the response body. "
        "TTFT is the first text chunk's arrival; ITL is the gap between successive chunks. Note "
        "that a chunk is a framing unit, not necessarily one token: M* and vLLM-Omni emit one "
        "token per chunk while SGLang-Omni bursts several, so ITL is comparable within a runtime "
        "and only indicative across them. Token counts come from the usage field."
    ),
    notes=(
        "Constant per-request work by construction, which makes this the clearest check that the "
        "per-request cooldown gate is actually preventing drift."
    ),
)
