#!/usr/bin/env python3
"""Qwen3-Omni text-to-speech on Seed-TTS: M*, SGLang-Omni and vLLM-Omni.

The failure this task is built around
-------------------------------------
Round 2 sent the bare target sentence as the prompt. Qwen3-Omni is an
instruction-following omni model, so the Thinker *answered* the sentence and
the Talker spoke the answer. M* cells returned roughly 1,140 bytes of reply
text for a one-sentence target -- fifteen to twenty times the intended work --
and that was reported as a latency comparison.

Two arms, and why
-----------------
`verbatim`     An explicit read-aloud instruction, plus a per-response contract
               check: the returned text must resemble the target sentence, and
               the audio duration must fall inside a speech-rate band for its
               character count. A response that answers instead of reading
               fails both and voids the cell. This is the only arm that
               supports an equal-work ranking.

`voice_clone`  Reference-audio conditioning, which vLLM-Omni and SGLang-Omni
               support and **M* does not**. Recorded as descriptive only; M* is
               marked not-applicable rather than given a fabricated equivalent.

Contract canary
---------------
Before the measured repeats, three requests per arm/runtime/language are issued
and checked. Running a full sweep on a broken contract only multiplies the
defect by N, which is what round 2 did.

Scale
-----
EN 100 / ZH 200 stratified samples of the Seed-TTS eval set, a tenth of the
paper's size, with a fixed seed and every row id recorded in the fixture
manifest.

Topology
--------
Single H200 with Thinker, Talker and Code2Wav colocated. The paper uses two
H200s with Thinker and Talker on separate devices; that is not reproduced here
and no result should be read as if it were.
"""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

#: Read-aloud instruction. Explicit about all three failure modes seen in the
#: round-2 transcripts: answering the prompt, embellishing it, and adding
#: conversational filler around it.
READ_ALOUD_INSTRUCTION = {
    "en": (
        "Read the following text aloud exactly as written. Do not answer it, do not add or omit "
        "anything, and say nothing else.\n\n{target}"
    ),
    "zh": "请逐字朗读以下文本。不要回答它，不要增加或省略任何内容，也不要说其他任何话。\n\n{target}",
}

#: Plausible speech-rate bands, characters per second. Deliberately wide: the
#: job is to catch an 84-second answer to a 12-word sentence, not to judge
#: prosody. The English ceiling was 30; vLLM-Omni read a 43-character sentence in
#: 1.419 s (30.3 chars/s) and voided a whole cell, while M* reads the same sentence
#: at about 24 chars/s, so it is 36 now.
CHARS_PER_SECOND = {"en": (6.0, 36.0), "zh": (1.5, 10.0)}

#: Below this similarity between returned text and target, the runtime answered
#: rather than read.
TEXT_SIMILARITY_FLOOR = 0.60

#: Greedy everywhere, 256-token Thinker cap -- the paper-appendix profile.
MAX_NEW_TOKENS = 256
SEED = 42

MSTAR_VOICE_CLONE_NOT_APPLICABLE = (
    "M* exposes no reference-audio conditioning for Qwen3-Omni. Round 2's alternative was to "
    "send it the bare sentence, which produced a different workload entirely. A missing cell is "
    "recorded instead."
)


#: Concurrency sweep (paper reports B=1..32 throughput/RTF on 2xH200). Run via explicit `--case`
#: only. Offered concurrency is B per gated unit; the reported e2e is the wave wall time.
CONCURRENCY_BATCH_SIZES = (4, 8, 12, 16)


def _concurrency_cases() -> tuple[Case, ...]:
    return tuple(
        Case(
            name=f"verbatim-en-b{batch}",
            label=f"verbatim read-aloud EN, {batch} concurrent requests per gated unit",
            params={"arm": "verbatim", "language": "en", "batch_size": batch},
            profiles=(),
        )
        for batch in CONCURRENCY_BATCH_SIZES
    )


TASK = TaskSpec(
    name="qwen_t2s",
    title="Qwen3-Omni text-to-speech (Seed-TTS)",
    frameworks=("mstar", "sglang-omni", "vllm-omni"),
    cases=(
        Case(
            name="verbatim-en",
            label="verbatim read-aloud, Seed-TTS EN 100",
            params={"arm": "verbatim", "language": "en"},
        ),
        Case(
            name="verbatim-zh",
            label="verbatim read-aloud, Seed-TTS ZH 200",
            params={"arm": "verbatim", "language": "zh"},
        ),
        Case(
            name="voice-clone-en",
            label="reference-audio conditioning, Seed-TTS EN (descriptive only)",
            params={"arm": "voice_clone", "language": "en"},
            profiles=("full",),
            not_applicable={"mstar": MSTAR_VOICE_CLONE_NOT_APPLICABLE},
        ),
        Case(
            name="voice-clone-zh",
            label="reference-audio conditioning, Seed-TTS ZH (descriptive only)",
            params={"arm": "voice_clone", "language": "zh"},
            profiles=("full",),
            not_applicable={"mstar": MSTAR_VOICE_CLONE_NOT_APPLICABLE},
        ),
        *_concurrency_cases(),
    ),
    runner_module="exps.tasks.qwen_t2s.cell",
    comparison_label="work_matched (verbatim) / descriptive (voice_clone)",
    timing_boundary=(
        "Anchored immediately before the HTTP send and closed at the end of the response body. "
        "Audio decoding, duration computation, contract checking and disk writes happen after."
    ),
    notes=(
        "Verbatim responses are contract-checked per request against the target text and a "
        "speech-rate duration band. Single-GPU colocated topology, unlike the paper's 2xH200."
    ),
)
