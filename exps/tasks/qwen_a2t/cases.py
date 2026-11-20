#!/usr/bin/env python3
"""Qwen3-Omni audio-to-text transcription on FLEURS, English and Mandarin.

Two languages because the audio encoder and the Thinker behave differently on
them: Mandarin produces more tokens per second of speech than English, so a
single-language result would not say whether a ranking holds generally.

Equal work here means equal audio
---------------------------------
Transcription latency scales with input duration, so the two engines must
receive the same clips in the same order. The fixture pins both: a frozen
100-row FLEURS sample per language with a recorded SHA-256 per file, consumed
positionally.

The frozen clips are mono 16 kHz WAVs (IEEE float32 in this fixture) and are
not resampled by the experiment client. Each runtime's request adapter handles
the same source clip, so the fixture itself does not introduce a resampling
difference.
"""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

#: FLEURS config names, which are also the audio subdirectory names.
LANGUAGES = {"en": "en_us", "zh": "cmn_hans_cn"}

#: Transcription of a single utterance needs far fewer tokens than a chat
#: completion. Capped so a runaway decode cannot masquerade as a slow engine.
MAX_OUTPUT_TOKENS = 448

EN_SHORT_AUDIO = "en_us/00003_1728.wav"
EN_LONG_AUDIO = "en_us/00505_1901.wav"
ZH_SHORT_AUDIO = "cmn_hans_cn/00057_1728.wav"
ZH_LONG_AUDIO = "cmn_hans_cn/00584_1690.wav"


def _long_audio_probe_cases(
    language: str,
    fleurs_config: str,
    language_tag: str,
    short_audio: str,
    long_audio: str,
    shape_matched_other_audio: str,
) -> tuple[Case, ...]:
    base = {"language": language, "fleurs_config": fleurs_config}
    return (
        Case(
            name=f"a2t-{language_tag}-long-no-precondition",
            label=f"FLEURS {language_tag} long audio, no client precondition",
            params={
                **base,
                "probe_kind": "long-audio-precondition",
                "precondition_audio_relpaths": (),
                "measured_audio_relpaths": (long_audio,),
            },
            profiles=(),
        ),
        Case(
            name=f"a2t-{language_tag}-long-short-precondition",
            label=f"FLEURS {language_tag} short precondition then long audio",
            params={
                **base,
                "probe_kind": "long-audio-precondition",
                "precondition_audio_relpaths": (short_audio,),
                "measured_audio_relpaths": (long_audio,),
            },
            profiles=(),
        ),
        Case(
            name=f"a2t-{language_tag}-long-matched-precondition",
            label=f"FLEURS {language_tag} matched long-audio precondition",
            params={
                **base,
                "probe_kind": "long-audio-precondition",
                "precondition_audio_relpaths": (long_audio,),
                "measured_audio_relpaths": (long_audio,),
            },
            profiles=(),
        ),
        Case(
            name=f"a2t-{language_tag}-long-shape-matched-other-audio-precondition",
            label=f"FLEURS {language_tag} different audio, feature-shape-matched precondition",
            params={
                **base,
                "probe_kind": "long-audio-precondition",
                "precondition_audio_relpaths": (shape_matched_other_audio,),
                "measured_audio_relpaths": (long_audio,),
                "precondition_match_measured_shape": True,
            },
            profiles=(),
        ),
    )


PROBE_CASES = (
    *_long_audio_probe_cases(
        "en",
        "en_us",
        "en",
        EN_SHORT_AUDIO,
        EN_LONG_AUDIO,
        "en_us/00352_2008.wav",
    ),
    *_long_audio_probe_cases(
        "zh",
        "cmn_hans_cn",
        "zh",
        ZH_SHORT_AUDIO,
        ZH_LONG_AUDIO,
        "cmn_hans_cn/00111_1789.wav",
    ),
)


STAGE_TIMING_CASES = (
    Case(
        name="a2t-en-stage-timing",
        label="FLEURS en_us stage timing (audio encoder vs Thinker)",
        params={
            "language": "en",
            "fleurs_config": "en_us",
            "probe_kind": "stage-timing",
            "precondition_audio_relpaths": ("en_us/00352_2008.wav",),
            "precondition_match_measured_shape": True,
            "measured_audio_relpaths": (EN_LONG_AUDIO,),
        },
        profiles=(),
    ),
    Case(
        name="a2t-zh-stage-timing",
        label="FLEURS cmn_hans_cn stage timing (audio encoder vs Thinker)",
        params={
            "language": "zh",
            "fleurs_config": "cmn_hans_cn",
            "probe_kind": "stage-timing",
            "precondition_audio_relpaths": ("cmn_hans_cn/00111_1789.wav",),
            "precondition_match_measured_shape": True,
            "measured_audio_relpaths": (ZH_LONG_AUDIO,),
        },
        profiles=(),
    ),
)


#: What-if control (M* only): the same transcription cohort with M*'s fused-MoE tile config replaced by
#: the tuned H200 config vLLM ships (hook: tools/mstar_moe_tuned_hook). Tests whether M*'s ~2x Thinker MoE
#: kernel time in the nsys profile (fixed default tiles vs per-shape tuned JSON in vLLM / SGLang) explains
#: its steady-state A2T gap. Not a configuration M* ships.
MOE_TUNED_NA = {
    "sglang-omni": "the what-if swaps M*'s own MoE tile config; the other engines already use tuned configs",
    "vllm-omni": "the what-if swaps M*'s own MoE tile config; the other engines already use tuned configs",
}
MOE_TUNED_CASES = tuple(
    Case(
        name=f"a2t-{tag}-moe-tuned",
        label=f"FLEURS {cfg} transcription, M* with tuned MoE tile config (what-if)",
        params={"language": tag, "fleurs_config": cfg, "moe_tuned": True},
        profiles=(),
        not_applicable=dict(MOE_TUNED_NA),
    )
    for tag, cfg in (("en", "en_us"), ("zh", "cmn_hans_cn"))
)

#: Trace control (M* only): the ordinary cohort with the hook logging the token count of every MoE call,
#: to show how many Thinker passes a request makes and with how many tokens each (the nsys kernel grids
#: suggest M* runs 3 prefill-sized passes where vLLM / SGLang run 1).
MOE_TRACE_CASES = (
    Case(
        name="a2t-en-moe-trace",
        label="FLEURS en_us transcription, M* with MoE call tracing (diagnostic)",
        params={"language": "en", "fleurs_config": "en_us", "moe_trace": True},
        profiles=(),
        not_applicable=dict(MOE_TUNED_NA),
    ),
)


TASK = TaskSpec(
    name="qwen_a2t",
    title="Qwen3-Omni audio-to-text (FLEURS)",
    frameworks=("mstar", "sglang-omni", "vllm-omni"),
    cases=(
        Case(
            name="a2t-en",
            label="FLEURS en_us transcription",
            params={"language": "en", "fleurs_config": "en_us"},
        ),
        Case(
            name="a2t-zh",
            label="FLEURS cmn_hans_cn transcription",
            params={"language": "zh", "fleurs_config": "cmn_hans_cn"},
        ),
        Case(
            name="a2t-en-1tok",
            label="FLEURS en_us, completion capped at 1 token (E2E ~ first-token latency: audio encoder + prefill + first sample)",
            params={"language": "en", "fleurs_config": "en_us", "max_output_tokens": 1},
            profiles=(),
        ),
        *PROBE_CASES,
        *STAGE_TIMING_CASES,
        *MOE_TUNED_CASES,
        *MOE_TRACE_CASES,
    ),
    runner_module="exps.tasks.qwen_a2t.cell",
    comparison_label="work_matched",
    timing_boundary=(
        "Anchored immediately before the HTTP send and closed at the end of the response body, "
        "including the audio upload. TTFT is the first text chunk. Audio file reads and base64 "
        "encoding are done by RequestInput before the timer starts, so no engine is charged for "
        "the client's file I/O."
    ),
    notes=(
        "Fixture rows are consumed positionally with per-file SHA-256 recorded, so both engines "
        "transcribe the same clips in the same order."
    ),
)
