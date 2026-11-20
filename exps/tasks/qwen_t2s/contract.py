#!/usr/bin/env python3
"""Per-response contract checks for Qwen3-Omni text-to-speech.

Two independent signals, because either alone can be fooled:

* **Text similarity.** A runtime that answered the prompt instead of reading it
  returns text bearing no resemblance to the target. This is the decisive
  signal when the text channel is populated.
* **Duration band.** Some configurations return audio with an empty text
  channel, where similarity cannot be computed. A spoken sentence has a
  plausible duration for its character count; an 84-second response to a
  12-word target does not, whatever the text channel says.

Both are deliberately permissive. The job is to separate "read the sentence"
from "answered the sentence", not to grade pronunciation.
"""
from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from exps.core import adapters
from exps.core.codecs import ContractViolation
from exps.tasks.qwen_t2s.cases import CHARS_PER_SECOND, TEXT_SIMILARITY_FLOOR


def normalise(text: str) -> str:
    """Fold case, width, punctuation and whitespace before comparing.

    Without this, a correct reading that differs only in smart quotes or
    full-width punctuation scores as a mismatch and voids a valid cell.
    """
    folded = unicodedata.normalize("NFKC", text or "").casefold()
    return re.sub(r"[\s\W_]+", "", folded)


def similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def spoken_text(result) -> str:
    """The text channel of a response, as assembled by the adapter."""
    return adapters.response_text(result)


def verify_verbatim(
    result,
    row: dict[str, Any],
    *,
    language: str,
    audio_duration_s: float | None,
    text_similarity_floor: float = TEXT_SIMILARITY_FLOOR,
) -> dict[str, Any]:
    """Raise unless the response is a reading of the target sentence."""
    target = row["synthesis_text"]
    returned = spoken_text(result)
    ratio = similarity(normalise(returned), normalise(target))

    if not audio_duration_s:
        raise ContractViolation("the response carried no audio, so nothing was read aloud")

    low_rate, high_rate = CHARS_PER_SECOND[language]
    expected_low = len(target) / high_rate
    expected_high = len(target) / low_rate
    in_band = expected_low <= audio_duration_s <= expected_high

    audit = {
        "contract": "verbatim",
        "target_text": target,
        "target_characters": len(target),
        "returned_text": returned,
        "returned_text_bytes": len(returned.encode("utf-8")),
        "text_similarity_to_target": round(ratio, 4),
        "text_similarity_floor": text_similarity_floor,
        "audio_duration_s": audio_duration_s,
        "expected_duration_band_s": [round(expected_low, 2), round(expected_high, 2)],
        "duration_in_band": in_band,
        "contract_honoured": True,
    }

    if returned.strip() and ratio < text_similarity_floor:
        raise ContractViolation(
            f"the runtime returned text {ratio:.0%} similar to the target (floor "
            f"{text_similarity_floor:.0%}): it answered the sentence instead of reading it. "
            f"target={target!r} returned={returned[:300]!r}"
        )
    if not in_band:
        raise ContractViolation(
            f"audio is {audio_duration_s:.1f} s for a {len(target)}-character target, outside the "
            f"{expected_low:.1f}-{expected_high:.1f} s band. Whatever was spoken, it was not the "
            "target sentence, so this is not equal work with the other runtimes."
        )
    return audit


def verify_voice_clone(
    result,
    row: dict[str, Any],
    *,
    language: str,
    audio_duration_s: float | None,
) -> dict[str, Any]:
    """Looser check for the descriptive arm: audio exists and is plausible.

    No similarity floor here. The voice-clone arm has no M* counterpart, so it
    is never ranked as equal work, and holding it to the verbatim contract
    would void cells whose only purpose is description.
    """
    if not audio_duration_s:
        raise ContractViolation("the response carried no audio")
    target = row["synthesis_text"]
    low_rate, high_rate = CHARS_PER_SECOND[language]
    return {
        "contract": "voice_clone",
        "comparison_label": "descriptive",
        "target_text": target,
        "target_characters": len(target),
        "returned_text": spoken_text(result),
        "audio_duration_s": audio_duration_s,
        "expected_duration_band_s": [
            round(len(target) / high_rate, 2),
            round(len(target) / low_rate, 2),
        ],
        "reference_audio": row.get("prompt_audio_path"),
        "contract_honoured": True,
    }
