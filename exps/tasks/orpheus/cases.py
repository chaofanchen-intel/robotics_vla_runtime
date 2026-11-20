#!/usr/bin/env python3
"""Orpheus text-to-speech: M* against VoxServe, with RTF on both sides.

What was wrong before
---------------------
The previous round drove M* through its own in-tree `benchmark.runner` and
VoxServe through a separately written client. The two disagreed on the request
protocol, on which sentences were spoken and in what order, and on which
metrics came out -- the M* side reported **no real-time factor at all**, which
is the headline metric for a speech system and the one the paper reports.

What this round does
--------------------
Both runtimes go through the same adapter family in M*'s `benchmark.request`
(`OurSystem` and `VoxServe`), reading the same frozen Seed-TTS EN text list in
the same order, and both produce RTF, TTFA, inter-chunk latency, e2e and audio
duration from the same code.

Sampling parity
---------------
M* fixes temperature 0.6, top_p 0.8, repetition_penalty 1.3 and
max_new_tokens 4096 in `OrpheusModelConfig`. VoxServe's defaults differ -- its
temperature is unset, which its sampler reads as 1.0 -- so the serve command
passes those four values explicitly. Without that, the two engines would decode
different distributions, producing different token counts, different audio
lengths and therefore incomparable latencies.

Batch points
------------
A case is a batch size: B concurrent requests offered as one gated unit. The
wave is the measured object, because that is what a batch point asks about --
how quickly the engine clears B requests at once. Per-request latencies and
per-request RTFs are retained underneath the wave figure.

Protocol asymmetry, recorded rather than hidden
-----------------------------------------------
M* returns NDJSON with base64-framed PCM chunks; VoxServe returns a WAV body.
Base64 costs about 33% more bytes plus per-chunk JSON parsing on the client.
That is a real difference between the systems, not a measurement artefact, but
it is a transport difference rather than a synthesis one, so every summary
records both protocols explicitly.
"""
from __future__ import annotations

from exps.core.registry import Case, TaskSpec

BATCH_SIZES = (1, 2, 4, 8, 16)

#: M*'s OrpheusModelConfig values, passed to VoxServe so both decode alike.
SAMPLING = {
    "temperature": 0.6,
    "top_p": 0.8,
    "repetition_penalty": 1.3,
    "max_tokens": 4096,
}

#: Both default to the `tara` preset, and neither clones a reference voice.
VOICE = "tara"

#: Orpheus and the SNAC decoder emit 24 kHz mono PCM16.
SAMPLE_RATE_HZ = 24_000


TASK = TaskSpec(
    name="orpheus",
    title="Orpheus TTS on Seed-TTS EN",
    frameworks=("mstar", "voxserve"),
    cases=tuple(
        Case(
            name=f"b{batch}",
            label=f"{batch} concurrent request{'s' if batch > 1 else ''} per gated unit",
            params={"batch_size": batch},
            # Keep the entire concurrency curve in both standard profiles.
            profiles=("nightly", "full"),
        )
        for batch in BATCH_SIZES
    ),
    runner_module="exps.tasks.orpheus.cell",
    comparison_label="work_matched",
    timing_boundary=(
        "Anchored immediately before the HTTP send and closed at the end of the response body, by "
        "the same adapter family for both runtimes. WAV parsing, duration computation, hashing and "
        "disk writes happen after the timer. For B>1 the reported e2e is the wall time to clear "
        "the whole wave; per-request latencies are kept alongside it."
    ),
    notes=(
        "Both sides report RTF. Sampling parameters are forced to M*'s OrpheusModelConfig values "
        "on the VoxServe side so the two engines decode the same distribution."
    ),
)
