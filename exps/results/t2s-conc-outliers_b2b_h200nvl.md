# Qwen3-Omni T2S at B = 1 / 4 / 16: how often does a response fail the read-aloud contract (record, do not void)?

**Question.** t2s-concurrency-fair voided two B = 16 cells in the precondition phase: M* returned the right text but 20.5 s of audio for a 64-character sentence (wave 5, request 2), vLLM-Omni returned 1.42 s of audio for a 43-character sentence (band floor 1.43 s, so a 0.7% miss: the 30 chars/s ceiling was too tight and is now 36). Is the M* long-audio response a rare random event or tied to concurrency, and is anything similar happening in vLLM-Omni / SGLang-Omni?

**Pre-registered reading.** analyze_t2s_outliers.py: every response's chars per second against the band, per framework and B, with the fixture row of each outlier. Outliers at B = 16 only = concurrency-dependent (state shared between requests in a wave); outliers at B = 1 too = sampling or the Talker's stop condition; the same fixture row failing again = prompt-specific. The contract runs in record mode (`qwen.contract=record`): responses are saved and flagged, never void the cell.

**Caveats.** Durations are the only judge of 'not the target sentence' besides the returned-text similarity (M* returned the right text for the 20.5 s response, so duration is what caught it); nobody listened to the audio. 30 measured waves + 10 precondition waves per cell: 640 responses at B = 16.

Campaigns: `T2S_CONC_OUTLIERS_RECORD_01_B2B` (record, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs record |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | record | 1 | 30 | 83.3 [83.3–83.3] | 100% | 74 | 1,785 | - |
| mstar | verbatim-en-b4 | record | 1 | 30 | 36.1 [36.1–36.1] | 100% | 78 | 1,785 | - |
| mstar | verbatim-en-b16 | record | 1 | 30 | 16.2 [16.2–16.2] | 47% | 83 | 1,675 | - |
| sglang-omni | verbatim-en | record | 1 | 30 | 102 [102–102] | 100% | 71 | 1,785 | - |
| sglang-omni | verbatim-en-b4 | record | 1 | 30 | 42.1 [42.1–42.1] | 100% | 73 | 1,785 | - |
| sglang-omni | verbatim-en-b16 | record | 1 | 30 | 18.1 [18.1–18.1] | 100% | 75 | 1,785 | - |
| vllm-omni | verbatim-en | record | 1 | 30 | 161 [161–161] | 100% | 67 | 1,785 | - |
| vllm-omni | verbatim-en-b4 | record | 1 | 30 | 69.4 [69.4–69.4] | 100% | 71 | 1,785 | - |
| vllm-omni | verbatim-en-b16 | record | 1 | 30 | 25.6 [25.6–25.6] | 100% | 78 | 1,785 | - |
