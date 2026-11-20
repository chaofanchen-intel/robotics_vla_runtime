# Qwen3-Omni T2S: E2E and time-to-first-audio against audio duration (20 sentences, shapes pre-warmed)

**Question.** T2S: M* is 1.18x ahead of SGLang-Omni and ~1.5x ahead of vLLM-Omni per audio second, and M*'s own launch chain explains only 1.24x of it. Existing data (5 sentences) say: vLLM-Omni has ~1 in 4 requests with a time-to-first-audio stall (~400 ms) and, off the stalls, a per-audio-second cost of 110-124 ms against M*'s 67-73 and SGLang's 89-95; first audio is ~20 ms later. Which part is fixed latency and which the Talker's frame loop, over a wider set of sentence lengths, with the stalls removed by warming?

**Pre-registered reading.** analyze_t2s_fit.py: `a` (fixed ms) and `b` (ms per audio second) per framework from E2E = a + b * audio_s over 20 distinct sentences; `stalls` should be 0 when the rows are pre-warmed; if vLLM still stalls it is not first-seen-shape JIT. vllm-talker-greedy: vLLM's Talker sampled at temperature 0.9 / top_k 50 in the baseline (the request's talker_* fields do not reach it); if b drops to M*'s with a greedy Talker, the gap was the sampler / longer audio, not the engine. sglang is run with the Thinker radix cache off, vLLM and M* have none, so no framework gets a cache hit on the shared instruction.

**Caveats.** Audio length still differs a little across frameworks for the same sentence (the Talkers do not decode identically); the fit uses the realised duration. If the greedy-Talker vLLM server hangs (repetition loop) that is recorded, not retried.

Campaigns: `QWEN_T2S_FIT_MSTAR_02_B2B` (mstar, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | mstar | 2 | 40 | 335 [328–344] | 0% | 46 | 2,674 | - |
