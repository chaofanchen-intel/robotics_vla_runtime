# Qwen3-Omni T2S: B = 1 / 4 / 16 concurrent requests with the shapes pre-warmed (no first-seen-shape JIT in the measured waves)

**Question.** t2s-concurrency ran with cold rows: qwen-t2s-fit showed vLLM-Omni pays inference-time JIT on every first-seen shape, which a wave of 16 different sentences hits many times. With the rows cycled during preconditions (qwen.rows=cycle), what is the real scaling per audio second at B = 16?

**Pre-registered reading.** analyze_concurrency.py: per-audio-second time and requests/s against B. Compare with t2s-concurrency (cold rows): a framework whose B = 16 number falls a lot here was measuring JIT before. vLLM-Omni runs the max_num_seqs=16 variant of its recipe (a failed start is a result).

**Caveats.** 10 preconditions per start is the number of waves, not requests: at B = 16 that is 160 requests over the 20-sentence fixture, so every sentence is seen before measurement; at B = 1 only 10 of the 20 are.

Campaigns: `T2S_CONCURRENCY_FAIR_WARM_01_B2B` (warm, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs warm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | warm | 1 | 6 | 75.5 [75.5–75.5] | 100% | 73 | 1,785 | - |
| mstar | verbatim-en-b4 | warm | 1 | 6 | 38.1 [38.1–38.1] | 100% | 75 | 1,785 | - |
| mstar | verbatim-en-b16 | warm | 1 | 6 | 14.9 [14.9–14.9] | 83% | 79 | 1,784 | - |
| sglang-omni | verbatim-en | warm | 1 | 6 | 129 [129–129] | 100% | 69 | 1,785 | - |
| sglang-omni | verbatim-en-b4 | warm | 1 | 6 | 42.7 [42.7–42.7] | 100% | 71 | 1,785 | - |
| sglang-omni | verbatim-en-b16 | warm | 1 | 6 | 17.0 [17.0–17.0] | 100% | 73 | 1,785 | - |
| vllm-omni | verbatim-en | warm | 1 | 6 | 139 [139–139] | 100% | 65 | 1,785 | - |
| vllm-omni | verbatim-en-b4 | warm | 1 | 6 | 74.9 [74.9–74.9] | 100% | 68 | 1,785 | - |
| vllm-omni | verbatim-en-b16 | warm | 1 | 6 | 24.4 [24.4–24.4] | 100% | 72 | 1,785 | - |
