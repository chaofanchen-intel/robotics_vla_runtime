# T2S: B = 1 / 4 / 8 / 16 concurrent requests, three frameworks

**Question.** The paper's headline T2S claim is B=16 throughput; the C=1 cells cannot test it. How does each framework scale?

**Pre-registered reading.** Per-audio-second time against B per framework; the ratio between frameworks at B=16 is the throughput claim. vLLM-Omni runs the max_num_seqs=16 variant of its best recipe (untested at that setting: a failed start is a result).

Campaigns: `T2S_CONCURRENCY_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | main | 2 | 16 | 80.9 [79.6–81.3] | 100% | 72 | 1,785 | - |
| mstar | verbatim-en-b4 | main | 2 | 16 | 36.7 [36.2–37.1] | 100% | 74 | 1,785 | - |
| mstar | verbatim-en-b16 | main | 2 | 16 | 15.2 [15.0–15.6] | 94% | 76 | 1,785 | - |
| sglang-omni | verbatim-en | main | 2 | 16 | 99.7 [93.2–102.2] | 100% | 68 | 1,785 | - |
| sglang-omni | verbatim-en-b4 | main | 2 | 16 | 41.1 [39.9–41.4] | 100% | 70 | 1,785 | - |
| sglang-omni | verbatim-en-b16 | main | 2 | 16 | 17.5 [17.4–18.1] | 100% | 71 | 1,785 | - |
| vllm-omni | verbatim-en | main | 2 | 16 | 158 [155–162] | 100% | 64 | 1,785 | - |
| vllm-omni | verbatim-en-b4 | main | 2 | 16 | 66.1 [65.4–66.1] | 100% | 66 | 1,785 | - |
| vllm-omni | verbatim-en-b16 | main | 2 | 16 | 23.8 [23.5–24.1] | 100% | 70 | 1,785 | - |
