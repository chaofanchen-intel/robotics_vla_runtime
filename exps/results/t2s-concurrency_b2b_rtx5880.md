# T2S: B = 1 / 4 / 8 / 16 concurrent requests, three frameworks

**Question.** The paper's headline T2S claim is B=16 throughput; the C=1 cells cannot test it. How does each framework scale?

**Pre-registered reading.** Per-audio-second time against B per framework; the ratio between frameworks at B=16 is the throughput claim. vLLM-Omni runs the max_num_seqs=16 variant of its best recipe (untested at that setting: a failed start is a result).

Campaigns: `T2S_CONCURRENCY_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | main | 2 | 16 | 94.6 [94.6–94.7] | 0% | 48 | 2,671 | - |
| mstar | verbatim-en-b4 | main | 2 | 16 | 46.3 [46.1–46.4] | 0% | 47 | 2,685 | - |
| mstar | verbatim-en-b16 | main | 2 | 16 | 24.3 [24.0–24.5] | 0% | 44 | 2,685 | - |
| sglang-omni | verbatim-en | main | 2 | 16 | 111 [111–113] | 0% | 41 | 2,618 | - |
| sglang-omni | verbatim-en-b4 | main | 2 | 16 | 47.7 [47.7–47.8] | 0% | 42 | 2,685 | - |
| sglang-omni | verbatim-en-b16 | main | 2 | 16 | 20.6 [20.4–20.6] | 0% | 43 | 2,692 | - |
| vllm-omni | verbatim-en | main | 2 | 16 | 151 [149–151] | 0% | 37 | 2,670 | - |
| vllm-omni | verbatim-en-b4 | main | 2 | 16 | 65.0 [64.5–66.2] | 0% | 38 | 2,700 | - |
| vllm-omni | verbatim-en-b16 | main | 2 | 16 | 29.7 [29.6–29.8] | 0% | 38 | 2,693 | - |
