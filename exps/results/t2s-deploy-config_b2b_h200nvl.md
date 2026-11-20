# T2S: what vLLM-Omni's async-chunk + Code2Wav graph and SGLang's partial start are worth

**Question.** The baselines run their best recipe. How far behind is each in its plain single-GPU recipe?

**Pre-registered reading.** plain / best per framework: the part of the baselines' T2S numbers that is configuration, not engine.

Campaigns: `T2S_DEPLOY_CONFIG_BEST_01_B2B` (best, back-to-back), `T2S_DEPLOY_CONFIG_PLAIN_01_B2B` (plain, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs best |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | verbatim-en | best | 3 | 15 | 108 [105–108] | 100% | 66 | 1,785 | - |
| sglang-omni | verbatim-en | plain | 3 | 15 | 116 [115–117] | 100% | 64 | 1,785 | 1.08x |
| vllm-omni | verbatim-en | best | 3 | 15 | 154 [151–163] | 100% | 62 | 1,785 | - |
| vllm-omni | verbatim-en | plain | 3 | 15 | 225 [203–250] | 100% | 61 | 1,785 | 1.46x |
