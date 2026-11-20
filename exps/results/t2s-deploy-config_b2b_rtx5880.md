# T2S: what vLLM-Omni's async-chunk + Code2Wav graph and SGLang's partial start are worth

**Question.** The baselines run their best recipe. How far behind is each in its plain single-GPU recipe?

**Pre-registered reading.** plain / best per framework: the part of the baselines' T2S numbers that is configuration, not engine.

Campaigns: `T2S_DEPLOY_CONFIG_BEST_01_B2B` (best, back-to-back), `T2S_DEPLOY_CONFIG_PLAIN_01_B2B` (plain, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs best |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | verbatim-en | best | 3 | 15 | 122 [121–122] | 0% | 39 | 2,643 | - |
| sglang-omni | verbatim-en | plain | 3 | 15 | 127 [125–127] | 0% | 39 | 2,572 | 1.04x |
| vllm-omni | verbatim-en | best | 3 | 15 | 157 [157–170] | 0% | 37 | 2,683 | - |
| vllm-omni | verbatim-en | plain | 3 | 15 | 206 [206–209] | 0% | 37 | 2,584 | 1.31x |
