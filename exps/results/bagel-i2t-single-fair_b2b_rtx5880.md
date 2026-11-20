# BAGEL I2T, fair profile: vLLM-Omni single-stage against M* with its prefix cache on and off

**Question.** The fair I2T comparison so far is two-stage only (vllm-fair against m-fair). Single-stage runs understanding inside the DiT engine: its KV is a NaiveCache, there is no Thinker prefix cache to switch on, and text is decoded greedily. What are its E2E, TTFT and ITL under the same protocol, and how does it compare with M* once M*'s prefix cache is off too, the only cache state single-stage can have?

**Pre-registered reading.** E2E first, TTFT and ITL as extra (analyze_decode_itl.py). single-stage against m-cache-off is the cache-equal pair: neither side has a prefix cache. single-stage against m-fair (cache on) is reference only, because one side has a cache the other cannot have. The sampler needs no switch on the single-stage side: its text decode is greedy unless do_sample is set (default false), which matches M*'s neutral arm (top_k 1).

**Caveats.** One repeated prompt + image, so every request after the first is a hot prefix for M*'s cache-on arm. Lengths are pinned (128 tokens, ignore_eos). The single-stage deploy has no prefix cache option (enable_prefix_caching false in bagel_single_stage.yaml); its gap to two-stage is the whole decode path, not only the cache. Read TTFT and ITL separately from E2E.

Campaigns: `BAGEL_I2T_SINGLE_FAIR_SINGLE_STAGE_01_B2B` (single-stage, back-to-back), `BAGEL_I2T_SINGLE_FAIR_M_FAIR_01_B2B` (m-fair, back-to-back), `BAGEL_I2T_SINGLE_FAIR_M_CACHE_OFF_01_B2B` (m-cache-off, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs single-stage |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-single-stage | i2t-512 | single-stage | 2 | 10 | 6,020 [5,993–6,044] | 0% | 45 | 2,530 | - |
| mstar | i2t-512 | m-fair | 2 | 10 | 2,296 [2,280–2,305] | 0% | 46 | 2,629 | - |
| mstar | i2t-512 | m-cache-off | 2 | 10 | 2,923 [2,913–2,924] | 0% | 48 | 2,374 | - |
