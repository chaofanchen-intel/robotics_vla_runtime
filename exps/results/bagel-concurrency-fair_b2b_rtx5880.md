# BAGEL I2T: B = 1 / 4 / 16 concurrent requests with the Thinker prefix cache ON in both M* and vLLM-Omni two-stage

**Question.** bagel-concurrency ran vLLM-Omni two-stage with its Thinker prefix cache OFF (the upstream default) and M* with its cache ON; bagel-sampling-parity showed that is the whole I2T TTFT gap at B = 1. Does the batch scaling (wave time against B) look the same once the cache state is equal?

**Pre-registered reading.** analyze_concurrency.py per arm: wave wall time, requests/s and throughput scaling against B = 1. m-cache-on is M* (cache on by default); vllm-cache-on is two-stage with `enable_prefix_caching: true` on the Thinker. Compare the scaling curves, not only B = 1.

**Caveats.** The fixture repeats one prompt + image, so every request after the first is a hot prefix (best case for any cache). vLLM-Omni single-stage has no Thinker prefix cache option and is not in this run.

Campaigns: `BAGEL_CONCURRENCY_FAIR_M_CACHE_ON_01_B2B` (m-cache-on, back-to-back), `BAGEL_CONCURRENCY_FAIR_VLLM_CACHE_ON_01_B2B` (vllm-cache-on, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m-cache-on |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2t-512 | m-cache-on | 2 | 10 | 2,293 [2,292–2,293] | 0% | 50 | 2,614 | - |
| mstar | i2t-512-b4 | m-cache-on | 2 | 10 | 2,665 [2,662–2,668] | 0% | 51 | 2,682 | - |
| mstar | i2t-512-b16 | m-cache-on | 2 | 10 | 3,647 [3,639–3,661] | 0% | 55 | 2,471 | - |
| vllm-omni-two-stage | i2t-512 | vllm-cache-on | 2 | 10 | 2,258 [2,258–2,259] | 0% | 44 | 2,685 | - |
| vllm-omni-two-stage | i2t-512-b4 | vllm-cache-on | 2 | 10 | 4,545 [4,543–4,547] | 0% | 45 | 2,683 | - |
| vllm-omni-two-stage | i2t-512-b16 | vllm-cache-on | 2 | 10 | 13,647 [13,646–13,648] | 0% | 57 | 2,667 | - |
