# BAGEL: does vLLM-Omni get the same TTFT when its Thinker prefix cache is on?

**Question.** bagel-prefix-cache showed M*'s I2T lead (TTFT 138 vs 805 ms) is its KV prefix cache hitting a repeated image: with the cache off M* takes 777 ms. Is that an M* feature vLLM-Omni cannot match, or a switch the baseline left off?

**Pre-registered reading.** TTFT per arm. vllm-cache-on near m-cache-on (~140 ms): the lead is a deploy flag, not the engine, and the fair comparison is cache-on against cache-on or off against off. TTFT unchanged (or the server fails; read server.log): vLLM-Omni's BAGEL path cannot reuse the multimodal prefix, so the cache is a real M* capability, only one the fixture (one repeated image) overstates. I2I: any change here is the Thinker context being reused.

**Caveats.** If vLLM-Omni's BAGEL Thinker does not support prefix caching with image inputs the cell may fail or serve stale context: check the output text against vllm-cache-off before trusting a faster TTFT.

Campaigns: `BAGEL_VLLM_PREFIX_CACHE_VLLM_CACHE_OFF_01_B2B` (vllm-cache-off, back-to-back), `BAGEL_VLLM_PREFIX_CACHE_VLLM_CACHE_ON_01_B2B` (vllm-cache-on, back-to-back), `BAGEL_VLLM_PREFIX_CACHE_M_CACHE_ON_01_B2B` (m-cache-on, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs vllm-cache-off |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2t-512 | vllm-cache-off | 2 | 10 | 2,979 [2,978–2,980] | 0% | 44 | 2,321 | - |
| vllm-omni-two-stage | i2t-512 | vllm-cache-on | 2 | 10 | 2,256 [2,255–2,258] | 0% | 42 | 2,693 | 0.76x |
| vllm-omni-two-stage | i2i-512 | vllm-cache-off | 2 | 10 | 28,075 [28,049–28,102] | 0% | 35 | 1,823 | - |
| vllm-omni-two-stage | i2i-512 | vllm-cache-on | 2 | 10 | 27,152 [27,104–27,200] | 0% | 34 | 1,827 | 0.97x |
| mstar | i2t-512 | m-cache-on | 2 | 10 | 2,289 [2,284–2,298] | 0% | 49 | 2,567 | - |
| mstar | i2i-512 | m-cache-on | 2 | 10 | 23,447 [23,410–23,484] | 0% | 66 | 1,256 | - |
