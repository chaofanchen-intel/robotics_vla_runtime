# BAGEL I2T / I2I: M*'s KV prefix cache on vs off, against vLLM-Omni (prefix caching off)

**Question.** Every request of a cell sends the same prompt and the same image. M* opens a KV prefix cache by default; every vLLM-Omni deploy config sets enable_prefix_caching: false. nsys shows M*'s I2T LLM prefill at ~28 ms of GPU work against ~740 ms for vLLM-Omni's 4,935-token prefill, and the I2T TTFT gap (130 vs 808 ms) is exactly that. Is the gap a prefix-cache hit on a repeated input, and how much of the I2I gap is too?

**Pre-registered reading.** TTFT and E2E per arm (analyze_decode_itl.py). m-cache-off TTFT near vllm's (~800 ms for I2T) means the TTFT lead was the cache, not the engine: the equal-work comparison is m-cache-off against vllm. Unchanged TTFT means the cache is not the reason and the profile above is read wrong.

**Caveats.** Not a claim that caching is unfair in general (a served system keeps it), but the baseline configs forbid it and the fixture repeats one input, so the cell measures the cache. A bagel.prefix_cache=off arm is recorded in the manifest's overrides; the proof it took effect is the TTFT change itself.

Campaigns: `BAGEL_PREFIX_CACHE_M_CACHE_ON_01_B2B` (m-cache-on, back-to-back), `BAGEL_PREFIX_CACHE_M_CACHE_OFF_01_B2B` (m-cache-off, back-to-back), `BAGEL_PREFIX_CACHE_VLLM_01_B2B` (vllm, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m-cache-on |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2t-512 | m-cache-on | 2 | 10 | 2,293 [2,283–2,303] | 0% | 48 | 2,576 | - |
| mstar | i2t-512 | m-cache-off | 2 | 10 | 2,934 [2,925–2,947] | 0% | 55 | 2,370 | 1.28x |
| mstar | i2i-512 | m-cache-on | 2 | 10 | 23,528 [23,433–23,624] | 0% | 65 | 1,255 | - |
| mstar | i2i-512 | m-cache-off | 2 | 10 | 23,583 [23,573–23,593] | 0% | 66 | 1,245 | 1.00x |
| vllm-omni-two-stage | i2t-512 | vllm | 2 | 10 | 2,979 [2,974–2,989] | 0% | 45 | 2,360 | - |
| vllm-omni-two-stage | i2i-512 | vllm | 2 | 10 | 28,230 [28,134–39,717] | 0% | 38 | 1,610 | - |
