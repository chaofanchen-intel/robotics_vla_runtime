# BAGEL I2I with every known difference removed: Mooncake connector + Thinker prefix cache on, against M* with its cache on and off

**Question.** vLLM-Omni two-stage I2I still has a ~1.6 s fixed gap to M* after the connector is swapped, and its stage 0 alone takes ~2.2 s before the first KV branch is out. The Thinker's prefill of the two 5k-token branches is ~0.74 s each (I2T nsys) and every vLLM-Omni deploy config has prefix caching off while M* opens its cache by default. Whether M* actually hits its cache on an I2I request is not logged, and the 50-step cache on/off pair (23,528 / 23,583 ms) cannot tell. With the Thinker cache on in vLLM and M*'s cache switched on and off at 5 steps, how much of the fixed gap is prefill?

**Pre-registered reading.** Per case, E2E by arm (analyze_decode_itl.py). M* cases: `i2i-512-srv-steps5` (M* takes the step count from the server, request-level steps are inert) and `i2i-512` (50 steps); vLLM cases `i2i-512-steps5` and `i2i-512`. m-cache-off vs m-cache-on at 5 steps = the prefill M* skips (or not). vllm-mooncake vs vllm-mooncake-prefix = the prefill vLLM skips when its Thinker caches the repeated input. Equal-state pairs: m-cache-on vs vllm-mooncake-prefix and m-cache-off vs vllm-mooncake. If the 5-step M* pair differs by ~1.5 s, the two-stage gap in every earlier I2I number included M*'s cache hit, and the I2I row of the fairness audit changes from `no difference` to `M*`.

**Caveats.** One fixture repeated (a hot prefix, best case for every cache). The 50-step cells cross the thermal knee on this box; the 5-step pair is the clean one.

Campaigns: `BAGEL_I2I_FAIR_M_CACHE_ON_01_B2B` (m-cache-on, back-to-back), `BAGEL_I2I_FAIR_M_CACHE_OFF_01_B2B` (m-cache-off, back-to-back), `BAGEL_I2I_FAIR_VLLM_MOONCAKE_01_B2B` (vllm-mooncake, back-to-back), `BAGEL_I2I_FAIR_VLLM_MOONCAKE_PREFIX_01_B2B` (vllm-mooncake-prefix, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m-cache-on |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2i-512-srv-steps5 | m-cache-on | 2 | 10 | 2,760 [2,737–2,784] | 0% | 49 | 1,610 | - |
| mstar | i2i-512-srv-steps5 | m-cache-off | 2 | 10 | 2,764 [2,749–2,767] | 0% | 53 | 1,614 | 1.00x |
| mstar | i2i-512 | m-cache-on | 2 | 10 | 23,506 [23,454–23,557] | 0% | 64 | 1,249 | - |
| mstar | i2i-512 | m-cache-off | 2 | 10 | 23,550 [23,546–23,554] | 0% | 66 | 1,249 | 1.00x |
| vllm-omni-two-stage | i2i-512 | vllm-mooncake | 2 | 10 | 25,964 [25,958–25,970] | 0% | 35 | 1,808 | - |
| vllm-omni-two-stage | i2i-512 | vllm-mooncake-prefix | 2 | 10 | 24,959 [24,943–24,974] | 0% | 34 | 1,834 | - |
| vllm-omni-two-stage | i2i-512-steps5 | vllm-mooncake | 2 | 10 | 4,564 [4,525–4,580] | 0% | 40 | 2,147 | - |
| vllm-omni-two-stage | i2i-512-steps5 | vllm-mooncake-prefix | 2 | 10 | 3,523 [3,516–3,530] | 0% | 37 | 2,430 | - |
