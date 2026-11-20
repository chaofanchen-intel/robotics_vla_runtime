# BAGEL I2T: vLLM-Omni two-stage Thinker with async scheduling off

**Question.** BAGEL I2T decodes at 16.9 ms/token on M* and 17.1 on vLLM-Omni two-stage; M* with its launch chain off takes 19.5. Does vLLM lose the same ~2.5 ms/token when its async scheduling is off?

**Pre-registered reading.** ITL per arm: vllm-async-off near 19.5-20 means vLLM already has the same overlap and the per-token parity is explained; unchanged means vLLM's per-token cost has another reason.

Campaigns: `BAGEL_ASYNC_SCHED_VLLM_ASYNC_ON_01_B2B` (vllm-async-on, back-to-back), `BAGEL_ASYNC_SCHED_VLLM_ASYNC_OFF_01_B2B` (vllm-async-off, back-to-back), `BAGEL_ASYNC_SCHED_MSTAR_SPEC_ON_01_B2B` (mstar-spec-on, back-to-back), `BAGEL_ASYNC_SCHED_MSTAR_SPEC_OFF_01_B2B` (mstar-spec-off, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs vllm-async-on |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2t-512 | vllm-async-on | 2 | 10 | 2,982 [2,982–2,982] | 0% | 45 | 2,355 | - |
| vllm-omni-two-stage | i2t-512 | vllm-async-off | 2 | 10 | 3,123 [3,122–3,126] | 0% | 43 | 2,418 | 1.05x |
| mstar | i2t-512 | mstar-spec-on | 2 | 10 | 2,284 [2,282–2,285] | 0% | 48 | 2,572 | - |
| mstar | i2t-512 | mstar-spec-off | 2 | 10 | 2,599 [2,597–2,602] | 0% | 51 | 2,630 | - |
