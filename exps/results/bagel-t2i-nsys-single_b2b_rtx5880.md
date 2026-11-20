# BAGEL T2I 512: GPU busy per denoise step on one card, M* against vLLM-Omni single-stage (nsys, one session)

**Question.** The two-stage vLLM-Omni per-step numbers were read from its DiT card. On one card, in one session, with the same nsys capture: does vLLM-Omni's single-stage denoise loop cost the same GPU time per step as M*?

**Pre-registered reading.** GPU busy per step and span per step, by framework, at -nocfg and -cfg3. Busy within about 2% of M*: the denoise loop is equal on the 5880 whatever the stage split, so the per-step gap is not where M*'s lead comes from. Busy clearly above M*: M* does less GPU work per step on one card, which the two-stage comparison could not show. Busy clearly below M*: check the kernel names before reading anything into it.

**Caveats.** nsys inflates host time: the GPU busy column is the evidence, not the span. One card (the one the run is pinned to), one start per framework.

Campaigns: `BAGEL_T2I_NSYS_SINGLE_NSYS_01_B2B` (nsys, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | nsys | 1 | 2 | 7,738 [7,738–7,738] | 0% | 44 | 1,307 | - |
| mstar | t2i-512-cfg3 | nsys | 1 | 2 | 19,484 [19,484–19,484] | 0% | 53 | 1,165 | - |
| vllm-omni-single-stage | t2i-512-nocfg | nsys | 1 | 2 | 7,900 [7,900–7,900] | 0% | 44 | 1,432 | - |
| vllm-omni-single-stage | t2i-512-cfg3 | nsys | 1 | 2 | 19,892 [19,892–19,892] | 0% | 55 | 1,242 | - |
