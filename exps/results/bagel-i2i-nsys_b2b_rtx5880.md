# BAGEL I2I: GPU busy / idle per card (nsys), M* vs vLLM-Omni single- and two-stage

**Question.** Where do the extra ~4 s of vLLM-Omni two-stage I2I go: GPU work (Thinker prefill, more attention per step), idle time between the Thinker card and the DiT card (KV hand-off), or the denoise loop itself?

**Pre-registered reading.** analyze_nsys_burst_gaps.py, one table per GPU. Two-stage: the Thinker card's burst, the gap, then the DiT card's burst (the hand-off is the idle time between them); busy/span of the DiT card against single-stage's single burst says whether the loop itself differs. nsys inflates host time (M* most): structure only.

Campaigns: `BAGEL_I2I_NSYS_NSYS_01_B2B` (nsys, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2i-512 | nsys | 1 | 2 | 22,836 [22,836–22,836] | 0% | 55 | 1,316 | - |
| vllm-omni-two-stage | i2i-512 | nsys | 1 | 2 | 27,914 [27,914–27,914] | 0% | 38 | 2,415 | - |
| vllm-omni-single-stage | i2i-512 | nsys | 1 | 2 | 23,410 [23,410–23,410] | 0% | 54 | 1,365 | - |
