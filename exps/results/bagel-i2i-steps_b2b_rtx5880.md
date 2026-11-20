# BAGEL I2I: fixed and per-step cost, three deployments

**Question.** I2I takes 23.8 s on M*, 24.2 s on vLLM-Omni single-stage and 28.2 s on two-stage (T2I is within 3% on all three). Is the two-stage 4.4 s a per-request fixed cost (KV hand-off from the Thinker over the shared-memory connector, the source-image prefill) or a per-denoise-step cost?

**Pre-registered reading.** Fit E2E = a + b*steps per framework (the I2I table of the fit). Gap in `a`: hand-off / prefill, once per request. Gap in `b`: every step is slower (attention over the longer image context, more KV per step). A zero gap to single-stage in both is the 2% M* lead and nothing else.

Campaigns: `BAGEL_I2I_STEPS_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2i-512 | main | 2 | 10 | 23,643 [23,624–23,661] | 0% | 66 | 1,248 | - |
| mstar | i2i-512-steps5 | main | 2 | 10 | 23,568 [23,546–23,589] | 0% | 66 | 1,252 | - |
| mstar | i2i-512-steps25 | main | 2 | 10 | 23,515 [23,469–23,562] | 0% | 65 | 1,257 | - |
| mstar | i2i-512-srv-steps5 | main | 2 | 10 | 2,761 [2,759–2,763] | 0% | 53 | 1,606 | - |
| mstar | i2i-512-srv-steps25 | main | 2 | 10 | 11,784 [11,783–11,785] | 0% | 59 | 1,343 | - |
| vllm-omni-two-stage | i2i-512 | main | 2 | 10 | 28,161 [28,158–28,165] | 0% | 36 | 1,812 | - |
| vllm-omni-two-stage | i2i-512-steps5 | main | 2 | 10 | 6,865 [6,856–6,874] | 0% | 41 | 2,218 | - |
| vllm-omni-two-stage | i2i-512-steps25 | main | 2 | 10 | 16,059 [16,034–16,084] | 0% | 39 | 2,363 | - |
| vllm-omni-single-stage | i2i-512 | main | 2 | 10 | 23,926 [23,907–23,945] | 0% | 61 | 1,332 | - |
| vllm-omni-single-stage | i2i-512-steps5 | main | 2 | 10 | 2,909 [2,907–2,911] | 0% | 43 | 1,736 | - |
| vllm-omni-single-stage | i2i-512-steps25 | main | 2 | 10 | 12,030 [12,026–12,033] | 0% | 54 | 1,411 | - |
