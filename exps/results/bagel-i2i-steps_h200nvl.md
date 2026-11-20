# BAGEL I2I: fixed and per-step cost, three deployments

**Question.** I2I takes 23.8 s on M*, 24.2 s on vLLM-Omni single-stage and 28.2 s on two-stage (T2I is within 3% on all three). Is the two-stage 4.4 s a per-request fixed cost (KV hand-off from the Thinker over the shared-memory connector, the source-image prefill) or a per-denoise-step cost?

**Pre-registered reading.** Fit E2E = a + b*steps per framework (the I2I table of the fit). Gap in `a`: hand-off / prefill, once per request. Gap in `b`: every step is slower (attention over the longer image context, more KV per step). A zero gap to single-stage in both is the 2% M* lead and nothing else.

Campaigns: `BAGEL_I2I_STEPS_MAIN_01` (main, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2i-512 | main | 2 | 10 | 5,116 [5,114–5,117] | 0% | 70 | 1,443 | - |
| mstar | i2i-512-steps5 | main | 2 | 10 | 5,113 [5,112–5,115] | 0% | 70 | 1,446 | - |
| mstar | i2i-512-steps25 | main | 2 | 10 | 5,114 [5,107–5,124] | 0% | 70 | 1,442 | - |
| mstar | i2i-512-srv-steps5 | main | 2 | 10 | 825 [825–826] | 0% | 66 | 1,612 | - |
| mstar | i2i-512-srv-steps25 | main | 2 | 10 | 2,720 [2,720–2,724] | 0% | 69 | 1,492 | - |
| vllm-omni-two-stage | i2i-512 | main | 2 | 10 | 10,449 [10,411–10,558] | 0% | 70 | 1,676 | - |
| vllm-omni-two-stage | i2i-512-steps5 | main | 2 | 10 | 4,988 [4,956–5,013] | 10% | 67 | 1,772 | - |
| vllm-omni-two-stage | i2i-512-steps25 | main | 2 | 10 | 7,535 [7,500–7,593] | 0% | 69 | 1,719 | - |
| vllm-omni-single-stage | i2i-512 | main | 2 | 10 | 6,504 [6,502–6,505] | 0% | 69 | 1,610 | - |
| vllm-omni-single-stage | i2i-512-steps5 | main | 2 | 10 | 1,180 [1,145–1,187] | 0% | 64 | 1,768 | - |
| vllm-omni-single-stage | i2i-512-steps25 | main | 2 | 10 | 3,596 [3,575–3,616] | 0% | 69 | 1,669 | - |
