# Pi0.5: GPU timeline of one action request in M*, SGLang core and vLLM-Omni (nsys)

**Question.** baseline-pi05: M* 104 ms, SGLang core 164, vLLM-Omni (eager) 220 for the same 10 flow-matching steps; M* with every ablation off (mstar-ablation-pi05) is still 120 ms. Nothing says where the 40-100 ms go. Is it GPU work (different kernels per step, un-fused attention / GEMM, a larger vision encoder pass) or idle GPU between launches (launch gaps, host-side image / state handling)?

**Pre-registered reading.** analyze_nsys_burst_gaps.py, last burst per process: GPU busy against span and launch count per framework. Same busy and a longer span = launch / host gaps; different busy = different kernels, then read the top-kernel list (vision encoder, VLM prefill, action expert x 10 steps). Structure only: nsys inflates host time, never read E2E from it.

**Caveats.** No fairness audit of the Pi0.5 configs exists yet (attention backend, dtype, compile, image preprocessing, number of denoise steps is the only equalised setting); this run only says where the time is.

Campaigns: `PI05_NSYS_MSTAR_01_B2B` (mstar, back-to-back), `PI05_NSYS_SGLANG_CORE_01_B2B` (sglang-core, back-to-back), `PI05_NSYS_VLLM_01_B2B` (vllm, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row | mstar | 1 | 4 | 113 [113–113] | 0% | 37 | 2,460 | - |
| sglang-core | droid-46row | sglang-core | 1 | 4 | 180 [180–180] | 0% | 35 | 2,186 | - |
| vllm-omni | droid-46row | vllm | 1 | 4 | 274 [274–274] | 0% | 35 | 2,535 | - |
