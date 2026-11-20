# Cosmos3: GPU kernel mix of M* (sequential CFG, eager, no compile) vs vLLM-Omni at 256x256 (nsys)

**Question.** With CFG, graph and compile all matched (m-all-off) vLLM-Omni is still 1.17x (~440 ms) slower than M*. Is that GPU work (a different attention / GEMM / VAE kernel) or time the GPU is idle?

**Pre-registered reading.** Last burst per cell (analyze_nsys_burst_gaps.py): GPU busy against span says idle vs work; the top-kernel lists say which kernel family carries the 440 ms. Structure only, nsys inflates host time.

Campaigns: `COSMOS3_NSYS_M_ALL_OFF_01_B2B` (m-all-off, back-to-back), `COSMOS3_NSYS_VLLM_01_B2B` (vllm, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m-all-off |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | m-all-off | 1 | 4 | 3,407 [3,407–3,407] | 100% | 61 | 1,785 | - |
| vllm-omni | t2i-256 | vllm | 1 | 4 | 1,736 [1,736–1,736] | 100% | 61 | 1,785 | - |
