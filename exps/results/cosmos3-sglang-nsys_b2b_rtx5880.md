# Cosmos3-Nano 256x256: GPU kernel mix and idle of M* (default), vLLM-Omni and SGLang-Omni (nsys)

**Question.** cosmos3-nsys: against M* with sequential CFG / eager / no compile, vLLM-Omni is still 1.17x slower and the extra time is idle GPU between launches. Where is SGLang-Omni, and does it batch the two CFG branches (one forward per step like M*, or two like vLLM)?

**Pre-registered reading.** analyze_nsys_burst_gaps.py, the last burst per process: GPU busy against span and the number of DiT forwards (attention launches / 36 layers) per request. ~50 forwards = batched CFG, ~100 = two sequential branches. Compare busy first (same work?), then span minus busy (launch gaps). nsys inflates host time; structure only.

**Caveats.** The SGLang-Omni generation stage is a child process of the launcher; nsys follows children by default, if a report is empty read server.log first.

Campaigns: `COSMOS3_SGLANG_NSYS_MSTAR_01_B2B` (mstar, back-to-back), `COSMOS3_SGLANG_NSYS_VLLM_01_B2B` (vllm, back-to-back), `COSMOS3_SGLANG_NSYS_SGLANG_01_B2B` (sglang, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | mstar | 1 | 4 | 1,669 [1,669–1,669] | 0% | 42 | 2,640 | - |
| vllm-omni | t2i-256 | vllm | 1 | 4 | 3,734 [3,734–3,734] | 0% | 40 | 2,698 | - |
| sglang-omni | t2i-256 | sglang | 1 | 4 | 5,161 [5,161–5,161] | 0% | 41 | 2,694 | - |
