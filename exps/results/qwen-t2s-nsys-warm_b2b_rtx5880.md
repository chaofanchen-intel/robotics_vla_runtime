# T2S: per-stage GPU timeline (nsys), three frameworks, rows pre-warmed

**Question.** qwen-t2s-fit: after the first audio chunk the frame loop costs M* 54, SGLang 65, vLLM 100 ms per audio second. M*'s own launch chain explains 1.24x; vLLM's extra 1.27x has only a structural nsys hint (3x launches in the Talker, measured on cold rows) and its mechanism is unproven. Where is the per-frame time of vLLM-Omni's Talker + Code2Wav loop on the GPU when no request carries first-seen-shape JIT?

**Pre-registered reading.** analyze_nsys_t2s_stages.py: busy per audio second by stage (Thinker / Talker / Code2Wav) and the idle remainder. vLLM Talker busy per audio second near M*'s but a long idle remainder = host launch / scheduling gaps in the frame loop (the chain M* has); Talker busy itself ~1.3x = more or slower kernels per frame (no CUDA graph, unfused sampling). nsys inflates the timeline 1.6-2.2x, M* most: structure only, never E2E.

Campaigns: `QWEN_T2S_NSYS_WARM_NSYS_01_B2B` (nsys, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | nsys | 1 | 5 | 377 [377–377] | 0% | 44 | 2,685 | - |
| sglang-omni | verbatim-en | nsys | 1 | 5 | 549 [549–549] | 0% | 39 | 2,691 | - |
| vllm-omni | verbatim-en | nsys | 1 | 5 | 585 [585–585] | 0% | 37 | 2,700 | - |
