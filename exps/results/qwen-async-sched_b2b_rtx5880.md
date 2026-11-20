# Qwen3-Omni TXT: is M*'s speculative launch chain just vLLM's async scheduling?

**Question.** On a pinned 256-token decode M* takes 5.85 ms/token with its launch chain and 7.9 without; vLLM-Omni takes 5.93 (its Thinker runs OmniARAsyncScheduler) and SGLang-Omni 8.1 (its Omni scheduler refuses the overlap loop). Is the chain the same CPU/GPU-overlap mechanism vLLM already has?

**Pre-registered reading.** ITL (analyze_decode_itl.py) per arm. If vllm-async-off lands near mstar-spec-off (~7.9-8.2 ms/token) and vllm-async-on near mstar-spec-on, the lever is generic overlap scheduling, not something only a Walk / graph runtime can do. If vLLM stays at ~5.9 with async off, the two mechanisms differ and this reading is wrong.

**Caveats.** 5880 only (3 GPUs, Thinker TP=2, M* without torch.compile on sm_89): host overhead per token is larger with TP=2, so the size of the lever is not transferable to a single-GPU Thinker.

Campaigns: `QWEN_ASYNC_SCHED_VLLM_ASYNC_ON_01_B2B` (vllm-async-on, back-to-back), `QWEN_ASYNC_SCHED_VLLM_ASYNC_OFF_01_B2B` (vllm-async-off, back-to-back), `QWEN_ASYNC_SCHED_MSTAR_SPEC_ON_01_B2B` (mstar-spec-on, back-to-back), `QWEN_ASYNC_SCHED_MSTAR_SPEC_OFF_01_B2B` (mstar-spec-off, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs vllm-async-on |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni | txt-1k | vllm-async-on | 2 | 10 | 1,613 [1,611–1,618] | 0% | 38 | 2,700 | - |
| vllm-omni | txt-1k | vllm-async-off | 2 | 10 | 1,956 [1,952–1,958] | 0% | 39 | 2,698 | 1.21x |
| mstar | txt-1k | mstar-spec-on | 2 | 10 | 1,643 [1,643–1,644] | 0% | 48 | 2,685 | - |
| mstar | txt-1k | mstar-spec-off | 2 | 10 | 2,166 [2,156–2,179] | 0% | 48 | 2,684 | - |
