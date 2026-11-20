# Qwen3-Omni A2T, one output token: does vLLM's slow audio TTFT with the Thinker prefix cache on survive three warm passes?

**Question.** qwen-fair-a2t: vLLM-Omni with the Thinker prefix cache on has TTFT 481 ms against 77 ms with it off (SGLang 119 on vs 88 off), although the measured phase repeats clips the precondition phase has already served once. 3 of the 7 slow requests were fast (95-107 ms) on the first pass and 3 were ~500 ms on both. With the cache on, the second pass prefills only the uncached tail, which is a new token count for every clip: Triton JIT of the fused-MoE kernel for first-seen shapes would fit, a real cost of the cached path would not.

**Pre-registered reading.** Preconditions = 30 = three passes over the ten clips (`qwen.rows=cycle`), the measured phase is the fourth. vllm-on at ~77-90 ms = the 481 ms was shapes first seen on the cached path (JIT), and the earlier arm just under-warmed; vllm-on still ~480 ms = a real cost of the cached audio path. Count the `Triton kernel JIT compilation during inference` lines in each server.log by phase. sglang-on vs sglang-off 30 ms: same reading.

**Caveats.** M* is not in this experiment (its cache is inert on audio prompts of this length: 252 vs 255 ms). One output token, so E2E is the first token.

Campaigns: `QWEN_A2T_CACHE_WARM_VLLM_OFF_01_B2B` (vllm-off, back-to-back), `QWEN_A2T_CACHE_WARM_VLLM_ON_01_B2B` (vllm-on, back-to-back), `QWEN_A2T_CACHE_WARM_SGLANG_OFF_01_B2B` (sglang-off, back-to-back), `QWEN_A2T_CACHE_WARM_SGLANG_ON_01_B2B` (sglang-on, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs vllm-off |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni | a2t-en-1tok | vllm-off | 2 | 20 | 75.9 [74.2–78.0] | 0% | 36 | 2,460 | - |
| vllm-omni | a2t-en-1tok | vllm-on | 2 | 20 | 77.4 [76.7–78.4] | 0% | 35 | 2,460 | 1.02x |
| sglang-omni | a2t-en-1tok | sglang-off | 2 | 20 | 89.7 [89.5–92.3] | 0% | 37 | 2,460 | - |
| sglang-omni | a2t-en-1tok | sglang-on | 2 | 20 | 121 [121–121] | 0% | 37 | 2,460 | - |
