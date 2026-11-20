# Qwen3-Omni first-token latency, text: fixed cost vs prefill (tiny and 1K prompts, one output token, cache off)

**Question.** TXT: M*'s TTFT is 149 ms against vLLM-Omni's 99 (both without a prefix cache) and SGLang-Omni's ~108 (cache on). Is the extra ~50 ms a per-request fixed cost (front end, scheduling, TP sync, sampling, first chunk) or the 1K-token prefill (MoE tiles, attention)?

**Pre-registered reading.** analyze_ttft_parts.py: `fixed` = E2E of a ~15-token prompt with one output token; `per 1k tokens` = the extra for the 1K prompt. M* ahead of vLLM in neither column = noise; M* behind in `fixed` = the per-request path (the BAGEL I2T cache-hit path also left ~100 ms unaccounted); M* behind in `per 1k tokens` = prefill kernels (MoE tile tables, attention).

**Caveats.** Linear prefill model between ~15 and 1,024 tokens. One output token, so ITL says nothing here.

Campaigns: `QWEN_TTFT_TXT_SGLANG_01_B2B` (sglang, back-to-back), `QWEN_TTFT_TXT_VLLM_01_B2B` (vllm, back-to-back), `QWEN_TTFT_TXT_MSTAR_01_B2B` (mstar, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | txt-tiny-1tok | sglang | 2 | 20 | 56.4 [56.4–56.6] | 0% | 38 | 2,460 | - |
| sglang-omni | txt-1k-1tok | sglang | 2 | 20 | 118 [117–119] | 0% | 38 | 2,604 | - |
| vllm-omni | txt-tiny-1tok | vllm | 2 | 20 | 57.1 [56.9–60.1] | 0% | 35 | 2,460 | - |
| vllm-omni | txt-1k-1tok | vllm | 2 | 20 | 99.6 [99.1–99.8] | 0% | 35 | 2,460 | - |
| mstar | txt-tiny-1tok | mstar | 2 | 20 | 64.9 [64.9–65.0] | 0% | 44 | 2,460 | - |
| mstar | txt-1k-1tok | mstar | 2 | 20 | 156 [155–156] | 0% | 47 | 2,685 | - |
