# Qwen3-Omni TXT: what the 1k-token prefill is made of in each framework (nsys, one output token, cache off)

**Question.** qwen-ttft-txt: the fixed cost is the same in all three (M* 54 / SGLang 55 / vLLM 57 ms, one ~15-token prompt) but the 1k-token prefill is M* 86, SGLang 62, vLLM 42 ms per 1k tokens (TTFT 146 / 117 / 99 ms). That is the only TXT / A2T place where M* is clearly behind. Is it the MoE kernels (M*'s own fused-MoE tile tables against vLLM's tuned Triton ones), the attention kernel, the TP=2 all-reduce, or launch gaps?

**Pre-registered reading.** analyze_nsys_burst_gaps.py, one table per process (both TP ranks) and the top kernels of the last burst, per framework. Compare GPU busy against span (span >> busy = launch / sync gaps, not kernels), then the top kernels' ms: a MoE GEMM that is ~2x vLLM's = the tile table (`mstar-moe-tune` re-sweeps it); attention or NCCL dominating = a different lever. A framework whose burst has a CUDA-graph replay shows few launches: say so. nsys inflates host time, kernel ms are the evidence.

**Caveats.** One prompt length (1,072 tokens); the per-1k-token figure came from a two-point line between ~15 and ~1,072 tokens.

Campaigns: `QWEN_PREFILL_NSYS_SGLANG_01_B2B` (sglang, back-to-back), `QWEN_PREFILL_NSYS_VLLM_01_B2B` (vllm, back-to-back), `QWEN_PREFILL_NSYS_MSTAR_01_B2B` (mstar, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | txt-1k-1tok | sglang | 1 | 5 | 119 [119–119] | 0% | 38 | 2,700 | - |
| vllm-omni | txt-1k-1tok | vllm | 1 | 5 | 101 [101–101] | 0% | 37 | 2,460 | - |
| mstar | txt-1k-1tok | mstar | 1 | 5 | 155 [155–155] | 0% | 46 | 2,685 | - |
