# Qwen3-Omni A2T, fair profile: prefix cache off/on in all three frameworks, SGLang lookahead eligible, shapes pre-warmed

**Question.** Same as qwen-fair-txt for audio input, plus: the precondition phase serves the SAME ten clips the measured phase will send (`qwen.rows=cycle`), because vLLM-Omni JIT-compiles Triton fused-MoE kernels at inference time for input shapes it has not seen (server.log: 'Triton kernel JIT compilation during inference: fused_moe_kernel ... latency spike') and 2 of 10 measured A2T-zh requests per start were such first-seen shapes (TTFT 507 / 629 ms against ~100).

**Pre-registered reading.** As qwen-fair-txt. The `vllm-pinned` / `mstar-pinned` arms use the baseline request but the same warm rows. A2T's cache `on` arms only help if the framework caches audio-token prefixes at all; no TTFT change is itself a finding (TTFT unchanged on vs off).

**Caveats.** The warm profile removes first-seen-shape JIT from the measured phase; the baseline numbers (6 preconditions, rows sequential) are the cold-shape view and stay valid as such.

Campaigns: `QWEN_FAIR_A2T_SGLANG_OFF_01_B2B` (sglang-off, back-to-back), `QWEN_FAIR_A2T_SGLANG_ON_01_B2B` (sglang-on, back-to-back), `QWEN_FAIR_A2T_SGLANG_ON_ASYNC_OFF_01_B2B` (sglang-on-async-off, back-to-back), `QWEN_FAIR_A2T_VLLM_OFF_01_B2B` (vllm-off, back-to-back), `QWEN_FAIR_A2T_VLLM_ON_01_B2B` (vllm-on, back-to-back), `QWEN_FAIR_A2T_MSTAR_OFF_01_B2B` (mstar-off, back-to-back), `QWEN_FAIR_A2T_MSTAR_ON_01_B2B` (mstar-on, back-to-back), `QWEN_FAIR_A2T_VLLM_PINNED_01_B2B` (vllm-pinned, back-to-back), `QWEN_FAIR_A2T_MSTAR_PINNED_01_B2B` (mstar-pinned, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang-off |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | a2t-en | sglang-off | 2 | 20 | 193 [188–202] | 100% | 71 | 1,785 | - |
| sglang-omni | a2t-en | sglang-on | 2 | 20 | 255 [255–255] | 100% | 69 | 1,785 | 1.32x |
| sglang-omni | a2t-en | sglang-on-async-off | 2 | 20 | 317 [317–320] | 100% | 69 | 1,785 | 1.64x |
| vllm-omni | a2t-en | vllm-off | 2 | 20 | 208 [198–212] | 100% | 66 | 1,785 | - |
| vllm-omni | a2t-en | vllm-on | 2 | 20 | 429 [427–429] | 100% | 64 | 1,785 | - |
| vllm-omni | a2t-en | vllm-pinned | 2 | 20 | 206 [194–226] | 100% | 67 | 1,785 | - |
| mstar | a2t-en | mstar-off | 2 | 20 | 184 [182–184] | 100% | 74 | 1,785 | - |
| mstar | a2t-en | mstar-on | 2 | 20 | 178 [176–181] | 100% | 73 | 1,785 | - |
| mstar | a2t-en | mstar-pinned | 2 | 20 | 181 [181–184] | 100% | 73 | 1,785 | - |
