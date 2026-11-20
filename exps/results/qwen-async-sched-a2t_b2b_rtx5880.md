# Qwen3-Omni A2T: is M*'s speculative launch chain just vLLM's async scheduling? (the TXT arms on audio input)

**Question.** The same question as qwen-async-sched on audio input. The per-token table has M* chain on / off and vLLM async on, but the vLLM async-off arm was only run for TXT, so A2T has no vLLM async-off number.

**Pre-registered reading.** ITL (analyze_decode_itl.py) per arm, same reading as qwen-async-sched: vllm-async-off near mstar-spec-off and vllm-async-on near mstar-spec-on means generic overlap scheduling; vLLM staying flat with async off means the mechanisms differ.

**Caveats.** 5880 only (3 GPUs, Thinker TP=2, M* without torch.compile on sm_89). Audio input adds the audio encoder to the prompt path; ITL is per output token and is not affected by it.

Campaigns: `QWEN_ASYNC_SCHED_A2T_VLLM_ASYNC_ON_01_B2B` (vllm-async-on, back-to-back), `QWEN_ASYNC_SCHED_A2T_VLLM_ASYNC_OFF_01_B2B` (vllm-async-off, back-to-back), `QWEN_ASYNC_SCHED_A2T_MSTAR_SPEC_ON_01_B2B` (mstar-spec-on, back-to-back), `QWEN_ASYNC_SCHED_A2T_MSTAR_SPEC_OFF_01_B2B` (mstar-spec-off, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs vllm-async-on |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni | a2t-en | vllm-async-on | 2 | 10 | 674 [658–689] | 0% | 35 | 2,700 | - |
| vllm-omni | a2t-en | vllm-async-off | 2 | 10 | 705 [697–712] | 0% | 36 | 2,700 | 1.05x |
| mstar | a2t-en | mstar-spec-on | 2 | 10 | 263 [262–264] | 0% | 48 | 2,685 | - |
| mstar | a2t-en | mstar-spec-off | 2 | 10 | 319 [315–323] | 0% | 49 | 2,685 | - |
