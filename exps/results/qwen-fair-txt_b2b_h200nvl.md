# Qwen3-Omni TXT, fair profile: prefix cache off/on in all three frameworks, SGLang lookahead eligible

**Question.** The baseline TXT cells were not equal: SGLang-Omni's Thinker had its radix cache ON while M* (prefix_cache_tp off at TP=2) and vLLM-Omni (upstream enable_prefix_caching: false) had none, the prompt repeats, and our seed / min_tokens switched SGLang's own lookahead decode off. With the cache OFF in all three, and again ON in all three, and a request SGLang's lookahead accepts, does M* still lead, and by how much?

**Pre-registered reading.** Compare within a cache state: off-vs-off (a cold prefill every request) and on-vs-on (a hot prefix). TTFT carries the cache, ITL the decode loop. sglang-on-async-off must be slower than sglang-on in ITL, or the lookahead was never engaged. vllm-pinned / mstar-pinned against vllm-off / mstar-off say whether dropping seed + min_tokens changes them (expected: no). Anything the arms leave unequal is listed in the fairness audit, not assumed away.

**Caveats.** `on` = the same prompt repeated, i.e. a hot prefix (best case for every cache); `off` = a cold prefill each request. There is no M* `on` arm: with the Thinker's prefix cache enabled at TP=2 (`prefix_cache_tp: true`, offload 0) M* serves the first 1k-token request and the SECOND (identical prompt) kills worker_1 with `TP ranks diverged: KV thinker_kv: rank 0 leased 8 cached pages ...` (QWEN_FAIR_TXT_MSTAR_ON_01_B2B, both repeats). M* ships that switch off at world size 2, so M* has no hot-prefix TXT number on this box. On h200nvl (one rank) M* opens the Thinker cache by itself: its hot-prefix number is the M* cell of baseline-qwen-txt (default config, server.log 'whole prompt already cached True'), and `mstar-off` / `mstar-pinned` here run the cache-off recipe (resources.thinker_kv.prefix_cache: false).

Campaigns: `QWEN_FAIR_TXT_SGLANG_OFF_01_B2B` (sglang-off, back-to-back), `QWEN_FAIR_TXT_SGLANG_ON_01_B2B` (sglang-on, back-to-back), `QWEN_FAIR_TXT_SGLANG_ON_ASYNC_OFF_01_B2B` (sglang-on-async-off, back-to-back), `QWEN_FAIR_TXT_VLLM_OFF_01_B2B` (vllm-off, back-to-back), `QWEN_FAIR_TXT_VLLM_ON_01_B2B` (vllm-on, back-to-back), `QWEN_FAIR_TXT_MSTAR_OFF_01_B2B` (mstar-off, back-to-back), `QWEN_FAIR_TXT_VLLM_PINNED_01_B2B` (vllm-pinned, back-to-back), `QWEN_FAIR_TXT_MSTAR_PINNED_01_B2B` (mstar-pinned, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang-off |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | txt-1k | sglang-off | 2 | 10 | 1,093 [1,090–1,099] | 100% | 68 | 1,785 | - |
| sglang-omni | txt-1k | sglang-on | 2 | 10 | 1,137 [1,131–1,139] | 100% | 71 | 1,785 | 1.04x |
| sglang-omni | txt-1k | sglang-on-async-off | 2 | 10 | 1,433 [1,432–1,434] | 100% | 71 | 1,785 | 1.31x |
| vllm-omni | txt-1k | vllm-off | 2 | 10 | 1,152 [1,149–1,181] | 100% | 68 | 1,785 | - |
| vllm-omni | txt-1k | vllm-on | 2 | 10 | 1,164 [1,161–1,164] | 100% | 66 | 1,785 | - |
| vllm-omni | txt-1k | vllm-pinned | 2 | 10 | 1,151 [1,150–1,152] | 100% | 68 | 1,785 | - |
| mstar | txt-1k | mstar-off | 2 | 10 | 1,088 [1,087–1,089] | 100% | 75 | 1,785 | - |
| mstar | txt-1k | mstar-pinned | 2 | 10 | 1,088 [1,088–1,092] | 100% | 75 | 1,785 | - |
