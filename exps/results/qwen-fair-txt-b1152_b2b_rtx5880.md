# Qwen3-Omni TXT, fair profile (cache off): does M*'s shipped ladder or the 1,152 rung decide the 'M* is 2.4% slower' headline? (256 and 1 output tokens)

**Question.** qwen-fair-txt: with every prefix cache off M* is 2.4% slower than vLLM on the 1,072-token TXT request (1,648 against 1,610 ms), all of it first-token latency. qwen-prefill-len: ~32 of the 47 ms TTFT gap at 1,072 tokens is the default prefill-graph ladder (1,024 + 48 in two forwards, or one step padded to 2,048 at budget 4,096); a 1,152 rung with budget 4,096 gives TTFT 113 against 145 ms. Does the 256-token E2E headline follow, i.e. does M* with the rung catch up with or pass vLLM?

**Pre-registered reading.** analyze_decode_itl.py: E2E (txt-1k) and TTFT (txt-1k-1tok) per arm. mstar-off is the shipped config, mstar-b1152 the counterfactual (budget 4,096 + 1,152 rung), sglang-off / vllm-off the same arms as qwen-fair-txt (repeat check against its numbers). mstar-b1152 E2E within ~1% of vllm-off or below = the TXT lag is a configuration, report both rows; still ~2% behind = a kernel-level residual (qwen-prefill-len: ~11-15 ms from 1,024 tokens up). ITL must be equal to mstar-off (the ladder only changes prefill).

**Caveats.** Same protocol as qwen-fair-txt (cache off, lookahead-eligible request, 2 starts); C=1 only. The 1,152 config also sets the chunk budget to 4,096, so a concurrent decode row shares its step with a longer prefill: invisible here. It displaces the 2,048 prefill graph (qwen-prefill-len, section 9.6).

Campaigns: `QWEN_FAIR_TXT_B1152_SGLANG_OFF_01_B2B` (sglang-off, back-to-back), `QWEN_FAIR_TXT_B1152_VLLM_OFF_01_B2B` (vllm-off, back-to-back), `QWEN_FAIR_TXT_B1152_MSTAR_OFF_01_B2B` (mstar-off, back-to-back), `QWEN_FAIR_TXT_B1152_MSTAR_B1152_01_B2B` (mstar-b1152, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang-off |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | txt-1k | sglang-off | 2 | 10 | 1,596 [1,596–1,596] | 0% | 40 | 2,700 | - |
| sglang-omni | txt-1k-1tok | sglang-off | 2 | 10 | 120 [120–122] | 0% | 38 | 2,628 | - |
| vllm-omni | txt-1k | vllm-off | 2 | 10 | 1,613 [1,602–1,615] | 0% | 37 | 2,699 | - |
| vllm-omni | txt-1k-1tok | vllm-off | 2 | 10 | 101 [100–105] | 0% | 34 | 2,464 | - |
| mstar | txt-1k | mstar-off | 2 | 10 | 1,645 [1,645–1,648] | 0% | 47 | 2,685 | - |
| mstar | txt-1k | mstar-b1152 | 2 | 10 | 1,612 [1,610–1,613] | 0% | 48 | 2,685 | - |
| mstar | txt-1k-1tok | mstar-off | 2 | 10 | 157 [156–157] | 0% | 47 | 2,685 | - |
| mstar | txt-1k-1tok | mstar-b1152 | 2 | 10 | 123 [123–124] | 0% | 46 | 2,604 | - |
