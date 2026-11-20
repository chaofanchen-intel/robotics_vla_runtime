# Qwen3-Omni audio-to-text

**Question.** Qwen3-Omni audio-to-text: how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

**Caveats.** Steady-state numbers: 6 preconditioning requests are not scored. vLLM-Omni's first ~4 requests after a server start take 0.6-1.5 s TTFB (DIAG_A2T_02/03/04: same with the default config, back-to-back, and the old eager config), so the cold path is not represented. A request whose input length is new can also be slow once (vLLM a2t-zh: 0.6-1.1 s TTFB on the 25 s clip in every start); the cause is not identified. The p50 is robust to those requests, the start range is not.

Campaigns: `BASELINE_QWEN_A2T_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | a2t-en | main | 3 | 30 | 183 [180–184] | 100% | 73 | 1,785 | - |
| mstar | a2t-zh | main | 3 | 30 | 153 [151–155] | 100% | 73 | 1,785 | - |
| sglang-omni | a2t-en | main | 3 | 30 | 251 [247–253] | 100% | 69 | 1,785 | - |
| sglang-omni | a2t-zh | main | 3 | 30 | 199 [194–212] | 100% | 70 | 1,785 | - |
| vllm-omni | a2t-en | main | 3 | 30 | 219 [208–225] | 100% | 66 | 1,785 | - |
| vllm-omni | a2t-zh | main | 3 | 30 | 234 [209–262] | 100% | 66 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| a2t-en | mstar | main | 183 [180–184] | 1.00x |
| a2t-en | sglang-omni | main | 251 [247–253] | 1.38x |
| a2t-en | vllm-omni | main | 219 [208–225] | 1.20x |
| a2t-en | **strongest baseline: vllm-omni** | | 219 [208–225] | **1.20x** |
| a2t-zh | mstar | main | 153 [151–155] | 1.00x |
| a2t-zh | sglang-omni | main | 199 [194–212] | 1.30x |
| a2t-zh | vllm-omni | main | 234 [209–262] | 1.53x |
| a2t-zh | **strongest baseline: sglang-omni** | | 199 [194–212] | **1.30x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
