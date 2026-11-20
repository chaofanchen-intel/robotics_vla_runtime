# Qwen3-Omni text-to-text

**Question.** Qwen3-Omni text-to-text: how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_QWEN_TXT_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1k | main | 3 | 15 | 1,052 [1,052–1,053] | 100% | 74 | 1,785 | - |
| sglang-omni | txt-1k | main | 3 | 15 | 1,476 [1,456–1,648] | 100% | 71 | 1,785 | - |
| vllm-omni | txt-1k | main | 3 | 15 | 1,159 [1,157–1,185] | 100% | 68 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| txt-1k | mstar | main | 1,052 [1,052–1,053] | 1.00x |
| txt-1k | sglang-omni | main | 1,476 [1,456–1,648] | 1.40x |
| txt-1k | vllm-omni | main | 1,159 [1,157–1,185] | 1.10x |
| txt-1k | **strongest baseline: vllm-omni** | | 1,159 [1,157–1,185] | **1.10x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
