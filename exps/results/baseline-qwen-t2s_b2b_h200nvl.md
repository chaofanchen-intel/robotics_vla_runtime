# Qwen3-Omni text-to-speech

**Question.** Qwen3-Omni text-to-speech: how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_QWEN_T2S_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | main | 3 | 15 | 84.2 [82.5–86.3] | 100% | 72 | 1,785 | - |
| mstar | verbatim-zh | main | 3 | 15 | 79.7 [79.7–80.7] | 100% | 72 | 1,785 | - |
| sglang-omni | verbatim-en | main | 3 | 15 | 107 [106–111] | 100% | 68 | 1,785 | - |
| sglang-omni | verbatim-zh | main | 3 | 15 | 100 [93–101] | 100% | 69 | 1,785 | - |
| vllm-omni | verbatim-en | main | 3 | 15 | 168 [153–175] | 100% | 65 | 1,785 | - |
| vllm-omni | verbatim-zh | main | 3 | 15 | 144 [137–157] | 100% | 65 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| verbatim-en | mstar | main | 84.2 [82.5–86.3] | 1.00x |
| verbatim-en | sglang-omni | main | 107 [106–111] | 1.27x |
| verbatim-en | vllm-omni | main | 168 [153–175] | 1.99x |
| verbatim-en | **strongest baseline: sglang-omni** | | 107 [106–111] | **1.27x** |
| verbatim-zh | mstar | main | 79.7 [79.7–80.7] | 1.00x |
| verbatim-zh | sglang-omni | main | 100 [93–101] | 1.26x |
| verbatim-zh | vllm-omni | main | 144 [137–157] | 1.80x |
| verbatim-zh | **strongest baseline: sglang-omni** | | 100 [93–101] | **1.26x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
