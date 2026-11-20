# Qwen3-Omni audio-to-text

**Question.** Qwen3-Omni audio-to-text: how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_QWEN_A2T_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | a2t-en | main | 3 | 30 | 251 [251–256] | 0% | 49 | 2,685 | - |
| mstar | a2t-zh | main | 3 | 30 | 210 [210–211] | 0% | 49 | 2,685 | - |
| sglang-omni | a2t-en | main | 3 | 30 | 309 [308–313] | 0% | 43 | 2,700 | - |
| sglang-omni | a2t-zh | main | 3 | 30 | 258 [250–258] | 0% | 44 | 2,690 | - |
| vllm-omni | a2t-en | main | 3 | 30 | 270 [266–275] | 0% | 39 | 2,700 | - |
| vllm-omni | a2t-zh | main | 3 | 30 | 264 [260–269] | 0% | 39 | 2,700 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| a2t-en | mstar | main | 251 [251–256] | 1.00x |
| a2t-en | sglang-omni | main | 309 [308–313] | 1.23x |
| a2t-en | vllm-omni | main | 270 [266–275] | 1.08x |
| a2t-en | **strongest baseline: vllm-omni** | | 270 [266–275] | **1.08x** |
| a2t-zh | mstar | main | 210 [210–211] | 1.00x |
| a2t-zh | sglang-omni | main | 258 [250–258] | 1.23x |
| a2t-zh | vllm-omni | main | 264 [260–269] | 1.26x |
| a2t-zh | **strongest baseline: sglang-omni** | | 258 [250–258] | **1.23x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
