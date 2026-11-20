# Qwen3-Omni text-to-speech

**Question.** Qwen3-Omni text-to-speech: how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_QWEN_T2S_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | main | 3 | 15 | 103 [99–103] | 0% | 46 | 2,640 | - |
| mstar | verbatim-zh | main | 3 | 15 | 93.3 [92.1–93.8] | 0% | 46 | 2,679 | - |
| sglang-omni | verbatim-en | main | 3 | 15 | 121 [121–122] | 0% | 41 | 2,628 | - |
| sglang-omni | verbatim-zh | main | 3 | 15 | 106 [105–110] | 0% | 41 | 2,663 | - |
| vllm-omni | verbatim-en | main | 3 | 15 | 155 [146–159] | 0% | 37 | 2,642 | - |
| vllm-omni | verbatim-zh | main | 3 | 15 | 146 [144–150] | 0% | 38 | 2,684 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| verbatim-en | mstar | main | 103 [99–103] | 1.00x |
| verbatim-en | sglang-omni | main | 121 [121–122] | 1.18x |
| verbatim-en | vllm-omni | main | 155 [146–159] | 1.51x |
| verbatim-en | **strongest baseline: sglang-omni** | | 121 [121–122] | **1.18x** |
| verbatim-zh | mstar | main | 93.3 [92.1–93.8] | 1.00x |
| verbatim-zh | sglang-omni | main | 106 [105–110] | 1.14x |
| verbatim-zh | vllm-omni | main | 146 [144–150] | 1.56x |
| verbatim-zh | **strongest baseline: sglang-omni** | | 106 [105–110] | **1.14x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
