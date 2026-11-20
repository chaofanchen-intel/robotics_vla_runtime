# Qwen3-Omni text-to-text

**Question.** Qwen3-Omni text-to-text: how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_QWEN_TXT_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1k | main | 3 | 15 | 1,647 [1,646–1,647] | 0% | 48 | 2,685 | - |
| sglang-omni | txt-1k | main | 3 | 15 | 2,181 [2,167–2,189] | 0% | 42 | 2,692 | - |
| vllm-omni | txt-1k | main | 3 | 15 | 1,612 [1,610–1,613] | 0% | 38 | 2,699 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| txt-1k | mstar | main | 1,647 [1,646–1,647] | 1.00x |
| txt-1k | sglang-omni | main | 2,181 [2,167–2,189] | 1.32x |
| txt-1k | vllm-omni | main | 1,612 [1,610–1,613] | 0.98x |
| txt-1k | **strongest baseline: vllm-omni** | | 1,612 [1,610–1,613] | **0.98x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
