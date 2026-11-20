# Cosmos3-Nano T2I 256x256 (M* vs vLLM-Omni)

**Question.** Cosmos3-Nano T2I 256x256 (M* vs vLLM-Omni): how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_COSMOS3_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | main | 3 | 15 | 1,290 [1,275–1,355] | 0% | 42 | 2,312 | - |
| vllm-omni | t2i-256 | main | 3 | 15 | 3,133 [3,072–3,196] | 0% | 41 | 2,693 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-256 | mstar | main | 1,290 [1,275–1,355] | 1.00x |
| t2i-256 | vllm-omni | main | 3,133 [3,072–3,196] | 2.43x |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
