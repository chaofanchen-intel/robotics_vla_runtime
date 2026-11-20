# Orpheus TTS (M* vs VoxServe)

**Question.** Orpheus TTS (M* vs VoxServe): how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_ORPHEUS_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | b1 | main | 3 | 15 | 2,531 [2,527–2,536] | 0% | 49 | 2,656 | - |
| mstar | b4 | main | 3 | 15 | 3,979 [3,977–3,986] | 0% | 51 | 2,678 | - |
| mstar | b16 | main | 3 | 15 | 5,000 [4,823–5,404] | 0% | 52 | 2,656 | - |
| voxserve | b1 | main | 3 | 15 | 2,751 [2,675–2,875] | 0% | 50 | 2,473 | - |
| voxserve | b4 | main | 3 | 15 | 5,159 [4,862–5,324] | 0% | 51 | 2,490 | - |
| voxserve | b16 | main | 3 | 15 | 6,925 [6,759–7,106] | 0% | 54 | 2,406 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| b1 | mstar | main | 2,531 [2,527–2,536] | 1.00x |
| b1 | voxserve | main | 2,751 [2,675–2,875] | 1.09x |
| b4 | mstar | main | 3,979 [3,977–3,986] | 1.00x |
| b4 | voxserve | main | 5,159 [4,862–5,324] | 1.30x |
| b16 | mstar | main | 5,000 [4,823–5,404] | 1.00x |
| b16 | voxserve | main | 6,925 [6,759–7,106] | 1.38x |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
