# V-JEPA 2-AC rollout (M* vs Meta-native, both BF16)

**Question.** V-JEPA 2-AC rollout (M* vs Meta-native, both BF16): how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

Campaigns: `BASELINE_VJEPA_AC_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | h4 | main | 3 | 15 | 205 [182–210] | 0% | 35 | 2,422 | - |
| mstar | h15 | main | 3 | 15 | 394 [392–394] | 0% | 34 | 2,431 | - |
| mstar | h30 | main | 3 | 15 | 661 [646–669] | 0% | 36 | 2,547 | - |
| meta-native | h4 | main | 3 | 15 | 342 [336–349] | 0% | 35 | 2,471 | - |
| meta-native | h15 | main | 3 | 15 | 1,133 [1,100–1,136] | 0% | 40 | 2,656 | - |
| meta-native | h30 | main | 3 | 15 | 2,830 [2,819–2,879] | 0% | 43 | 2,391 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| h4 | mstar | main | 205 [182–210] | 1.00x |
| h4 | meta-native | main | 342 [336–349] | 1.67x |
| h15 | mstar | main | 394 [392–394] | 1.00x |
| h15 | meta-native | main | 1,133 [1,100–1,136] | 2.88x |
| h30 | mstar | main | 661 [646–669] | 1.00x |
| h30 | meta-native | main | 2,830 [2,819–2,879] | 4.28x |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
