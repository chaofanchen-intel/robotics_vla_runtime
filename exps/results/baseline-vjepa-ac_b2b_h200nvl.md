# V-JEPA 2-AC rollout (M* vs Meta-native, both BF16)

**Question.** V-JEPA 2-AC rollout (M* vs Meta-native, both BF16): how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

**Caveats.** Both clients read frozen copies of the DROID videos (exps/inputs/VJ-AC/frozen_videos), so the source bytes are equal across engines and starts; the `*_pairs_*` audit must show every start valid before a ratio is quoted. Timing boundaries differ: M* is HTTP E2E, Meta-native is in-process, and equal decoded tensors are not proven.

Campaigns: `BASELINE_VJEPA_AC_MAIN_02_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | h4 | main | 3 | 15 | 167 [166–170] | 100% | 58 | 1,785 | - |
| mstar | h15 | main | 3 | 15 | 393 [386–412] | 100% | 56 | 1,785 | - |
| mstar | h30 | main | 3 | 15 | 789 [730–794] | 100% | 57 | 1,785 | - |
| meta-native | h4 | main | 3 | 15 | 394 [277–525] | 100% | 56 | 1,785 | - |
| meta-native | h15 | main | 3 | 15 | 955 [858–985] | 100% | 58 | 1,785 | - |
| meta-native | h30 | main | 3 | 15 | 2,110 [1,999–2,203] | 100% | 64 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| h4 | mstar | main | 167 [166–170] | 1.00x |
| h4 | meta-native | main | 394 [277–525] | 2.35x |
| h15 | mstar | main | 393 [386–412] | 1.00x |
| h15 | meta-native | main | 955 [858–985] | 2.43x |
| h30 | mstar | main | 789 [730–794] | 1.00x |
| h30 | meta-native | main | 2,110 [1,999–2,203] | 2.67x |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
