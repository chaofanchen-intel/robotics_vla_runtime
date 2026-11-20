# Orpheus TTS (M* vs VoxServe)

**Question.** Orpheus TTS (M* vs VoxServe): how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

**Caveats.** E2E latency of the whole request (a wave of B requests) only: no RTF or tokens-per-second, so this table does not answer a throughput or real-time-factor claim.

Campaigns: `BASELINE_ORPHEUS_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | b1 | main | 3 | 15 | 945 [886–991] | 100% | 67 | 1,785 | - |
| mstar | b4 | main | 3 | 15 | 1,594 [1,573–1,597] | 100% | 66 | 1,785 | - |
| mstar | b16 | main | 3 | 15 | 2,265 [2,122–2,406] | 100% | 67 | 1,785 | - |
| voxserve | b1 | main | 3 | 15 | 1,602 [1,326–1,607] | 100% | 65 | 1,785 | - |
| voxserve | b4 | main | 3 | 15 | 2,570 [2,390–2,599] | 100% | 65 | 1,785 | - |
| voxserve | b16 | main | 3 | 15 | 3,525 [3,055–3,725] | 100% | 67 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| b1 | mstar | main | 945 [886–991] | 1.00x |
| b1 | voxserve | main | 1,602 [1,326–1,607] | 1.70x |
| b4 | mstar | main | 1,594 [1,573–1,597] | 1.00x |
| b4 | voxserve | main | 2,570 [2,390–2,599] | 1.61x |
| b16 | mstar | main | 2,265 [2,122–2,406] | 1.00x |
| b16 | voxserve | main | 3,525 [3,055–3,725] | 1.56x |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
