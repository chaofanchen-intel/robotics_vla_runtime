# Pi0.5 on the 46-row DROID fixture (M* vs SGLang core vs vLLM-Omni eager)

**Question.** Pi0.5 end-to-end action latency at equal work (10 flow steps, [50,32] action).

**Pre-registered reading.** Per-case ratios under the protocol named in the campaign (`_B2B` = back-to-back). The protocol changes this task's ratios a lot (see protocol-ab-pi05), so always report both. vLLM-Omni is measured eager only.

**Caveats.** vLLM-Omni's graph variant (`pi05-vllm-graph`) did not change latency; no better vLLM recipe is known.

Campaigns: `BASELINE_PI05_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row | main | 3 | 138 | 92.9 [92.4–93.9] | 100% | 59 | 1,785 | - |
| sglang-core | droid-46row | main | 3 | 138 | 142 [140–144] | 100% | 58 | 1,785 | - |
| vllm-omni | droid-46row | main | 3 | 138 | 192 [190–195] | 100% | 59 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| droid-46row | mstar | main | 92.9 [92.4–93.9] | 1.00x |
| droid-46row | sglang-core | main | 142 [140–144] | 1.53x |
| droid-46row | vllm-omni | main | 192 [190–195] | 2.06x |
| droid-46row | **strongest baseline: sglang-core** | | 142 [140–144] | **1.53x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
