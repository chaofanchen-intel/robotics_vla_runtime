# BAGEL at 1024x1024 (the paper's size), equal work

**Question.** Same as baseline-bagel at the paper's resolution. Needs stable cooling: on the throttling H200 NVL one request at this size crosses the thermal knee and measures the chassis.

**Pre-registered reading.** As baseline-bagel; read the clock column first: a clock that drops across starts invalidates the ratio.

Campaigns: `BASELINE_BAGEL_1024_MAIN_01` (main, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-1024-nocfg | main | 2 | 6 | 6,098 [6,084–6,103] | 0% | 70 | 1,434 | - |
| mstar | t2i-1024-cfg3 | main | 2 | 6 | 22,495 [22,410–23,268] | 0% | 70 | 909 | - |
| mstar | i2i-1024 | main | 2 | 6 | 40,902 [40,713–41,134] | 0% | 70 | 676 | - |
| mstar | i2t-1024 | main | 2 | 6 | 771 [770–772] | 100% | 65 | 1,785 | - |
| vllm-omni-two-stage | t2i-1024-nocfg | main | 2 | 6 | 7,522 [7,430–7,530] | 0% | 70 | 1,503 | - |
| vllm-omni-two-stage | t2i-1024-cfg3 | main | 2 | 6 | 31,526 [31,375–31,972] | 0% | 70 | 829 | - |
| vllm-omni-two-stage | i2i-1024 | main | 2 | 6 | 57,399 [57,385–58,005] | 0% | 70 | 749 | - |
| vllm-omni-two-stage | i2t-1024 | main | 2 | 6 | 1,050 [1,050–1,050] | 0% | 64 | 1,778 | - |
| vllm-omni-single-stage | t2i-1024-nocfg | main | 2 | 6 | 7,624 [7,598–7,650] | 0% | 69 | 1,498 | - |
| vllm-omni-single-stage | t2i-1024-cfg3 | main | 2 | 6 | 31,317 [31,284–32,048] | 0% | 70 | 834 | - |
| vllm-omni-single-stage | i2i-1024 | main | 2 | 6 | 49,803 [49,796–50,296] | 0% | 70 | 647 | - |
| vllm-omni-single-stage | i2t-1024 | main | 2 | 6 | 4,510 [4,499–4,522] | 0% | 64 | 1,756 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-1024-nocfg | mstar | main | 6,098 [6,084–6,103] | 1.00x |
| t2i-1024-nocfg | vllm-omni-two-stage | main | 7,522 [7,430–7,530] | 1.23x |
| t2i-1024-nocfg | vllm-omni-single-stage | main | 7,624 [7,598–7,650] | 1.25x |
| t2i-1024-nocfg | **strongest baseline: vllm-omni-two-stage** | | 7,522 [7,430–7,530] | **1.23x** |
| t2i-1024-cfg3 | mstar | main | 22,495 [22,410–23,268] | 1.00x |
| t2i-1024-cfg3 | vllm-omni-two-stage | main | 31,526 [31,375–31,972] | 1.40x |
| t2i-1024-cfg3 | vllm-omni-single-stage | main | 31,317 [31,284–32,048] | 1.39x |
| t2i-1024-cfg3 | **strongest baseline: vllm-omni-single-stage** | | 31,317 [31,284–32,048] | **1.39x** |
| i2i-1024 | mstar | main | 40,902 [40,713–41,134] | 1.00x |
| i2i-1024 | vllm-omni-two-stage | main | 57,399 [57,385–58,005] | 1.40x |
| i2i-1024 | vllm-omni-single-stage | main | 49,803 [49,796–50,296] | 1.22x |
| i2i-1024 | **strongest baseline: vllm-omni-single-stage** | | 49,803 [49,796–50,296] | **1.22x** |
| i2t-1024 | mstar | main | 771 [770–772] | 1.00x |
| i2t-1024 | vllm-omni-two-stage | main | 1,050 [1,050–1,050] | 1.36x |
| i2t-1024 | vllm-omni-single-stage | main | 4,510 [4,499–4,522] | 5.85x |
| i2t-1024 | **strongest baseline: vllm-omni-two-stage** | | 1,050 [1,050–1,050] | **1.36x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
