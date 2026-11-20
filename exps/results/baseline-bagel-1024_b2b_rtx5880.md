# BAGEL at 1024x1024 (the paper's size), equal work

**Question.** Same as baseline-bagel at the paper's resolution. Needs stable cooling: on the throttling H200 NVL one request at this size crosses the thermal knee and measures the chassis.

**Pre-registered reading.** As baseline-bagel; read the clock column first: a clock that drops across starts invalidates the ratio.

Campaigns: `BASELINE_BAGEL_1024_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-1024-nocfg | main | 2 | 6 | 28,610 [28,559–28,662] | 0% | 66 | 1,216 | - |
| mstar | t2i-1024-cfg3 | main | 2 | 6 | 78,611 [78,511–78,799] | 0% | 71 | 1,127 | - |
| mstar | i2i-1024 | main | 2 | 6 | 99,939 [99,926–100,049] | 0% | 71 | 1,170 | - |
| mstar | i2t-1024 | main | 2 | 6 | 2,350 [2,348–2,352] | 0% | 53 | 2,599 | - |
| vllm-omni-two-stage | t2i-1024-nocfg | main | 2 | 6 | 29,331 [29,317–29,345] | 0% | 35 | 1,576 | - |
| vllm-omni-two-stage | t2i-1024-cfg3 | main | 2 | 6 | 80,224 [80,223–80,224] | 0% | 32 | 713 | - |
| vllm-omni-two-stage | i2i-1024 | main | 2 | 6 | 107,739 [107,683–107,978] | 0% | 32 | 681 | - |
| vllm-omni-two-stage | i2t-1024 | main | 2 | 6 | 3,016 [3,015–3,018] | 0% | 46 | 2,395 | - |
| vllm-omni-single-stage | t2i-1024-nocfg | main | 2 | 6 | 28,811 [28,782–28,839] | 0% | 59 | 1,287 | - |
| vllm-omni-single-stage | t2i-1024-cfg3 | main | 2 | 6 | 80,430 [80,286–80,574] | 0% | 68 | 1,154 | - |
| vllm-omni-single-stage | i2i-1024 | main | 2 | 6 | 102,491 [102,483–102,498] | 0% | 70 | 1,198 | - |
| vllm-omni-single-stage | i2t-1024 | main | 2 | 6 | 7,963 [7,963–7,965] | 0% | 51 | 2,506 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-1024-nocfg | mstar | main | 28,610 [28,559–28,662] | 1.00x |
| t2i-1024-nocfg | vllm-omni-two-stage | main | 29,331 [29,317–29,345] | 1.03x |
| t2i-1024-nocfg | vllm-omni-single-stage | main | 28,811 [28,782–28,839] | 1.01x |
| t2i-1024-nocfg | **strongest baseline: vllm-omni-single-stage** | | 28,811 [28,782–28,839] | **1.01x** |
| t2i-1024-cfg3 | mstar | main | 78,611 [78,511–78,799] | 1.00x |
| t2i-1024-cfg3 | vllm-omni-two-stage | main | 80,224 [80,223–80,224] | 1.02x |
| t2i-1024-cfg3 | vllm-omni-single-stage | main | 80,430 [80,286–80,574] | 1.02x |
| t2i-1024-cfg3 | **strongest baseline: vllm-omni-two-stage** | | 80,224 [80,223–80,224] | **1.02x** |
| i2i-1024 | mstar | main | 99,939 [99,926–100,049] | 1.00x |
| i2i-1024 | vllm-omni-two-stage | main | 107,739 [107,683–107,978] | 1.08x |
| i2i-1024 | vllm-omni-single-stage | main | 102,491 [102,483–102,498] | 1.03x |
| i2i-1024 | **strongest baseline: vllm-omni-single-stage** | | 102,491 [102,483–102,498] | **1.03x** |
| i2t-1024 | mstar | main | 2,350 [2,348–2,352] | 1.00x |
| i2t-1024 | vllm-omni-two-stage | main | 3,016 [3,015–3,018] | 1.28x |
| i2t-1024 | vllm-omni-single-stage | main | 7,963 [7,963–7,965] | 3.39x |
| i2t-1024 | **strongest baseline: vllm-omni-two-stage** | | 3,016 [3,015–3,018] | **1.28x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
