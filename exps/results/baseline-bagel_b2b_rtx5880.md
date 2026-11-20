# BAGEL at 512x512: M* vs vLLM-Omni (two deployments), equal work

**Question.** At equal work and with vLLM-Omni's MoT GEMM tuned for this GPU, how do M* and vLLM-Omni compare per task? 512x512 is the main operating point (256x256 is launch-bound and drift-prone; it is not run by default).

**Pre-registered reading.** Per case: best vLLM-Omni deployment / M*. T2I `-nocfg` is one branch on both (same output semantics); `-cfg3` is three on both. A tuned MoT config must be loaded in every vLLM-Omni server log (checked).

**Caveats.** vLLM-Omni's tuner covers the M values 258 x {1,3,4,12,16,48}: 256..1024 at one and three branches.

Campaigns: `BASELINE_BAGEL_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | main | 3 | 15 | 7,509 [7,439–7,553] | 0% | 57 | 1,364 | - |
| mstar | t2i-512-cfg3 | main | 3 | 15 | 18,994 [18,822–19,057] | 0% | 65 | 1,209 | - |
| mstar | i2i-512 | main | 3 | 15 | 23,810 [23,787–23,825] | 0% | 67 | 1,236 | - |
| mstar | i2t-512 | main | 3 | 15 | 2,293 [2,286–2,301] | 0% | 54 | 2,516 | - |
| vllm-omni-two-stage | t2i-512-nocfg | main | 3 | 15 | 7,636 [7,591–7,654] | 0% | 38 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | main | 3 | 15 | 19,230 [19,188–19,337] | 0% | 37 | 2,283 | - |
| vllm-omni-two-stage | i2i-512 | main | 3 | 15 | 28,171 [28,053–28,194] | 0% | 37 | 1,863 | - |
| vllm-omni-two-stage | i2t-512 | main | 3 | 15 | 2,983 [2,982–2,986] | 0% | 47 | 2,362 | - |
| vllm-omni-single-stage | t2i-512-nocfg | main | 3 | 15 | 7,527 [7,464–7,569] | 0% | 51 | 1,448 | - |
| vllm-omni-single-stage | t2i-512-cfg3 | main | 3 | 15 | 18,972 [18,907–18,974] | 0% | 60 | 1,281 | - |
| vllm-omni-single-stage | i2i-512 | main | 3 | 15 | 24,179 [24,167–24,204] | 0% | 62 | 1,318 | - |
| vllm-omni-single-stage | i2t-512 | main | 3 | 15 | 6,049 [6,017–6,110] | 0% | 51 | 2,474 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-512-nocfg | mstar | main | 7,509 [7,439–7,553] | 1.00x |
| t2i-512-nocfg | vllm-omni-two-stage | main | 7,636 [7,591–7,654] | 1.02x |
| t2i-512-nocfg | vllm-omni-single-stage | main | 7,527 [7,464–7,569] | 1.00x |
| t2i-512-nocfg | **strongest baseline: vllm-omni-single-stage** | | 7,527 [7,464–7,569] | **1.00x** |
| t2i-512-cfg3 | mstar | main | 18,994 [18,822–19,057] | 1.00x |
| t2i-512-cfg3 | vllm-omni-two-stage | main | 19,230 [19,188–19,337] | 1.01x |
| t2i-512-cfg3 | vllm-omni-single-stage | main | 18,972 [18,907–18,974] | 1.00x |
| t2i-512-cfg3 | **strongest baseline: vllm-omni-single-stage** | | 18,972 [18,907–18,974] | **1.00x** |
| i2i-512 | mstar | main | 23,810 [23,787–23,825] | 1.00x |
| i2i-512 | vllm-omni-two-stage | main | 28,171 [28,053–28,194] | 1.18x |
| i2i-512 | vllm-omni-single-stage | main | 24,179 [24,167–24,204] | 1.02x |
| i2i-512 | **strongest baseline: vllm-omni-single-stage** | | 24,179 [24,167–24,204] | **1.02x** |
| i2t-512 | mstar | main | 2,293 [2,286–2,301] | 1.00x |
| i2t-512 | vllm-omni-two-stage | main | 2,983 [2,982–2,986] | 1.30x |
| i2t-512 | vllm-omni-single-stage | main | 6,049 [6,017–6,110] | 2.64x |
| i2t-512 | **strongest baseline: vllm-omni-two-stage** | | 2,983 [2,982–2,986] | **1.30x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
