# BAGEL at 512x512: M* vs vLLM-Omni (two deployments), equal work

**Question.** At equal work and with vLLM-Omni's MoT GEMM tuned for this GPU, how do M* and vLLM-Omni compare per task? 512x512 is the main operating point (256x256 is launch-bound and drift-prone; it is not run by default).

**Pre-registered reading.** Per case: best vLLM-Omni deployment / M*. T2I `-nocfg` is one branch on both (same output semantics); `-cfg3` is three on both. The tuned MoT config is passed to every vLLM-Omni server; only the single-stage server log shows it loading (the two-stage DiT subprocess logs nothing to the server log), so `No tuned MoT config` is checked absent.

**Caveats.** vLLM-Omni's tuner covers the M values 258 x {1,3,4,12,16,48}: 256..1024 at one and three branches.

Campaigns: `BASELINE_BAGEL_MAIN_01` (main, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | main | 3 | 15 | 2,136 [2,054–2,261] | 0% | 67 | 1,710 | - |
| mstar | t2i-512-cfg3 | main | 3 | 15 | 3,869 [3,860–3,888] | 0% | 70 | 1,449 | - |
| mstar | i2i-512 | main | 3 | 15 | 5,113 [5,110–5,124] | 0% | 70 | 1,445 | - |
| mstar | i2t-512 | main | 3 | 15 | 699 [698–703] | 100% | 66 | 1,785 | - |
| vllm-omni-two-stage | t2i-512-nocfg | main | 3 | 15 | 2,737 [2,310–2,942] | 67% | 67 | 1,778 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | main | 3 | 15 | 5,143 [5,093–5,217] | 0% | 69 | 1,660 | - |
| vllm-omni-two-stage | i2i-512 | main | 3 | 15 | 10,409 [10,408–10,496] | 0% | 70 | 1,674 | - |
| vllm-omni-two-stage | i2t-512 | main | 3 | 15 | 995 [994–995] | 0% | 65 | 1,778 | - |
| vllm-omni-single-stage | t2i-512-nocfg | main | 3 | 15 | 2,939 [2,767–3,122] | 93% | 67 | 1,785 | - |
| vllm-omni-single-stage | t2i-512-cfg3 | main | 3 | 15 | 5,216 [5,194–5,396] | 0% | 69 | 1,663 | - |
| vllm-omni-single-stage | i2i-512 | main | 3 | 15 | 6,504 [6,404–6,598] | 0% | 70 | 1,615 | - |
| vllm-omni-single-stage | i2t-512 | main | 3 | 15 | 4,712 [4,503–5,011] | 0% | 65 | 1,783 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-512-nocfg | mstar | main | 2,136 [2,054–2,261] | 1.00x |
| t2i-512-nocfg | vllm-omni-two-stage | main | 2,737 [2,310–2,942] | 1.28x |
| t2i-512-nocfg | vllm-omni-single-stage | main | 2,939 [2,767–3,122] | 1.38x |
| t2i-512-nocfg | **strongest baseline: vllm-omni-two-stage** | | 2,737 [2,310–2,942] | **1.28x** |
| t2i-512-cfg3 | mstar | main | 3,869 [3,860–3,888] | 1.00x |
| t2i-512-cfg3 | vllm-omni-two-stage | main | 5,143 [5,093–5,217] | 1.33x |
| t2i-512-cfg3 | vllm-omni-single-stage | main | 5,216 [5,194–5,396] | 1.35x |
| t2i-512-cfg3 | **strongest baseline: vllm-omni-two-stage** | | 5,143 [5,093–5,217] | **1.33x** |
| i2i-512 | mstar | main | 5,113 [5,110–5,124] | 1.00x |
| i2i-512 | vllm-omni-two-stage | main | 10,409 [10,408–10,496] | 2.04x |
| i2i-512 | vllm-omni-single-stage | main | 6,504 [6,404–6,598] | 1.27x |
| i2i-512 | **strongest baseline: vllm-omni-single-stage** | | 6,504 [6,404–6,598] | **1.27x** |
| i2t-512 | mstar | main | 699 [698–703] | 1.00x |
| i2t-512 | vllm-omni-two-stage | main | 995 [994–995] | 1.42x |
| i2t-512 | vllm-omni-single-stage | main | 4,712 [4,503–5,011] | 6.74x |
| i2t-512 | **strongest baseline: vllm-omni-two-stage** | | 995 [994–995] | **1.42x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
