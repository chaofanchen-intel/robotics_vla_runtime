# Pi0.5 at 1 / 5 / 10 / 20 flow steps: how much of each framework's latency is the denoise loop?

**Question.** baseline-pi05 runs 10 flow-matching steps, so its ratios say nothing about the denoise loop. Each step is one action-expert forward on the prefix KV, so E2E = fixed + steps x per-step. Which framework pays more per step, and which pays more fixed cost (images, prefix, request path)? Does M*'s lead come from the step, the fixed part, or both?

**Pre-registered reading.** Per case (step count), the ratio best baseline / M* as in baseline-pi05, pooled over starts with the per-start range beside it. Then the slope per framework: (p50 at 20 steps - p50 at 1 step) / 19. A lead that grows with the step count is a per-step lead; a lead the same at 1 and at 20 steps is a fixed-cost lead. The intercept (p50 at 1 step minus one slope) is the fixed cost.

**Caveats.** Steps 1, 5 and 20 are counterfactual M* services (`exps/configs/mstar/pi05_bfloat16_steps{N}.yaml`, identical to the shipped 10-step file except `num_flow_steps`). vLLM-Omni and SGLang core take the count per request, so their configs stay at 10 (verified in source: vLLM-Omni `sampling_params.num_inference_steps`, SGLang core payload `num_inference_steps`). Latency only: the actions are shape-checked, not compared across step counts. The 1-step point is a timing point, not a usable policy.

Campaigns: `PI05_STEPS_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row-s1 | main | 2 | 24 | 62.1 [61.9–62.2] | 100% | 61 | 1,785 | - |
| mstar | droid-46row-s5 | main | 2 | 24 | 74.1 [73.1–75.2] | 100% | 58 | 1,785 | - |
| mstar | droid-46row | main | 2 | 24 | 94.3 [94.0–94.3] | 100% | 58 | 1,785 | - |
| mstar | droid-46row-s20 | main | 2 | 24 | 130 [128–131] | 100% | 59 | 1,785 | - |
| sglang-core | droid-46row-s1 | main | 2 | 24 | 44.3 [43.9–44.6] | 100% | 59 | 1,785 | - |
| sglang-core | droid-46row-s5 | main | 2 | 24 | 93.4 [90.7–96.9] | 100% | 58 | 1,785 | - |
| sglang-core | droid-46row | main | 2 | 24 | 154 [153–154] | 100% | 57 | 1,785 | - |
| sglang-core | droid-46row-s20 | main | 2 | 24 | 263 [252–268] | 100% | 58 | 1,785 | - |
| vllm-omni | droid-46row-s1 | main | 2 | 24 | 91.2 [89.8–107.0] | 100% | 59 | 1,785 | - |
| vllm-omni | droid-46row-s5 | main | 2 | 24 | 137 [136–137] | 100% | 59 | 1,785 | - |
| vllm-omni | droid-46row | main | 2 | 24 | 190 [188–193] | 100% | 58 | 1,785 | - |
| vllm-omni | droid-46row-s20 | main | 2 | 24 | 296 [295–300] | 100% | 58 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| droid-46row-s1 | mstar | main | 62.1 [61.9–62.2] | 1.00x |
| droid-46row-s1 | sglang-core | main | 44.3 [43.9–44.6] | 0.71x |
| droid-46row-s1 | vllm-omni | main | 91.2 [89.8–107.0] | 1.47x |
| droid-46row-s1 | **strongest baseline: sglang-core** | | 44.3 [43.9–44.6] | **0.71x** |
| droid-46row-s5 | mstar | main | 74.1 [73.1–75.2] | 1.00x |
| droid-46row-s5 | sglang-core | main | 93.4 [90.7–96.9] | 1.26x |
| droid-46row-s5 | vllm-omni | main | 137 [136–137] | 1.85x |
| droid-46row-s5 | **strongest baseline: sglang-core** | | 93.4 [90.7–96.9] | **1.26x** |
| droid-46row | mstar | main | 94.3 [94.0–94.3] | 1.00x |
| droid-46row | sglang-core | main | 154 [153–154] | 1.63x |
| droid-46row | vllm-omni | main | 190 [188–193] | 2.01x |
| droid-46row | **strongest baseline: sglang-core** | | 154 [153–154] | **1.63x** |
| droid-46row-s20 | mstar | main | 130 [128–131] | 1.00x |
| droid-46row-s20 | sglang-core | main | 263 [252–268] | 2.03x |
| droid-46row-s20 | vllm-omni | main | 296 [295–300] | 2.29x |
| droid-46row-s20 | **strongest baseline: sglang-core** | | 263 [252–268] | **2.03x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
