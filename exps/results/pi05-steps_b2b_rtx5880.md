# Pi0.5 at 1 / 5 / 10 / 20 flow steps: how much of each framework's latency is the denoise loop?

**Question.** baseline-pi05 runs 10 flow-matching steps, so its ratios say nothing about the denoise loop. Each step is one action-expert forward on the prefix KV, so E2E = fixed + steps x per-step. Which framework pays more per step, and which pays more fixed cost (images, prefix, request path)? Does M*'s lead come from the step, the fixed part, or both?

**Pre-registered reading.** Per case (step count), the ratio best baseline / M* as in baseline-pi05, pooled over starts with the per-start range beside it. Then the slope per framework: (p50 at 20 steps - p50 at 1 step) / 19. A lead that grows with the step count is a per-step lead; a lead the same at 1 and at 20 steps is a fixed-cost lead. The intercept (p50 at 1 step minus one slope) is the fixed cost.

**Caveats.** Steps 1, 5 and 20 are counterfactual M* services (`exps/configs/mstar/pi05_bfloat16_steps{N}.yaml`, identical to the shipped 10-step file except `num_flow_steps`). vLLM-Omni and SGLang core take the count per request, so their configs stay at 10 (verified in source: vLLM-Omni `sampling_params.num_inference_steps`, SGLang core payload `num_inference_steps`). Latency only: the actions are shape-checked, not compared across step counts. The 1-step point is a timing point, not a usable policy.

Campaigns: `PI05_STEPS_MAIN_02_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row-s1 | main | 2 | 24 | 71.6 [70.0–73.9] | 0% | 37 | 2,460 | - |
| mstar | droid-46row-s5 | main | 2 | 24 | 116 [89–156] | 0% | 40 | 2,230 | - |
| mstar | droid-46row | main | 2 | 24 | 99.9 [98.5–100.8] | 0% | 37 | 2,460 | - |
| mstar | droid-46row-s20 | main | 2 | 24 | 122 [122–122] | 0% | 44 | 2,460 | - |
| sglang-core | droid-46row-s1 | main | 2 | 24 | 52.2 [51.7–52.7] | 0% | 34 | 2,186 | - |
| sglang-core | droid-46row-s5 | main | 2 | 24 | 96.5 [96.1–98.3] | 0% | 34 | 2,198 | - |
| sglang-core | droid-46row | main | 2 | 24 | 161 [160–163] | 0% | 41 | 2,174 | - |
| sglang-core | droid-46row-s20 | main | 2 | 24 | 307 [307–308] | 0% | 35 | 2,209 | - |
| vllm-omni | droid-46row-s1 | main | 2 | 24 | 90.7 [90.4–91.1] | 0% | 34 | 2,295 | - |
| vllm-omni | droid-46row-s5 | main | 2 | 24 | 145 [143–146] | 0% | 35 | 2,434 | - |
| vllm-omni | droid-46row | main | 2 | 24 | 211 [206–213] | 0% | 37 | 2,529 | - |
| vllm-omni | droid-46row-s20 | main | 2 | 24 | 346 [340–349] | 0% | 37 | 2,328 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| droid-46row-s1 | mstar | main | 71.6 [70.0–73.9] | 1.00x |
| droid-46row-s1 | sglang-core | main | 52.2 [51.7–52.7] | 0.73x |
| droid-46row-s1 | vllm-omni | main | 90.7 [90.4–91.1] | 1.27x |
| droid-46row-s1 | **strongest baseline: sglang-core** | | 52.2 [51.7–52.7] | **0.73x** |
| droid-46row-s5 | mstar | main | 116 [89–156] | 1.00x |
| droid-46row-s5 | sglang-core | main | 96.5 [96.1–98.3] | 0.83x |
| droid-46row-s5 | vllm-omni | main | 145 [143–146] | 1.24x |
| droid-46row-s5 | **strongest baseline: sglang-core** | | 96.5 [96.1–98.3] | **0.83x** |
| droid-46row | mstar | main | 99.9 [98.5–100.8] | 1.00x |
| droid-46row | sglang-core | main | 161 [160–163] | 1.61x |
| droid-46row | vllm-omni | main | 211 [206–213] | 2.11x |
| droid-46row | **strongest baseline: sglang-core** | | 161 [160–163] | **1.61x** |
| droid-46row-s20 | mstar | main | 122 [122–122] | 1.00x |
| droid-46row-s20 | sglang-core | main | 307 [307–308] | 2.51x |
| droid-46row-s20 | vllm-omni | main | 346 [340–349] | 2.83x |
| droid-46row-s20 | **strongest baseline: sglang-core** | | 307 [307–308] | **2.51x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
