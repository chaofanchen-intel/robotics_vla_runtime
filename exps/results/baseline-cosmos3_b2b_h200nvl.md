# Cosmos3-Nano T2I 256x256 (M* vs vLLM-Omni)

**Question.** Cosmos3-Nano T2I 256x256 (M* vs vLLM-Omni): how do the frameworks compare at equal work, each at its best configuration?

**Pre-registered reading.** Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the per-start range beside it. A ratio is only a claim about the configurations named in the table.

**Caveats.** Equal work (50 steps, 256x256, guidance_interval [400,1000], same prompts). The ~2.4x gap is M*'s design at this launch/weight-bound size, not unequal work (nsys, RTX 5880): M* packs cond+uncond into ONE forward per step (denoise_step_batched_cfg), vLLM-Omni runs two sequential forwards (~90 passes vs ~50) at ~83k vs ~40k kernels per request; GPU time per forward is about equal (~20 ms). Whether the 256x256 step is CUDA-graph replayed is NOT established: M*'s captured tiers are 192x320 and 480x832 (server log), so 256x256 may run eager; `cosmos3-decompose` measures graph / batched CFG / compile one by one. Expect the gap to shrink at larger sizes, where the GEMMs are compute-bound.

Campaigns: `BASELINE_COSMOS3_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | main | 3 | 15 | 1,073 [1,031–1,077] | 100% | 61 | 1,785 | - |
| vllm-omni | t2i-256 | main | 3 | 15 | 1,349 [1,288–1,374] | 100% | 61 | 1,785 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-256 | mstar | main | 1,073 [1,031–1,077] | 1.00x |
| t2i-256 | vllm-omni | main | 1,349 [1,288–1,374] | 1.26x |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
