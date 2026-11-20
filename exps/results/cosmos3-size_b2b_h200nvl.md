# Cosmos3-Nano T2I at 256 / 512 / 1,024: does the M* lead shrink as the GEMMs become compute-bound?

**Question.** At 256x256 a denoise step is weight- and launch-bound (nsys: GPU time per forward equal, vLLM-Omni has 2x the kernels per request); the host-gap and batched-CFG terms of the 2.46x should shrink when the per-step GEMMs grow with the pixel count. The doc says so; no larger size was ever run (only one case existed). How does the ratio move?

**Pre-registered reading.** Per-size ratio vLLM-Omni / M* and SGLang-Omni / M*, with the per-start range. The M* batched-CFG term (1.54x at 256) is a weight-reuse effect and should fall toward 1 as the GEMMs fill the GPU; a ratio that stays ~2x at 1,024 means the lead is not launch-bound. M*'s denoise CUDA graph is capped by graph_max_latent_area (2,000): the larger sizes run eager in M*.

**Caveats.** Same 5 prompts at every size; 1,024x1,024 x 50 steps is slow (tens of seconds per request) so only 3 measured requests per start. A framework that runs out of memory at 1,024 is a result. Image size is verified from the PNG header at every size.

Campaigns: `COSMOS3_SIZE_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | main | 2 | 6 | 1,101 [1,099–1,180] | 100% | 61 | 1,785 | - |
| mstar | t2i-512 | main | 2 | 6 | 1,117 [1,112–1,124] | 100% | 64 | 1,785 | - |
| mstar | t2i-1024 | main | 2 | 6 | 2,778 [2,769–2,794] | 0% | 71 | 1,517 | - |
| vllm-omni | t2i-256 | main | 2 | 6 | 1,302 [1,286–1,310] | 100% | 63 | 1,785 | - |
| vllm-omni | t2i-512 | main | 2 | 6 | 1,457 [1,407–1,508] | 100% | 66 | 1,785 | - |
| vllm-omni | t2i-1024 | main | 2 | 6 | 2,894 [2,888–2,928] | 0% | 69 | 1,542 | - |
| sglang-omni | t2i-256 | main | 2 | 6 | 998 [974–1,034] | 100% | 63 | 1,785 | - |
| sglang-omni | t2i-512 | main | 2 | 6 | 1,070 [1,066–1,074] | 0% | 68 | 1,746 | - |
| sglang-omni | t2i-1024 | main | 2 | 6 | 2,678 [2,677–2,678] | 0% | 73 | 1,458 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-256 | mstar | main | 1,101 [1,099–1,180] | 1.00x |
| t2i-256 | vllm-omni | main | 1,302 [1,286–1,310] | 1.18x |
| t2i-256 | sglang-omni | main | 998 [974–1,034] | 0.91x |
| t2i-256 | **strongest baseline: sglang-omni** | | 998 [974–1,034] | **0.91x** |
| t2i-512 | mstar | main | 1,117 [1,112–1,124] | 1.00x |
| t2i-512 | vllm-omni | main | 1,457 [1,407–1,508] | 1.31x |
| t2i-512 | sglang-omni | main | 1,070 [1,066–1,074] | 0.96x |
| t2i-512 | **strongest baseline: sglang-omni** | | 1,070 [1,066–1,074] | **0.96x** |
| t2i-1024 | mstar | main | 2,778 [2,769–2,794] | 1.00x |
| t2i-1024 | vllm-omni | main | 2,894 [2,888–2,928] | 1.04x |
| t2i-1024 | sglang-omni | main | 2,678 [2,677–2,678] | 0.96x |
| t2i-1024 | **strongest baseline: sglang-omni** | | 2,678 [2,677–2,678] | **0.96x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
