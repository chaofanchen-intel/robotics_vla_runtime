# Cosmos3-Nano T2I at 256 / 512 / 1,024: does the M* lead shrink as the GEMMs become compute-bound?

**Question.** At 256x256 a denoise step is weight- and launch-bound (nsys: GPU time per forward equal, vLLM-Omni has 2x the kernels per request); the host-gap and batched-CFG terms of the 2.46x should shrink when the per-step GEMMs grow with the pixel count. The doc says so; no larger size was ever run (only one case existed). How does the ratio move?

**Pre-registered reading.** Per-size ratio vLLM-Omni / M* and SGLang-Omni / M*, with the per-start range. The M* batched-CFG term (1.54x at 256) is a weight-reuse effect and should fall toward 1 as the GEMMs fill the GPU; a ratio that stays ~2x at 1,024 means the lead is not launch-bound. M*'s denoise CUDA graph is capped by graph_max_latent_area (2,000): the larger sizes run eager in M*.

**Caveats.** Same 5 prompts at every size; 1,024x1,024 x 50 steps is slow (tens of seconds per request) so only 3 measured requests per start. A framework that runs out of memory at 1,024 is a result. Image size is verified from the PNG header at every size.

Campaigns: `COSMOS3_SIZE_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | main | 2 | 6 | 1,315 [1,285–1,350] | 0% | 41 | 2,330 | - |
| mstar | t2i-512 | main | 2 | 6 | 3,672 [3,669–3,676] | 0% | 43 | 1,618 | - |
| mstar | t2i-1024 | main | 2 | 6 | 13,517 [13,463–13,571] | 0% | 55 | 1,264 | - |
| vllm-omni | t2i-256 | main | 2 | 6 | 3,092 [3,067–3,106] | 0% | 41 | 2,692 | - |
| vllm-omni | t2i-512 | main | 2 | 6 | 4,056 [4,053–4,060] | 0% | 43 | 1,428 | - |
| vllm-omni | t2i-1024 | main | 2 | 6 | 13,465 [13,403–13,528] | 0% | 53 | 1,282 | - |
| sglang-omni | t2i-256 | main | 2 | 6 | 4,753 [4,744–4,753] | 0% | 40 | 2,678 | - |
| sglang-omni | t2i-512 | main | 2 | 6 | 6,536 [6,535–6,537] | 0% | 43 | 1,952 | - |
| sglang-omni | t2i-1024 | main | 2 | 6 | 16,512 [16,500–16,524] | 0% | 55 | 1,454 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-256 | mstar | main | 1,315 [1,285–1,350] | 1.00x |
| t2i-256 | vllm-omni | main | 3,092 [3,067–3,106] | 2.35x |
| t2i-256 | sglang-omni | main | 4,753 [4,744–4,753] | 3.61x |
| t2i-256 | **strongest baseline: vllm-omni** | | 3,092 [3,067–3,106] | **2.35x** |
| t2i-512 | mstar | main | 3,672 [3,669–3,676] | 1.00x |
| t2i-512 | vllm-omni | main | 4,056 [4,053–4,060] | 1.10x |
| t2i-512 | sglang-omni | main | 6,536 [6,535–6,537] | 1.78x |
| t2i-512 | **strongest baseline: vllm-omni** | | 4,056 [4,053–4,060] | **1.10x** |
| t2i-1024 | mstar | main | 13,517 [13,463–13,571] | 1.00x |
| t2i-1024 | vllm-omni | main | 13,465 [13,403–13,528] | 1.00x |
| t2i-1024 | sglang-omni | main | 16,512 [16,500–16,524] | 1.22x |
| t2i-1024 | **strongest baseline: vllm-omni** | | 13,465 [13,403–13,528] | **1.00x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
