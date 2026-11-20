# Pi0.5: does a prefix cache change the action latency? SGLang core cache on / off, M*'s default, vLLM-Omni as control

**Question.** A Pi0.5 request carries three images, the prompt and the discretised state; a prefix cache reuses the prefix KV only when a later request repeats it. The 46-row fixture never repeats a row, so this case sends row 0 on every request of a start. Is SGLang core's global prefix cache on in its shipped config (no: enable_global_prefix_cache=false), and does M*'s default KV prefix cache change Pi0.5 latency (source says it never opens for Pi0.5)?

**Pre-registered reading.** sglang-core against sglang-cache on the same row: a lower p50 for sglang-cache is the prefix cache, and the difference is the saving per request. The log line shows the setting took effect, not that it hit: a hit shows as a fall after the first request of each start, so read the per-request column. M*: its Pi0.5 prefix cache is not opened (no keyed KV label, no checkpoint name, no prefix_cache_salt), so its row-0 latency should match its 46-row latency; a fall on repeated rows would mean the source reading is wrong. vLLM-Omni has no Pi0.5 prefix cache and is the control for a repeated row.

**Caveats.** No M* cache-off arm: M*'s pi05 prefix cache is closed by construction, so an off arm would be the same configuration. The shipped SGLang config has enable_global_prefix_cache=false; the cache also needs conditioning_cache_max_size_mb > 0 (default 512). The `Pi05 memory config` log lines are source-verified (sglang multimodal_gen pipelines/pi05.py), not yet seen in a server.log: the first run confirms them.

Campaigns: `PI05_KV_MSTAR_02_B2B` (mstar, back-to-back), `PI05_KV_SGLANG_CORE_02_B2B` (sglang-core, back-to-back), `PI05_KV_SGLANG_CACHE_02_B2B` (sglang-cache, back-to-back), `PI05_KV_VLLM_OMNI_02_B2B` (vllm-omni, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-row0-x20 | mstar | 2 | 40 | 97.9 [96.0–99.1] | 0% | 40 | 2,460 | - |
| sglang-core | droid-row0-x20 | sglang-core | 2 | 40 | 91.5 [89.1–92.4] | 0% | 35 | 2,178 | - |
| sglang-core | droid-row0-x20 | sglang-cache | 2 | 40 | 50.3 [49.8–50.4] | 0% | 33 | 2,460 | - |
| vllm-omni | droid-row0-x20 | vllm-omni | 2 | 40 | 210 [209–211] | 0% | 35 | 2,502 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| droid-row0-x20 | mstar | mstar | 97.9 [96.0–99.1] | 1.00x |
| droid-row0-x20 | sglang-core | sglang-cache | 50.3 [49.8–50.4] | 0.51x |
| droid-row0-x20 | vllm-omni | vllm-omni | 210 [209–211] | 2.14x |
| droid-row0-x20 | **strongest baseline: sglang-core** | | 50.3 [49.8–50.4] | **0.51x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
