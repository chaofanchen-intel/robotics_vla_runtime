# BAGEL I2I two-stage at 1024^2, 5 steps: does the connector saving grow with the KV size?

**Question.** At 512^2 Mooncake-TCP saves 2.25 s per request and the earlier 1024^2 gap to M* was 1.36x against 1.18x at 512^2. The KV is ~4x larger at 1024^2 (about 1.3 GB per branch, if it scales with tokens). Does the saving scale with the KV, and is the 1.36x also mostly the default connector?

**Pre-registered reading.** E2E per arm. saving(1024) / saving(512) near 4 = the hand-off cost scales with bytes and the connector is the lever at every size; near 1 = a fixed orchestration cost the connector does not touch. The Mooncake arm must not log a pool / put failure (4 GiB pinned pool vs ~2.7 GB of in-flight KV): read the server.log before trusting it. vLLM-Omni only: M* ignores request-level steps, compare against the 50-step M* number from bagel-i2i-steps.

**Caveats.** Single request at a time on one pair of cards; one fixture. 1024^2 is explicit-only elsewhere because a long request crosses the thermal knee: with 5 steps a request is ~10-20 s.

Campaigns: `BAGEL_CONNECTOR_1024_SHM_01_B2B` (shm, back-to-back), `BAGEL_CONNECTOR_1024_MOONCAKE_TCP_01_B2B` (mooncake-tcp, back-to-back), `BAGEL_CONNECTOR_1024_NIXL_01_B2B` (nixl, back-to-back), `BAGEL_CONNECTOR_1024_SINGLE_STAGE_01_B2B` (single-stage, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs shm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2i-1024-steps5 | shm | 2 | 10 | 16,178 [16,178–16,178] | 0% | 39 | 2,320 | - |
| vllm-omni-two-stage | i2i-1024-steps5 | mooncake-tcp | 2 | 10 | 12,764 [12,740–12,803] | 0% | 39 | 2,289 | 0.79x |
| vllm-omni-two-stage | i2i-1024-steps5 | nixl | 2 | 10 | 14,085 [13,922–14,495] | 0% | 39 | 2,282 | 0.87x |
| vllm-omni-single-stage | i2i-1024-steps5 | single-stage | 2 | 10 | 9,897 [9,821–9,938] | 0% | 53 | 1,480 | - |
