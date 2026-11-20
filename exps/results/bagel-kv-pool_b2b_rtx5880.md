# BAGEL on M*, one 48 GiB card: KV pool 1,024 pages against the shipped 512, at one request at a time

**Question.** The 5880 runs M* BAGEL with a 512-page KV pool (~65k tokens) where the H200 default is 2,048 pages. Does a pool twice as big change the latency of one request at a time (1024x1024 three-branch T2I, 512 I2I, 512 I2T)?

**Pre-registered reading.** E2E p50 per case, kv1024 against the two shipped-pool arms around it (A1 / A2 is the drift check). kv1024 inside the A1-A2 spread: the pool is not limiting one request, so the smaller 5880 advantage is not a pool effect. kv1024 faster than both by more than the spread: the 512-page pool is limiting something at B = 1 (eviction or allocation waits; read server.log). kv1024 failing on memory: the 1,024-page pool does not fit beside the weights on this card, which is a memory limit, not a latency one.

**Caveats.** B = 1 only: the pool matters more at B = 4 / 16, which this does not run. The 2,048-page default is not run (its config comment says it does not fit beside the weights). 1,024 pages add ~3.75 GB to the ~12 GB the log reports free after graph capture, so a 1,024-pixel request could still fail on activation memory.

Campaigns: `BAGEL_KV_POOL_A1_01_B2B` (A1, back-to-back), `BAGEL_KV_POOL_KV1024_01_B2B` (kv1024, back-to-back), `BAGEL_KV_POOL_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-1024-cfg3 | A1 | 2 | 6 | 83,505 [83,478–83,531] | 0% | 69 | 1,051 | - |
| mstar | t2i-1024-cfg3 | kv1024 | 2 | 6 | 83,336 [83,283–83,388] | 0% | 69 | 1,057 | 1.00x † |
| mstar | t2i-1024-cfg3 | A2 | 2 | 6 | 83,297 [83,259–83,336] | 0% | 68 | 1,053 | 1.00x |
| mstar | i2i-512 | A1 | 2 | 6 | 25,225 [25,201–25,250] | 0% | 63 | 1,151 | - |
| mstar | i2i-512 | kv1024 | 2 | 6 | 25,189 [25,172–25,206] | 0% | 64 | 1,155 | 1.00x † |
| mstar | i2i-512 | A2 | 2 | 6 | 25,107 [25,020–25,193] | 0% | 62 | 1,164 | 1.00x |
| mstar | i2t-512 | A1 | 2 | 6 | 2,294 [2,285–2,296] | 0% | 52 | 2,434 | - |
| mstar | i2t-512 | kv1024 | 2 | 6 | 2,291 [2,291–2,294] | 0% | 50 | 2,526 | 1.00x † |
| mstar | i2t-512 | A2 | 2 | 6 | 2,286 [2,284–2,290] | 0% | 52 | 2,385 | 1.00x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar t2i-1024-cfg3 0%; mstar i2i-512 0%; mstar i2t-512 0%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
