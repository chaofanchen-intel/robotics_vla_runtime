# BAGEL I2I two-stage: the Thinker -> DiT KV hand-off over SharedMemory vs Mooncake (TCP) vs NIXL (UCX)

**Question.** The two-stage I2I is 4.4 s slower than M* because the Thinker's KV (2 x 340 MB) is serialised to bytes (~0.47 s per branch), copied through shared memory at ~0.9 GB/s and copied back, with ~3.7 s of idle GPU between the stages. vLLM-Omni also ships connectors that carry the GPU tensor itself (Mooncake Transfer Engine, NIXL). Is the hand-off an architectural cost of two stages, or of the default connector? Installed in a user directory by exps/tools/install_omni_connectors.sh, not in the baseline environment.

**Pre-registered reading.** analyze_kv_handoff.py: sender-side serialise + transfer per request, and E2E, per connector. `steps5` exposes the fixed cost (E2E 6.9 s on shm, 2.9 s single-stage); `i2i-512` shows it against the 24-28 s total. mooncake / nixl E2E near single-stage = the gap was the default connector, and 'M* has no stage boundary' is not an architectural advantage; E2E near shm = the gap is elsewhere (receiver side, orchestration, the Thinker's three separate prefills) and the connector is not the lever. A connector that fails to start is a result (read server.log): the box has no RDMA NIC and no NVLink, so Mooncake runs its TCP transport and UCX picks host-staged or PCIe paths between the two cards.

**Caveats.** One pair of cards on PCIe without NVLink; the harness picks the lowest free pair, which may sit on different NUMA nodes. Mooncake uses its TCP transport (no RDMA device). The hand-off cost seen here is the sender side plus whatever the client sees; a faster connector does not change the Thinker's three separate prefills.

Campaigns: `BAGEL_CONNECTOR_SHM_01_B2B` (shm, back-to-back), `BAGEL_CONNECTOR_MOONCAKE_TCP_01_B2B` (mooncake-tcp, back-to-back), `BAGEL_CONNECTOR_NIXL_01_B2B` (nixl, back-to-back), `BAGEL_CONNECTOR_SINGLE_STAGE_01_B2B` (single-stage, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs shm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2i-512-steps5 | shm | 2 | 10 | 6,809 [6,765–6,830] | 0% | 37 | 2,229 | - |
| vllm-omni-two-stage | i2i-512-steps5 | mooncake-tcp | 2 | 10 | 4,558 [4,551–4,563] | 0% | 38 | 2,148 | 0.67x |
| vllm-omni-two-stage | i2i-512-steps5 | nixl | 2 | 10 | 5,308 [5,301–5,322] | 0% | 38 | 2,208 | 0.78x |
| vllm-omni-two-stage | i2i-512 | shm | 2 | 10 | 28,190 [28,188–28,192] | 0% | 35 | 1,814 | - |
| vllm-omni-two-stage | i2i-512 | mooncake-tcp | 2 | 10 | 25,943 [25,891–25,995] | 0% | 35 | 1,814 | 0.92x |
| vllm-omni-two-stage | i2i-512 | nixl | 2 | 10 | 26,832 [26,831–26,833] | 0% | 35 | 1,837 | 0.95x |
| vllm-omni-single-stage | i2i-512-steps5 | single-stage | 2 | 10 | 2,936 [2,926–2,975] | 0% | 46 | 1,793 | - |
| vllm-omni-single-stage | i2i-512 | single-stage | 2 | 10 | 24,122 [24,018–24,227] | 0% | 62 | 1,320 | - |
