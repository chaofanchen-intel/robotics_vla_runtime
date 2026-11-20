# BAGEL I2I two-stage, 5 steps: millisecond timeline of one request per connector (and two UCX transports for NIXL)

**Question.** bagel-connector showed Mooncake-TCP saves 2.25 s of the 3.9 s two-stage gap to single-stage and NIXL 1.5 s; 1.6 s (Mooncake) and 2.4 s (NIXL) are still unexplained, and the log stamps there have one-second resolution. Where does a request's time go (client -> engine, Thinker, send, orchestrator gap, DiT side), and does NIXL's UCX pick a slow path (host-staged TCP) where cuda_ipc exists? nvidia-smi topo -p2p r reports P2P OK between every pair of cards.

**Pre-registered reading.** analyze_i2i_timeline.py: the segment that differs between connectors is the cost of the connector; the segment that is the same across all arms (stage 0, pre) is not. nixl-cuda-ipc faster than nixl means UCX's default transport choice, not NIXL, was the limit; equal to nixl means the default already uses it. `DiT side` includes the receiver's KV load, so a difference there is the receiver cost. Neither connector is expected to remove stage 0 (three separate Thinker prefills), which stays in every two-stage arm.

**Caveats.** Single-stage has no KV markers in its log and is read from bagel-connector. UCX_TLS names the transports UCX may use; an unavailable one is dropped silently, so a nixl-cuda-ipc that equals nixl is a result, not a failed switch (the arm's server.log would show UCX's own transport lines at higher log level).

Campaigns: `BAGEL_CONNECTOR_TIMELINE_SHM_01_B2B` (shm, back-to-back), `BAGEL_CONNECTOR_TIMELINE_MOONCAKE_TCP_01_B2B` (mooncake-tcp, back-to-back), `BAGEL_CONNECTOR_TIMELINE_NIXL_01_B2B` (nixl, back-to-back), `BAGEL_CONNECTOR_TIMELINE_NIXL_CUDA_IPC_01_B2B` (nixl-cuda-ipc, back-to-back), `BAGEL_CONNECTOR_TIMELINE_NIXL_TCP_01_B2B` (nixl-tcp, back-to-back), `BAGEL_CONNECTOR_TIMELINE_MOONCAKE_TCP_PREFIX_01_B2B` (mooncake-tcp-prefix, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs shm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2i-512-steps5 | shm | 2 | 10 | 6,862 [6,823–6,870] | 0% | 37 | 2,266 | - |
| vllm-omni-two-stage | i2i-512-steps5 | mooncake-tcp | 2 | 10 | 4,542 [4,535–4,548] | 0% | 39 | 2,188 | 0.66x |
| vllm-omni-two-stage | i2i-512-steps5 | nixl | 2 | 10 | 5,701 [5,688–5,721] | 0% | 38 | 2,196 | 0.83x |
| vllm-omni-two-stage | i2i-512-steps5 | nixl-cuda-ipc | 2 | 10 | 5,486 [5,281–5,704] | 0% | 38 | 2,175 | 0.80x |
| vllm-omni-two-stage | i2i-512-steps5 | nixl-tcp | 2 | 10 | 5,276 [5,266–5,288] | 0% | 38 | 2,169 | 0.77x |
| vllm-omni-two-stage | i2i-512-steps5 | mooncake-tcp-prefix | 2 | 10 | 3,528 [3,520–3,535] | 0% | 39 | 2,438 | 0.51x |
