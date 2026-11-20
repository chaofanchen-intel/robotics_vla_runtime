# Cosmos3-Nano T2I, SGLang-Omni on the 48 GB card: DiT kept resident against the auto offload policy

**Question.** cosmos3-sglang-nsys: on the 5880 the SGLang-Omni run copies about 29 GB of DiT weights host-to-device and back on every request. The server ran the auto memory policy with dit_cpu_offload on (47 GiB free, below the 90 GiB keep-resident threshold). Is the ~2.9 s per request that puts SGLang-Omni at 3.6x M* at 256 this copy, and does pinning the DiT resident bring it near the H200 result?

**Pre-registered reading.** Same protocol as cosmos3-size, same session: auto arm (the default config) against resident arm (`--generation.factory.server_args_overrides.dit_cpu_offload false`; `sgl-omni serve` has no bare `--dit-cpu-offload` flag). If the copy is the cost, the resident arm drops by about the same ~2.9 s at every size (a constant term, not a compute term), and the SGLang / M* ratio moves toward the H200 range (0.9-1.0). Read dit_cpu_offload in each start's server.log.

**Caveats.** 5880 only. The resident arm needs the 27 GB DiT plus activations inside 48 GB; a 1,024 failure is a result. This card throttles (sw_power_cap) during these runs, as in cosmos3-size.

Campaigns: `COSMOS3_SGLANG_RESIDENT_AUTO_01_B2B` (auto, back-to-back), `COSMOS3_SGLANG_RESIDENT_RESIDENT_01_B2B` (resident, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs auto |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | t2i-256 | auto | 2 | 6 | 4,748 [4,741–4,754] | 0% | 40 | 2,645 | - |
| sglang-omni | t2i-256 | resident | 2 | 6 | 2,062 [2,059–2,068] | 0% | 41 | 2,628 | 0.43x |
| sglang-omni | t2i-512 | auto | 2 | 6 | 6,774 [6,753–6,788] | 0% | 41 | 1,777 | - |
| sglang-omni | t2i-512 | resident | 2 | 6 | 4,082 [4,061–4,094] | 0% | 43 | 1,405 | 0.60x |
| sglang-omni | t2i-1024 | auto | 2 | 6 | 17,280 [17,244–17,317] | 0% | 50 | 1,399 | - |
| sglang-omni | t2i-1024 | resident | 2 | 6 | 14,604 [14,565–14,643] | 0% | 51 | 1,225 | 0.85x |
