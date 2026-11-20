# BAGEL CFG-parallel, three branches: M* and vLLM-Omni on the same tensor transport (B=1, 512 and 1024)

**Question.** baseline-bagel-cfgpar pairs M*'s shared-memory tensor manager with vLLM-Omni's shared-memory Thinker -> DiT connector. Does M* move when its tensors go over Mooncake TCP, and does vLLM-Omni move when its one stage edge does?

**Pre-registered reading.** Per case, E2E by arm. Same-transport pairs: mstar-shm with vllm-shm, and mstar-tcp with vllm-tcp. The branch exchange is not switched: vLLM-Omni all_gathers the branches with NCCL inside its DiT group, and M* hands branch tensors over its tensor manager. So the transport arms cover the Thinker -> DiT hand-off on vLLM-Omni and M*'s own tensor hand-off, not the branch exchange.

**Caveats.** Needs 3 cards for M* and 4 for vLLM-Omni two-stage on 48 GiB cards. No NVLink on this box: all branch traffic crosses PCIe.

Campaigns: `TRANSPORT_BAGEL_CFG_MSTAR_SHM_01_B2B` (mstar-shm, back-to-back), `TRANSPORT_BAGEL_CFG_MSTAR_TCP_01_B2B` (mstar-tcp, back-to-back), `TRANSPORT_BAGEL_CFG_VLLM_SHM_01_B2B` (vllm-shm, back-to-back), `TRANSPORT_BAGEL_CFG_VLLM_TCP_01_B2B` (vllm-tcp, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar-shm |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-1024-cfg3 | mstar-shm | 2 | 6 | 30,446 [30,352–30,540] | 0% | 60 | 1,152 | - |
| mstar | t2i-1024-cfg3 | mstar-tcp | 2 | 6 | 30,636 [30,577–30,695] | 0% | 60 | 1,177 | 1.01x |
| mstar | i2i-1024 | mstar-shm | 2 | 6 | 37,396 [37,393–37,399] | 0% | 63 | 1,240 | - |
| mstar | i2i-1024 | mstar-tcp | 2 | 6 | 38,181 [38,070–38,292] | 0% | 62 | 1,277 | 1.02x |
| mstar | t2i-512-cfg3 | mstar-shm | 2 | 6 | 7,980 [7,975–7,986] | 0% | 55 | 1,343 | - |
| mstar | t2i-512-cfg3 | mstar-tcp | 2 | 6 | 7,991 [7,987–7,994] | 0% | 52 | 1,397 | 1.00x |
| mstar | i2i-512 | mstar-shm | 2 | 6 | 9,665 [9,633–9,698] | 0% | 53 | 1,445 | - |
| mstar | i2i-512 | mstar-tcp | 2 | 6 | 10,097 [10,000–10,211] | 0% | 52 | 1,565 | 1.04x |
| vllm-omni-two-stage | t2i-1024-cfg3 | vllm-shm | 2 | 6 | 30,520 [30,496–30,543] | 0% | 32 | 1,534 | - |
| vllm-omni-two-stage | t2i-1024-cfg3 | vllm-tcp | 2 | 6 | 30,440 [30,431–30,450] | 0% | 31 | 1,527 | - |
| vllm-omni-two-stage | i2i-1024 | vllm-shm | 2 | 6 | 46,827 [46,716–46,868] | 0% | 33 | 1,300 | - |
| vllm-omni-two-stage | i2i-1024 | vllm-tcp | 2 | 6 | 43,708 [43,545–43,872] | 0% | 32 | 1,254 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | vllm-shm | 2 | 6 | 7,821 [7,806–7,836] | 0% | 34 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | vllm-tcp | 2 | 6 | 7,862 [7,852–7,871] | 0% | 34 | 2,460 | - |
| vllm-omni-two-stage | i2i-512 | vllm-shm | 2 | 6 | 16,362 [16,096–16,446] | 0% | 35 | 2,337 | - |
| vllm-omni-two-stage | i2i-512 | vllm-tcp | 2 | 6 | 13,871 [13,844–13,899] | 0% | 35 | 2,323 | - |
