| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BAGEL_ASYNC_SCHED_VLLM_ASYNC_ON_01_B2B | vllm-omni-two-stage | i2t-512 | 10 | 128 | 2,982 | 807 | 17.16 | 17.12 |
| BAGEL_ASYNC_SCHED_VLLM_ASYNC_OFF_01_B2B | vllm-omni-two-stage | i2t-512 | 10 | 128 | 3,123 | 801 | 18.24 | 18.28 |
| BAGEL_ASYNC_SCHED_MSTAR_SPEC_ON_01_B2B | mstar | i2t-512 | 10 | 128 | 2,284 | 130 | 16.94 | 16.97 |
| BAGEL_ASYNC_SCHED_MSTAR_SPEC_OFF_01_B2B | mstar | i2t-512 | 10 | 128 | 2,599 | 130 | 19.42 | 19.44 |
