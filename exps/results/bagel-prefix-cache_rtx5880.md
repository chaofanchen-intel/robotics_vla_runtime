| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BAGEL_PREFIX_CACHE_M_CACHE_ON_01_B2B | mstar | i2i-512 | 10 | - | 23,528 | - | - | - |
| BAGEL_PREFIX_CACHE_M_CACHE_ON_01_B2B | mstar | i2t-512 | 10 | 128 | 2,293 | 138 | 16.93 | 16.97 |
| BAGEL_PREFIX_CACHE_M_CACHE_OFF_01_B2B | mstar | i2i-512 | 10 | - | 23,583 | - | - | - |
| BAGEL_PREFIX_CACHE_M_CACHE_OFF_01_B2B | mstar | i2t-512 | 10 | 128 | 2,934 | 777 | 16.94 | 16.98 |
| BAGEL_PREFIX_CACHE_VLLM_01_B2B | vllm-omni-two-stage | i2i-512 | 10 | 3 | 28,230 | - | - | - |
| BAGEL_PREFIX_CACHE_VLLM_01_B2B | vllm-omni-two-stage | i2t-512 | 10 | 128 | 2,979 | 805 | 17.15 | 17.12 |
