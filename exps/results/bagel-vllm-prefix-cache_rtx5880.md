| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BAGEL_VLLM_PREFIX_CACHE_VLLM_CACHE_OFF_01_B2B | vllm-omni-two-stage | i2i-512 | 10 | 3 | 28,075 | - | - | - |
| BAGEL_VLLM_PREFIX_CACHE_VLLM_CACHE_OFF_01_B2B | vllm-omni-two-stage | i2t-512 | 10 | 128 | 2,979 | 806 | 17.17 | 17.11 |
| BAGEL_VLLM_PREFIX_CACHE_VLLM_CACHE_ON_01_B2B | vllm-omni-two-stage | i2i-512 | 10 | 3 | 27,152 | - | - | - |
| BAGEL_VLLM_PREFIX_CACHE_VLLM_CACHE_ON_01_B2B | vllm-omni-two-stage | i2t-512 | 10 | 128 | 2,256 | 86 | 17.10 | 17.08 |
| BAGEL_VLLM_PREFIX_CACHE_M_CACHE_ON_01_B2B | mstar | i2i-512 | 10 | - | 23,447 | - | - | - |
| BAGEL_VLLM_PREFIX_CACHE_M_CACHE_ON_01_B2B | mstar | i2t-512 | 10 | 128 | 2,289 | 135 | 16.94 | 16.96 |
