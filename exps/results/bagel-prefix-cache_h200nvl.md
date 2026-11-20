| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BAGEL_PREFIX_CACHE_M_CACHE_ON_01 | mstar | i2i-512 | 10 | - | 5,131 | - | - | - |
| BAGEL_PREFIX_CACHE_M_CACHE_ON_01 | mstar | i2t-512 | 10 | 128 | 708 | 105 | 4.70 | 4.76 |
| BAGEL_PREFIX_CACHE_M_CACHE_OFF_01 | mstar | i2i-512 | 10 | - | 5,115 | - | - | - |
| BAGEL_PREFIX_CACHE_M_CACHE_OFF_01 | mstar | i2t-512 | 10 | 128 | 838 | 234 | 4.70 | 4.76 |
| BAGEL_PREFIX_CACHE_VLLM_01 | vllm-omni-two-stage | i2i-512 | 10 | 3 | 10,229 | - | - | - |
| BAGEL_PREFIX_CACHE_VLLM_01 | vllm-omni-two-stage | i2t-512 | 10 | 128 | 995 | 297 | 5.50 | 5.50 |
