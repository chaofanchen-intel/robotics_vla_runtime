| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BAGEL_I2T_SINGLE_FAIR_SINGLE_STAGE_01_B2B | vllm-omni-single-stage | i2t-512 | 10 | 128 | 6,020 | 6,020 | - | 0.00 |
| BAGEL_I2T_SINGLE_FAIR_M_FAIR_01_B2B | mstar | i2t-512 | 10 | 128 | 2,296 | 144 | 16.90 | 16.95 |
| BAGEL_I2T_SINGLE_FAIR_M_CACHE_OFF_01_B2B | mstar | i2t-512 | 10 | 128 | 2,923 | 768 | 16.91 | 16.97 |
