| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| QWEN_FAIR_TXT_B1152_SGLANG_OFF_01_B2B | sglang-omni | txt-1k | 10 | 256 | 1,596 | 119 | 5.82 | 5.79 |
| QWEN_FAIR_TXT_B1152_SGLANG_OFF_01_B2B | sglang-omni | txt-1k-1tok | 10 | 1 | 120 | 120 | - | - |
| QWEN_FAIR_TXT_B1152_VLLM_OFF_01_B2B | vllm-omni | txt-1k | 10 | 256 | 1,613 | 101 | 5.96 | 5.93 |
| QWEN_FAIR_TXT_B1152_VLLM_OFF_01_B2B | vllm-omni | txt-1k-1tok | 10 | 1 | 101 | 101 | - | - |
| QWEN_FAIR_TXT_B1152_MSTAR_OFF_01_B2B | mstar | txt-1k | 10 | 256 | 1,645 | 148 | 5.85 | 5.87 |
| QWEN_FAIR_TXT_B1152_MSTAR_OFF_01_B2B | mstar | txt-1k-1tok | 10 | 2 | 157 | 147 | 8.13 | 9.26 |
| QWEN_FAIR_TXT_B1152_MSTAR_B1152_01_B2B | mstar | txt-1k | 10 | 256 | 1,612 | 113 | 5.86 | 5.88 |
| QWEN_FAIR_TXT_B1152_MSTAR_B1152_01_B2B | mstar | txt-1k-1tok | 10 | 2 | 123 | 113 | 8.32 | 9.54 |
