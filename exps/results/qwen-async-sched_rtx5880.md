| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |
|---|---|---|---:|---:|---:|---:|---:|---:|
| QWEN_ASYNC_SCHED_VLLM_ASYNC_ON_01_B2B | vllm-omni | txt-1k | 10 | 256 | 1,613 | 104 | 5.92 | 5.92 |
| QWEN_ASYNC_SCHED_VLLM_ASYNC_OFF_01_B2B | vllm-omni | txt-1k | 10 | 256 | 1,956 | 98 | 7.25 | 7.29 |
| QWEN_ASYNC_SCHED_MSTAR_SPEC_ON_01_B2B | mstar | txt-1k | 10 | 256 | 1,643 | 147 | 5.85 | 5.87 |
| QWEN_ASYNC_SCHED_MSTAR_SPEC_OFF_01_B2B | mstar | txt-1k | 10 | 256 | 2,166 | 152 | 7.86 | 7.90 |
