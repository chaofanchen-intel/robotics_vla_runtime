# BAGEL attention backend: probe counters and ratio matrix

## What the worker-side probe counted (M*)

| campaign | case | start | probe procs | pip flash_attn importable | ViT flash calls | ViT SDPA-fallback calls | attention managers |
|---|---|---|---:|---|---:|---:|---|
| BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | i2i-512 | r01 | 3 | [True] | 156 | 0 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | i2i-512 | r02 | 3 | [True] | 156 | 0 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | i2t-512 | r01 | 3 | [True] | 156 | 0 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | i2t-512 | r02 | 3 | [True] | 156 | 0 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | t2i-512-nocfg | r01 | 3 | [True] | 0 | 0 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | t2i-512-nocfg | r02 | 3 | [True] | 0 | 0 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | i2i-512 | r01 | 3 | [False] | 0 | 156 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | i2i-512 | r02 | 3 | [False] | 0 | 156 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | i2t-512 | r01 | 3 | [False] | 0 | 156 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | i2t-512 | r02 | 3 | [False] | 0 | 156 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | t2i-512-nocfg | r01 | 3 | [False] | 0 | 0 | FlashInferManager |
| BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | t2i-512-nocfg | r02 | 3 | [False] | 0 | 0 | FlashInferManager |

## E2E p50 (ms, pooled) and vLLM-Omni / M* for every pairing

| case | BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 | BAGEL_ATTN_BACKEND_VLLM_SDPA_01 |
|---|---:|---:|---:|---:|
| i2i-512 | 5,159 | 5,133 | 10,268 | 13,949 |
| i2t-512 | 732 | 710 | 997 | 997 |
| t2i-512-nocfg | 2,000 | 2,078 | 2,953 | 2,613 |

| case | vLLM arm | M* arm | vLLM / M* |
|---|---|---|---:|
| i2i-512 | BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 | BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | 1.99x |
| i2i-512 | BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 | BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | 2.00x |
| i2i-512 | BAGEL_ATTN_BACKEND_VLLM_SDPA_01 | BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | 2.70x |
| i2i-512 | BAGEL_ATTN_BACKEND_VLLM_SDPA_01 | BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | 2.72x |
| i2t-512 | BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 | BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | 1.36x |
| i2t-512 | BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 | BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | 1.40x |
| i2t-512 | BAGEL_ATTN_BACKEND_VLLM_SDPA_01 | BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | 1.36x |
| i2t-512 | BAGEL_ATTN_BACKEND_VLLM_SDPA_01 | BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | 1.40x |
| t2i-512-nocfg | BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 | BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | 1.48x |
| t2i-512-nocfg | BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 | BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | 1.42x |
| t2i-512-nocfg | BAGEL_ATTN_BACKEND_VLLM_SDPA_01 | BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 | 1.31x |
| t2i-512-nocfg | BAGEL_ATTN_BACKEND_VLLM_SDPA_01 | BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 | 1.26x |
