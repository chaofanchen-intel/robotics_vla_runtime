| framework | tiny E2E ms (fixed) | 1k E2E ms | prompt tokens 1k | prefill, ms per 1k tokens | A2T E2E ms | A2T input tokens | A2T excess over text prefill ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | - | - | - | - | 101 | - | - |
| sglang-omni | 56 | 118 | 1,072 | 61.0 | 93 | 188 | 28 |
| vllm-omni | 57 | 100 | 1,072 | 41.8 | 80 | 188 | 17 |

Token counts are the server's reported prompt tokens when the client recorded them (`-` otherwise: the per-token columns need them). Single GPU set, C=1; a linear model of prefill in prompt length, adequate between ~15 and ~1,000 tokens.
