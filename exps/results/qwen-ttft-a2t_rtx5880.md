| framework | tiny TTFT ms (fixed) | 1k TTFT ms | prompt tokens 1k | prefill, ms per 1k tokens | A2T TTFT ms | A2T input tokens | A2T excess over text prefill ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | - | - | - | - | 93 | - | - |
| sglang-omni | - | - | - | - | 92 | 188 | - |
| vllm-omni | - | - | - | - | 79 | 188 | - |

Token counts are the server's reported prompt tokens when the client recorded them (`-` otherwise: the per-token columns need them). Single GPU set, C=1; a linear model of prefill in prompt length, adequate between ~15 and ~1,000 tokens.
