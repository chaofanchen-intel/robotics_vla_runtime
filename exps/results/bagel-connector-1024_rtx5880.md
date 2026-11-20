| campaign | framework | case | n | E2E ms | KV logged requests | KV MB per request | serialise ms | transfer ms | sender-side total ms | MB/s (bytes / transfer) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BAGEL_CONNECTOR_1024_SHM_01_B2B | vllm-omni-two-stage | i2i-1024-steps5 | 10 | 16,178 | 12 | 1,033 | 1,408 | 1,108 | 2,516 | 933 |
| BAGEL_CONNECTOR_1024_MOONCAKE_TCP_01_B2B | vllm-omni-two-stage | i2i-1024-steps5 | 10 | 12,764 | 12 | 1,033 | 2 | 42 | 44 | 24,601 |
| BAGEL_CONNECTOR_1024_NIXL_01_B2B | vllm-omni-two-stage | i2i-1024-steps5 | 10 | 14,085 | 12 | 1,033 | 2 | 1 | 3 | 1,033,243 |
| BAGEL_CONNECTOR_1024_SINGLE_STAGE_01_B2B | vllm-omni-single-stage | i2i-1024-steps5 | 10 | 9,897 | 0 | - | - | - | - | - |

serialise / transfer are the sender-side times vLLM-Omni logs, summed over a request's branches; the receiver's time is not logged, so compare E2E (and the single-stage row, which has no hand-off) for the full cost. Single request at a time, one pair of cards without NVLink or RDMA.
