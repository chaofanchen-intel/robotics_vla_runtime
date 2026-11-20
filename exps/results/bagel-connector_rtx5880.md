| campaign | framework | case | n | E2E ms | KV logged requests | KV MB per request | serialise ms | transfer ms | sender-side total ms | MB/s (bytes / transfer) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BAGEL_CONNECTOR_SHM_01_B2B | vllm-omni-two-stage | i2i-512 | 10 | 28,190 | 12 | 681 | 948 | 725 | 1,673 | 939 |
| BAGEL_CONNECTOR_SHM_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 6,809 | 12 | 681 | 946 | 726 | 1,672 | 937 |
| BAGEL_CONNECTOR_MOONCAKE_TCP_01_B2B | vllm-omni-two-stage | i2i-512 | 10 | 25,943 | 12 | 681 | 2 | 26 | 28 | 26,189 |
| BAGEL_CONNECTOR_MOONCAKE_TCP_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 4,558 | 12 | 681 | 2 | 26 | 28 | 26,189 |
| BAGEL_CONNECTOR_NIXL_01_B2B | vllm-omni-two-stage | i2i-512 | 10 | 26,832 | 12 | 681 | 2 | 0 | 2 | - |
| BAGEL_CONNECTOR_NIXL_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 5,308 | 12 | 681 | 2 | 0 | 2 | - |
| BAGEL_CONNECTOR_SINGLE_STAGE_01_B2B | vllm-omni-single-stage | i2i-512 | 10 | 24,122 | 0 | - | - | - | - | - |
| BAGEL_CONNECTOR_SINGLE_STAGE_01_B2B | vllm-omni-single-stage | i2i-512-steps5 | 10 | 2,936 | 0 | - | - | - | - | - |

serialise / transfer are the sender-side times vLLM-Omni logs, summed over a request's branches; the receiver's time is not logged, so compare E2E (and the single-stage row, which has no hand-off) for the full cost. Single request at a time, one pair of cards without NVLink or RDMA.
