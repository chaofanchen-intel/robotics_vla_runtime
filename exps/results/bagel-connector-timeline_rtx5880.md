| campaign | framework | case | n | E2E ms | pre (client to engine) | stage 0 (to first KV branch) | send (all branches) | gap (to DiT add request) | DiT side (KV load + denoise + VAE) | engine (all after pre) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BAGEL_CONNECTOR_TIMELINE_SHM_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 6,862 | 292 | 2,666 | 1,235 | 25 | 2,626 | 6,539 |
| BAGEL_CONNECTOR_TIMELINE_MOONCAKE_TCP_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 4,542 | 274 | 2,188 | 56 | 25 | 1,998 | 4,269 |
| BAGEL_CONNECTOR_TIMELINE_NIXL_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 5,701 | 283 | 2,186 | 31 | 25 | 3,163 | 5,414 |
| BAGEL_CONNECTOR_TIMELINE_NIXL_CUDA_IPC_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 5,486 | 287 | 2,199 | 32 | 25 | 2,960 | 5,205 |
| BAGEL_CONNECTOR_TIMELINE_NIXL_TCP_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 5,276 | 290 | 2,191 | 31 | 25 | 2,745 | 4,986 |
| BAGEL_CONNECTOR_TIMELINE_MOONCAKE_TCP_PREFIX_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 10 | 3,528 | 292 | 1,222 | 55 | 23 | 1,939 | 3,239 |

Medians over the measured requests, matched to the log by time (the engine's `CFG expansion` marker falls inside the client's start..end). `DiT side` includes the receiver pulling the KV (NIXL's pull is asynchronous: its sender-side `transfer` is only the submit), so compare it across connectors rather than reading it as pure denoising time; the same denoising runs in every arm.
