# Concurrency curves

## QWEN_T2S

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | 1 | 1 | 247 | 4.06 | 12.56 | 0.071 | 1.00 |
| mstar | 4 | 1 | 541 | 7.40 | 26.01 | 0.139 | 1.82 |
| mstar | 16 | 1 | 834 | 19.17 | 68.48 | 0.207 | 4.73 |
| sglang-omni | 1 | 1 | 388 | 2.58 | 7.79 | 0.128 | 1.00 |
| sglang-omni | 4 | 1 | 559 | 7.15 | 26.33 | 0.139 | 2.77 |
| sglang-omni | 16 | 1 | 993 | 16.11 | 59.62 | 0.229 | 6.25 |
| vllm-omni | 1 | 1 | 456 | 2.19 | 7.15 | 0.138 | 1.00 |
| vllm-omni | 4 | 1 | 1,003 | 3.99 | 13.64 | 0.248 | 1.82 |
| vllm-omni | 16 | 1 | 1,344 | 11.91 | 41.98 | 0.317 | 5.43 |

M* requests/s divided by the other framework's (>1: M* higher throughput):

- B=1: vs sglang-omni 1.57×, vs vllm-omni 1.85×
- B=4: vs sglang-omni 1.03×, vs vllm-omni 1.85×
- B=16: vs sglang-omni 1.19×, vs vllm-omni 1.61×

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
