# Concurrency curves

## QWEN_T2S

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | 1 | 2 | 303 | 3.30 | 12.99 | 0.080 | 1.00 |
| mstar | 4 | 2 | 464 | 8.63 | 28.01 | 0.135 | 2.61 |
| mstar | 16 | 2 | 858 | 18.67 | 70.12 | 0.214 | 5.66 |
| sglang-omni | 1 | 2 | 334 | 3.00 | 9.77 | 0.097 | 1.00 |
| sglang-omni | 4 | 2 | 521 | 7.68 | 27.49 | 0.139 | 2.56 |
| sglang-omni | 16 | 2 | 1,025 | 15.61 | 58.01 | 0.241 | 5.21 |
| vllm-omni | 1 | 2 | 550 | 1.82 | 6.38 | 0.155 | 1.00 |
| vllm-omni | 4 | 2 | 880 | 4.54 | 14.18 | 0.225 | 2.49 |
| vllm-omni | 16 | 2 | 1,396 | 11.47 | 42.98 | 0.316 | 6.29 |

M* requests/s divided by the other framework's (>1: M* higher throughput):

- B=1: vs sglang-omni 1.10×, vs vllm-omni 1.81×
- B=4: vs sglang-omni 1.12×, vs vllm-omni 1.90×
- B=16: vs sglang-omni 1.20×, vs vllm-omni 1.63×

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
