# Concurrency curves

## QWEN_T2S

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | 1 | 2 | 364 | 2.74 | 10.47 | 0.094 | 1.00 |
| mstar | 4 | 2 | 584 | 6.85 | 22.98 | 0.169 | 2.49 |
| mstar | 16 | 2 | 1,378 | 11.61 | 43.08 | 0.328 | 4.23 |
| sglang-omni | 1 | 2 | 417 | 2.40 | 8.79 | 0.110 | 1.00 |
| sglang-omni | 4 | 2 | 594 | 6.73 | 21.27 | 0.161 | 2.80 |
| sglang-omni | 16 | 2 | 1,202 | 13.31 | 50.28 | 0.265 | 5.55 |
| vllm-omni | 1 | 2 | 517 | 1.93 | 6.53 | 0.149 | 1.00 |
| vllm-omni | 4 | 2 | 880 | 4.54 | 15.59 | 0.225 | 2.35 |
| vllm-omni | 16 | 2 | 1,744 | 9.17 | 34.10 | 0.378 | 4.74 |

M* requests/s divided by the other framework's (>1: M* higher throughput):

- B=1: vs sglang-omni 1.14×, vs vllm-omni 1.42×
- B=4: vs sglang-omni 1.02×, vs vllm-omni 1.51×
- B=16: vs sglang-omni 0.87×, vs vllm-omni 1.27×

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
