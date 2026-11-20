# Concurrency curves

## BAGEL

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | 1 | 2 | 2,281 | 0.44 | n/a | n/a | 1.00 |
| mstar | 4 | 2 | 2,661 | 1.50 | n/a | n/a | 3.43 |
| mstar | 16 | 2 | 3,640 | 4.40 | n/a | n/a | 10.02 |
| vllm-omni-single-stage | 1 | 2 | 6,028 | 0.17 | n/a | n/a | 1.00 |
| vllm-omni-single-stage | 4 | 2 | 24,226 | 0.17 | n/a | n/a | 1.00 |
| vllm-omni-single-stage | 16 | 2 | 97,418 | 0.16 | n/a | n/a | 0.99 |
| vllm-omni-two-stage | 1 | 2 | 2,978 | 0.34 | n/a | n/a | 1.00 |
| vllm-omni-two-stage | 4 | 2 | 5,535 | 0.72 | n/a | n/a | 2.15 |
| vllm-omni-two-stage | 16 | 2 | 16,126 | 0.99 | n/a | n/a | 2.95 |

M* requests/s divided by the other framework's (>1: M* higher throughput):

- B=1: vs vllm-omni-single-stage 2.64×, vs vllm-omni-two-stage 1.31×
- B=4: vs vllm-omni-single-stage 9.10×, vs vllm-omni-two-stage 2.08×
- B=16: vs vllm-omni-single-stage 26.76×, vs vllm-omni-two-stage 4.43×

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
