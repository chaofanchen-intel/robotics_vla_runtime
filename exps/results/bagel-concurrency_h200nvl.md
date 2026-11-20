# Concurrency curves

## BAGEL

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | 1 | 2 | 699 | 1.43 | n/a | n/a | 1.00 |
| mstar | 4 | 2 | 960 | 4.17 | n/a | n/a | 2.91 |
| mstar | 16 | 2 | 1,625 | 9.86 | n/a | n/a | 6.90 |
| vllm-omni-single-stage | 1 | 2 | 4,808 | 0.21 | n/a | n/a | 1.00 |
| vllm-omni-single-stage | 4 | 2 | 17,875 | 0.22 | n/a | n/a | 1.08 |
| vllm-omni-single-stage | 16 | 2 | 79,646 | 0.20 | n/a | n/a | 0.97 |
| vllm-omni-two-stage | 1 | 2 | 998 | 1.00 | n/a | n/a | 1.00 |
| vllm-omni-two-stage | 4 | 2 | 1,794 | 2.23 | n/a | n/a | 2.22 |
| vllm-omni-two-stage | 16 | 2 | 5,160 | 3.10 | n/a | n/a | 3.09 |

M* requests/s divided by the other framework's (>1: M* higher throughput):

- B=1: vs vllm-omni-single-stage 6.87×, vs vllm-omni-two-stage 1.43×
- B=4: vs vllm-omni-single-stage 18.62×, vs vllm-omni-two-stage 1.87×
- B=16: vs vllm-omni-single-stage 49.10×, vs vllm-omni-two-stage 3.18×

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
