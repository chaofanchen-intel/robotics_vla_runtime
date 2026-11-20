# Concurrency curves

## BAGEL

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | 1 | 2 | 2,257 | 0.44 | n/a | n/a | 1.00 |
| vllm-omni-two-stage | 4 | 2 | 2,346 | 1.71 | n/a | n/a | 3.85 |
| vllm-omni-two-stage | 16 | 2 | 3,217 | 4.97 | n/a | n/a | 11.23 |

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
