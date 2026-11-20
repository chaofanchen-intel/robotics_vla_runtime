# Concurrency curves

## BAGEL

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | 1 | 2 | 2,251 | 0.44 | n/a | n/a | 1.00 |
| vllm-omni-two-stage | 4 | 2 | 4,536 | 0.88 | n/a | n/a | 1.98 |
| vllm-omni-two-stage | 16 | 2 | 13,668 | 1.17 | n/a | n/a | 2.63 |

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
