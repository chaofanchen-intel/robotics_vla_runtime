# Concurrency curves

## BAGEL

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | 1 | 2 | 2,293 | 0.44 | n/a | n/a | 1.00 |
| mstar | 4 | 2 | 2,665 | 1.50 | n/a | n/a | 3.44 |
| mstar | 16 | 2 | 3,650 | 4.38 | n/a | n/a | 10.05 |

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
