# Concurrency curves

## QWEN_T2S

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | 1 | 1 | 312 | 3.20 | 10.44 | 0.094 | 1.00 |
| mstar | 4 | 1 | 633 | 6.32 | 21.98 | 0.158 | 1.97 |
| mstar | 16 | 1 | 1,257 | 12.73 | 45.60 | 0.293 | 3.97 |

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
