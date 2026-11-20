# Concurrency curves

## QWEN_T2S

| framework | B | starts | wave wall p50 (ms) | req/s | audio-s/s | RTF p50 | scaling vs B=1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni | 1 | 2 | 487 | 2.05 | 6.44 | 0.153 | 1.00 |

Limits: one cell = one start on one card, offered load = B per wave (no arrival process), the card heats across a wave; `starts` is the number of starts behind each point. Scaling ≈ 1.0 usually means the engine serialised the wave; check max_num_seqs (vLLM-Omni's single-stage BAGEL cannot batch at all).
