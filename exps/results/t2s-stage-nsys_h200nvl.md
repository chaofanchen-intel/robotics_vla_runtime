# T2S per-stage GPU timeline (nsys, C=1; medians over measured requests)

## T2S_STAGE_NSYS_NSYS_01_B2B/mstar

requests: 3, client E2E (nsys-run) 295 ms, audio 3.50 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| node:Talker | -50 | 342 | 187 | 49716 |
| outside-node | -25 | 343 | 0 | 73 |
| node:Code2Wav | -25 | 345 | 12 | 1347 |
| node:Thinker | 76 | 190 | 62 | 7967 |

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| node:Code2Wav / node:Talker | 0.0 | 358 |
| node:Code2Wav / node:Thinker | 0.0 | 113 |
| node:Code2Wav / outside-node | 0.0 | 356 |
| node:Talker / node:Thinker | 0.0 | 113 |
| node:Talker / outside-node | 0.0 | 356 |
| node:Thinker / outside-node | 0.0 | 113 |

## T2S_STAGE_NSYS_NSYS_01_B2B/vllm-omni

requests: 3, client E2E (nsys-run) 656 ms, audio 3.82 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker | -50 | 695 | 346 | 93004 |
| thinker | 70 | 238 | 90 | 12215 |

No kernels recorded for: code2wav (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| talker / thinker | 11.6 | 167 |

## T2S_STAGE_NSYS_NSYS_01_B2B/sglang-omni

requests: 3, client E2E (nsys-run) 470 ms, audio 3.26 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker_ar | -50 | 388 | 129 | 29586 |
| thinker | 79 | 201 | 45 | 7813 |

No kernels recorded for: audio_encoder, decode, image_encoder, preprocessing (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| talker_ar / thinker | 4.2 | 121 |

Limits: nsys inflates host time (BAGEL +6-18%, A2T +34%, T2S x1.6-2.2 and most for M*, the engine with the most launches; gaps and overlap fractions are structural, not absolute), one start per cell; "both executing" needs two stages to have a kernel on the GPU at the same instant (processes time-slice the GPU without MPS, one-stream engines cannot), so span overlap is the structural indicator.
