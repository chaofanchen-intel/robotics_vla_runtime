# T2S per-stage GPU timeline (nsys, C=1; medians over measured requests)

## T2S_STAGE_NSYS_NSYS_01_B2B/mstar

requests: 3, client E2E (nsys-run) 408 ms, audio 3.66 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| node:Talker | -50 | 451 | 101 | 18226 |
| outside-node | -49 | 458 | 68 | 11043 |
| node:Code2Wav | -26 | 441 | 9 | 1033 |
| node:Thinker | 204 | 320 | 107 | 17853 |

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| node:Code2Wav / node:Talker | 0.0 | 402 |
| node:Code2Wav / node:Thinker | 0.0 | 112 |
| node:Code2Wav / outside-node | 0.0 | 450 |
| node:Talker / node:Thinker | 18.9 | 104 |
| node:Talker / outside-node | 1.7 | 447 |
| node:Thinker / outside-node | 26.6 | 112 |

## T2S_STAGE_NSYS_NSYS_01_B2B/vllm-omni

requests: 3, client E2E (nsys-run) 654 ms, audio 3.50 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker | -50 | 704 | 217 | 57299 |
| pid3973192 | 239 | 385 | 120 | 12910 |
| pid3973193 | 239 | 385 | 120 | 12910 |

No kernels recorded for: code2wav, thinker (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| pid3973192 / pid3973193 | 99.6 | 146 |
| pid3973192 / talker | 9.4 | 146 |
| pid3973193 / talker | 9.2 | 146 |

## T2S_STAGE_NSYS_NSYS_01_B2B/sglang-omni

requests: 3, client E2E (nsys-run) 491 ms, audio 3.66 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker_ar | -49 | 554 | 142 | 24538 |
| thinker | 269 | 295 | 21 | 1979 |

No kernels recorded for: audio_encoder, code2wav, decode, image_encoder, preprocessing (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| talker_ar / thinker | 0.0 | 26 |

Limits: nsys inflates host time (BAGEL +6-18%, A2T +34%, T2S x1.6-2.2 and most for M*, the engine with the most launches; gaps and overlap fractions are structural, not absolute), one start per cell; "both executing" needs two stages to have a kernel on the GPU at the same instant (processes time-slice the GPU without MPS, one-stream engines cannot), so span overlap is the structural indicator.
