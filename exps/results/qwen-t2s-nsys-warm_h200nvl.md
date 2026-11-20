# T2S per-stage GPU timeline (nsys, C=1; medians over measured requests)

## QWEN_T2S_NSYS_WARM_NSYS_01_B2B/mstar

requests: 5, client E2E (nsys-run) 348 ms, audio 3.66 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| node:Talker | -50 | 396 | 174 | 46037 |
| outside-node | -24 | 323 | 0 | 73 |
| node:Code2Wav | -24 | 326 | 11 | 1247 |
| node:Thinker | 79 | 213 | 62 | 7967 |

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| node:Code2Wav / node:Talker | 0.0 | 340 |
| node:Code2Wav / node:Thinker | 0.0 | 135 |
| node:Code2Wav / outside-node | 0.0 | 337 |
| node:Talker / node:Thinker | 0.0 | 135 |
| node:Talker / outside-node | 0.0 | 337 |
| node:Thinker / outside-node | 0.0 | 135 |

## QWEN_T2S_NSYS_WARM_NSYS_01_B2B/vllm-omni

requests: 5, client E2E (nsys-run) 588 ms, audio 3.18 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker | -50 | 626 | 249 | 67603 |
| thinker | 67 | 226 | 65 | 8951 |

No kernels recorded for: code2wav (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| talker / thinker | 2.1 | 160 |

## QWEN_T2S_NSYS_WARM_NSYS_01_B2B/sglang-omni

requests: 5, client E2E (nsys-run) 421 ms, audio 3.26 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker_ar | -50 | 418 | 159 | 35046 |
| thinker | 68 | 237 | 46 | 7939 |

No kernels recorded for: audio_encoder, decode, image_encoder, preprocessing (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| talker_ar / thinker | 16.4 | 170 |

Limits: nsys inflates host time (BAGEL +6-18%, A2T +34%, T2S x1.6-2.2 and most for M*, the engine with the most launches; gaps and overlap fractions are structural, not absolute), one start per cell; "both executing" needs two stages to have a kernel on the GPU at the same instant (processes time-slice the GPU without MPS, one-stream engines cannot), so span overlap is the structural indicator.
