# T2S per-stage GPU timeline (nsys, C=1; medians over measured requests)

## QWEN_T2S_NSYS_WARM_NSYS_01_B2B/mstar

requests: 5, client E2E (nsys-run) 377 ms, audio 3.66 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| node:Talker | -50 | 425 | 127 | 22983 |
| outside-node | -45 | 427 | 40 | 6393 |
| node:Code2Wav | -14 | 359 | 5 | 523 |
| node:Thinker | 227 | 333 | 107 | 18603 |

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| node:Code2Wav / node:Talker | 0.0 | 384 |
| node:Code2Wav / node:Thinker | 0.0 | 117 |
| node:Code2Wav / outside-node | 0.0 | 384 |
| node:Talker / node:Thinker | 26.0 | 117 |
| node:Talker / outside-node | 0.0 | 473 |
| node:Thinker / outside-node | 10.9 | 117 |

## QWEN_T2S_NSYS_WARM_NSYS_01_B2B/vllm-omni

requests: 5, client E2E (nsys-run) 585 ms, audio 3.10 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker | -50 | 635 | 188 | 47881 |
| pid583506 | 221 | 368 | 111 | 12910 |
| pid583507 | 221 | 368 | 126 | 12910 |

No kernels recorded for: code2wav, thinker (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| pid583506 / pid583507 | 100.2 | 155 |
| pid583506 / talker | 14.2 | 155 |
| pid583507 / talker | 15.8 | 156 |

## QWEN_T2S_NSYS_WARM_NSYS_01_B2B/sglang-omni

requests: 5, client E2E (nsys-run) 549 ms, audio 3.66 s

| group | first kernel ms | last kernel ms | GPU busy ms | launches |
|---|---:|---:|---:|---:|
| talker_ar | -49 | 563 | 146 | 25408 |
| thinker | 231 | 412 | 115 | 16614 |

No kernels recorded for: audio_encoder, code2wav, decode, image_encoder, preprocessing (process traced but silent; cannot say whether it ran concurrently).

| pair | both executing ms | [first,last] span overlap ms |
|---|---:|---:|
| talker_ar / thinker | 14.6 | 182 |

Limits: nsys inflates host time (BAGEL +6-18%, A2T +34%, T2S x1.6-2.2 and most for M*, the engine with the most launches; gaps and overlap fractions are structural, not absolute), one start per cell; "both executing" needs two stages to have a kernel on the GPU at the same instant (processes time-slice the GPU without MPS, one-stream engines cannot), so span overlap is the structural indicator.
