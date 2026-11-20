# Qwen3-Omni TXT, one output token: TTFT against prompt length (56 / 512 / 1,024 / 1,072 / 2,048 tokens) for M* with three chunk / graph-ladder settings, and vLLM-Omni

**Question.** qwen-prefill-chunk falsified the chunk-budget reading of the 1,072-token TTFT gap: budget 4,096 made M* SLOWER (154 against 138 ms), not 40 ms faster. M*'s Thinker prefill replays CUDA graphs on a 128...2048 ladder and pads a step to the smallest bucket that fits, so a 1,072-token step goes to 2,048 when it is not cut. Where does M*'s TTFT sit against vLLM's as the prompt length moves across the 1,024 budget and the ladder's rungs, and does a ladder with a rung near the prompt (chunk-4096-b1152) remove the penalty at 1,072?

**Pre-registered reading.** analyze_decode_itl.py, TTFT (ms) per case and arm; read TTFT, not E2E (M* returns 2 tokens for a 1-token cap). Plot TTFT against server-side prompt tokens (tiny = 56, 512, 1,024, 1,072, 2,048). vLLM's line is the reference. If M* matches vLLM at 512 and 1,024 (both exact rungs and within one chunk) the 1,072 gap is a boundary position, not slow kernels; chunk-4096 slower than chunk-1024 at 1,072 but equal at 1,024 and 2,048 = the padding story; chunk-4096-b1152 near vLLM at 1,072 = the ladder is the lever. M* behind vLLM at 512 or 1,024 too = the gap is kernels / per-step cost after all and the padding story is wrong. chunk-1024-again against chunk-1024 = order drift.

**Caveats.** The 512 / 1,024 / 2,048 prompts are the 1k prompt's words cut or cycled (body = target - 48 template tokens, the overhead read off SGLang-Omni's log); the server-side length is not asserted by the harness. One start per arm and case. 2,048 is the ladder's top rung: past it a step runs eagerly (not measured). C=1 only: the chunk budget's effect on decode rows sharing a step with a prompt is invisible here.

Campaigns: `QWEN_PREFILL_LEN_CHUNK_1024_01_B2B` (chunk-1024, back-to-back), `QWEN_PREFILL_LEN_CHUNK_4096_01_B2B` (chunk-4096, back-to-back), `QWEN_PREFILL_LEN_CHUNK_4096_B1152_01_B2B` (chunk-4096-b1152, back-to-back), `QWEN_PREFILL_LEN_VLLM_01_B2B` (vllm, back-to-back), `QWEN_PREFILL_LEN_CHUNK_1024_AGAIN_01_B2B` (chunk-1024-again, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs chunk-1024 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-tiny-1tok | chunk-1024 | 1 | 10 | 62.7 [62.7–62.7] | 0% | 42 | 2,460 | - |
| mstar | txt-tiny-1tok | chunk-4096 | 1 | 10 | 61.6 [61.6–61.6] | 0% | 44 | 2,460 | 0.98x |
| mstar | txt-tiny-1tok | chunk-4096-b1152 | 1 | 10 | 63.1 [63.1–63.1] | 0% | 45 | 2,460 | 1.01x |
| mstar | txt-tiny-1tok | chunk-1024-again | 1 | 10 | 62.2 [62.2–62.2] | 0% | 43 | 2,460 | 0.99x |
| mstar | txt-512-1tok | chunk-1024 | 1 | 10 | 81.2 [81.2–81.2] | 0% | 43 | 2,460 | - |
| mstar | txt-512-1tok | chunk-4096 | 1 | 10 | 89.4 [89.4–89.4] | 0% | 44 | 2,460 | 1.10x |
| mstar | txt-512-1tok | chunk-4096-b1152 | 1 | 10 | 85.4 [85.4–85.4] | 0% | 44 | 2,460 | 1.05x |
| mstar | txt-512-1tok | chunk-1024-again | 1 | 10 | 89.1 [89.1–89.1] | 0% | 45 | 2,460 | 1.10x |
| mstar | txt-1024-1tok | chunk-1024 | 1 | 10 | 118 [118–118] | 0% | 45 | 2,685 | - |
| mstar | txt-1024-1tok | chunk-4096 | 1 | 10 | 114 [114–114] | 0% | 44 | 2,700 | 0.97x |
| mstar | txt-1024-1tok | chunk-4096-b1152 | 1 | 10 | 118 [118–118] | 0% | 45 | 2,685 | 1.00x |
| mstar | txt-1024-1tok | chunk-1024-again | 1 | 10 | 117 [117–117] | 0% | 46 | 2,685 | 0.99x |
| mstar | txt-1k-1tok | chunk-1024 | 1 | 10 | 154 [154–154] | 0% | 46 | 2,684 | - |
| mstar | txt-1k-1tok | chunk-4096 | 1 | 10 | 168 [168–168] | 0% | 46 | 2,685 | 1.09x |
| mstar | txt-1k-1tok | chunk-4096-b1152 | 1 | 10 | 123 [123–123] | 0% | 46 | 2,685 | 0.80x |
| mstar | txt-1k-1tok | chunk-1024-again | 1 | 10 | 152 [152–152] | 0% | 48 | 2,684 | 0.99x |
| mstar | txt-2048-1tok | chunk-1024 | 1 | 10 | 202 [202–202] | 0% | 48 | 2,685 | - |
| mstar | txt-2048-1tok | chunk-4096 | 1 | 10 | 175 [175–175] | 0% | 46 | 2,679 | 0.87x |
| mstar | txt-2048-1tok | chunk-4096-b1152 | 1 | 10 | 178 [178–178] | 0% | 47 | 2,685 | 0.88x |
| mstar | txt-2048-1tok | chunk-1024-again | 1 | 10 | 200 [200–200] | 0% | 49 | 2,685 | 0.99x |
| vllm-omni | txt-tiny-1tok | vllm | 1 | 10 | 59.7 [59.7–59.7] | 0% | 34 | 2,460 | - |
| vllm-omni | txt-512-1tok | vllm | 1 | 10 | 72.5 [72.5–72.5] | 0% | 35 | 2,460 | - |
| vllm-omni | txt-1024-1tok | vllm | 1 | 10 | 96.5 [96.5–96.5] | 0% | 36 | 2,469 | - |
| vllm-omni | txt-1k-1tok | vllm | 1 | 10 | 98.8 [98.8–98.8] | 0% | 36 | 2,481 | - |
| vllm-omni | txt-2048-1tok | vllm | 1 | 10 | 152 [152–152] | 0% | 38 | 2,700 | - |
