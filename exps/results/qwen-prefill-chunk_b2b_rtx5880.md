# Qwen3-Omni TXT, one output token: M*'s 1,024-token chunked-prefill budget against 4,096 (the 1,072-token prompt as one pass)

**Question.** qwen-prefill-nsys: M*'s request has three 48-layer forwards, not one. Thinker MAX_BATCH_TOKENS is 1,024, so the 1,072-token prompt is cut 1,024 + 48; the first chunk takes ~77 ms (vLLM's whole 1,072-token pass takes 77 ms), the 48-token chunk another ~40 ms (every expert's weights are read for 48 tokens), the third forward is the second output token (M* returns 2 tokens for max_output_tokens=1). vLLM's Thinker budget is 8,192. Is the whole TXT prefill gap (M* TTFT 146 against vLLM 99 / SGLang 117) this one chunk boundary?

**Pre-registered reading.** E2E and TTFT by arm and case (analyze_decode_itl.py). chunk-4096 on txt-1k-1tok close to vLLM's ~100 ms (about 40 ms below chunk-1024) = the whole gap is the chunk budget, i.e. M*'s prefill kernels are not slower; unchanged = the cause is elsewhere. txt-tiny-1tok must not move (a 15-token prompt is one chunk either way); if it moves, the budget changes something besides chunking. A raised budget costs decode steps that share a step with a long prompt, which this C=1 test cannot see.

**Caveats.** One long prompt length (1,072 tokens, 48 past the budget). The prediction for other lengths is a step function of ceil(tokens / budget); the Thinker budget of 4,096 is a counterfactual, not M*'s shipped config.

Campaigns: `QWEN_PREFILL_CHUNK_CHUNK_1024_01_B2B` (chunk-1024, back-to-back), `QWEN_PREFILL_CHUNK_CHUNK_4096_01_B2B` (chunk-4096, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs chunk-1024 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1k-1tok | chunk-1024 | 1 | 10 | 148 [148–148] | 0% | 45 | 2,685 | - |
| mstar | txt-1k-1tok | chunk-4096 | 1 | 10 | 163 [163–163] | 0% | 45 | 2,685 | 1.10x |
| mstar | txt-tiny-1tok | chunk-1024 | 1 | 10 | 61.7 [61.7–61.7] | 0% | 44 | 2,460 | - |
| mstar | txt-tiny-1tok | chunk-4096 | 1 | 10 | 62.5 [62.5–62.5] | 0% | 44 | 2,460 | 1.01x |
