# Qwen3-Omni TXT 1,072-token prompt on M*: forwards per request and kernel time with chunk 1,024 / 4,096 / 4,096 + a 1,152 graph rung (nsys)

**Question.** qwen-prefill-chunk: budget 4,096 did not remove the second forward's cost (TTFT 154 against 138 ms). Is the 1,072-token prompt really ONE forward there, and how long is it: a graph padded to 2,048 tokens (about twice the dense / attention work), or something else? And does a 1,152 rung shorten it?

**Pre-registered reading.** analyze_nsys_burst_gaps.py per arm, both TP ranks. Count the attention launches in a burst (144 = three 48-layer forwards in the default; 96 = two). Chunk-4096 with one prompt forward + a decode forward whose prompt forward is ~100+ ms = padding (compare its GEMM / attention kernel ms against chunk-1024's first pass: ~2x = a 2,048 graph); chunk-4096 still showing three forwards = the budget did not take effect. chunk-4096-b1152's prompt forward near 80 ms = the ladder is the lever.

**Caveats.** One prompt length (1,072 tokens). nsys inflates host time; kernel ms are the evidence.

Campaigns: `QWEN_PREFILL_CHUNK_NSYS_CHUNK_1024_01_B2B` (chunk-1024, back-to-back), `QWEN_PREFILL_CHUNK_NSYS_CHUNK_4096_01_B2B` (chunk-4096, back-to-back), `QWEN_PREFILL_CHUNK_NSYS_CHUNK_4096_B1152_01_B2B` (chunk-4096-b1152, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs chunk-1024 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1k-1tok | chunk-1024 | 1 | 5 | 153 [153–153] | 0% | 46 | 2,595 | - |
| mstar | txt-1k-1tok | chunk-4096 | 1 | 5 | 123 [123–123] | 0% | 45 | 2,676 | 0.81x |
| mstar | txt-1k-1tok | chunk-4096-b1152 | 1 | 5 | 126 [126–126] | 0% | 45 | 2,685 | 0.83x |
