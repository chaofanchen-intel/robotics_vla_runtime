# Qwen3-Omni TXT, one output token: start-to-start repeatability of the 512- and 1,072-token TTFT, M* default against the 1,152 rung (3 starts)

**Question.** qwen-prefill-len's default arm, run twice, agreed to 0.2-3.9 ms everywhere except 512 tokens (71.3 against 78.7 ms), and every conclusion rests on one start per arm. How large is the start-to-start spread at 512 and 1,072 tokens, and does the 113 against 145 ms (1,152 rung against default) survive 3 starts each?

**Pre-registered reading.** Per-start TTFT p50 (analyze_decode_itl.py lists the start range). A spread of 5+ ms at 512 = differences below ~8 ms in the length sweeps are not readable; 1,152 below default in all 3 starts at 1,072 = the 32 ms is not an artefact of one start.

**Caveats.** Starts are separate server launches back to back, not independent days.

Campaigns: `QWEN_PREFILL_LEN_REPEAT_CHUNK_1024_01_B2B` (chunk-1024, back-to-back), `QWEN_PREFILL_LEN_REPEAT_CHUNK_4096_B1152_01_B2B` (chunk-4096-b1152, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs chunk-1024 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-512-1tok | chunk-1024 | 3 | 30 | 88.1 [86.1–89.2] | 0% | 44 | 2,460 | - |
| mstar | txt-512-1tok | chunk-4096-b1152 | 3 | 30 | 87.6 [85.0–89.5] | 0% | 45 | 2,460 | 0.99x |
| mstar | txt-1k-1tok | chunk-1024 | 3 | 30 | 156 [153–160] | 0% | 46 | 2,684 | - |
| mstar | txt-1k-1tok | chunk-4096-b1152 | 3 | 30 | 123 [121–123] | 0% | 46 | 2,684 | 0.79x |
