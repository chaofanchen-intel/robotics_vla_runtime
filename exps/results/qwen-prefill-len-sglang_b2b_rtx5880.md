# Qwen3-Omni TXT, one output token: SGLang-Omni's TTFT against prompt length (56 / 512 / 1,024 / 1,072 / 2,048), cache off

**Question.** qwen-prefill-len compares M* with vLLM-Omni only. SGLang-Omni's 1,072-token TTFT was 117 ms (qwen-ttft-txt, M* 146, vLLM 99). Is SGLang's line a straight 56 -> 2,048 line like vLLM's or does it also have a step, and where does it sit between the two?

**Pre-registered reading.** analyze_decode_itl.py, TTFT per case against qwen-prefill-len's vllm arm and M* arms (same cases, same protocol, same request). SGLang with the radix cache off (config no_radix) so every request is a cold prefill.

**Caveats.** SGLang-Omni returns the first text chunk differently from M* / vLLM (a chunk can carry several tokens); with one output token TTFT is the first chunk. One start per case.

Campaigns: `QWEN_PREFILL_LEN_SGLANG_SGLANG_01_B2B` (sglang, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | txt-tiny-1tok | sglang | 1 | 10 | 56.2 [56.2–56.2] | 0% | 36 | 2,460 | - |
| sglang-omni | txt-512-1tok | sglang | 1 | 10 | 83.0 [83.0–83.0] | 0% | 36 | 2,460 | - |
| sglang-omni | txt-1024-1tok | sglang | 1 | 10 | 108 [108–108] | 0% | 37 | 2,460 | - |
| sglang-omni | txt-1k-1tok | sglang | 1 | 10 | 120 [120–120] | 0% | 38 | 2,676 | - |
| sglang-omni | txt-2048-1tok | sglang | 1 | 10 | 157 [157–157] | 0% | 39 | 2,700 | - |
