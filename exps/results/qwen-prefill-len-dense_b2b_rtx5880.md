# Qwen3-Omni TXT, one output token: TTFT at 1,100 / 1,300 / 1,536 / 1,800 prompt tokens, M* ladders against vLLM-Omni

**Question.** qwen-prefill-len: a 1,152 rung fixed 1,072 (TTFT 113 against 145 / 156 ms) and displaced the 2,048 graph. Between the rungs the shipped ladder pads every step to the next captured graph, so the TTFT line against prompt length should be a staircase. Where are the steps for each ladder (default / budget 4,096 / + 1,152 / + 1,152 + 1,536), and does a second rung at 1,536 get captured on this 48 GiB card?

**Pre-registered reading.** analyze_decode_itl.py, TTFT per case and arm, with vLLM as the reference line. Default: 1,100 = 1,024 + 76 (two forwards, ~+38 ms), 1,300 and 1,536 likewise, 1,800 = 1,024 + 776. Budget 4,096: all four padded to the 2,048 graph (flat ~155-165 ms). + 1,152: 1,100 on the 1,152 rung, 1,300 / 1,536 / 1,800 on a graph only if one exists (else eager at the actual size). + 1,152 + 1,536: 1,300 and 1,536 on the 1,536 rung IF the log shows it captured (`Captured accelerator graph for Thinker: prefill_text[bs=1,tokens=1536]`); if not, that arm is the + 1,152 arm and the card cannot hold another rung.

**Caveats.** Nominal server-side lengths (body = target - 48 template tokens, not asserted by the harness). One start per arm and case. After the run, read the capture list of each M* arm (`grep 'Captured accelerator graph for Thinker: prefill_text' server.log`, and the `captured N of M buckets` ERROR line): a rung that was not captured is not a rung. C=1 only.

Campaigns: `QWEN_PREFILL_LEN_DENSE_CHUNK_1024_01_B2B` (chunk-1024, back-to-back), `QWEN_PREFILL_LEN_DENSE_CHUNK_4096_01_B2B` (chunk-4096, back-to-back), `QWEN_PREFILL_LEN_DENSE_CHUNK_4096_B1152_01_B2B` (chunk-4096-b1152, back-to-back), `QWEN_PREFILL_LEN_DENSE_CHUNK_4096_B1152_1536_01_B2B` (chunk-4096-b1152-1536, back-to-back), `QWEN_PREFILL_LEN_DENSE_VLLM_01_B2B` (vllm, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs chunk-1024 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1100-1tok | chunk-1024 | 1 | 10 | 154 [154–154] | 0% | 46 | 2,685 | - |
| mstar | txt-1100-1tok | chunk-4096 | 1 | 10 | 165 [165–165] | 0% | 46 | 2,662 | 1.07x |
| mstar | txt-1100-1tok | chunk-4096-b1152 | 1 | 10 | 121 [121–121] | 0% | 46 | 2,685 | 0.79x |
| mstar | txt-1100-1tok | chunk-4096-b1152-1536 | 1 | 10 | 121 [121–121] | 0% | 46 | 2,685 | 0.78x |
| mstar | txt-1300-1tok | chunk-1024 | 1 | 10 | 172 [172–172] | 0% | 48 | 2,685 | - |
| mstar | txt-1300-1tok | chunk-4096 | 1 | 10 | 167 [167–167] | 0% | 48 | 2,685 | 0.97x |
| mstar | txt-1300-1tok | chunk-4096-b1152 | 1 | 10 | 140 [140–140] | 0% | 46 | 2,685 | 0.81x |
| mstar | txt-1300-1tok | chunk-4096-b1152-1536 | 1 | 10 | 134 [134–134] | 0% | 46 | 2,679 | 0.78x |
| mstar | txt-1536-1tok | chunk-1024 | 1 | 10 | 175 [175–175] | 0% | 48 | 2,682 | - |
| mstar | txt-1536-1tok | chunk-4096 | 1 | 10 | 168 [168–168] | 0% | 47 | 2,676 | 0.96x |
| mstar | txt-1536-1tok | chunk-4096-b1152 | 1 | 10 | 146 [146–146] | 0% | 46 | 2,685 | 0.84x |
| mstar | txt-1536-1tok | chunk-4096-b1152-1536 | 1 | 10 | 143 [143–143] | 0% | 46 | 2,685 | 0.82x |
| mstar | txt-1800-1tok | chunk-1024 | 1 | 10 | 196 [196–196] | 0% | 47 | 2,685 | - |
| mstar | txt-1800-1tok | chunk-4096 | 1 | 10 | 171 [171–171] | 0% | 47 | 2,680 | 0.87x |
| mstar | txt-1800-1tok | chunk-4096-b1152 | 1 | 10 | 164 [164–164] | 0% | 47 | 2,676 | 0.83x |
| mstar | txt-1800-1tok | chunk-4096-b1152-1536 | 1 | 10 | 165 [165–165] | 0% | 48 | 2,678 | 0.84x |
| vllm-omni | txt-1100-1tok | vllm | 1 | 10 | 101 [101–101] | 0% | 35 | 2,464 | - |
| vllm-omni | txt-1300-1tok | vllm | 1 | 10 | 111 [111–111] | 0% | 35 | 2,694 | - |
| vllm-omni | txt-1536-1tok | vllm | 1 | 10 | 121 [121–121] | 0% | 36 | 2,692 | - |
| vllm-omni | txt-1800-1tok | vllm | 1 | 10 | 133 [133–133] | 0% | 37 | 2,688 | - |
