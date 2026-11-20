# Qwen3-Omni first-token latency, audio: A2T with one output token (the audio path's share), cache off, shapes pre-warmed

**Question.** A2T: vLLM-Omni's TTFT is ~19 ms above M*'s on the typical clip (109 vs 90 ms). Which part of the audio path: the encoder, the prefill of the audio tokens, or the front end?

**Pre-registered reading.** Run with qwen-ttft-txt and read analyze_ttft_parts.py's `A2T excess over text prefill`: the A2T E2E minus what a text prompt of the same token count costs on that framework = the audio encoder and its hand-off. A gap in the excess is the audio path; a gap that is already in `fixed` / `per 1k tokens` is the text path.

**Caveats.** The text columns of analyze_ttft_parts.py come from qwen-ttft-txt; run its analyze with both campaigns' directories to fill them.

Campaigns: `QWEN_TTFT_A2T_SGLANG_01_B2B` (sglang, back-to-back), `QWEN_TTFT_A2T_VLLM_01_B2B` (vllm, back-to-back), `QWEN_TTFT_A2T_MSTAR_01_B2B` (mstar, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-omni | a2t-en-1tok | sglang | 2 | 20 | 92.8 [91.6–92.9] | 0% | 39 | 2,460 | - |
| vllm-omni | a2t-en-1tok | vllm | 2 | 20 | 79.5 [77.7–79.7] | 0% | 35 | 2,460 | - |
| mstar | a2t-en-1tok | mstar | 2 | 20 | 101 [101–103] | 0% | 44 | 2,460 | - |
