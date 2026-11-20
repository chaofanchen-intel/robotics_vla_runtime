# BAGEL: what is tuning vLLM-Omni's MoT GEMM worth end to end?

**Question.** Per-step GPU busy dropped 32-58% with tuned tiles on the H200; does E2E follow, and on this GPU?

**Pre-registered reading.** untuned / tuned. ~1.0 (what was seen at 512^2 on the H200) means the E2E gap is not mostly this kernel.

Campaigns: `BAGEL_MOT_TUNED_VS_DEFAULT_TUNED_01` (tuned, gated), `BAGEL_MOT_TUNED_VS_DEFAULT_UNTUNED_01` (untuned, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs tuned |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | t2i-512-nocfg | tuned | 2 | 10 | 2,575 [2,558–2,592] | 20% | 67 | 1,779 | - |
| vllm-omni-two-stage | t2i-512-nocfg | untuned | 2 | 10 | 3,085 [2,963–3,103] | 70% | 68 | 1,784 | 1.20x |
| vllm-omni-two-stage | t2i-512-cfg3 | tuned | 2 | 10 | 5,248 [5,221–5,249] | 0% | 69 | 1,667 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | untuned | 2 | 10 | 5,391 [5,374–5,391] | 0% | 70 | 1,675 | 1.03x |
