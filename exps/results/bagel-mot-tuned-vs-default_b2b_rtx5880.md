# BAGEL: what is tuning vLLM-Omni's MoT GEMM worth end to end?

**Question.** Per-step GPU busy dropped 32-58% with tuned tiles on the H200; does E2E follow, and on this GPU?

**Pre-registered reading.** untuned / tuned. ~1.0 (what was seen at 512^2 on the H200) means the E2E gap is not mostly this kernel.

Campaigns: `BAGEL_MOT_TUNED_VS_DEFAULT_TUNED_01_B2B` (tuned, back-to-back), `BAGEL_MOT_TUNED_VS_DEFAULT_UNTUNED_01_B2B` (untuned, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs tuned |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | t2i-512-nocfg | tuned | 2 | 10 | 7,643 [7,639–7,647] | 0% | 36 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-nocfg | untuned | 2 | 10 | 7,811 [7,804–7,818] | 0% | 35 | 2,460 | 1.02x |
| vllm-omni-two-stage | t2i-512-cfg3 | tuned | 2 | 10 | 19,325 [19,306–19,344] | 0% | 35 | 2,297 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | untuned | 2 | 10 | 20,020 [19,998–20,041] | 0% | 33 | 2,189 | 1.04x |
