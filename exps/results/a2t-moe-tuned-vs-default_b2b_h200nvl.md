# A2T: what is M*'s own MoE tile tuning worth end to end?

**Question.** vLLM and SGLang run tuned MoE tiles; M* only does if a table for this exact GPU exists (the package ships H100 / H200 / GB200 / B300; an H200 NVL or RTX 5880 matches none). How much does M*'s own sweep (`mstar-moe-tune`) buy on the Thinker's MoE layers?

**Pre-registered reading.** builtin / tuned. The old finding (M* ~1.6x the GPU time per token of the baselines on A2T, concentrated in MoE) was measured with built-in tiles; if tuning closes most of it, the A2T / TXT 'M* is slower' results were a tuning artefact, and every Qwen3-Omni ratio must be re-read with tuned M*.

Campaigns: `A2T_MOE_TUNED_VS_DEFAULT_TUNED_01_B2B` (tuned, back-to-back), `A2T_MOE_TUNED_VS_DEFAULT_BUILTIN_01_B2B` (builtin, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs tuned |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | a2t-en | tuned | 2 | 16 | 168 [163–170] | 100% | 73 | 1,785 | - |
| mstar | a2t-en | builtin | 2 | 16 | 168 [167–170] | 100% | 73 | 1,785 | 1.00x |
