# M* component ablation on Qwen3-Omni T2S

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_QWEN_T2S_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_QWEN_T2S_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_QWEN_T2S_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_QWEN_T2S_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_QWEN_T2S_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_QWEN_T2S_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_QWEN_T2S_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | A1 | 2 | 10 | 80.9 [77.7–81.5] | 100% | 72 | 1,785 | - |
| mstar | verbatim-en | C | 2 | 10 | 89.9 [89.7–90.0] | 100% | 72 | 1,785 | 1.11x |
| mstar | verbatim-en | S | 2 | 10 | 105 [104–106] | 100% | 72 | 1,785 | 1.30x |
| mstar | verbatim-en | SC | 2 | 10 | 116 [116–117] | 100% | 71 | 1,785 | 1.44x |
| mstar | verbatim-en | W | 2 | 10 | 92.6 [91.6–92.8] | 100% | 72 | 1,785 | 1.15x |
| mstar | verbatim-en | WC | 2 | 10 | 118 [110–125] | 100% | 71 | 1,785 | 1.46x |
| mstar | verbatim-en | A2 | 2 | 10 | 84.0 [82.6–85.5] | 100% | 72 | 1,785 | 1.04x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar verbatim-en 4%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
