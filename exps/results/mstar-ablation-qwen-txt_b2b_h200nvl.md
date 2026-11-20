# M* component ablation on Qwen3-Omni TXT

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_QWEN_TXT_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_QWEN_TXT_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_QWEN_TXT_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_QWEN_TXT_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_QWEN_TXT_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_QWEN_TXT_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_QWEN_TXT_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1k | A1 | 2 | 10 | 1,053 [1,053–1,053] | 100% | 75 | 1,785 | - |
| mstar | txt-1k | C | 2 | 10 | 1,160 [1,159–1,161] | 100% | 74 | 1,785 | 1.10x |
| mstar | txt-1k | S | 2 | 10 | 1,327 [1,326–1,446] | 100% | 74 | 1,785 | 1.26x |
| mstar | txt-1k | SC | 2 | 10 | 1,508 [1,412–1,530] | 100% | 73 | 1,785 | 1.43x |
| mstar | txt-1k | W | 2 | 10 | 1,530 [1,509–1,552] | 100% | 74 | 1,785 | 1.45x |
| mstar | txt-1k | WC | 2 | 10 | 1,441 [1,436–1,477] | 100% | 73 | 1,785 | 1.37x |
| mstar | txt-1k | A2 | 2 | 10 | 1,052 [1,051–1,054] | 100% | 74 | 1,785 | 1.00x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar txt-1k 0%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
