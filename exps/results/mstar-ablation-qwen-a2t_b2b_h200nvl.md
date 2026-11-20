# M* component ablation on Qwen3-Omni A2T

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_QWEN_A2T_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_QWEN_A2T_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_QWEN_A2T_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_QWEN_A2T_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_QWEN_A2T_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_QWEN_A2T_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_QWEN_A2T_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | a2t-en | A1 | 2 | 16 | 167 [164–167] | 100% | 73 | 1,785 | - |
| mstar | a2t-en | C | 2 | 16 | 178 [177–179] | 100% | 72 | 1,785 | 1.07x |
| mstar | a2t-en | S | 2 | 16 | 227 [226–227] | 100% | 72 | 1,785 | 1.35x |
| mstar | a2t-en | SC | 2 | 16 | 221 [215–227] | 100% | 72 | 1,785 | 1.32x |
| mstar | a2t-en | W | 2 | 16 | 204 [189–234] | 100% | 72 | 1,785 | 1.22x |
| mstar | a2t-en | WC | 2 | 16 | 242 [238–242] | 100% | 72 | 1,785 | 1.44x |
| mstar | a2t-en | A2 | 2 | 16 | 167 [167–168] | 100% | 72 | 1,785 | 1.00x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar a2t-en 0%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
