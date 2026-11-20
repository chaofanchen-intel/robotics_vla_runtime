# M* component ablation on Cosmos3

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_COSMOS3_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_COSMOS3_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_COSMOS3_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_COSMOS3_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_COSMOS3_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_COSMOS3_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_COSMOS3_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | A1 | 2 | 10 | 1,092 [1,052–1,095] | 100% | 60 | 1,785 | - |
| mstar | t2i-256 | C | 2 | 10 | 1,398 [1,389–1,406] | 100% | 61 | 1,785 | 1.28x |
| mstar | t2i-256 | S | 2 | 10 | 1,103 [1,088–1,363] | 100% | 61 | 1,785 | 1.01x † |
| mstar | t2i-256 | SC | 2 | 10 | 1,475 [1,393–1,497] | 100% | 61 | 1,785 | 1.35x |
| mstar | t2i-256 | W | 2 | 10 | 1,105 [1,081–1,116] | 100% | 62 | 1,785 | 1.01x † |
| mstar | t2i-256 | WC | 2 | 10 | 1,382 [1,381–1,383] | 100% | 60 | 1,785 | 1.27x |
| mstar | t2i-256 | A2 | 2 | 10 | 1,112 [1,104–1,112] | 100% | 60 | 1,785 | 1.02x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar t2i-256 2%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
