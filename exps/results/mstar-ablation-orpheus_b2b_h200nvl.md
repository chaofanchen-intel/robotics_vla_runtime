# M* component ablation on Orpheus

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_ORPHEUS_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_ORPHEUS_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_ORPHEUS_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_ORPHEUS_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_ORPHEUS_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_ORPHEUS_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_ORPHEUS_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | b1 | A1 | 2 | 10 | 910 [889–931] | 100% | 68 | 1,785 | - |
| mstar | b1 | C | 2 | 10 | 1,172 [1,127–1,217] | 100% | 64 | 1,785 | 1.29x |
| mstar | b1 | S | 2 | 10 | 1,371 [1,304–1,509] | 100% | 64 | 1,785 | 1.51x |
| mstar | b1 | SC | 2 | 10 | 1,636 [1,569–1,704] | 100% | 62 | 1,785 | 1.80x |
| mstar | b1 | W | 2 | 10 | 1,205 [1,201–1,210] | 100% | 64 | 1,785 | 1.32x |
| mstar | b1 | WC | 2 | 10 | 1,550 [1,495–1,558] | 100% | 62 | 1,785 | 1.70x |
| mstar | b1 | A2 | 2 | 10 | 962 [949–974] | 100% | 64 | 1,785 | 1.06x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar b1 6%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
