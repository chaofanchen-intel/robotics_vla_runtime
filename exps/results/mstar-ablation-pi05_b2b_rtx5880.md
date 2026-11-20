# M* component ablation on Pi0.5

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_PI05_A1_02_B2B` (A1, back-to-back), `MSTAR_ABLATION_PI05_C_02_B2B` (C, back-to-back), `MSTAR_ABLATION_PI05_S_02_B2B` (S, back-to-back), `MSTAR_ABLATION_PI05_SC_02_B2B` (SC, back-to-back), `MSTAR_ABLATION_PI05_W_02_B2B` (W, back-to-back), `MSTAR_ABLATION_PI05_WC_02_B2B` (WC, back-to-back), `MSTAR_ABLATION_PI05_A2_02_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row | A1 | 2 | 24 | 94.3 [93.4–94.8] | 0% | 36 | 2,460 | - |
| mstar | droid-46row | C | 2 | 24 | 105 [104–106] | 0% | 35 | 2,460 | 1.12x |
| mstar | droid-46row | S | 2 | 24 | 104 [100–110] | 0% | 36 | 2,460 | 1.10x |
| mstar | droid-46row | SC | 2 | 24 | 123 [122–123] | 0% | 35 | 2,460 | 1.30x |
| mstar | droid-46row | W | 2 | 24 | 113 [113–113] | 0% | 35 | 2,460 | 1.20x |
| mstar | droid-46row | WC | 2 | 24 | 123 [119–124] | 0% | 36 | 2,460 | 1.31x |
| mstar | droid-46row | A2 | 2 | 24 | 101 [96–103] | 0% | 36 | 2,460 | 1.07x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar droid-46row 7%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
