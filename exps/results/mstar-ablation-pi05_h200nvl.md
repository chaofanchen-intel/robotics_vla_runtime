# M* component ablation on Pi0.5

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_PI05_A1_01` (A1, gated), `MSTAR_ABLATION_PI05_C_01` (C, gated), `MSTAR_ABLATION_PI05_S_01` (S, gated), `MSTAR_ABLATION_PI05_SC_01` (SC, gated), `MSTAR_ABLATION_PI05_W_01` (W, gated), `MSTAR_ABLATION_PI05_WC_01` (WC, gated), `MSTAR_ABLATION_PI05_A2_01` (A2, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row | A1 | 2 | 24 | 101 [96–105] | 100% | 62 | 1,785 | - |
| mstar | droid-46row | C | 2 | 24 | 97.6 [96.9–98.0] | 100% | 61 | 1,785 | 0.96x † |
| mstar | droid-46row | S | 2 | 24 | 102 [98–105] | 100% | 61 | 1,785 | 1.00x † |
| mstar | droid-46row | SC | 2 | 24 | 105 [101–107] | 100% | 60 | 1,785 | 1.03x † |
| mstar | droid-46row | W | 2 | 24 | 102 [102–103] | 100% | 61 | 1,785 | 1.01x † |
| mstar | droid-46row | WC | 2 | 24 | 107 [107–107] | 100% | 61 | 1,785 | 1.05x † |
| mstar | droid-46row | A2 | 2 | 24 | 94.1 [94.0–94.7] | 100% | 62 | 1,785 | 0.93x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar droid-46row 7%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
