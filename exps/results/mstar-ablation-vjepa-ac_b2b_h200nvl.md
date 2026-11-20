# M* component ablation on V-JEPA 2-AC

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_VJEPA_AC_A1_02_B2B` (A1, back-to-back), `MSTAR_ABLATION_VJEPA_AC_C_02_B2B` (C, back-to-back), `MSTAR_ABLATION_VJEPA_AC_S_02_B2B` (S, back-to-back), `MSTAR_ABLATION_VJEPA_AC_SC_02_B2B` (SC, back-to-back), `MSTAR_ABLATION_VJEPA_AC_W_02_B2B` (W, back-to-back), `MSTAR_ABLATION_VJEPA_AC_WC_02_B2B` (WC, back-to-back), `MSTAR_ABLATION_VJEPA_AC_A2_02_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | h15 | A1 | 2 | 10 | 405 [349–409] | 100% | 53 | 1,785 | - |
| mstar | h15 | C | 2 | 10 | 396 [344–431] | 100% | 56 | 1,785 | 0.98x † |
| mstar | h15 | S | 2 | 10 | 412 [411–416] | 100% | 56 | 1,785 | 1.02x † |
| mstar | h15 | SC | 2 | 10 | 446 [445–447] | 100% | 56 | 1,785 | 1.10x |
| mstar | h15 | W | 2 | 10 | 425 [421–434] | 100% | 57 | 1,785 | 1.05x |
| mstar | h15 | WC | 2 | 10 | 438 [422–443] | 100% | 57 | 1,785 | 1.08x |
| mstar | h15 | A2 | 2 | 10 | 422 [414–423] | 100% | 57 | 1,785 | 1.04x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar h15 4%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
