# M* component ablation on BAGEL (one-branch T2I, and the AR path)

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_BAGEL_A1_01` (A1, gated), `MSTAR_ABLATION_BAGEL_C_01` (C, gated), `MSTAR_ABLATION_BAGEL_S_01` (S, gated), `MSTAR_ABLATION_BAGEL_SC_01` (SC, gated), `MSTAR_ABLATION_BAGEL_W_01` (W, gated), `MSTAR_ABLATION_BAGEL_WC_01` (WC, gated), `MSTAR_ABLATION_BAGEL_A2_01` (A2, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | A1 | 2 | 10 | 2,027 [1,937–2,153] | 0% | 66 | 1,713 | - |
| mstar | t2i-512-nocfg | C | 2 | 10 | 2,106 [2,089–2,253] | 0% | 66 | 1,729 | 1.04x |
| mstar | t2i-512-nocfg | S | 2 | 10 | 1,972 [1,961–1,983] | 0% | 66 | 1,703 | 0.97x |
| mstar | t2i-512-nocfg | SC | 2 | 10 | 2,263 [2,261–2,398] | 0% | 66 | 1,761 | 1.12x |
| mstar | t2i-512-nocfg | W | 2 | 10 | 2,095 [2,070–2,197] | 0% | 66 | 1,728 | 1.03x |
| mstar | t2i-512-nocfg | WC | 2 | 10 | 2,285 [2,222–2,413] | 10% | 65 | 1,759 | 1.13x |
| mstar | t2i-512-nocfg | A2 | 2 | 10 | 2,024 [1,999–2,444] | 0% | 65 | 1,728 | 1.00x |
| mstar | i2t-512 | A1 | 2 | 10 | 709 [699–715] | 100% | 65 | 1,785 | - |
| mstar | i2t-512 | C | 2 | 10 | 776 [775–777] | 100% | 65 | 1,785 | 1.09x |
| mstar | i2t-512 | S | 2 | 10 | 1,080 [1,073–1,088] | 100% | 65 | 1,785 | 1.52x |
| mstar | i2t-512 | SC | 2 | 10 | 1,198 [1,176–1,200] | 100% | 64 | 1,785 | 1.69x |
| mstar | i2t-512 | W | 2 | 10 | 1,003 [991–1,010] | 100% | 64 | 1,785 | 1.41x |
| mstar | i2t-512 | WC | 2 | 10 | 1,090 [1,087–1,115] | 100% | 63 | 1,785 | 1.54x |
| mstar | i2t-512 | A2 | 2 | 10 | 700 [700–703] | 100% | 64 | 1,785 | 0.99x |

**A1/A2 drift** (the reference arm measured before and after the others): mstar t2i-512-nocfg 0%; mstar i2t-512 1%. `†` marks an arm whose ratio to A1 is no larger than that drift: by the pre-registered reading it cannot be attributed to the switch. If the drift is larger than every arm's effect the ablation is inconclusive for that case and must be re-measured, not interpreted.
