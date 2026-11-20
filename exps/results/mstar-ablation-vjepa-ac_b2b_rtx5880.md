# M* component ablation on V-JEPA 2-AC

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_VJEPA_AC_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_VJEPA_AC_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_VJEPA_AC_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_VJEPA_AC_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_VJEPA_AC_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_VJEPA_AC_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_VJEPA_AC_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | h15 | A1 | 2 | 10 | 380 [379–386] | 0% | 36 | 2,556 | - |
| mstar | h15 | C | 2 | 10 | 394 [393–397] | 0% | 35 | 2,676 | 1.04x |
| mstar | h15 | S | 2 | 10 | 393 [381–393] | 0% | 35 | 2,564 | 1.04x |
| mstar | h15 | SC | 2 | 10 | 399 [397–404] | 0% | 35 | 2,586 | 1.05x |
| mstar | h15 | W | 2 | 10 | 387 [361–405] | 0% | 35 | 2,580 | 1.02x |
| mstar | h15 | WC | 2 | 10 | 410 [409–411] | 0% | 35 | 2,562 | 1.08x |
| mstar | h15 | A2 | 2 | 10 | 378 [369–385] | 0% | 35 | 2,568 | 1.00x |
