# M* component ablation on Qwen3-Omni A2T

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_QWEN_A2T_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_QWEN_A2T_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_QWEN_A2T_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_QWEN_A2T_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_QWEN_A2T_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_QWEN_A2T_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_QWEN_A2T_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | a2t-en | A1 | 2 | 16 | 235 [234–235] | 0% | 48 | 2,685 | - |
| mstar | a2t-en | C | 2 | 16 | 234 [233–234] | 0% | 48 | 2,685 | 1.00x |
| mstar | a2t-en | S | 2 | 16 | 283 [282–288] | 0% | 48 | 2,685 | 1.20x |
| mstar | a2t-en | SC | 2 | 16 | 290 [285–292] | 0% | 48 | 2,685 | 1.23x |
| mstar | a2t-en | W | 2 | 16 | 292 [290–295] | 0% | 48 | 2,685 | 1.24x |
| mstar | a2t-en | WC | 2 | 16 | 294 [294–295] | 0% | 48 | 2,685 | 1.25x |
| mstar | a2t-en | A2 | 2 | 16 | 235 [231–235] | 0% | 49 | 2,685 | 1.00x |
