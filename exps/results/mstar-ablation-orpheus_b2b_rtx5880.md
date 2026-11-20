# M* component ablation on Orpheus

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_ORPHEUS_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_ORPHEUS_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_ORPHEUS_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_ORPHEUS_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_ORPHEUS_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_ORPHEUS_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_ORPHEUS_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | b1 | A1 | 2 | 10 | 2,521 [2,500–2,542] | 0% | 45 | 2,677 | - |
| mstar | b1 | C | 2 | 10 | 2,368 [2,359–2,377] | 0% | 46 | 2,685 | 0.94x |
| mstar | b1 | S | 2 | 10 | 3,166 [3,163–3,169] | 0% | 47 | 2,686 | 1.26x |
| mstar | b1 | SC | 2 | 10 | 2,920 [2,913–2,927] | 0% | 45 | 2,688 | 1.16x |
| mstar | b1 | W | 2 | 10 | 3,236 [3,234–3,238] | 0% | 46 | 2,687 | 1.28x |
| mstar | b1 | WC | 2 | 10 | 2,991 [2,988–2,993] | 0% | 45 | 2,687 | 1.19x |
| mstar | b1 | A2 | 2 | 10 | 2,521 [2,518–2,524] | 0% | 48 | 2,671 | 1.00x |
