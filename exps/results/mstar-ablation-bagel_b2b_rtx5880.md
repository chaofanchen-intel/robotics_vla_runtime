# M* component ablation on BAGEL (one-branch T2I, and the AR path)

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_BAGEL_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_BAGEL_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_BAGEL_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_BAGEL_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_BAGEL_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_BAGEL_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_BAGEL_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | A1 | 2 | 10 | 7,423 [7,412–7,434] | 0% | 54 | 1,384 | - |
| mstar | t2i-512-nocfg | C | 2 | 10 | 7,713 [7,706–7,721] | 0% | 57 | 1,411 | 1.04x |
| mstar | t2i-512-nocfg | S | 2 | 10 | 7,409 [7,398–7,421] | 0% | 55 | 1,409 | 1.00x |
| mstar | t2i-512-nocfg | SC | 2 | 10 | 7,734 [7,724–7,744] | 0% | 57 | 1,411 | 1.04x |
| mstar | t2i-512-nocfg | W | 2 | 10 | 7,420 [7,411–7,429] | 0% | 56 | 1,432 | 1.00x |
| mstar | t2i-512-nocfg | WC | 2 | 10 | 7,699 [7,696–7,702] | 0% | 58 | 1,417 | 1.04x |
| mstar | t2i-512-nocfg | A2 | 2 | 10 | 7,441 [7,417–7,468] | 0% | 56 | 1,373 | 1.00x |
| mstar | i2t-512 | A1 | 2 | 10 | 2,286 [2,285–2,290] | 0% | 52 | 2,587 | - |
| mstar | i2t-512 | C | 2 | 10 | 2,315 [2,314–2,316] | 0% | 53 | 2,604 | 1.01x |
| mstar | i2t-512 | S | 2 | 10 | 2,609 [2,604–2,617] | 0% | 54 | 2,658 | 1.14x |
| mstar | i2t-512 | SC | 2 | 10 | 2,634 [2,624–2,638] | 0% | 55 | 2,606 | 1.15x |
| mstar | i2t-512 | W | 2 | 10 | 2,655 [2,653–2,662] | 0% | 52 | 2,593 | 1.16x |
| mstar | i2t-512 | WC | 2 | 10 | 2,684 [2,670–2,686] | 0% | 56 | 2,642 | 1.17x |
| mstar | i2t-512 | A2 | 2 | 10 | 2,288 [2,288–2,289] | 0% | 52 | 2,506 | 1.00x |
