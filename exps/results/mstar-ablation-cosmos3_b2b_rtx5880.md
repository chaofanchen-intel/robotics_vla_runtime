# M* component ablation on Cosmos3

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_COSMOS3_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_COSMOS3_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_COSMOS3_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_COSMOS3_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_COSMOS3_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_COSMOS3_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_COSMOS3_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | A1 | 2 | 10 | 1,273 [1,252–1,281] | 0% | 42 | 2,240 | - |
| mstar | t2i-256 | C | 2 | 10 | 1,687 [1,654–1,773] | 0% | 43 | 2,627 | 1.33x |
| mstar | t2i-256 | S | 2 | 10 | 1,297 [1,292–1,299] | 0% | 42 | 2,380 | 1.02x |
| mstar | t2i-256 | SC | 2 | 10 | 1,691 [1,687–1,695] | 0% | 42 | 2,630 | 1.33x |
| mstar | t2i-256 | W | 2 | 10 | 1,302 [1,296–1,376] | 0% | 43 | 2,418 | 1.02x |
| mstar | t2i-256 | WC | 2 | 10 | 1,704 [1,690–1,746] | 0% | 44 | 2,632 | 1.34x |
| mstar | t2i-256 | A2 | 2 | 10 | 1,269 [1,259–1,271] | 0% | 42 | 2,182 | 1.00x |
