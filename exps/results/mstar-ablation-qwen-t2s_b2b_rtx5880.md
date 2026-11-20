# M* component ablation on Qwen3-Omni T2S

**Question.** Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) account for its advantage on this model, and what is left when they are all off?

**Pre-registered reading.** Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null results, not missing data.

Campaigns: `MSTAR_ABLATION_QWEN_T2S_A1_01_B2B` (A1, back-to-back), `MSTAR_ABLATION_QWEN_T2S_C_01_B2B` (C, back-to-back), `MSTAR_ABLATION_QWEN_T2S_S_01_B2B` (S, back-to-back), `MSTAR_ABLATION_QWEN_T2S_SC_01_B2B` (SC, back-to-back), `MSTAR_ABLATION_QWEN_T2S_W_01_B2B` (W, back-to-back), `MSTAR_ABLATION_QWEN_T2S_WC_01_B2B` (WC, back-to-back), `MSTAR_ABLATION_QWEN_T2S_A2_01_B2B` (A2, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs A1 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | A1 | 2 | 10 | 103 [103–105] | 0% | 46 | 2,685 | - |
| mstar | verbatim-en | C | 2 | 10 | 103 [103–103] | 0% | 46 | 2,662 | 1.00x |
| mstar | verbatim-en | S | 2 | 10 | 128 [128–128] | 0% | 46 | 2,595 | 1.24x |
| mstar | verbatim-en | SC | 2 | 10 | 128 [126–130] | 0% | 46 | 2,685 | 1.24x |
| mstar | verbatim-en | W | 2 | 10 | 134 [134–134] | 0% | 46 | 2,612 | 1.30x |
| mstar | verbatim-en | WC | 2 | 10 | 130 [126–135] | 0% | 46 | 2,685 | 1.26x |
| mstar | verbatim-en | A2 | 2 | 10 | 102 [102–102] | 0% | 46 | 2,685 | 0.99x |
