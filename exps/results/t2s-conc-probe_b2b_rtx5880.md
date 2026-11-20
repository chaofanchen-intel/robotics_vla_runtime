# Qwen3-Omni T2S: what in a B = 16 wave makes seedtts_en_0590 (M*) / seedtts_en_0139 (SGLang-Omni) run to ~20.5 s of audio?

**Question.** t2s-conc-shuffle: with rows shuffled the failures stayed on one sentence per framework (M* 0590: 13 of 17 servings at B = 16; SGLang 0139: 3 of 18), clean at B = 1 / 4 (4 servings each at B = 4). So the sentence needs a large batch to fail. Is that the batch size itself, what the other 15 slots hold (long or short sentences, the same sentence), or the sentence's position in the wave?

**Pre-registered reading.** analyze_t2s_outliers.py per arm: the probe sentence is in every wave (40 measured + 5 precondition waves) at position wave % B, so each cell serves it 45 times and the failed / served column gives a rate per B. Rate rising with B under every fill = batch size; rate depends on fill (short or long neighbours, same sentence) = composition; rate depends on position (outlier table, position column) = slot. Neighbours are fixed per fill, so the 'random' fill is the shuffle-run condition at a known rate.

**Caveats.** Record-mode contract; one start per cell. The 'random' fill runs B = 4 / 8 / 12 / 16 (batch-size dependence), short / long / same only B = 16. B = 8 and 12 are new shapes for the warm-up. The probe sentence is not from the shuffle run's neighbour distribution, so a rate here is conditional on the fill, not the in-the-wild rate. 'same' fills the wave with 16 copies of one sentence (identical prompts in a batch).

Campaigns: `T2S_CONC_PROBE_M_SHORT_01_B2B` (m-short, back-to-back), `T2S_CONC_PROBE_M_LONG_01_B2B` (m-long, back-to-back), `T2S_CONC_PROBE_M_SAME_01_B2B` (m-same, back-to-back), `T2S_CONC_PROBE_M_RANDOM_01_B2B` (m-random, back-to-back), `T2S_CONC_PROBE_SG_SHORT_01_B2B` (sg-short, back-to-back), `T2S_CONC_PROBE_SG_LONG_01_B2B` (sg-long, back-to-back), `T2S_CONC_PROBE_SG_SAME_01_B2B` (sg-same, back-to-back), `T2S_CONC_PROBE_SG_RANDOM_01_B2B` (sg-random, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs m-short |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en-b16 | m-short | 1 | 40 | 31.2 [31.2–31.2] | 0% | 44 | 2,640 | - |
| mstar | verbatim-en-b16 | m-long | 1 | 40 | 31.6 [31.6–31.6] | 0% | 46 | 2,639 | 1.01x |
| mstar | verbatim-en-b16 | m-same | 1 | 40 | 14.1 [14.1–14.1] | 0% | 41 | 2,590 | 0.45x |
| mstar | verbatim-en-b16 | m-random | 1 | 40 | 32.1 [32.1–32.1] | 0% | 44 | 2,645 | 1.03x |
| mstar | verbatim-en-b4 | m-random | 1 | 40 | 46.7 [46.7–46.7] | 0% | 44 | 2,685 | - |
| mstar | verbatim-en-b8 | m-random | 1 | 40 | 34.0 [34.0–34.0] | 0% | 45 | 2,682 | - |
| mstar | verbatim-en-b12 | m-random | 1 | 40 | 28.9 [28.9–28.9] | 0% | 45 | 2,667 | - |
| sglang-omni | verbatim-en-b16 | sg-short | 1 | 40 | 19.0 [19.0–19.0] | 0% | 40 | 2,678 | - |
| sglang-omni | verbatim-en-b16 | sg-long | 1 | 40 | 20.0 [20.0–20.0] | 0% | 39 | 2,674 | - |
| sglang-omni | verbatim-en-b16 | sg-same | 1 | 40 | 14.1 [14.1–14.1] | 0% | 37 | 2,595 | - |
| sglang-omni | verbatim-en-b16 | sg-random | 1 | 40 | 21.3 [21.3–21.3] | 0% | 39 | 2,682 | - |
| sglang-omni | verbatim-en-b4 | sg-random | 1 | 40 | 47.1 [47.1–47.1] | 0% | 38 | 2,700 | - |
| sglang-omni | verbatim-en-b8 | sg-random | 1 | 40 | 31.1 [31.1–31.1] | 0% | 38 | 2,700 | - |
| sglang-omni | verbatim-en-b12 | sg-random | 1 | 40 | 24.3 [24.3–24.3] | 0% | 39 | 2,697 | - |
