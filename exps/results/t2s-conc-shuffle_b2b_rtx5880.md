# Qwen3-Omni T2S at B = 1 / 4 / 16 with shuffled rows: is the ~20.5 s runaway tied to a sentence, a wave composition, or chance?

**Question.** t2s-conc-outliers: at B = 16, M* returned ~20.5 s of audio for fixture row seedtts_en_0590 and SGLang-Omni for seedtts_en_0870, each failing in both waves with the same composition (rows cycle every 25 waves) and passing in the other 4 times the row was served; no framework failed at B = 1 / 4, but B = 1 only reached rows 0-39. Does a failure follow the sentence, the batch neighbours, or neither (a per-response chance that grows with B)?

**Pre-registered reading.** analyze_t2s_outliers.py on a campaign where every pass over the 100 sentences is a fresh seeded permutation (qwen.rows=shuffle): the 'Outlier rows across batch sizes' table gives failed / served per row and B. The same row failing in different compositions = prompt-specific; failures on many different rows at B = 16 and none at B = 1 / 4 = a per-response chance that grows with concurrency; no failure at all = the earlier two were tied to that exact wave composition. B = 1 now serves all 100 rows (110 responses).

**Caveats.** One start per cell. Shuffle changes the warm-up property of cycle mode: the precondition phase (10 waves) at B = 1 / 4 does not cover all rows, so a first-seen-shape JIT in vLLM-Omni can add latency spikes to the timing; read the outlier table, not the p50, for this campaign.

Campaigns: `T2S_CONC_SHUFFLE_SHUFFLE_01_B2B` (shuffle, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs shuffle |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | verbatim-en | shuffle | 1 | 100 | 96.6 [96.6–96.6] | 0% | 43 | 2,685 | - |
| mstar | verbatim-en-b4 | shuffle | 1 | 100 | 46.6 [46.6–46.6] | 0% | 46 | 2,685 | - |
| mstar | verbatim-en-b16 | shuffle | 1 | 100 | 25.0 [25.0–25.0] | 0% | 47 | 2,673 | - |
| sglang-omni | verbatim-en | shuffle | 1 | 100 | 115 [115–115] | 0% | 40 | 2,666 | - |
| sglang-omni | verbatim-en-b4 | shuffle | 1 | 100 | 48.9 [48.9–48.9] | 0% | 42 | 2,699 | - |
| sglang-omni | verbatim-en-b16 | shuffle | 1 | 100 | 20.3 [20.3–20.3] | 0% | 43 | 2,683 | - |
| vllm-omni | verbatim-en | shuffle | 1 | 100 | 155 [155–155] | 0% | 38 | 2,680 | - |
| vllm-omni | verbatim-en-b4 | shuffle | 1 | 100 | 65.8 [65.8–65.8] | 0% | 41 | 2,700 | - |
| vllm-omni | verbatim-en-b16 | shuffle | 1 | 100 | 29.6 [29.6–29.6] | 0% | 43 | 2,686 | - |
