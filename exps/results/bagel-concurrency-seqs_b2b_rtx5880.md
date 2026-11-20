# BAGEL I2T: B = 1 / 4 / 16 with vLLM-Omni's Thinker max_num_seqs 3 (upstream recipe) / 8 / 16, against M*

**Question.** bagel-concurrency-fair (both caches on): M* wave time 2,293 / 2,665 / 3,647 ms at B = 1 / 4 / 16, vLLM two-stage 2,258 / 4,545 / 13,647. The vLLM numbers are almost exactly 1x / 2x / 6x the B = 1 time, and the upstream recipe sets the Thinker's `max_num_seqs: 3` (sized for three CFG branches). Is the 3.7x at B = 16 a scheduler cap rather than an engine property?

**Pre-registered reading.** analyze_concurrency.py per arm: wave wall time against B. Prediction if it is the cap: seqs3 = ceil(B/3) rounds (B=4: 2x, B=16: 6x the B=1 time), seqs8 = ceil(B/8) rounds (B=16: 2 rounds), seqs16 = one round, and the vLLM curve moves to or below M*'s. If seqs16 still sits far above M* at B = 16, the cap is not (only) the cause and the Thinker stage itself is slower per batched step; then read bagel-i2t-nsys style timelines at B = 16.

**Caveats.** The fixture repeats one prompt + image, so every request after the first is a hot prefix. max_num_seqs also changes KV memory planning in vLLM; a start failure at 16 is a result (read server.log). This is a counterfactual deploy config, not the upstream recipe: the fair-profile comparison in the report keeps the upstream value and states the cap.

Campaigns: `BAGEL_CONCURRENCY_SEQS_M_CACHE_ON_01_B2B` (m-cache-on, back-to-back), `BAGEL_CONCURRENCY_SEQS_VLLM_SEQS3_01_B2B` (vllm-seqs3, back-to-back), `BAGEL_CONCURRENCY_SEQS_VLLM_SEQS8_01_B2B` (vllm-seqs8, back-to-back), `BAGEL_CONCURRENCY_SEQS_VLLM_SEQS16_01_B2B` (vllm-seqs16, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m-cache-on |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2t-512 | m-cache-on | 2 | 10 | 2,297 [2,294–2,300] | 0% | 48 | 2,590 | - |
| mstar | i2t-512-b4 | m-cache-on | 2 | 10 | 2,670 [2,666–2,675] | 0% | 49 | 2,684 | - |
| mstar | i2t-512-b16 | m-cache-on | 2 | 10 | 3,654 [3,648–3,654] | 0% | 54 | 2,470 | - |
| vllm-omni-two-stage | i2t-512 | vllm-seqs3 | 2 | 10 | 2,253 [2,248–2,253] | 0% | 46 | 2,681 | - |
| vllm-omni-two-stage | i2t-512 | vllm-seqs8 | 2 | 10 | 2,254 [2,253–2,255] | 0% | 45 | 2,684 | - |
| vllm-omni-two-stage | i2t-512 | vllm-seqs16 | 2 | 10 | 2,257 [2,256–2,258] | 0% | 43 | 2,689 | - |
| vllm-omni-two-stage | i2t-512-b4 | vllm-seqs3 | 2 | 10 | 4,537 [4,534–4,537] | 0% | 46 | 2,682 | - |
| vllm-omni-two-stage | i2t-512-b4 | vllm-seqs8 | 2 | 10 | 2,359 [2,355–2,364] | 0% | 42 | 2,698 | - |
| vllm-omni-two-stage | i2t-512-b4 | vllm-seqs16 | 2 | 10 | 2,349 [2,341–2,351] | 0% | 41 | 2,698 | - |
| vllm-omni-two-stage | i2t-512-b16 | vllm-seqs3 | 2 | 10 | 13,673 [13,654–13,682] | 0% | 57 | 2,668 | - |
| vllm-omni-two-stage | i2t-512-b16 | vllm-seqs8 | 2 | 10 | 4,893 [4,892–4,893] | 0% | 47 | 2,686 | - |
| vllm-omni-two-stage | i2t-512-b16 | vllm-seqs16 | 2 | 10 | 3,221 [3,194–3,240] | 0% | 44 | 2,659 | - |
