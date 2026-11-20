# BAGEL I2I (512x512, 5 steps): B = 1 / 4 / 16 against vLLM-Omni two-stage (Mooncake + Thinker cache) with Thinker max_num_seqs 3 and 16

**Question.** bagel-concurrency-seqs: the I2T 3.7x lead at B = 16 was the upstream Thinker max_num_seqs=3; with it at 16 vLLM two-stage is 12% ahead of M*. Image editing has a second stage: every request also hands three 5k-token KV branches to the DiT, whose stage runs max_num_seqs=1. Does the DiT stage serialise concurrent I2I requests in vLLM-Omni, and how does M*'s single engine scale?

**Pre-registered reading.** analyze_concurrency.py per arm: wave wall time against B. A serialised DiT shows as wave time ~ B x the B = 1 time in both vllm arms, whatever the Thinker cap; M* below that line = it overlaps or batches requests. If seqs16 sits well under seqs3 the Thinker cap matters for I2I too (the three CFG branches of one request already fill 3 slots).

**Caveats.** The fixture repeats one prompt + image, so every request after the first is a hot prefix in both engines. Counterfactual deploy configs for seqs16, not the upstream recipe. A start failure at B = 16 (KV memory for 16 x 3 branches) is a result: read server.log. Server-side 5 steps for M*, request-level 5 steps for vLLM.

Campaigns: `BAGEL_I2I_CONCURRENCY_M_01_B2B` (m, back-to-back), `BAGEL_I2I_CONCURRENCY_VLLM_SEQS3_01_B2B` (vllm-seqs3, back-to-back), `BAGEL_I2I_CONCURRENCY_VLLM_SEQS16_01_B2B` (vllm-seqs16, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2i-512-srv-steps5 | m | 2 | 10 | 2,757 [2,754–2,760] | 0% | 50 | 1,591 | - |
| mstar | i2i-512-srv-steps5-b4 | m | 2 | 10 | 11,156 [11,129–11,182] | 0% | 58 | 1,384 | - |
| mstar | i2i-512-srv-steps5-b16 | m | 2 | 10 | 46,427 [46,403–46,450] | 0% | 69 | 1,223 | - |
| vllm-omni-two-stage | i2i-512-steps5 | vllm-seqs3 | 2 | 10 | 3,524 [3,504–3,538] | 0% | 39 | 2,420 | - |
| vllm-omni-two-stage | i2i-512-steps5 | vllm-seqs16 | 2 | 10 | 3,515 [3,494–3,515] | 0% | 36 | 2,416 | - |
| vllm-omni-two-stage | i2i-512-steps5-b4 | vllm-seqs3 | 2 | 10 | 10,989 [10,960–10,989] | 0% | 41 | 2,199 | - |
| vllm-omni-two-stage | i2i-512-steps5-b4 | vllm-seqs16 | 2 | 10 | 13,487 [13,467–13,493] | 0% | 40 | 2,230 | - |
| vllm-omni-two-stage | i2i-512-steps5-b16 | vllm-seqs3 | 2 | 10 | 97,173 [96,991–97,221] | 0% | 36 | 1,054 | - |
| vllm-omni-two-stage | i2i-512-steps5-b16 | vllm-seqs16 | 2 | 10 | 226,561 [226,074–226,598] | 0% | 29 | 531 | - |
