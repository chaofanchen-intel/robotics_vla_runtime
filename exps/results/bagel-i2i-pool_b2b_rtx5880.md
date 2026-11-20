# BAGEL I2I B = 4 / 16 on vLLM-Omni two-stage with the Mooncake sender pool at 16 GiB: is the B = 16 collapse the pool or the receive timeout?

**Question.** At B = 16 the vLLM-Omni two-stage I2I wave took 97 s (Thinker cap 3) and 226 s (cap 16) against 46 s for M*, with 'Timeout waiting for KV cache ... after 30.0s' in server.log. The Mooncake sender pool is 4 GiB and each request puts about 680 MB into it (main KV plus the cfg_text branch), so a 16-request wave needs about 11 GB: 'Pool exhausted' appears in every B = 16 campaign, the 30 s ones and the 900 s ones alike. bagel-i2i-kvtimeout changed only the receive timeout, and its B = 16 cell failed on the pool. Is the collapse the pool, the 30 s receive timeout, or a slow DiT?

**Pre-registered reading.** Decisive pair at B = 16 with the pool at 16 GiB: seqs3-t30-pool16 against seqs3-t900-pool16. If the 30 s arm finishes near the 900 s arm, both near B x the B = 1 wave time of bagel-i2i-concurrency (this group has no B = 1 cell), and neither logs 'Pool exhausted', the earlier collapse was the pool and the receive timeout is not implicated. If the 30 s arm still collapses (wave well above the 900 s arm) with no 'Pool exhausted' line, the timeout is implicated. If both collapse alike, the DiT stage is the bottleneck. The seqs16 pair repeats the test at Thinker cap 16. Count the timeout lines per arm and compare output sha256 across arms, as in bagel-i2i-kvtimeout.

**Caveats.** Counterfactual configs (pool16g): the Mooncake sender pool is 16 GiB instead of 4 GiB. Besides that, the t900 arms differ from the t30 arms only in the DiT receive timeout, and the seqs16 arms only in Thinker max_num_seqs. The cap-16 arms take about 2 h because a 16-request wave takes about 226 s there; the cap-3 pair alone answers the question. The Expect forbids 'Pool exhausted' in every arm, so a cell whose wave exhausts the pool is rejected at analysis rather than counted.

Campaigns: `BAGEL_I2I_POOL_SEQS3_T30_POOL16_02_B2B` (seqs3-t30-pool16, back-to-back), `BAGEL_I2I_POOL_SEQS3_T900_POOL16_02_B2B` (seqs3-t900-pool16, back-to-back), `BAGEL_I2I_POOL_SEQS16_T30_POOL16_02_B2B` (seqs16-t30-pool16, back-to-back), `BAGEL_I2I_POOL_SEQS16_T900_POOL16_02_B2B` (seqs16-t900-pool16, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs seqs3-t30-pool16 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2i-512-steps5-b4 | seqs3-t30-pool16 | 2 | 10 | 10,991 [10,986–11,051] | 0% | 40 | 2,158 | - |
| vllm-omni-two-stage | i2i-512-steps5-b4 | seqs3-t900-pool16 | 2 | 10 | 11,034 [11,024–11,061] | 0% | 42 | 2,184 | 1.00x |
| vllm-omni-two-stage | i2i-512-steps5-b4 | seqs16-t30-pool16 | 2 | 10 | 13,483 [13,462–13,514] | 0% | 41 | 2,221 | 1.23x |
| vllm-omni-two-stage | i2i-512-steps5-b4 | seqs16-t900-pool16 | 2 | 10 | 13,513 [13,498–13,535] | 0% | 40 | 2,243 | 1.23x |
| vllm-omni-two-stage | i2i-512-steps5-b16 | seqs3-t30-pool16 | 2 | 10 | 37,224 [37,176–37,271] | 0% | 48 | 2,044 | - |
| vllm-omni-two-stage | i2i-512-steps5-b16 | seqs3-t900-pool16 | 2 | 10 | 37,076 [36,981–37,228] | 0% | 47 | 2,042 | 1.00x |
| vllm-omni-two-stage | i2i-512-steps5-b16 | seqs16-t30-pool16 | 2 | 10 | 44,099 [43,965–44,397] | 0% | 43 | 1,723 | 1.18x |
| vllm-omni-two-stage | i2i-512-steps5-b16 | seqs16-t900-pool16 | 2 | 10 | 44,193 [43,896–44,279] | 0% | 43 | 1,716 | 1.19x |
