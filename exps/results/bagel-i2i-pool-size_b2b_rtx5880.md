# BAGEL I2I B = 4 / 16 on vLLM-Omni two-stage (Thinker cap 3): how small can the Mooncake sender pool be?

**Question.** bagel-i2i-pool showed the 4 GiB sender pool exhausts at B = 16 (about 11 GB per wave) and 16 GiB removes the collapse. Which sizes in between serve B = 16: 8 GiB, 12 GiB? Each request puts about 680 MB into the pool.

**Pre-registered reading.** Count 'Pool exhausted' lines per arm and read the wave time. The pool-8 and pool-12 arms are NOT asserted to be free of exhaustion: an exhausted arm is the answer, recorded, not a failed cell. A size that serves B = 16 with no exhaustion line and a wave near the 16 GiB arm is the smallest working pool. If 12 GiB still exhausts, the working size is between 12 and 16 GiB.

**Caveats.** Same prompt and image in every request (hot prefix), as bagel-i2i-pool. The pool-size question is about the sender side only; the 900 s receive timeout is not varied here.

Campaigns: `BAGEL_I2I_POOL_SIZE_SEQS3_T30_POOL8_03_B2B` (seqs3-t30-pool8, back-to-back), `BAGEL_I2I_POOL_SIZE_SEQS3_T30_POOL12_03_B2B` (seqs3-t30-pool12, back-to-back), `BAGEL_I2I_POOL_SIZE_SEQS3_T30_POOL16_03_B2B` (seqs3-t30-pool16, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs seqs3-t30-pool8 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2i-512-steps5-b4 | seqs3-t30-pool8 | 2 | 10 | 10,979 [10,978–10,979] | 0% | 40 | 2,135 | - |
| vllm-omni-two-stage | i2i-512-steps5-b4 | seqs3-t30-pool12 | 2 | 10 | 11,031 [11,018–11,043] | 0% | 41 | 2,146 | 1.00x |
| vllm-omni-two-stage | i2i-512-steps5-b4 | seqs3-t30-pool16 | 2 | 10 | 11,018 [11,010–11,026] | 0% | 42 | 2,174 | 1.00x |
| vllm-omni-two-stage | i2i-512-steps5-b16 | seqs3-t30-pool8 | 2 | 10 | 37,242 [37,233–37,250] | 0% | 47 | 2,011 | - |
| vllm-omni-two-stage | i2i-512-steps5-b16 | seqs3-t30-pool12 | 2 | 10 | 37,040 [36,938–37,141] | 0% | 47 | 2,034 | 0.99x |
| vllm-omni-two-stage | i2i-512-steps5-b16 | seqs3-t30-pool16 | 2 | 10 | 37,513 [37,077–38,445] | 0% | 47 | 2,025 | 1.01x |
