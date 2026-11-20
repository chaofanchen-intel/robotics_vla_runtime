# BAGEL T2I (512x512, 5 steps): B = 1 / 4 / 16, M* against vLLM-Omni two-stage (Mooncake + Thinker cache), KV timeout 30 s and 900 s

**Question.** Concurrency has been measured for I2T and I2I only. T2I has no input image, so the Thinker prefill is small and the three CFG branches' KV is smaller than for I2I; does the DiT stage still serialise concurrent requests, and does vLLM two-stage collapse at B = 16 like I2I?

**Pre-registered reading.** analyze_concurrency.py per arm: wave time against B, throughput (req/s) flat = serialised. The t900 arm separates a timeout effect from DiT speed exactly as in bagel-i2i-kvtimeout.

**Caveats.** Server-side 5 steps for M*, request-level 5 steps for vLLM. The 30 s arm has no timeout Expect on purpose: timeouts there are a result. T2I here is the default two-branch CFG request.

Campaigns: `BAGEL_T2I_CONCURRENCY_M_01_B2B` (m, back-to-back), `BAGEL_T2I_CONCURRENCY_VLLM_SEQS3_01_B2B` (vllm-seqs3, back-to-back), `BAGEL_T2I_CONCURRENCY_VLLM_SEQS3_T900_01_B2B` (vllm-seqs3-t900, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-srv-steps5 | m | 2 | 10 | 1,200 [1,196–1,212] | 0% | 43 | 1,940 | - |
| mstar | t2i-512-srv-steps5-b4 | m | 2 | 10 | 4,644 [4,642–4,646] | 0% | 48 | 1,465 | - |
| mstar | t2i-512-srv-steps5-b16 | m | 2 | 10 | 19,035 [18,861–19,103] | 0% | 57 | 1,300 | - |
| vllm-omni-two-stage | t2i-512-steps5 | vllm-seqs3 | 2 | 10 | 1,408 [1,400–1,413] | 0% | 34 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-steps5 | vllm-seqs3-t900 | 2 | 10 | 1,417 [1,415–1,420] | 0% | 33 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-steps5-b4 | vllm-seqs3 | 2 | 10 | 4,984 [4,962–5,004] | 0% | 32 | 2,502 | - |
| vllm-omni-two-stage | t2i-512-steps5-b4 | vllm-seqs3-t900 | 2 | 10 | 5,010 [5,008–5,011] | 0% | 33 | 2,523 | - |
| vllm-omni-two-stage | t2i-512-steps5-b16 | vllm-seqs3 | 2 | 10 | 19,893 [19,863–19,924] | 0% | 32 | 2,237 | - |
| vllm-omni-two-stage | t2i-512-steps5-b16 | vllm-seqs3-t900 | 2 | 10 | 19,933 [19,929–19,936] | 0% | 33 | 2,249 | - |
