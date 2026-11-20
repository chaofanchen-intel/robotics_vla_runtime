# BAGEL on M*: the LLM's 1,024-token chunked-prefill budget against 8,192 (I2T TTFT and 5-step I2I)

**Question.** M* cuts BAGEL's ~5k-token branch prefill into 1,024-token chunks (nsys: 168 prefill-attention launches = 6 chunks x 28 layers) while vLLM-Omni prefills it in one pass. With both caches on, M*'s I2T TTFT is 134 ms against vLLM's 85 ms and nothing explains the 44 ms; for Qwen the same budget turns out to be one whole extra forward per chunk. A dense 7B chunk of 1,024 tokens is compute-bound, so the per-chunk cost should be far smaller than Qwen's MoE ~40 ms. How large is it here?

**Pre-registered reading.** TTFT of i2t-512 and E2E of i2i-512-srv-steps5 by arm (analyze_decode_itl.py). chunk-8192 TTFT well below chunk-1024 (towards vLLM's 85 ms) = part of D2 is M*'s chunk budget; no change = the 44 ms is elsewhere. Both arms have the prefix cache at its default, so a cached prompt may skip chunks: the cache hit and the chunk count interact, read the I2I E2E as well (it did not move with the cache switch).

**Caveats.** One image size (512²). The budget of 8,192 is a counterfactual, not M*'s shipped config; it also lengthens the decode steps that share a step with a prefill chunk.

Campaigns: `BAGEL_PREFILL_CHUNK_CHUNK_1024_01_B2B` (chunk-1024, back-to-back), `BAGEL_PREFILL_CHUNK_CHUNK_8192_01_B2B` (chunk-8192, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs chunk-1024 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | i2t-512 | chunk-1024 | 2 | 10 | 2,288 [2,283–2,294] | 0% | 46 | 2,598 | - |
| mstar | i2t-512 | chunk-8192 | 2 | 10 | 2,282 [2,280–2,287] | 0% | 50 | 2,618 | 1.00x |
| mstar | i2i-512-srv-steps5 | chunk-1024 | 2 | 10 | 2,739 [2,736–2,741] | 0% | 49 | 1,601 | - |
| mstar | i2i-512-srv-steps5 | chunk-8192 | 2 | 10 | 2,734 [2,713–2,735] | 0% | 49 | 1,562 | 1.00x |
