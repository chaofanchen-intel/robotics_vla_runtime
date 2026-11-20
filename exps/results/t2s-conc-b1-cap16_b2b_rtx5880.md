# Qwen3-Omni T2S: vLLM-Omni at B = 1 on the B > 1 deploy (max_num_seqs 16) against its C = 1 deploy (max_num_seqs 1)

**Question.** In t2s-concurrency-fair, vLLM-Omni's B = 1 point runs the C = 1 recipe (max_num_seqs 1) while B = 4 and B = 16 run the max_num_seqs 16 variant, so the first point of the curve comes from a different deploy. Does the B = 1 number move when the same single request runs on the 16-cap overlay?

**Pre-registered reading.** Compare the vllm-omni B = 1 row here with the B = 1 row of t2s-concurrency-fair (same warm rows and counts, two starts here against one there). If the two agree within the start spread, the cap does not touch the B = 1 point and the curve is fine as it is.

**Caveats.** Only the vLLM-Omni arm is run: M* and SGLang-Omni already take the same B = 1 cell in t2s-concurrency-fair, and their configs do not change with B. The cap does not limit a single request, so a difference here would come from graph capture sizes or KV pool size, not from scheduling.

Campaigns: `T2S_CONC_B1_CAP16_CAP16_01_B2B` (cap16, back-to-back)

Headline = pooled p50 of ms per audio-second over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | ms per audio-second p50 [start range] | clean | T °C | SM MHz | vs cap16 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni | verbatim-en | cap16 | 2 | 12 | 156 [150–160] | 0% | 37 | 2,700 | - |
