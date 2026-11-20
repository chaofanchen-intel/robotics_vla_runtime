# BAGEL I2T, fair profile: neutral sampler + prefix cache on in both, vLLM's yaml repetition penalty removed

**Question.** vLLM-Omni's BAGEL deploy yaml gives the Thinker repetition_penalty 1.05 / top_p 0.9 as server defaults and the request only sets temperature, so vLLM decodes with a repetition penalty M* does not apply (the two engines' captions differ for the same image). Does the penalty cost vLLM per-token time, and what is the fully fair I2T (neutral sampler, cache on in both)?

**Pre-registered reading.** ITL per arm. vllm-neutral below vllm-default: the penalty was a cost vLLM paid alone. vllm-fair (neutral + Thinker cache on) against m-fair (neutral + M* default cache) is the I2T comparison with nothing left unequal that we know of.

**Caveats.** Captions are not byte-identical across engines even with the same sampler (different kernels, near-tied tokens); lengths are pinned.

Campaigns: `BAGEL_SAMPLING_PARITY_VLLM_DEFAULT_01_B2B` (vllm-default, back-to-back), `BAGEL_SAMPLING_PARITY_VLLM_NEUTRAL_01_B2B` (vllm-neutral, back-to-back), `BAGEL_SAMPLING_PARITY_VLLM_FAIR_01_B2B` (vllm-fair, back-to-back), `BAGEL_SAMPLING_PARITY_M_DEFAULT_01_B2B` (m-default, back-to-back), `BAGEL_SAMPLING_PARITY_M_FAIR_01_B2B` (m-fair, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs vllm-default |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| vllm-omni-two-stage | i2t-512 | vllm-default | 2 | 10 | 2,976 [2,975–2,980] | 0% | 45 | 2,332 | - |
| vllm-omni-two-stage | i2t-512 | vllm-neutral | 2 | 10 | 2,978 [2,975–2,983] | 0% | 43 | 2,320 | 1.00x |
| vllm-omni-two-stage | i2t-512 | vllm-fair | 2 | 10 | 2,256 [2,256–2,256] | 0% | 43 | 2,689 | 0.76x |
| mstar | i2t-512 | m-default | 2 | 10 | 2,288 [2,283–2,293] | 0% | 48 | 2,571 | - |
| mstar | i2t-512 | m-fair | 2 | 10 | 2,280 [2,279–2,282] | 0% | 48 | 2,599 | - |
