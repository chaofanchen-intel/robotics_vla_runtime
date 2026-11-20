# Pi0.5 on SGLang core: is the shipped config its best? Prompt-length buckets against the default, 46-row fixture

**Question.** The shipped SGLang core Pi0.5 config leaves prompt_token_buckets empty. Its denoise CUDA graph is keyed by the exact prefix length and holds 4 signatures (action_cuda_graph_max_entries); once full, new signatures run eagerly ('VLA denoise CUDA graph capacity reached'). The 46 rows give several prefix lengths, so most denoise steps may run eager, while a repeated row (pi05-kv) hits the graph. Padding every prompt to one bucket gives one signature. Does the best config for SGLang core move its ratio to M*?

**Pre-registered reading.** sglang (shipped) against sglang-buckets (prompt_token_buckets [200], the config's max_token_len, so no prompt misses a bucket). A drop of 5% or more is a config effect and replaces the SGLang row of baseline-pi05; a smaller change means the eager fallback was not the cost. The buckets arm must log no capacity message.

**Caveats.** Counterfactual config = `pi05_pipeline.json` plus prompt_token_buckets [200] (exps/configs/sglang_core/). 200 is the config's max_token_len, not a tuned value: a tighter bucket could be faster and is not tested. The captured requests in the baseline log had 140 to 148 prompt tokens, so the padding is about 50 tokens. Latency only; actions are shape-checked.

Campaigns: `PI05_SGLANG_BUCKETS_SGLANG_02_B2B` (sglang, back-to-back), `PI05_SGLANG_BUCKETS_SGLANG_BUCKETS_02_B2B` (sglang-buckets, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-core | droid-46row | sglang | 2 | 24 | 167 [166–167] | 0% | 33 | 2,064 | - |
| sglang-core | droid-46row | sglang-buckets | 2 | 24 | 92.9 [92.2–93.3] | 0% | 33 | 2,127 | 0.56x |
