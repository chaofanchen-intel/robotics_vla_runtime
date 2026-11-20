# M* Qwen3-Omni TP=2 Thinker prefix cache with the whole prompt in one chunk (is the chunk boundary the crash trigger?)

**Question.** qwen3omni_3gpu_thinker_tp2_prefix_cache.yaml (prefix_cache_tp: true, offload 0, the documented opt-in) crashes on the first cache hit: `TP ranks diverged: KV thinker_kv: rank 0 leased 8 cached pages for 0/main, whose stream here already holds tokens or has no keys` (RuntimeError from KVManager.adopt_cached_prefix on the follower; the config passed M*'s own load-time checks). The 8 leased pages are 1,024 tokens, exactly the first chunk of the 1,072-token prompt. M*'s unit tests (test_kv_prefix_tp.py, 14 pass here) use stubs, not the Qwen3-Omni path, and no shipped TP config turns prefix_cache_tp on.

**Pre-registered reading.** Read server.log (grep 'TP ranks diverged'). No crash and the measured requests complete = the chunked-row path under TP is the trigger (the prompt is one chunk here); the same crash = the chunk boundary is not it and the follower's chain / stored_len state is. Either way it is an M* defect in an opt-in path, to be reported with the log; this probe only narrows where. If it runs, E2E here against qwen-prefill-chunk's chunk-4096 is the prefix cache's TTFT effect.

**Caveats.** The analyze step reports nothing when the cell crashed again; the evidence is then the server.log line and the cell's requests.jsonl (503s).

Campaigns: `MSTAR_TP_CACHE_PROBE_TP_CACHE_CHUNK4096_01_B2B` (tp-cache-chunk4096, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs tp-cache-chunk4096 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | txt-1k-1tok | tp-cache-chunk4096 | 1 | 10 | 78.9 [78.9–78.9] | 0% | 45 | 2,463 | - |
