# Pi0.5 on SGLang core: which of its two CUDA graphs (prefix or action expert) carries the latency?

**Question.** pi05-audit turned both SGLang core graphs off together (sglang-nograph). The prefix graph captures the images and the prompt; the action graph captures the 10-step expert loop. Which graph is the cost of running without graphs? M* captures both, so the answer bears on whether SGLang core's gap is a graph gap.

**Pre-registered reading.** Each arm against sglang (both graphs on). The arm that slows more holds the graph that matters; a change under 5% is not part of the gap (the pi05-audit rule). If the two single-graph penalties add up to pi05-audit's sglang-nograph, the graphs act independently; if they do not, the two interact.

**Caveats.** Counterfactual configs are copies of `pi05_pipeline.json` with one graph flag changed (exps/configs/sglang_core/). The M* split is not built: M*'s Pi0.5 graphs (SigLIP, prefill, action expert) are switched only together, by the `mstar_pi05_nograph` hook, and a per-graph split needs that hook to take a family list. Not built here.

Campaigns: `PI05_GRAPH_SPLIT_SGLANG_02_B2B` (sglang, back-to-back), `PI05_GRAPH_SPLIT_SGLANG_PREFIX_OFF_02_B2B` (sglang-prefix-off, back-to-back), `PI05_GRAPH_SPLIT_SGLANG_ACTION_OFF_02_B2B` (sglang-action-off, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs sglang |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| sglang-core | droid-46row | sglang | 2 | 24 | 162 [161–165] | 0% | 34 | 2,042 | - |
| sglang-core | droid-46row | sglang-prefix-off | 2 | 24 | 162 [161–163] | 0% | 34 | 1,990 | 1.00x |
| sglang-core | droid-46row | sglang-action-off | 2 | 24 | 162 [161–162] | 0% | 34 | 1,998 | 1.00x |
