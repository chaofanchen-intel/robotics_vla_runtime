# BAGEL: fixed and per-step cost (request-level steps; M* server-side steps)

**Question.** E2E = a + b*steps: does a framework's gap live in the per-request fixed cost or in the per-step cost?

**Pre-registered reading.** Fit per framework; `srv-steps` are M*-only (its steps are server config). Steps cases use the default request, so compare a and b within a framework, not the absolute E2E across frameworks.

Campaigns: `BAGEL_STEPS_MAIN_01` (main, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-default | main | 2 | 10 | 2,901 [2,885–2,920] | 0% | 69 | 1,570 | - |
| mstar | t2i-512-steps5 | main | 2 | 10 | 2,882 [2,845–2,888] | 0% | 69 | 1,584 | - |
| mstar | t2i-512-steps25 | main | 2 | 10 | 2,909 [2,896–2,922] | 0% | 69 | 1,570 | - |
| mstar | t2i-512-srv-steps5 | main | 2 | 10 | 453 [410–465] | 100% | 65 | 1,785 | - |
| mstar | t2i-512-srv-steps25 | main | 2 | 10 | 1,543 [1,515–1,561] | 0% | 66 | 1,625 | - |
| vllm-omni-two-stage | t2i-512-default | main | 2 | 10 | 3,896 [3,770–3,998] | 0% | 69 | 1,750 | - |
| vllm-omni-two-stage | t2i-512-steps5 | main | 2 | 10 | 713 [712–716] | 100% | 65 | 1,785 | - |
| vllm-omni-two-stage | t2i-512-steps25 | main | 2 | 10 | 2,238 [2,207–2,272] | 10% | 68 | 1,779 | - |
| vllm-omni-single-stage | t2i-512-default | main | 2 | 10 | 4,236 [4,228–4,244] | 0% | 69 | 1,766 | - |
| vllm-omni-single-stage | t2i-512-steps5 | main | 2 | 10 | 774 [771–775] | 90% | 62 | 1,784 | - |
| vllm-omni-single-stage | t2i-512-steps25 | main | 2 | 10 | 2,365 [2,358–2,414] | 10% | 67 | 1,777 | - |
