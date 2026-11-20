# BAGEL: fixed and per-step cost (request-level steps; M* server-side steps)

**Question.** E2E = a + b*steps: does a framework's gap live in the per-request fixed cost or in the per-step cost?

**Pre-registered reading.** Fit per framework; `srv-steps` are M*-only (its steps are server config). Steps cases use the default request, so compare a and b within a framework, not the absolute E2E across frameworks.

Campaigns: `BAGEL_STEPS_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-default | main | 2 | 10 | 13,009 [12,990–13,028] | 0% | 61 | 1,282 | - |
| mstar | t2i-512-steps5 | main | 2 | 10 | 12,938 [12,896–12,981] | 0% | 60 | 1,288 | - |
| mstar | t2i-512-steps25 | main | 2 | 10 | 12,910 [12,832–12,978] | 0% | 59 | 1,278 | - |
| mstar | t2i-512-srv-steps5 | main | 2 | 10 | 1,222 [1,220–1,228] | 0% | 48 | 1,907 | - |
| mstar | t2i-512-srv-steps25 | main | 2 | 10 | 6,262 [6,262–6,263] | 0% | 54 | 1,392 | - |
| vllm-omni-two-stage | t2i-512-default | main | 2 | 10 | 13,465 [13,443–13,487] | 0% | 36 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-steps5 | main | 2 | 10 | 1,389 [1,383–1,403] | 0% | 37 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-steps25 | main | 2 | 10 | 6,589 [6,589–6,590] | 0% | 37 | 2,460 | - |
| vllm-omni-single-stage | t2i-512-default | main | 2 | 10 | 13,043 [13,033–13,052] | 0% | 54 | 1,360 | - |
| vllm-omni-single-stage | t2i-512-steps5 | main | 2 | 10 | 1,316 [1,315–1,323] | 0% | 39 | 1,983 | - |
| vllm-omni-single-stage | t2i-512-steps25 | main | 2 | 10 | 6,362 [6,351–6,407] | 0% | 47 | 1,471 | - |
