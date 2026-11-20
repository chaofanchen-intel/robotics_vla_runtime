# BAGEL: GPU busy / kernel mix at equal work (nsys), M* vs vLLM two-stage

**Question.** Per-step GPU busy and launch count at equal work with vLLM-Omni tuned: is M*'s lead less GPU work per step?

**Pre-registered reading.** Busy per step and kernels by name; nsys E2E is not a benchmark number. The attention kernels' share says whether attention is the lever.

Campaigns: `BAGEL_NSYS_EQUAL_WORK_NSYS_01` (nsys, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | nsys | 1 | 2 | 2,682 [2,682–2,682] | 100% | 68 | 1,785 | - |
| mstar | t2i-512-cfg3 | nsys | 1 | 2 | 4,143 [4,143–4,143] | 0% | 68 | 1,493 | - |
| vllm-omni-two-stage | t2i-512-nocfg | nsys | 1 | 2 | 3,506 [3,506–3,506] | 100% | 66 | 1,785 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | nsys | 1 | 2 | 5,775 [5,775–5,775] | 0% | 69 | 1,720 | - |
