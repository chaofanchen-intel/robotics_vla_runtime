# BAGEL: GPU busy / kernel mix at equal work (nsys), M* vs vLLM two-stage

**Question.** Per-step GPU busy and launch count at equal work with vLLM-Omni tuned: is M*'s lead less GPU work per step?

**Pre-registered reading.** Busy per step and kernels by name; nsys E2E is not a benchmark number. The attention kernels' share says whether attention is the lever.

Campaigns: `BAGEL_NSYS_EQUAL_WORK_NSYS_01_B2B` (nsys, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs nsys |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | nsys | 1 | 2 | 7,310 [7,310–7,310] | 0% | 48 | 1,467 | - |
| mstar | t2i-512-cfg3 | nsys | 1 | 2 | 18,228 [18,228–18,228] | 0% | 53 | 1,267 | - |
| vllm-omni-two-stage | t2i-512-nocfg | nsys | 1 | 2 | 7,624 [7,624–7,624] | 0% | 37 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | nsys | 1 | 2 | 19,046 [19,046–19,046] | 0% | 37 | 2,460 | - |
