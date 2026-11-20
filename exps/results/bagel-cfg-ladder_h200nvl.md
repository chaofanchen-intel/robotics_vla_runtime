# BAGEL: E2E against the number of CFG branches (1 / default / 3)

**Question.** Does equal work (three branches on both) give equal time, and what does one extra branch cost per step on each framework? (Both engines compute the image-guidance branch only when cfg_img_scale > 1.)

**Pre-registered reading.** ms per extra branch per step per framework. `default` (cfg_img_scale=1.0) is two branches on both engines, `cfg3` three; the per-branch cost should agree across the two variants of one engine.

Campaigns: `BAGEL_CFG_LADDER_MAIN_01` (main, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | main | 2 | 10 | 2,152 [1,980–2,213] | 0% | 68 | 1,713 | - |
| mstar | t2i-512-default | main | 2 | 10 | 2,835 [2,830–2,899] | 0% | 69 | 1,583 | - |
| mstar | t2i-512-cfg3 | main | 2 | 10 | 3,868 [3,858–3,872] | 0% | 70 | 1,440 | - |
| vllm-omni-two-stage | t2i-512-nocfg | main | 2 | 10 | 2,532 [2,530–2,741] | 40% | 68 | 1,780 | - |
| vllm-omni-two-stage | t2i-512-default | main | 2 | 10 | 3,992 [3,849–4,204] | 0% | 69 | 1,759 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | main | 2 | 10 | 5,139 [5,138–5,141] | 0% | 69 | 1,661 | - |
| vllm-omni-single-stage | t2i-512-nocfg | main | 2 | 10 | 3,028 [2,940–3,046] | 100% | 67 | 1,785 | - |
| vllm-omni-single-stage | t2i-512-default | main | 2 | 10 | 4,108 [4,077–4,277] | 0% | 69 | 1,761 | - |
| vllm-omni-single-stage | t2i-512-cfg3 | main | 2 | 10 | 5,341 [5,330–5,352] | 0% | 69 | 1,672 | - |
