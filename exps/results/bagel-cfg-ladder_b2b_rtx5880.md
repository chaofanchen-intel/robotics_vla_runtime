# BAGEL: E2E against the number of CFG branches (1 / default / 3)

**Question.** Does equal work (three branches on both) give equal time, and what does one extra branch cost per step on each framework? (Both engines compute the image-guidance branch only when cfg_img_scale > 1.)

**Pre-registered reading.** ms per extra branch per step per framework. `default` (cfg_img_scale=1.0) is two branches on both engines, `cfg3` three; the per-branch cost should agree across the two variants of one engine.

Campaigns: `BAGEL_CFG_LADDER_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-512-nocfg | main | 2 | 10 | 7,412 [7,386–7,438] | 0% | 55 | 1,379 | - |
| mstar | t2i-512-default | main | 2 | 10 | 12,984 [12,966–13,003] | 0% | 60 | 1,283 | - |
| mstar | t2i-512-cfg3 | main | 2 | 10 | 18,834 [18,795–18,872] | 0% | 65 | 1,205 | - |
| vllm-omni-two-stage | t2i-512-nocfg | main | 2 | 10 | 7,642 [7,641–7,644] | 0% | 36 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-default | main | 2 | 10 | 13,442 [13,414–13,470] | 0% | 37 | 2,460 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | main | 2 | 10 | 19,335 [19,320–19,349] | 0% | 36 | 2,292 | - |
| vllm-omni-single-stage | t2i-512-nocfg | main | 2 | 10 | 7,437 [7,436–7,439] | 0% | 48 | 1,469 | - |
| vllm-omni-single-stage | t2i-512-default | main | 2 | 10 | 13,053 [13,039–13,066] | 0% | 54 | 1,371 | - |
| vllm-omni-single-stage | t2i-512-cfg3 | main | 2 | 10 | 18,821 [18,806–18,836] | 0% | 58 | 1,295 | - |
