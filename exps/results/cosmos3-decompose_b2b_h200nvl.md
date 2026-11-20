# Cosmos3: M*'s 2.4x over vLLM-Omni, switched off one implementation choice at a time

**Question.** M* takes 1.29 s and vLLM-Omni 3.13 s for the same 50-step 256x256 image. M* packs cond + uncond into one forward (batched CFG), may replay a CUDA graph (its captured tiers are 192x320 and 480x832: is 256x256 one?) and compiles. If M* is given vLLM's structure (two sequential forwards, no graph, no compile), does the 2.4x disappear?

**Pre-registered reading.** E2E per arm against `vllm`. m-nograph ~ m-control: the graph is not used at 256x256. m-seqcfg: what batching the two CFG branches is worth. m-all-off ~ vllm: the gap is these three choices and nothing else (then it is implementation, not architecture). m-capture256: what a 256x256 graph would add. A leftover gap after m-all-off is unexplained.

**Caveats.** Counterfactual M* arms are not baselines. Sequential CFG changes M*'s graph/eager path selection; outputs are not compared (M*'s own docs say the two paths do the same math).

Campaigns: `COSMOS3_DECOMPOSE_M_CONTROL_01_B2B` (m-control, back-to-back), `COSMOS3_DECOMPOSE_M_NOGRAPH_01_B2B` (m-nograph, back-to-back), `COSMOS3_DECOMPOSE_M_SEQCFG_01_B2B` (m-seqcfg, back-to-back), `COSMOS3_DECOMPOSE_M_SEQCFG_NOGRAPH_01_B2B` (m-seqcfg-nograph, back-to-back), `COSMOS3_DECOMPOSE_M_ALL_OFF_01_B2B` (m-all-off, back-to-back), `COSMOS3_DECOMPOSE_M_CAPTURE256_01_B2B` (m-capture256, back-to-back), `COSMOS3_DECOMPOSE_VLLM_01_B2B` (vllm, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs m-control |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | m-control | 3 | 30 | 1,096 [1,093–1,136] | 100% | 66 | 1,785 | - |
| mstar | t2i-256 | m-nograph | 3 | 30 | 1,133 [1,104–1,137] | 100% | 64 | 1,785 | 1.03x |
| mstar | t2i-256 | m-seqcfg | 3 | 30 | 1,915 [1,791–2,125] | 100% | 65 | 1,785 | 1.75x |
| mstar | t2i-256 | m-seqcfg-nograph | 3 | 30 | 1,925 [1,920–2,159] | 100% | 66 | 1,785 | 1.76x |
| mstar | t2i-256 | m-all-off | 3 | 30 | 2,427 [2,320–2,442] | 100% | 66 | 1,785 | 2.22x |
| mstar | t2i-256 | m-capture256 | 3 | 30 | 388 [388–389] | 3% | 71 | 1,758 | 0.35x |
| vllm-omni | t2i-256 | vllm | 3 | 30 | 1,357 [1,324–1,381] | 100% | 65 | 1,785 | - |
