# BAGEL at the paper's topology: 3-GPU CFG-parallel, M* vs vLLM-Omni (two deployments)

**Question.** The paper's headline BAGEL figure (T2I / I2I, B=1, 1024x1024) runs one rank per CFG branch on 3 GPUs (M* `Parallel` placement vs vLLM-Omni's CFG-parallel plugin); baseline-bagel* are its single-GPU Appendix E counterpart. Does M*'s lead survive the cross-GPU branch exchange?

**Pre-registered reading.** Per case: best vLLM-Omni deployment / M*, three branches on every engine. vLLM-Omni two-stage needs a 4th card on 48 GiB GPUs (Thinker cannot share a card with a DiT rank). No NVLink on the RTX 5880: the branch exchange crosses PCIe, so absolute numbers and possibly the ratio are not the paper's (NVLink H100).

**Caveats.** Needs >= 3 GPUs (4 for vLLM-Omni two-stage on 48 GiB cards): skipped on a single-GPU machine.

Campaigns: `BASELINE_BAGEL_CFGPAR_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests that started cool at full clock; T / clock = mean at request start / during it.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-1024-cfg3 | main | 2 | 6 | 29,845 [29,706–29,984] | 0% | 61 | 1,367 | - |
| mstar | i2i-1024 | main | 2 | 6 | 34,898 [34,822–34,973] | 0% | 64 | 1,380 | - |
| mstar | t2i-512-cfg3 | main | 2 | 6 | 7,878 [7,856–7,897] | 0% | 52 | 1,620 | - |
| mstar | i2i-512 | main | 2 | 6 | 9,253 [9,231–9,272] | 0% | 51 | 1,596 | - |
| vllm-omni-two-stage | t2i-1024-cfg3 | main | 2 | 6 | 30,725 [30,702–30,747] | 0% | 33 | 1,501 | - |
| vllm-omni-two-stage | i2i-1024 | main | 2 | 6 | 47,503 [47,306–47,646] | 0% | 34 | 1,255 | - |
| vllm-omni-two-stage | t2i-512-cfg3 | main | 2 | 6 | 7,908 [7,907–7,908] | 0% | 36 | 2,460 | - |
| vllm-omni-two-stage | i2i-512 | main | 2 | 6 | 16,654 [16,489–16,681] | 0% | 38 | 2,368 | - |
| vllm-omni-single-stage | t2i-1024-cfg3 | main | 2 | 6 | 31,065 [31,040–31,089] | 0% | 57 | 1,471 | - |
| vllm-omni-single-stage | i2i-1024 | main | 2 | 6 | 36,133 [36,042–36,223] | 0% | 60 | 1,425 | - |
| vllm-omni-single-stage | t2i-512-cfg3 | main | 2 | 6 | 7,938 [7,937–7,939] | 0% | 44 | 1,657 | - |
| vllm-omni-single-stage | i2i-512 | main | 2 | 6 | 9,529 [9,502–9,557] | 0% | 46 | 1,733 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-1024-cfg3 | mstar | main | 29,845 [29,706–29,984] | 1.00x |
| t2i-1024-cfg3 | vllm-omni-two-stage | main | 30,725 [30,702–30,747] | 1.03x |
| t2i-1024-cfg3 | vllm-omni-single-stage | main | 31,065 [31,040–31,089] | 1.04x |
| t2i-1024-cfg3 | **strongest baseline: vllm-omni-two-stage** | | 30,725 [30,702–30,747] | **1.03x** |
| i2i-1024 | mstar | main | 34,898 [34,822–34,973] | 1.00x |
| i2i-1024 | vllm-omni-two-stage | main | 47,503 [47,306–47,646] | 1.36x |
| i2i-1024 | vllm-omni-single-stage | main | 36,133 [36,042–36,223] | 1.04x |
| i2i-1024 | **strongest baseline: vllm-omni-single-stage** | | 36,133 [36,042–36,223] | **1.04x** |
| t2i-512-cfg3 | mstar | main | 7,878 [7,856–7,897] | 1.00x |
| t2i-512-cfg3 | vllm-omni-two-stage | main | 7,908 [7,907–7,908] | 1.00x |
| t2i-512-cfg3 | vllm-omni-single-stage | main | 7,938 [7,937–7,939] | 1.01x |
| t2i-512-cfg3 | **strongest baseline: vllm-omni-two-stage** | | 7,908 [7,907–7,908] | **1.00x** |
| i2i-512 | mstar | main | 9,253 [9,231–9,272] | 1.00x |
| i2i-512 | vllm-omni-two-stage | main | 16,654 [16,489–16,681] | 1.80x |
| i2i-512 | vllm-omni-single-stage | main | 9,529 [9,502–9,557] | 1.03x |
| i2i-512 | **strongest baseline: vllm-omni-single-stage** | | 9,529 [9,502–9,557] | **1.03x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
