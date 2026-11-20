# Cosmos3-Nano T2I 256x256: M* against vLLM-Omni and SGLang-Omni (direct-generation pipeline)

**Question.** baseline-cosmos3: M* 1,290 ms against vLLM-Omni 3,133 ms (2.46x), decomposed into batched CFG 1.54x x compile 1.38x x host launch gaps 1.17x. SGLang-Omni serves Cosmos3 through native SGLang multimodal_gen (Cosmos3PipelineConfig; docs/cookbook/cosmos3_nano.md) and was never run. Where does it sit, and is the 2.46x a vLLM-Omni property or an M* one?

**Pre-registered reading.** Same 5 prompts, 256x256, 50 steps, guidance 6.0, interval [400, 1000], seed 42, one image per response, size verified from the PNG header. SGLang-Omni guardrails are off (SGLANG_DISABLE_COSMOS3_GUARDRAILS=1, as vLLM-Omni's `guardrails: false`). SGLang between the two = the structure story holds in a third implementation; SGLang near vLLM = the gap is what M* does (batched CFG, compile); SGLang at or below M* = vLLM-Omni's Cosmos3 pipeline is the slow one. First start of the SGLang arm may include native warmup: check start 1 against 2-3.

**Caveats.** First run of the SGLang-Omni Cosmos3 arm in this harness: if the server does not come up, read server.log (this is a result). The direct-generation pipeline, not the interleaved Reasoner pipeline. The M* config is the rtx5880 one-GPU overlay; the SGLang stage is placed on GPU 0 of the cell's visible devices.

Campaigns: `COSMOS3_SGLANG_MAIN_01_B2B` (main, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs main |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | t2i-256 | main | 3 | 15 | 1,321 [1,279–1,359] | 0% | 43 | 2,348 | - |
| vllm-omni | t2i-256 | main | 3 | 15 | 3,127 [3,082–3,170] | 0% | 42 | 2,694 | - |
| sglang-omni | t2i-256 | main | 3 | 15 | 4,743 [4,741–4,759] | 0% | 43 | 2,671 | - |

## Each framework at its best configuration, per case

| case | framework | best arm | p50 | S = framework / M* |
|---|---|---|---:|---:|
| t2i-256 | mstar | main | 1,321 [1,279–1,359] | 1.00x |
| t2i-256 | vllm-omni | main | 3,127 [3,082–3,170] | 2.37x |
| t2i-256 | sglang-omni | main | 4,743 [4,741–4,759] | 3.59x |
| t2i-256 | **strongest baseline: vllm-omni** | | 3,127 [3,082–3,170] | **2.37x** |

S > 1 means M* is faster. The strongest-baseline row is the claim a headline may quote; the others are per-framework context. All rows share the protocol named in the campaigns.
