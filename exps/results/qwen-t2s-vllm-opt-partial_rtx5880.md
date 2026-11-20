| campaign | framework | case | n | rows | stalls | audio s | E2E ms | ms / audio s | TTFA ms (steady) | after first audio, ms / audio s | fit a ms (all / steady) | fit b ms per audio s (all / steady) | R² (steady) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| QWEN_T2S_VLLM_OPT_VLLM_DEFAULT_03_B2B | vllm-omni | verbatim-en | 40 | 20 | 0 | 3.50 | 530 | 153.9 | 178 | 102.6 | 135 / 135 | 114.8 / 114.8 | 0.99 |
| QWEN_T2S_VLLM_OPT_VLLM_CODEPRED_03_B2B | vllm-omni | verbatim-en | 40 | 20 | 0 | 3.34 | 483 | 145.0 | 175 | 90.0 | 141 / 141 | 100.6 / 100.6 | 0.99 |

stall = a request whose time-to-first-audio exceeds twice its cell's median. `steady` excludes stalls. a = fixed latency, b = ms per second of audio produced. Single GPU set, C=1, one fixture family; localises the gap, does not name the kernel.

## Status (run 03, 2026-11-16)

- Arms analysed: `vllm-default` and `vllm-codepred` (2 starts × 20 sentences each, same batch). Columns above come from `analyze_t2s_fit.py` on these two campaigns only.
- `vllm-mrv2-codepred`: FAILED on both starts. Server ready after about 400 s; the first request hung (Thinker worker produced no reply for 60+ s, request aborted; later requests failed with no audio). No MPS control daemon was running, while the config comment requires MPS for stages 1 and 2 on the shared card. Cause not confirmed.
- `vllm-default-again` and `mstar`: skipped by the A/B/A order rule after the MRv2 failure. This batch has no M* comparison.
- Rerun: start MPS, then `ATTRIB_RUN_ID=04 ./run_attrib.sh qwen-t2s-vllm-opt` (new run id, whole group).
