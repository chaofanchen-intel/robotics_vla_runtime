| campaign | framework | case | n | rows | stalls | audio s | E2E ms | ms / audio s | TTFA ms (steady) | after first audio, ms / audio s | fit a ms (all / steady) | fit b ms per audio s (all / steady) | R² (steady) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|
| QWEN_T2S_VLLM_OPT_VLLM_DEFAULT_04_B2B_r02only | vllm-omni | verbatim-en | 20 | 20 | 0 | 3.46 | 529 | 153.4 | 172 | 100.5 | 139 / 139 | 111.4 / 111.4 | 0.97 |
| QWEN_T2S_VLLM_OPT_VLLM_CODEPRED_04_B2B | vllm-omni | verbatim-en | 40 | 20 | 0 | 3.42 | 478 | 140.2 | 174 | 89.2 | 143 / 143 | 97.4 / 97.4 | 0.96 |
| QWEN_T2S_VLLM_OPT_MSTAR_04_B2B | mstar | verbatim-en | 40 | 20 | 0 | 3.38 | 326 | 95.1 | 155 | 50.3 | 99 / 99 | 66.4 / 66.4 | 1.00 |

Reference only, not used for the comparison: QWEN_T2S_VLLM_OPT_VLLM_DEFAULT_04_B2B pooled over both starts (r01 from 2026-11-15 19:51 UTC, an earlier session, and r02 from 2026-11-18): n 40, E2E 535 ms, 153.7 ms / audio s, fit b 111.9.

stall = a request whose time-to-first-audio exceeds twice its cell's median. `steady` excludes stalls. a = fixed latency, b = ms per second of audio produced. Single GPU set (physical GPUs 1, 2, 3), C=1, one fixture family, back-to-back; localises the gap, does not name the kernel. E2E ms is the median over all measured requests in the rows shown (`exps/tools/analyze_t2s_fit.py`); the per-cell headline p50 in each campaign's summary uses a different subset and is not used here.

Command: `envs/mstar/bin/python exps/tools/analyze_t2s_fit.py <default r02 only> exps/runs/QWEN_T2S_VLLM_OPT_VLLM_CODEPRED_04_B2B exps/runs/QWEN_T2S_VLLM_OPT_MSTAR_04_B2B`. The default r02-only row uses a symlink directory in /tmp that points at `exps/runs/QWEN_T2S_VLLM_OPT_VLLM_DEFAULT_04_B2B/QWEN_T2S/vllm-omni/verbatim-en/r02`.

Runs: the default and code-predictor arms came from `ATTRIB_RUN_ID=04 ATTRIB_GPUS=1,2,3 ./run_attrib.sh qwen-t2s-vllm-opt` (stopped at the MRv2 arm). The M* arm was run alone with `envs/mstar/bin/python -m exps.suite run qwen-t2s-vllm-opt --hardware rtx5880 --protocol back-to-back --run-id 04 --gpus 1,2,3 --arms mstar --execute`.

## Status (run 04, 2026-11-18)

- Arms analysed: `vllm-default` (same-session start r02 only), `vllm-codepred` (2 starts), `mstar` (2 starts, default config with torch.compile on for T2S). All on physical GPUs 1, 2, 3, no MPS daemon running.
- Gain from the code-predictor switches, same session: 153.4 to 140.2 ms per audio second (−8.6%); E2E 529 to 478 ms (−9.6%). The earlier batch (run 03, 2026-11-16, see `qwen-t2s-vllm-opt-partial_rtx5880.md`) gave 153.9 to 145.0 (−5.8%).
- vLLM default to M*: 1.61× per audio second, 1.62× E2E. Code predictor on: 1.47×.
- `vllm-mrv2-codepred`: FAILED on both starts. The first request got no response for 120 s; the stage-0 engine then exited and all later requests failed. Same pattern as run 03. No MPS daemon was running for these starts; whether MRv2 works with MPS is untested. Not investigated further (decision on 2026-11-18).
- `vllm-default-again`: not run, skipped by the A/B/A order after the MRv2 failure.
- Earlier attempt of this group on 2026-11-18 01:00 (leftover MPS daemon on GPUs 1, 2, 3): the default r02 start failed (a stage-1 worker reported "No CUDA GPUs are available"). That attempt is not used; its server log is `exps/runs/QWEN_T2S_VLLM_OPT_VLLM_DEFAULT_04_B2B/_service_lifecycles/qwen_t2s_vllm-omni_verbatim-en_r02/server.failed-under-mps.log`.
