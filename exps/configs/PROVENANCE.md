# 服务配置来源与偏离说明

> **运行版本（2026-11-16）：**H200 与本机的 exps/ 运行使用同一组源码 refs：M* `9c6f1ad`（main）、vLLM-Omni v0.32.0（`5917730`）、SGLang-Omni v0.2.0（`e8c635e`）。H200 的文件由本机 scp 同步，版本没有差异。当前 refs 与环境兼容性见 [SOURCE-RECORD](../../external/SOURCE-RECORD.md)。
>
> 本文下表中的 v0.2.1、v0.30.0、v0.1.7 只是 2026-10-16 **复制配置文件时的出处**，不是 H200 或本机当前运行的版本。历史性能结论按各自保留的 run manifest 解释；当前 exps 不读取已删除的旧轮次目录。

运行时使用本目录中的配置副本，不在测量时直接读取 `external/` 或任何旧实验目录的配置。以下为 2026-10-16 复制时的来源及源码版本（即上面所说的出处）。改变配置会改变实验条件；每次运行应结合 `manifest.json` 中的服务命令、配置路径及外部仓库状态核对结果，不能只凭配置文件名判断是否等工作量。

## M\*：`mstar/`

来源 checkout：`b333630e5f1f6d2d0f899232a9024bce2f76bf47`（v0.2.1）。

| 本地文件 | 原始来源 | 是否原样复制 |
|---|---|---|
| `bagel_single_gpu.yaml` | `external/mstar/configs/bagel_single_gpu.yaml` | 是 |
| `vjepa2_ac.yaml` | `external/mstar/configs/vjepa2_ac.yaml` | 是 |
| `orpheus_colocated.yaml` | `external/mstar/configs/orpheus_colocated.yaml` | 是 |
| `cosmos3_nano.yaml` | `external/mstar/configs/cosmos3_nano.yaml` | 是 |
| `qwen3omni_1gpu.yaml` | 本机第一轮单卡适配配方；当前可运行快照在本目录 | 是 |
| `pi05_bfloat16.yaml` | 经过验证的 Pi0.5 BF16 配方；原始输入快照不随 exps 分发 | **否**，显式固定 `model_kwargs.num_flow_steps: 10` |
| `pi05_bfloat16_steps{1,5,20}.yaml` | 由 `pi05_bfloat16.yaml` 复制，仅 `model_kwargs.num_flow_steps` 改为 1 / 5 / 20（`pi05-steps` 反事实） | **否**，仅改步数 |

Pi0.5 的 M\* flow loop 在服务端配置，客户端请求无法改变它；另外两端请求显式传 10 步。若更改步数，必须同时修改服务配置与各端请求，并重新验证可比性。

**源码补丁不同于上述配置副本：**`exps/patches/mstar-04-bagel-vllm-request-parity.diff` 修改 pinned M\* 的 `benchmark/request.py` vLLM-Omni 适配器，保留每请求的 BAGEL 宽高等覆盖值，并在请求的 `model` 字段优先使用显式服务别名。`bootstrap.sh --verify` 会检查该补丁和哈希；缺失时不能运行本轮 BAGEL 等工作量比较。

## vLLM-Omni：`vllm_omni/`

来源 checkout：`a8576ccb725c4e21cd13c3eb5f9a546b21149d2b`（v0.30.0）。

| 本地文件 | 原始来源 | 是否原样复制 |
|---|---|---|
| `bagel_single_stage.yaml` | `external/vllm-omni/vllm_omni/deploy/bagel_single_stage.yaml` | 是 |
| `bagel.yaml` | `external/vllm-omni/vllm_omni/deploy/bagel.yaml`，双阶段配方 | 是 |
| `cosmos3_omni.yaml` | `external/vllm-omni/vllm_omni/deploy/cosmos3_omni.yaml` | 是 |
| `qwen3_omni_full_1gpu.yaml` | 单卡 Thinker/Talker/Code2Wav 配方；当前可运行快照在本目录 | 是 |
| `pi05_bfloat16.yaml` | 经过验证的 Pi0.5 BF16 配方；原始输入快照不随 exps 分发 | 是 |
| `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_pool16g.yaml` | `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache.yaml` | **否**，仅 Mooncake `memory_pool_size` 4 → 16 GiB（bagel-i2i-pool 反事实，不是基线） |
| `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_pool16g_kvtimeout900.yaml` | `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_kvtimeout900.yaml` | **否**，仅 `memory_pool_size` 4 → 16 GiB（bagel-i2i-pool 反事实） |
| `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16_pool16g.yaml` | `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16.yaml` | **否**，仅 `memory_pool_size` 4 → 16 GiB（bagel-i2i-pool 反事实） |
| `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16_pool16g_kvtimeout900.yaml` | `hardware/rtx5880/vllm_omni/bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16_kvtimeout900.yaml` | **否**，仅 `memory_pool_size` 4 → 16 GiB（bagel-i2i-pool 反事实） |

**BAGEL 双阶段注意事项：**`bagel.yaml` 中 stage 0 的种子取自请求，stage 1 扩散仍使用 YAML 种子；默认 `max_tokens: 2048`、`temperature: 0.4`。I2T 客户端会逐请求覆盖采样参数并断言实际输出长度；种子不对称影响像素，但不改变步数。修改这些值之后不要把新旧 run 混作同一配方。

**源码补丁（不是配置）：**`exps/patches/vllm-03-worker-connector-kv-transfer-port.diff` 修改 pinned vLLM-Omni 的 `omni_connector_runtime.py`，使 worker 侧 connector 使用 `kv_transfer` 端口段。它修复 Qwen Mooncake TCP 下 scheduler 与 worker 同绑 50051 的冲突。`vllm-01`、`vllm-02` 见 SOURCE-RECORD。`bootstrap.sh --verify` 校验其哈希与 ledger。

## SGLang-Omni：`sglang_omni/`

来源 checkout：`d58ca4f85b03ed40c7851cda9bde750bec3fcf8a`（v0.1.7）。

| 本地文件或目录 | 原始来源 | 是否原样复制 |
|---|---|---|
| `qwen3_omni_colocated_h200.yaml` | 本机 H200 colocated 配方 | **否**，带本机 H200 override |
| `moe_h200nvl/` | H200 Triton MoE kernel 配置 | 是 |

`talker_ar.engine` 下的 `moe_runner_backend: triton` 与 `disable_flashinfer_autotune: true` 是本机 H200 override；否则 Talker 首次预热可能在绑定 API 端口前触发长时间 FlashInfer JIT autotune。该改动保存在**本地配置**而非 `external/sglang-omni` 源码中；启动命令以 `SGLANG_MOE_CONFIG_DIR` 指向 `moe_h200nvl/`，外部 checkout 因此可保持干净。

## SGLang core：`sglang_core/`

同一 SGLang 来源 commit。`pi05_pipeline.json` 是 exps 内维护的冻结 Pi0.5 pipeline 配置；其原始输入快照已不随本目录分发。`pi05_pipeline_graph_off.json`、`pi05_pipeline_prefix_cache.json`、`pi05_pipeline_prefix_graph_off.json`、`pi05_pipeline_action_graph_off.json` 均从该本地基线派生，各只改一个开关（分别为 pi05-audit、pi05-kv、pi05-graph-split 的反事实）。`pi05_pipeline_buckets200.json` 同样派生，只加 `prompt_token_buckets: [200]`（等于 `max_token_len`，不是调过的值），是 pi05-sglang-buckets 的反事实。Pi0.5 经 `sglang.multimodal_gen.runtime.launch_server` 启动 `Pi05Pipeline`；这是 **SGLang core，不是 SGLang-Omni Stage**，结果不可改标为后者。
