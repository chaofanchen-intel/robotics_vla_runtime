# 硬件覆盖配置（config overlay）

`--hardware rtx5880`（或自动检测到 5880）时，所有服务配置先在 `exps/configs/hardware/rtx5880/` 里找 overlay，找不到才用 `exps/configs/` 里的默认。overlay 的文件名写明拓扑（如 `qwen3omni_3gpu_thinker_tp2.yaml`），默认配置名到 overlay 文件名的对应在档案 `profiles/hardware/rtx5880.json` 的 `config_overlay_files` 里；没有对应项的配置按**同样的相对路径**找。解析后的路径写进每个 cell 的服务命令（manifest 里可查）。不改默认配置，就能换拓扑 / 显存预算。

也可以不放文件，直接用命令行替换某个框架家族的配置：`--opt config.mstar=PATH`、`--opt config.vllm-omni=PATH`、`--opt config.sglang=PATH`（对该次运行里这个家族的所有 cell 生效，所以一次只跑一个任务）。5880 工作站是共享机器，所有物理卡都可用：不传 `--gpus` 时每个 cell 自动取空闲卡（张数见档案的 `gpus_per_task`），传了就固定使用所给的卡，例如 `--gpus 2,3,5`。配置里的 `ranks` / `devices` 是 `CUDA_VISIBLE_DEVICES` 重新编号后的逻辑序号。

## 为什么 5880 需要它

- **Qwen3-Omni（T2S / TXT / A2T）**：30B BF16 约 61 GB，**一张 48 GB 的 5880 放不下**。`rtx5880` 档案把这三个任务标成 `needs_overlay_for_tasks`：没有 overlay 配置就拒绝规划，不会等到起服务才 OOM。
- **BAGEL**：M* 与 vLLM single-stage 应能放进 48 GB；vLLM two-stage 可能在两个 stage 各存一份 LLM 权重（≈2×29 GB），一张卡可能放不下，OOM 时给它两张卡。
- 论文里的拓扑（Qwen3-Omni 2×H200 把 Thinker / Talker 分卡，BAGEL 3×H100 做 CFG 并行）只能在多卡上复现。

## RTX 5880 多卡 overlays

| 路径 | 拓扑 |
|---|---|
| `mstar/qwen3omni_3gpu_thinker_tp2.yaml`（替代 `mstar/qwen3omni_1gpu.yaml`） | 3 卡：Thinker TP=2（卡 0–1）、Talker + Code2Wav（卡 2）；小 KV 池，最大序列长度 8192 |
| `vllm_omni/qwen3_omni_3gpu_thinker_tp2_async_c2wgraph.yaml`（替代 `..._1gpu_default_async_c2wgraph.yaml`） | 3 卡：Thinker TP=2（卡 0–1）、Talker + Code2Wav 共用卡 2（显存占比 0.6 + 0.2） |
| `sglang_omni/qwen3_omni_3gpu_thinker_tp2_partial_start.yaml`（替代 `..._colocated_h200_partial_start.yaml`） | 3 卡，非 colocated speech pipeline：Thinker TP=2（卡 0–1）、Talker（卡 2），编码器与 Thinker rank 0 同卡、Code2Wav 与 Talker 同卡；关闭 custom all-reduce，避免 Ada 上 CUDA-graph IPC illegal access |
| `mstar/bagel_single_gpu_kv512.yaml`（替代 `mstar/bagel_single_gpu.yaml`） | 1 卡，KV 池限到 512 页：默认 2048 页（约 15 GB）加 29 GB 权重在 48 GiB 卡上 OOM |
| `mstar/bagel_cfg_parallel_3gpu_kv512.yaml`（替代 `mstar/bagel_cfg_parallel.yaml`） | 3 卡 CFG 并行（论文拓扑），同样 512 页 KV 池 |
| `vllm_omni/bagel_cfg_parallel_4gpu.yaml`（替代 `vllm_omni/bagel_cfg_parallel.yaml`） | 4 卡：Thinker 独占一张，DiT 阶段 `cfg_parallel_size: 3` 占三张 |
| `vllm_omni/bagel_two_stage_2gpu.yaml`（替代 `vllm_omni/bagel.yaml`） | 2 卡：Thinker 在卡 0，DiT 在卡 1（Thinker 阶段自己就加载 27.5 GiB 完整 LLM，放不进与 DiT 共卡的 45% 显存） |
| `vllm_omni/bagel_batch_two_stage_2gpu.yaml`（替代 `vllm_omni/bagel_batch.yaml`） | 同上 2 卡，Thinker `max_num_seqs=16`（BAGEL 并发 I2T 用）；原配置把两个阶段放同一卡，在 48 GiB 卡上加载 Thinker 时 OOM |
| `vllm_omni/qwen3_omni_3gpu_thinker_tp2_thinker_sync_sched.yaml`、`vllm_omni/bagel_two_stage_2gpu_thinker_sync_sched.yaml` | 反事实：对应基线配置，仅 Thinker stage 的 `async_scheduling: false`；只经 `--opt config.vllm-omni=` 使用（`qwen-async-sched` / `bagel-async-sched`），不是 baseline |
| `mstar/cosmos3_nano_1gpu_kv512.yaml`（替代 `mstar/cosmos3_nano.yaml`） | 1 卡：KV 池从 1024 页缩到 512 页（默认的 1024 页在 31 GB DiT 权重旁放不下） |

M* Qwen 在 5880 上默认带 `TORCHDYNAMO_DISABLE=1`：第一次 smoke 在 Talker 的 `_cp_linear_kernel`（M* 自己的 Triton kernel，按 H200 调的 tile）上报 `OutOfResources: shared memory, Required: 147456, Hardware limit: 101376`。**这个开关并不是修复**：`SMOKE_5880_03` 已经带着它，仍然同样失败；当前 pin 的 M* 在 `cp_linear` 里对 OutOfResources 做了 fewer-stages / 更短 K 步的回退，`SMOKE_5880_05` 才跑通。所以它大概率是过时的 workaround，不应把 eager 当作最佳性能配方；`qwen-mstar-compile-{txt,a2t,t2s}` 实验测 compile 开 / 关（`--opt qwen.mstar_compile=on` 或启动环境里 `EXPS_QWEN_MSTAR_COMPILE=on`）。Qwen overlays 使用 3 张卡；SGLang-Omni 的多卡配置不能加 `--colocate`，其启动命令已按 5880 档案自动使用非 colocated pipeline。上面的卡号是 `CUDA_VISIBLE_DEVICES` 重新编号后的逻辑序号，物理卡由 harness 取当时空闲的。其它任务先使用已有单卡配方，若 smoke 确认 OOM 再为具体任务添加 overlay。

## 反事实配置按机器取文件：`{variant:KEY}`

catalogue 里的反事实配置（前缀缓存开、Thinker `max_num_seqs` 16、Mooncake / NIXL 连接器、chunk 预算、Thinker 同步调度……）不再写某台机器的路径，而是用与机器无关的 key（`{variant:vllm_omni/bagel_two_stage_prefix_cache}`）；`profiles/hardware/<机器>.json` 的 `config_variants` 把 key 映射到该机器自己的文件（相对 `exps/configs/`）。`{variant_file:KEY}` 是文件名（`command_has` 检查用），`{variant_abs:KEY}` 是绝对路径（服务读的环境变量用），`{hw:NAME}` 取档案 `tokens` 里的一段文字（空则丢弃，所以一个检查可以只属于一台机器）。

- **rtx5880** 的映射就是上面表里的多卡 overlay，展开后的选项、环境变量和 `command_has` / `log_has` 与改动前逐字节相同，所以旧 campaign 仍然能 `analyze`。
- **h200nvl** 的映射在 `exps/configs/hardware/h200nvl/`：一张卡，Qwen3-Omni 是 Thinker TP=1 与 Talker 同卡，BAGEL 的两段 vLLM 同卡（Thinker 0.45，和默认 `bagel.yaml` 一样）。这些文件**不手写**：`python -m exps.tools.h200_variants` 生成，内容 = H200 默认配置 + 5880 版本相对其基线的同一处改动（`--check` 检查是否过期，`exps/tests/test_hardware_variants.py` 同时验证“改动相同”）。要改某个反事实，改 5880 的文件或 `h200_variants.py` 的 SPECS，再重新生成。
- 机器没有某个 key，就拒绝规划用到它的实验（`hardware profile 'x' has no config variant 'k'`），不会换成另一台机器的拓扑。`--opt config.<family>=PATH` 指向别的机器的 overlay 目录也会被拒绝（`hardware.check_config_path`）；2026-11-14 之前 H200 的 plan 会无声地带上 5880 的 3 卡 TP=2 配置（65 处）。
- 单卡不是 5880 的拓扑：H200 的连接器实验里两段在同一张卡上（NIXL 的 `cuda_ipc` 是同卡传输），Qwen 没有 TP all-reduce。结论里要写“H200 单卡变体”，不要和 5880 的数字当同一结论。`baseline-bagel-cfgpar`（3 卡）和 `mstar-tp-cache-probe`（TP=2）按 `min_gpus` 在 H200 上跳过。

**M* 的 Thinker 前缀缓存在单 rank 上默认是开的**（日志 `KV thinker_kv: prefix cache open for main on Thinker`，第二个相同 prompt 起 `whole prompt already cached True`），TP=2 上是关的（`prefix cache off at world size 2`）。5880 的 TTFT / prefill / fair 实验靠 TP=2 得到“每个请求冷 prefill”，所以在 H200 上这些 M* 臂用 `resources.thinker_kv.prefix_cache: false` 的配方（`h200nvl/mstar/qwen3omni_1gpu_prefix_cache_off.yaml` 及 chunk / bucket 变体）；`{hw:mstar_qwen_cache_off_*}` 让同一个臂在 5880 上不带这个选项、在 H200 上带，并各自检查自己的日志证据（5880：`prefix cache off at world size`；H200：没有 `prefix cache open for`）。H200 上已有的 `baseline-qwen-txt` / `a2t` 的 M* 单元用的是默认配置，即热前缀。

## 各框架可以当起点的上游多卡配置（**未在本机验证，用前自己核对显存预算与 rank**）

| 目标 | M\*（`external/mstar/configs/`） | vLLM-Omni（`external/vllm-omni/vllm_omni/deploy/`） | SGLang-Omni（`external/sglang-omni/examples/configs/`） |
|---|---|---|---|
| Qwen3-Omni 2 卡（论文拓扑：Thinker 与 Talker 分卡） | `qwen3omni_2gpu.yaml`（针对 80 GB；48 GB 要减小 KV 页数）、`qwen3omni_full_tp2.yaml`（TP2）、`qwen3omni_thinker_tp2.yaml` | `qwen3_omni_moe.yaml`（上游 2 卡默认）、`qwen3_omni_moe_thinking.yaml` | `qwen3_omni_speech_replica2.yaml`（2 卡 speech）；单卡 colocated 变体按 H100 / H200 / H20 区分 |
| BAGEL CFG 并行 | `bagel_cfg_parallel_2gpu.yaml`、`bagel_cfg_parallel.yaml`（3 卡） | `bagel.yaml`（two-stage，stage 的设备在 yaml 里） | — |
| Cosmos3 | `cosmos3_nano_{dp2,sp2,tp2,tp4,sp4,tp2_sp2}.yaml` | `cosmos3_omni.yaml` | — |
| Orpheus | `orpheus_tp2.yaml`、`orpheus_pd_disaggregated.yaml` | — | — |

5880 的 Qwen / BAGEL 已有上述配置，可用 `--hardware rtx5880` 做 smoke（自动取空闲卡）。其它模型若需多卡，复制上游配置到 `exps/configs/hardware/rtx5880/<框架目录>/<与默认同名的文件>`，调整 `ranks` / KV 页数 / 显存比例后验证。**每个多卡配置都是一个新的被测条件**：换配置就换 campaign，结果里要写明拓扑。

## 多卡运行时的门控范围

热 / 时钟 / 利用率门只读主 GPU（`--gpus` 的第一个）；其余 GPU 在起服务前和停服务后要求没有计算进程，`lifecycle.json` 里的 `gpus_in_use` 和 `state_at_ready_all` 记录每张卡的状态。稳定散热的机器默认不用冷却门（见 `PROTOCOL.md`），这个范围限制在那里不影响结论。
