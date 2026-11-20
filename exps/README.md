# exps/：自包含的对比与消融测试框架

> 自 2026-11-04 起，`exps/` 是当前唯一可执行的实验 harness；旧轮次目录已删除，当前代码、配置和输入不从它们加载文件。冻结 fixture 位于 `exps/inputs/`（文件因体积/授权原因被 Git 忽略，部署数据要求见 [inputs/README.md](inputs/README.md)）。
>
> 三框架源码 checkout 见 [SOURCE-RECORD](../external/SOURCE-RECORD.md)；旧性能结论不能和这里重跑的数据混比。命令均在**仓库根目录**执行；模型权重和 tokenizer 缓存在 `models/cache/`，解释器在各框架独立的 `envs/`，源码 checkout 在 `external/`。

## 三条原则（headline 数字必须同时满足）

1. **等工作量**：任务包的 profile 里只有等工作量的 case。BAGEL T2I 是 `-nocfg`（一分支）和 `-cfg3`（两边都三分支）；默认请求 `t2i-<res>-default`（M\* 算 3 个分支、vLLM-Omni 算 2 个）只作参考，不进任何 profile 和 baseline。
2. **各自最优配置，能调优的都调优**：cell 默认用每个框架的最优配方，并要求调优配置已加载（缺了就报错或在分析里失败）。哪些能调、哪些不能（M\* 的 MoE 没有调优入口），见 [GUIDE](GUIDE.md) §3 和 `python -m exps.tuning status`。偏离最优的 arm 是命名的反事实。
3. **协议是参数**：每个实验、每个任务都能用 gated、back-to-back 或固定间隔运行，换协议自动换 campaign；默认值由硬件档案决定（会降频的 H200 用 gated，散热稳定的 5880 用 back-to-back）。见 [PROTOCOL](PROTOCOL.md)。

## 目录

```
exps/
  GUIDE.md               测试指南 / 计划：smoke → 调优 → 基线 → 消融（先读这个）
  run.py                 单个 task / case / framework 的 cell 运行入口（list、task、sweep、analyze）
  suite/                 目录驱动的实验：catalogue.py（全部对比与消融，数据）、runner.py、analysis.py
  hardware.py            硬件档案（profiles/hardware/*.json）：GPU 数量、协议默认值、显存门限、配置 overlay
  tuning.py              `python -m exps.tuning status`：每个可调优的 kernel 在本机有没有调优配置
  core/                  测量循环（measure.py）、热门控（thermal.py）、协议参数（cli.py）、覆盖项（overrides.py）…
  frameworks/            各框架的 serve 命令构造（M\*、vLLM-Omni、SGLang-Omni/core、VoxServe、Meta-native）
  tasks/<任务>/          cases.py（case 表）、cell.py（服务与客户端计划）、client.py
  tools/                 hook（改源码之外的 what-if 与探针）与 CPU 分析工具
  inputs/ configs/ patches/ profiles/
  configs/hardware/      各机器的配置 overlay（5880 的多卡配置放这里）
  results/               分析输出（markdown，进 Git）；runs/ 是原始数据（被 Git 忽略）
  tests/                 pytest，全部 CPU
```

## 快速开始

在 8×RTX 5880 工作站上先运行 `bash envs/setup.sh` 建立独立 Python 环境；环境脚本和 `rtx5880` 档案均限制在物理 GPU 4–7。若只希望使用其中一部分，可传 `CUDA_VISIBLE_DEVICES=4,5`。详细说明见 [envs/README.md](../envs/README.md)。

```bash
envs/mstar/bin/python -m exps.hardware                   # 检测到的 GPU、硬件档案、默认协议、显存门限
envs/mstar/bin/python -m exps.doctor                     # 只读自检：源码 ref、补丁、fixture、GPU 遥测
envs/mstar/bin/python -m exps.tuning status              # 每个可调优 kernel 在本机有没有调优配置、缺了怎么生成
envs/mstar/bin/python -m pytest exps/tests -q            # CPU 用例；BAGEL 规划测试需先生成本机 MoT 调优文件

envs/mstar/bin/python -m exps.suite list                 # 全部实验，按组
envs/mstar/bin/python -m exps.suite show bagel-attn-backend     # 问题、预登记读法、arm
envs/mstar/bin/python -m exps.suite plan 'baseline-*'    # cell 数、估算时长、每个 arm 的 exps.run 命令（不占 GPU）
envs/mstar/bin/python -m exps.suite run  protocol-ab-pi05 --execute   # 不加 --execute 等同 plan；baseline 要先有协议决定
envs/mstar/bin/python -m exps.suite decide-protocol                 # 由 protocol-ab 的结果决定每个任务的协议
envs/mstar/bin/python -m exps.suite run  baseline-bagel --execute
envs/mstar/bin/python -m exps.suite analyze baseline-bagel          # 校验每个 cell，再写 exps/results/<name>[_b2b]_<硬件档案>.md（文件名带 `rtx5880` / `h200nvl`，两台机器的结果不会互相覆盖）
```

`exps.doctor` 仍会拦截未登记的外部运行时代码改动；比较 SGLang 已知补丁文件集时，忽略 `external/sglang-omni/CLAUDE.md` 这类本地 agent 元数据。该路径仍会记录在运行 provenance 中。

- 机器相关的开关对 `exps.run` 和 `exps.suite` 都一样：`--hardware rtx5880 --gpus 4,5`、`--protocol back-to-back [--gap-s N]`、`--cases a,b`（替换所有 arm 的 case）、`--frameworks a,b`（限制框架）、`--run-id 02`。不传 `--gpus` 时，5880 档案对**每个 cell**挑当时空闲的物理 GPU（编号最小的），张数取 `gpus_per_task`（该部署能跑的最少卡数：Qwen3-Omni 3 卡，vLLM-Omni BAGEL two-stage 2 卡，其余 1 卡）；空闲卡不够时按 `--gpu-wait-s` / `EXPS_GPU_WAIT_S` 等待，仍不够则该 cell 失败。显式 `--gpus` 则固定使用所给的卡。
- `run` 按 arm 顺序逐个执行（A1 / B / A2 的顺序有意义），某个 arm 失败则跳过该实验的后续 arm；每个 arm 是**一条** `python -m exps.run …` 命令，`plan` 会原样打印，手敲得到同样的 manifest。
- `--resume` 默认开启（成功 cell 不重跑）；**失败的 cell 要换新 campaign**：`--run-id 02`，不要手改旧目录。
- 名字支持通配：`'baseline-*'`、`'mstar-ablation-*'`。
- 时长是 harness 的**保守估计**（按每请求 180 s 冷却等待算），背靠背实际短得多。
- GPU 锁：同一时刻只允许一个运行（`runs/.gpu.lock`）。需要独占 GPU，且不能有残留的服务。

## 各框架的最优配置（cell 默认值）

| 任务 | M\* | vLLM-Omni | SGLang | 其他 |
|---|---|---|---|---|
| BAGEL | `bagel_single_gpu.yaml`，compile 开 | **two-stage** 与 single-stage 都跑，取每个 case 里更快的；**加载调优 MoT 配置**（`configs/vllm_omni/tuned/`，按 GPU 名匹配；本 GPU 的文件缺失就报错；`--opt bagel.vllm_mot=default` 才用未调优） | 无 BAGEL 路径 | — |
| Qwen3-Omni T2S / TXT / A2T | `qwen3omni_1gpu.yaml` | `qwen3_omni_full_1gpu_default_async_c2wgraph.yaml`：`async_chunk: true` + Code2Wav `enforce_eager: false`（upstream 默认；并发用 `..._batch_async_c2wgraph.yaml`） | `..._partial_start.yaml`：Talker partial start 开；MoE 表在 `configs/sglang_omni/moe_<档案名>/` | 原始配方保留，用 `--opt qwen.vllm_deploy_config=…` / `qwen.sglang_config=…` 才选；baseline 的服务日志里出现 “Using default MoE config” 就判失败 |
| V-JEPA 2-AC | BF16 autocast（引擎默认） | — | — | Meta-native **仅 BF16**（FP32 不测） |
| Pi0.5 | `pi05_bfloat16.yaml` | eager（graph 变体 `--opt pi05.vllm_graph=1` 上次不改变延迟，无更优配方） | SGLang core `Pi05Pipeline`（不是 SGLang-Omni） | — |
| Cosmos3 | `cosmos3_nano.yaml` | `cosmos3_omni.yaml` | — | 本地能力对照，不是论文基准 |
| Orpheus | `orpheus_colocated.yaml` | — | — | 基线是 VoxServe（不是 vLLM-Omni） |

> **升级与换机后必须重做的**：所有调优配置是按 GPU 名和（SGLang）Triton 版本匹配的，换 GPU、升级 vLLM-Omni / Triton 后都要重新生成；`configs/` 里各 yaml 是旧版本配方的副本，未在新源码上逐个验证（见 [configs/PROVENANCE.md](configs/PROVENANCE.md)）。**M\* 也有调优器**（升级后的源码里：`external/mstar/benchmark/kernels/sweep.py`，按精确 torch 设备名查 `MSTAR_KERNEL_TABLES`，包里只带 H100 / H200 / GB200 / B300 的表）：H200 **NVL** 和 RTX 5880 都匹配不上，要自己扫（`suite run mstar-moe-tune`）。vLLM 与 SGLang 的 MoE 调优器来自 core 源码，已 checkout 到 `external/vllm`（v0.32.0）和 `external/sglang`（v0.5.22），版本与 env 一致，对应 `suite run vllm-moe-tune` / `sglang-moe-tune`。三家都调完（`exps.tuning status --require`）才是各自最优；`a2t-moe-tuned-vs-default` 量出 M\* 自己的 MoE 调优在 E2E 上值多少。

## 实验目录（`suite list` 的全貌，按「论文复现 / 找 M\* 优势来源」裁剪过）

**baseline（论文复现：等工作量 + 各自最优；协议可选）**：`baseline-bagel`（512，主工作点）、`baseline-bagel-1024`（论文的分辨率，需要稳定散热）、`baseline-bagel-cfgpar`（论文主图的 3 卡 CFG 并行拓扑，至少 3 卡）、`baseline-qwen-t2s`、`baseline-qwen-txt`、`baseline-qwen-a2t`、`baseline-orpheus`、`baseline-vjepa-ac`、`baseline-cosmos3`、`baseline-pi05`。输出 `exps/results/<名字>[_b2b]_<硬件档案>.md`，含「每个框架在各自最优 arm 下」和「最强基线 / M\*」。

**advantage（M\* 的优势来自哪里）**：

| 实验 | 回答 |
|---|---|
| `mstar-ablation-<任务>`（8 个，每个任务一个） | 同一个矩阵：A1 / C（关 compile）/ S（关推测发射链）/ SC / W（walk-blind：关链 + 关预规划 + 开步同步）/ WC / A2。哪个组件占多少、是否相乘、全关后还剩什么（WC 对比 baseline 最强基线）。**只用 M\* 源码里已有的环境变量**；没有 CUDA graph / 分页 KV / FlashInfer 的开关，这些不声称 |
| `bagel-attn-backend` | flash-attn 核对（见下） |
| `bagel-cfg-ladder` | E2E 随 CFG 分支数（1 / default / 3），每额外分支每步多少 ms：M\* 多算被丢弃分支是实现选择 |
| `bagel-steps` | E2E = a + b·steps：差距在每请求固定开销还是每步 |
| `bagel-concurrency`、`t2s-concurrency` | 论文的 I2T 批量与 T2S B=16 吞吐主张 |
| `qwen-async-sched`、`bagel-async-sched` | M\* 的推测发射链是不是 vLLM 异步调度的同一种 CPU/GPU 重叠：vLLM-Omni Thinker 关 / 开 async scheduling，对照 M\* 链关 / 开，看每 token 延迟（`analyze_decode_itl.py`）；仅 5880 配置 |
| `cosmos3-decompose` | Cosmos3 上 M\* 的 2.4× 能不能逐项拆掉：顺序 CFG（hook）/ 关 CUDA graph / 关 compile / 给 256² 加 graph，对照 vLLM |
| `bagel-i2t-nsys`（profile） | BAGEL I2T 里 vLLM-Omni 首 token 多出的约 650 ms 在 GPU 还是 CPU（`analyze_nsys_burst_gaps.py`）；已回答：是 GPU 上 4,935 token 的 prefill |
| `bagel-prefix-cache` | M\* 的 KV prefix cache 开/关（`--opt bagel.prefix_cache=off`，请求带 `prefix_cache=false`）对照 vLLM-Omni（配置里全部关了 prefix caching），BAGEL I2T / I2I |
| `bagel-vllm-prefix-cache` | vLLM-Omni two-stage 的 Thinker 打开 prefix caching（反事实配置）对照 M*，BAGEL I2T / I2I：M* 的 TTFT 领先是引擎能力还是 baseline 少开了一个开关 |
| `bagel-i2i-steps` | BAGEL I2I 的 E2E 对去噪步数（5/25/50）拟合，三种部署：两阶段多出的 4.4 s 是每请求固定成本还是每步成本 |
| `bagel-i2i-nsys` | BAGEL I2I 三种部署的每卡 GPU busy / 空闲（nsys，两阶段按 GPU 分表）：KV 交接的空闲有多长 |
| `qwen-fair-txt` | Qwen TXT 公平档：三个框架 prefix cache 各关/各开（SGLang radix、vLLM `enable_prefix_caching`、M* `prefix_cache_tp`），请求去掉 seed/min_tokens 使 SGLang 的 lookahead 可用，另有 lookahead 强制关闭对照和原请求对照 |
| `qwen-fair-a2t` | 同上，A2T，且 precondition 与测量用同一批音频（`qwen.rows=cycle`），去掉 vLLM 对新输入 shape 的推理期 Triton JIT |
| `qwen-t2s-fit` | T2S：20 句，E2E = a + b×音频秒的拟合，按 stall 拆开；含 vLLM Talker greedy 对照 |
| `qwen-ttft-txt` / `qwen-ttft-a2t` | 首 token 拆解：15 token 提示（固定成本）、1K 提示（prefill）、A2T（音频路径），都只生成 1 个 token |
| `bagel-sampling-parity` | BAGEL I2T：去掉 vLLM yaml 的 repetition_penalty 1.05，两边都开 prefix cache |
| `bagel-connector` | BAGEL I2I two-stage 的 Thinker→DiT KV 交接：SharedMemory / Mooncake（TCP）/ NIXL（UCX）/ single-stage 对照；依赖由 `exps/tools/install_omni_connectors.sh` 装到 `~/.local/opt/omni-connectors`（不进基线环境） |
| `bagel-connector-timeline` | 同上 + 毫秒级日志时间戳（`VLLM_LOGGING_CONFIG_PATH`）把 5 步请求拆成 client→engine / Thinker / 发送 / 编排间隙 / DiT 侧，并给 NIXL 加 `UCX_TLS=cuda_ipc` / `tcp` 两个传输对照；`exps/tools/analyze_i2i_timeline.py` |
| `bagel-connector-1024` | 1024² / 5 步下同样的 connector 对照（vLLM-Omni only）：交接节省是否随 KV 变大而按字节增长 |
| `bagel-connector-nsys` | 5 步 I2I 的 nsys：single-stage / two-stage SharedMemory / Mooncake / Mooncake + Thinker 缓存开 / M*；`--handoff` 列出每个请求的 Thinker busy、hand-off、DiT（Mooncake 已测：Thinker 2.05 s busy、hand-off 0.28 s、DiT 1.77 s） |
| `bagel-i2i-fair` | I2I 在 Mooncake + vLLM Thinker 缓存开 的条件下对 M* 缓存开 / 关（5 步、50 步）。已测：vLLM 缓存省 1,041 ms，M* 缓存开关无影响 |
| `qwen-prefill-nsys` | Qwen TXT 1k-token prefill（一个输出 token）的 nsys，三家。**结果：M\* 一个请求 3 次 forward（1,024 + 48 token 两块 + 第二个输出 token），第 1 块与 vLLM 的整个提示同为 77 ms；「M\* prefill 慢 2×」是 chunk 预算 1,024 造成的假象** |
| `qwen-prefill-chunk` | M\* Thinker chunk 预算 1,024（默认）对 4,096（反事实）。**结果：预测（TTFT 降约 40 ms）被推翻，1,072-token 的 TTFT 138 → 154 ms（更慢）**；`txt-tiny-1tok` 不变（`run_attrib.sh prefill-chunk`） |
| `qwen-prefill-chunk-nsys` | M\* 默认 / 预算 4,096 / 预算 4,096 + 1,152 图桶 的 nsys：一个请求几次 forward、prompt forward 多长（`run_attrib.sh prefill-chunk-nsys`，待跑） |
| `qwen-prefill-len` | TTFT 随服务端提示长度（56 / 512 / 1,024 / 1,072 / 2,048）：M\* 三个 chunk / 图桶设置 + vLLM（`run_attrib.sh prefill-len`，待跑，约 1.3 h） |
| `bagel-prefill-chunk` | M\* BAGEL LLM chunk 预算 1,024 对 8,192。**结果：I2T TTFT 降约 6–7 ms（差距约 44–49 ms 的 15%），E2E 与 5 步 I2I 不动（1.00×）**（`run_attrib.sh bagel-chunk`） |
| `mstar-tp-cache-probe` | M\* TP=2 前缀缓存（`prefix_cache_tp`）+ 4,096 预算（整段提示一个 chunk）：`TP ranks diverged` 崩溃还在不在，缩小根因范围（`run_attrib.sh tp-cache-probe`）。**已跑：不崩**，崩溃与提示被切成多个 chunk 有关 |
| `qwen-a2t-cache-warm` | A2T 一个输出 token，三遍预热。**结果：vLLM 开缓存 77.4 对关 75.9 ms，481 ms 是预热不足；SGLang 开缓存稳定慢 31 ms（89.7 → 121）** |
| `qwen-t2s-nsys-warm` | T2S 各 stage 的 GPU 时间线（预热行）。**结果：GPU 2 上 vLLM 每音频秒 15.4k 个 kernel、M\* 8.2k（1.9×），busy / span 都约 27%，帧循环 launch 受限** |
| `qwen-mstar-compile-{txt,a2t,t2s}` | M* 在 Qwen 上 torch.compile 开 / 关（5880 默认关，原因是过时的 workaround）：A1 / compile-on / A2 |
| `cosmos3-nsys`（profile） | Cosmos3 在 CFG/graph/compile 全对齐（m-all-off）后 vLLM 仍慢 1.17× 的 kernel 构成 |

**fairness（基线自己的配置 / 调优值多少，防止基线是稻草人）**：`bagel-mot-tuned-vs-default`（调优 MoT 在 E2E 上值多少）、`t2s-deploy-config`（vLLM async-chunk + Code2Wav graph、SGLang partial start 各值多少）、`a2t-moe-tuned-vs-default`（M\* 自己的 MoE 调优值多少；builtin 臂是 M\* 的内置 tile，不是它的最优配置）。

**protocol（先于 baseline）**：`protocol-ab-<任务>`，8 个任务各一个（Pi0.5、BAGEL、V-JEPA、Orpheus、Cosmos3、Qwen T2S / TXT / A2T），arm = `gated` / `b2b`（Pi0.5 另有 `gap30`），每个只测一个代表 case、2 次 start × 5 个请求。`python -m exps.suite decide-protocol` 按每个框架的 `背靠背 / 门控` 比值决定每个任务的协议（不敏感 → 背靠背；框架间偏移不同 → 两种都跑），写进 `profiles/protocol/<档案>.json`；**没有决定的任务，`baseline-*` 拒绝规划**（`--protocol` 显式指定可绕过）。规则见 [GUIDE](GUIDE.md) §0。

**profile / tool**：`bagel-nsys-equal-work`、`t2s-stage-nsys`（nsys，等工作量下每步 GPU busy / kernel，只看结构）；`bagel-mot-tune`、`mstar-moe-tune`（生成调优文件）、`thermal-probe`。

**已删除的旧实验及原因**：BAGEL 256² 作为 headline（host-bound、易漂移，512 为主；256 的 case 仍在，`--cases` 可选）；`t2s-spec-compile` / `t2s-walk-blind` / `t2s-runtime-switches` / `bagel-nocompile` / `bagel-runtime-switches` / `a2t-nocompile`（并进 `mstar-ablation-*`）；`bagel-recompile-fix`（hook 反事实，上次一分支无 E2E 变化，M\* 的短板不是优势来源）；`t2s-codec-chunk`（只影响 TTFA）；`bagel-vllm-graph` / `pi05-vllm-graph`（对手的诊断，已有结果）；`bagel-i2t-tokens`；`thermal-history`（对旧数据的审计）；V-JEPA FP32。相应 hook 仍在 `tools/`，需要时在 `suite/catalogue.py` 里加一条（`Arm(env=("mstar:PYTHONPATH={hooks}/…", …))`）。

**旧阶段 → 新实验**：`t2s-compile-aba` / `t2s-spec-compile-aba` / `spec-aba` / `preplan-aba` / `step-sync-aba` / `t2s-walk-blind-aba` → `mstar-ablation-*`；`pi05-protocol-ab` → `protocol-ab-pi05`；`bagel-equal-work-nsys` / `bagel-mot-tuned-nsys` → `bagel-nsys-equal-work`；`bagel-mot-tuned-e2e-{256,512}` → `baseline-bagel`；`vjepa-bf16-paired` → `baseline-vjepa-ac`；`run_bagel_ablation.sh` → `bagel-steps` / `bagel-cfg-ladder`；`run_concurrency_sweep.sh` → `t2s-concurrency` / `bagel-concurrency`。

## 覆盖项（取代旧的 `EXPERIMENTS_*` 环境变量）

```bash
--env  mstar:TORCHDYNAMO_DISABLE=1           # 某个框架家族的服务进程环境变量（* = 全部）；家族：mstar | vllm-omni | sglang | voxserve
--serve-arg mstar:--enable-nvtx              # 追加到服务命令的参数
--opt  pi05.vllm_graph=1                     # 任务选项：pi05.vllm_graph、bagel.vllm_mot、qwen.vllm_deploy_config、qwen.sglang_config、…
--opt  config.mstar=PATH                     # 替换某个框架家族的服务配置（多卡拓扑）；也可放进 configs/hardware/<档案>/ 做 overlay
--nsys-dir DIR [--nsys-flush-ms 1000]        # 用 nsys profile 包住服务命令（只用于 profile cell）
--campaign NAME                              # run 目录名；换协议 / 覆盖项 / 配置就换名字
--hardware rtx5880 --gpus 4,5                # 5880 机器从允许的物理 GPU 中选择
```

覆盖项写进每个 manifest 的 `overrides`，`suite analyze` 校验它与 arm 声明一致，并检查服务命令、服务日志、hook proof 里确实有该开关的痕迹——「日志里有这行」与「开关生效」是两回事，所以 hook arm 必须有 proof 文件，否则分析报错。继承了旧的 `EXPERIMENTS_*` / `QWEN_*` 环境变量时，`suite run --execute` 直接拒绝运行。

## 换机器（8×RTX 5880 等）

硬件档案 `profiles/hardware/<名>.json` 决定：用几张卡、默认协议（散热稳定就不需要冷却门）、空载 / 载入的显存门限、配置 overlay 目录、哪些任务单卡放不下。`--gpus` 选物理卡，第一张是门控读温度 / 时钟的主 GPU，其余卡要求起停时没有计算进程；5880 档案是共享机器（`auto_select_gpus`），所有物理卡都可用，只取空闲的。在其他机器上选做哪些见 [GUIDE](GUIDE.md) §9，多卡配置见 [configs/hardware/README.md](configs/hardware/README.md)。

## flash-attn 核对：`bagel-attn-backend`

起因：M\* 的 README 写 "To enable flash-attn support (recommended for BAGEL)"，怀疑 M\* 用了 flash-attn 而对手没用，不装它优势就没了。**源码和旧日志已经指向「不是」**（见 [`docs/当前总结论.md`](../docs/当前总结论.md) §3.1.7）：pip 包 `flash_attn` 只在 BAGEL 的 ViT eager 路径里被 import，T2I 没有输入图所以根本不调用；LLM 注意力走 M\* 自己的 `AttentionSpec(AUTO)` → FlashInfer；vLLM-Omni 的 T2I 生成步日志是 `FLASH_ATTN`（vLLM 自带的 FlashAttention）、ViT 也是 `FLASH_ATTN`、理解阶段是 `TRITON_ATTN`。这个实验把它变成运行时证据：

```bash
envs/mstar/bin/python -m exps.suite plan bagel-attn-backend                       # 4 个 arm，各 9 cell
envs/mstar/bin/python -m exps.suite run  bagel-attn-backend --execute
envs/mstar/bin/python -m exps.suite analyze bagel-attn-backend                    # 先校验全部 proof，再出表
```

| arm | 做什么 | 校验什么 |
|---|---|---|
| `mstar-probe` | M\* worker 里装探针（`tools/mstar_attn_probe_hook`）：计数 ViT 的 flash-attn 调用与 SDPA 回退，并记录每个注意力资源最终解析成哪个 manager | T2I：ViT 调用 = 0；I2T / I2I：flash 调用 > 0；LLM 注意力是 `FlashInferManager`；pip 包可导入 |
| `mstar-no-flash-attn` | 同上，再让 ViT 表现得像没装 `flash_attn`（走 SDPA 回退） | `flash_attn` 不可用、flash 调用 = 0；I2T / I2I 的 SDPA 回退调用 > 0 |
| `vllm-default` | vLLM-Omni two-stage 默认 | 日志里 `Resolved diffusion attention backend 'FLASH_ATTN'` |
| `vllm-sdpa` | `DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA`，把生成步注意力换成 PyTorch SDPA | 命令里有该变量，日志里 `'SDPA'` |

读法（预登记）：T2I 上 `mstar-no-flash-attn` 与 `mstar-probe` 在漂移内相同 → pip 包与 T2I 结论无关；I2T / I2I 只慢一个 ViT 的份额；`vllm-sdpa` 若 E2E 几乎不变 → 注意力不是差距所在（kernel 数据已显示 ≤4%）。输出里的 vLLM/M\* 比值矩阵把每个 vLLM arm 与每个 M\* arm 两两配对：四种配对下都成立的倍数就不是 flash-attn 效应。**探针本身已在 CPU 上验证**（`tests/test_suite.py::test_attn_probe_proofs`，且在升级后的 M\* 上 import 成功、两种模式都生效）；GPU 运行结果待跑。

## 不在 `exps/` 里的东西

不再提供：V-JEPA FP32 baseline、旧的轮次脚本（`run_round5…8`、`run_missing`、`run_remaining`、`run_followup`、`run_attribution_experiments` 等，全部并进 `suite/catalogue.py`）及其测试、一次性 fixture 生成脚本、靠 `EXPERIMENTS_*` 环境变量传递的覆盖项。`calibrate.py`、`doctor.py`、补丁 ledger、`smoke / nightly / full` 三个 profile 保留（profile 只含等工作量 case）。
