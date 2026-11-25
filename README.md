# robotics_vla_runtime

机器人 / VLA（Pi0.5）实时运行时的实测、上游源码分析与 Level 1 / Level 2 边界。与基准仓库
[Inference_Engine_Investigation_Repo](../Inference_Engine_Investigation_Repo/)（benchmark、报告、realtime 研究）并列，
后续的机器人实验在这里进行，与基准仓库的 benchmark 区分开。

设计文档与结论：[docs/design.md](docs/design.md)（第 0 节为结论，第 2.6 节为同协议探测，第 2.7 节为 Level 1 测量，第 7 节为核对状态）。

当前状态：Level 1 已测完（`pi05_droid`、droid_100 多频回放、三个框架，见 §2.7 与 `runs/2026-11-20_level1/level1_summary.md`）；Level 2（引擎内部改动）尚未开始。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/design.md` | 设计文档：Pi0.5 三框架实测（§2）、上游源码（§3）、Level 1 / Level 2（§4–§6）、核对状态（§7）、来源（§8） |
| `src/runtime/` | Level 1 Runtime，与推理框架无关：`dataset.py`（droid_100 加载、状态归一化与插值）、`observation.py`（多频输入流与"每路 ≤t 最新一条"快照）、`scheduler.py`（触发策略）、`action_manager.py`（动作块、代数号、对齐、欠载）、`controller.py`（控制时钟与请求循环） |
| `src/backends/` | 推理框架适配器：`base.py`（协议、连接、回复解析）、`mstar.py`、`vllm_omni.py`、`sglang_core.py`、`checkpoints.py`（`pi05_base` 50 步、`pi05_droid` 15 步）、`serving.py`（启动、等待就绪、停止服务） |
| `src/metrics/` | 指标：`stats.py`（分位数）、`level1.py`（E2E、输入年龄、动作年龄、预定时刻滞后、欠载） |
| `src/scenarios/` | 实验场景：`level1_periodic.py`（Level 1，十个 cell）、`protocol_probes.py`（协议探测：e2e、period、burst、cancel、concurrency） |
| `scripts/` | 启动器与汇总：`run_probes.py`、`run_level1.py`、`level1_report.py` |
| `tests/` | `src/runtime` 与 `src/backends` 的测试（假服务器，手算预期；CPU） |
| `data/` | droid_100 解码缓存（`droid_100_cache/`，不进 git，可重新生成） |
| `configs/mstar_pi05_openpi.yaml` | M\* 最新版的 Pi0.5 OpenPI WebSocket 配置（`openpi:` 块） |
| `runs/` | 实验输出：JSON、服务日志、GPU 快照（说明见 [runs/README.md](runs/README.md)） |
| `agents_results/` | 本目录的研究子代理记录（说明见 [agents_results/README.md](agents_results/README.md)） |
| `share/robotics_vla_report/` | 汇报稿（16:9 幻灯片，44 页）：框架能力按六项判据逐页比较（M\*、vLLM-Omni、SGLang-Omni）；Pi0.5 的输入输出、实时方案、测量结果、Level 2 缺口；Qwen3-Omni 的输入输出与三层路线；计划与待确定事项（说明见 [share/robotics_vla_report/README.md](share/robotics_vla_report/README.md)） |
| `external/` | 上游框架最新源码快照，只读；M\* 实验运行的就是其中的 `mstar`（说明见 [external/README.md](external/README.md)） |
| `pinned/` | 实验运行的 vLLM-Omni 与 SGLang-Omni 源码，固定到测量时的版本，含本地 overlay（说明见 [pinned/README.md](pinned/README.md)） |
| `build/` | M\* Rust 组件的编译产物：`mstar_rust` 的 wheel 与解包目录、`mstar-server` 二进制（说明见 [build/README.md](build/README.md)） |
| `exps/` | 基准仓库 `exps/` 的整体副本，不含 `runs/`：框架启动命令（`exps/frameworks`）、Pi0.5 任务（`exps/tasks/pi05`）、输入夹具（`exps/inputs/PI05`）、配置与硬件 profile |
| `envs/` | Python 环境 `mstar`、`vllm-omni`、`sglang-omni`，从基准仓库复制（约 31 GB，不进 git）；可编辑安装已指向本目录 |
| `models/` | 符号链接 → `../Inference_Engine_Investigation_Repo/models`（模型权重缓存，220 GB，不复制） |

## 与基准仓库的关系

- **本目录自包含**：代码、配置、夹具、实验输出、Python 环境、实际运行的框架源码（`external/mstar`、`pinned/`）、M\* Rust 编译产物。
- **共享的部分**：只有模型权重缓存（`models/` 符号链接）。
- **运行的代码**：
  - M\*（Python 与 Rust 前端）：`external/mstar`（2c8ffde8）。`envs/mstar` 以可编辑方式安装它；`mstar_rust` 是 `build/mstar_rust`（从 `external/mstar/rust` 编译，导入时校验源码哈希）。
  - vLLM-Omni：`pinned/vllm-omni`（5917730 + 本地 overlay）。与报告和设计文档 §2.6 的测量版本相同。
  - SGLang core：`envs/sglang-omni` 中的 pip 包 sglang 0.5.22。其中的 `sglang-omni` 可编辑安装指向 `pinned/sglang-omni`，SGLang core 的 Pi0.5 实验不导入它。
- `external/` 中的 vLLM-Omni（2e3ad0a4）与 SGLang-Omni（92a4c132）是最新源码，只用于读源码，不是服务实际运行的代码。要让实验改用新版本，需要重新测量。
- **环境复制的内容**：从基准仓库的 `envs/{mstar,vllm-omni,sglang-omni}` 整体复制（文件数、字节数与原件一致）。只改了两类内容：
  可编辑安装的路径（`__editable__*finder.py` 与 `direct_url.json`）和控制台脚本的 shebang（`bin/`）。它们改指向本目录；
  `envs/mstar` 的 `mstar_rust` 换成了 `build/mstar_rust_wheel` 的 wheel。其余包未动。
  复制后 `sys.path` 中不再有基准仓库的路径，`pip check` 与原环境一致。
- **基准仓库仍保留**：`share/`（报告、realtime 研究）、`external/`（campaign 检出，见 `external/SOURCE-RECORD.md`）、
  `external_latest/`、`envs/`（benchmark 自用的环境，本目录不修改）、`agents_results/`（其他研究记录）。

## 复现

```bash
cd /home/chaofan/workspaces/robotics_vla_runtime
nvidia-smi                                   # 确认 GPU 0 空闲（服务固定使用 CUDA_VISIBLE_DEVICES=0）
# 协议探测（46 行夹具，一次一条或突发）
envs/mstar/bin/python scripts/run_probes.py --framework mstar-rust --run-dir runs/<日期>_<名称>
# Level 1：三个框架交错，每轮各启动一次服务，跑全部 cell
envs/mstar/bin/python scripts/run_level1.py --run-dir runs/<日期>_level1 --repeats 3
envs/mstar/bin/python scripts/level1_report.py --run-dir runs/<日期>_level1
# 测试
envs/mstar/bin/python -m pytest tests -q
```

- `--framework` 可选 `mstar`（Python 前端，已弃用，只作对照）、`mstar-rust`（默认前端）、`vllm-omni`、`sglang-core`。
- `--run-dir` 必填，指定一个新目录。脚本不设默认值，避免覆盖 `runs/` 中已有的结果。
- `--modes` 可选 `e2e period burst cancel concurrency`，默认全部。
- 端口固定：M\* 18101（Python）/ 18104（Rust），vLLM-Omni 18102，SGLang core 18103。

## 约定

- 文档中的行内路径（如 `runs/...`、`src/runtime/...`）相对于本目录根；链接相对于所在文件。
- `runs/` 中较早的 `serve_command.txt` 与服务日志保存的是迁移前的绝对路径（`/home/chaofan/workspaces/Inference_Engine_Investigation_Repo/...`），没有改写，它们是当时实际执行的命令记录。
  2026-11-20 前的 M\* 命令带有 `PYTHONPATH=` 覆盖，那是复制环境之前的做法；此后的运行不再需要它。
- 服务进程的 socket 与上传目录使用 `/tmp/exps-robot-probe`，这是运行时产物，不是构建产物。
