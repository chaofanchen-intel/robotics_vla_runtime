# robotics_vla_runtime

机器人 / VLA（Pi0.5）实时运行时的实测、上游源码分析与 Level 1 / Level 2 边界。与基准仓库
[Inference_Engine_Investigation_Repo](../Inference_Engine_Investigation_Repo/)（benchmark、报告、realtime 研究）并列，
后续的机器人实验在这里进行，与基准仓库的 benchmark 区分开。

设计文档与结论：[docs/design.md](docs/design.md)（第 0 节为结论，第 2.6 节为同协议探测，第 7 节为核对状态）。

## 目录

| 路径 | 内容 |
|---|---|
| `docs/design.md` | 设计文档：Pi0.5 三框架实测（§2）、上游源码（§3）、Level 1 / Level 2（§4–§6）、核对状态（§7）、来源（§8） |
| `harness/run_all.py` | 为每个框架启动服务、运行探测、记录 GPU 快照 |
| `harness/probes.py` | 客户端探测（e2e、period、burst、cancel、concurrency 五种模式） |
| `configs/mstar_pi05_openpi.yaml` | M\* 最新版的 Pi0.5 OpenPI WebSocket 配置（`openpi:` 块） |
| `runs/` | 实验输出：JSON、服务日志、GPU 快照（说明见 [runs/README.md](runs/README.md)） |
| `agents_results/` | 本目录的研究子代理记录（说明见 [agents_results/README.md](agents_results/README.md)） |
| `external/` | 上游框架最新源码快照，只读（说明见 [external/README.md](external/README.md)） |
| `build/` | M\* Rust 组件的编译产物：`mstar_rust` 解包的 wheel、`mstar-server` 二进制（说明见 [build/README.md](build/README.md)） |
| `exps/` | 基准仓库 `exps/` 的整体副本，不含 `runs/`：框架启动命令（`exps/frameworks`）、Pi0.5 任务（`exps/tasks/pi05`）、输入夹具（`exps/inputs/PI05`）、配置与硬件 profile |
| `envs/` | 符号链接 → `../Inference_Engine_Investigation_Repo/envs`（Python 环境，不复制） |
| `models/` | 符号链接 → `../Inference_Engine_Investigation_Repo/models`（模型权重缓存，不复制） |

## 与基准仓库的关系

- **本目录自包含的部分**：代码、配置、夹具、实验输出、上游最新源码快照、M\* Rust 编译产物。
- **共享的部分（符号链接）**：`envs/` 和 `models/`。
- **需要注意的耦合**：`envs/` 中的可编辑安装指向基准仓库的 `external/`（2026 年 campaign 的检出版本）：
  `mstar`（9c6f1ad）、`vllm-omni`（5917730）、`sglang-omni`。因此：
  - M\* 的最新版实验通过 `PYTHONPATH` 把本目录的 `external/mstar`（2c8ffde8）放在前面，覆盖可编辑安装；
  - vLLM-Omni 与 SGLang core 的实验运行的仍是基准仓库 campaign 的代码（与报告使用的版本相同）。
  - 本目录的 `external/vllm-omni` 与 `external/sglang-omni` 是最新源码，只用于读源码，不是服务端实际运行的代码。
- **基准仓库仍保留**：`share/`（报告、realtime 研究）、`external/`（campaign 检出，见 `external/SOURCE-RECORD.md`）、
  `external_latest/`（基准仓库的 M\*、vLLM-Omni、SGLang-Omni 最新快照）、`agents_results/`（其他研究记录）。

## 复现

```bash
cd /home/chaofan/workspaces/robotics_vla_runtime
nvidia-smi                                   # 确认 GPU 0 空闲（服务固定使用 CUDA_VISIBLE_DEVICES=0）
envs/mstar/bin/python harness/run_all.py --framework mstar-rust --run-dir runs/<日期>_<名称>
envs/mstar/bin/python harness/run_all.py --framework vllm-omni  --run-dir runs/<日期>_<名称>
envs/mstar/bin/python harness/run_all.py --framework sglang-core --run-dir runs/<日期>_<名称>
```

- `--framework` 可选 `mstar`（Python 前端，已弃用，只作对照）、`mstar-rust`（默认前端）、`vllm-omni`、`sglang-core`。
- 不加 `--run-dir` 时输出到 `runs/2026-11-20_ws-probe/`，会覆盖那里的同名文件。新实验请总是指定新的目录名。
- `--modes` 可选 `e2e period burst cancel concurrency`，默认全部。
- 端口固定：M\* 18101（Python）/ 18104（Rust），vLLM-Omni 18102，SGLang core 18103。

## 约定

- 文档中的行内路径（如 `runs/...`、`harness/...`）相对于本目录根；链接相对于所在文件。
- `runs/` 中的命令记录（`serve_command.txt`）和服务日志保存的是迁移前的绝对路径（`/home/chaofan/workspaces/Inference_Engine_Investigation_Repo/...`），没有改写。
- 服务进程的 socket 与上传目录使用 `/tmp/exps-robot-probe`，这是运行时产物，不是构建产物。
