# external：上游框架最新源码快照

只读的源码快照，用于读源码和核对结论。M\* 的实验运行的就是这里的 `mstar`；vLLM-Omni 与 SGLang-Omni 的运行副本在
[pinned/](../pinned/README.md)，这里的快照是最新版，不是服务实际运行的代码。
目录内容不纳入本仓库的 git 跟踪（见 `.gitignore`），这里记录每个快照的来源与提交。

## 快照

| 框架 | 上游 | 分支 / 标签 | 提交 | 提交日期 | 获取方式 | 在本目录的用途 |
|---|---|---|---|---|---|---|
| M\* | https://github.com/mstar-project/mstar.git | `main` | `2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175` | 2026-11-15 | 从基准仓库 `external_latest/mstar` 复制（含完整历史） | M\* 最新版实验：`envs/mstar` 以可编辑方式安装本目录的 `mstar`，`harness/run_all.py` 直接运行它；`mstar_rust` 由 `build/` 的 wheel 安装，源码哈希与本目录的 `external/mstar/rust` 一致 |
| vLLM-Omni | https://github.com/vllm-project/vllm-omni.git | `main` | `2e3ad0a4ff6e7cd143b3d6e979ba66f6009cb8d5` | 2026-11-18 | 从基准仓库 `external_latest/vllm-omni` 复制（含完整历史） | 只读源码（OpenPI 的 `entrypoints/openpi`、Pi0.5 配方）。服务实际运行的是 `pinned/vllm-omni`（5917730 + 本地 overlay） |
| SGLang-Omni | https://github.com/sgl-project/sglang-omni.git | `main` | `92a4c132be3588d501bf5b209db78ea6aad98cf2` | 2026-11-19 | 从基准仓库 `external_latest/sglang-omni` 复制（含完整历史） | 只读源码。SGLang core 的 Pi0.5 实验运行的是 `envs/sglang-omni` 中的 pip 包（sglang 0.5.22）；`pinned/sglang-omni`（e8c635e）是可编辑安装的检出，这些实验不导入它 |
| LeRobot | https://github.com/huggingface/lerobot.git | tag `v0.7.0` | `410c095eacc0011d998a73383eb94abed2715af0` | 2026-11-05 | 浅克隆（`--depth 1`，无历史），迁移自基准仓库 `external_latest/lerobot` | 只读源码（remote inference、ActionQueue、RTC、Pi0.5 状态分箱） |
| OpenPI | https://github.com/Physical-Intelligence/openpi.git | `main` | `215abfb217dbac7d5f1273282331b9b1866c0479` | 2026-08-25 | 浅克隆（`--depth 1`），迁移自基准仓库 `external_latest/openpi` | 只读源码（WebSocket 协议、参考客户端、pi05_droid 配置）。与 vLLM-Omni CI 固定的版本相同 |
| ROS 2 message_filters | https://github.com/ros2/message_filters.git | tag `8.0.5` | `995b9b20e55e641197d7d69059e5066b3ca2d7b2` | 2026-10-23 | 浅克隆（`--depth 1`），迁移自基准仓库 `external_latest/message_filters` | 只读源码（Cache、ApproximateTime、ExactTime）。未安装 ROS 2，未运行 |

注意：

- LeRobot 的浅克隆是最新标签 v0.7.0，不含已移除的 `async_inference` 包；它的历史需要完整克隆才能核对（v0.6.1 之前的提交见 `agents_results/2026-11-20_lerobot-async-rtc.md`）。
- 快照只含源码。依赖、模型权重和运行环境没有在这里安装或验证。
- `mstar/rust/target`（构建缓存，约 197 MB）没有复制。`mstar_rust` 与 `mstar-server` 的编译产物在 `build/`，见 [build/README.md](../build/README.md)。

## Level 2 调研的源码（`external/l2-research/`）

2026-11-21 为 Level 2 调研下载的开源仓库与 HF 文件，也在本目录下（被 `.gitignore` 忽略）。每个子目录的提交号与许可证记录在 [agents_results/2026-11-21_level2-research-deep.md](../agents_results/2026-11-21_level2-research-deep.md) 的"范围与版本"表中。`arxiv/` 下是论文 PDF 与 `pdftotext` 抽取的文本。这些仓库仅供阅读，不属于运行的代码。

## 不在本目录的内容

- 基准仓库的 campaign 检出（`external/`：`mstar` 9c6f1ad、`sglang`、`vjepa2`、`vllm` 等）未复制。`vllm-omni` 与 `sglang-omni` 的运行副本在 `pinned/`。
  campaign 的版本记录仍在基准仓库的 `external/SOURCE-RECORD.md`。
- `envs/` 的可编辑安装已改指向本目录的 `external/mstar` 与 `pinned/`，不再指向基准仓库（见 [README.md](../README.md) 的"与基准仓库的关系"）。
- 本目录的 `exps/` 继承了基准仓库的 `core/manifest.py`，其中的 pin 检查读的是基准仓库的 `external/SOURCE-RECORD.md`；本目录的 `scripts/` 不调用这些检查。

## 更新快照

快照是某一时刻的上游状态。更新时：在对应目录执行 `git fetch` 或重新浅克隆，并更新上表的提交号与日期；
同时检查 `build/README.md` 中的 `mstar_rust` 源码哈希是否还与 `external/mstar/rust` 一致（不一致时服务会在启动时报错）。
更新 `external/mstar` 后，要按 `build/README.md` 重新编译 `mstar_rust` 并装入 `envs/mstar`。
`pinned/` 的检出不随快照更新：它固定在测量时的版本，换版本需要重新测量。
