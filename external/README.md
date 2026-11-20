# external：上游框架最新源码快照

只读的源码快照，用于读源码和核对结论，不是服务实际运行的代码（M\* 的实验例外，见下）。
目录内容不纳入本仓库的 git 跟踪（见 `.gitignore`），这里记录每个快照的来源与提交。

## 快照

| 框架 | 上游 | 分支 / 标签 | 提交 | 提交日期 | 获取方式 | 在本目录的用途 |
|---|---|---|---|---|---|---|
| M\* | https://github.com/mstar-project/mstar.git | `main` | `2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175` | 2026-11-15 | 从基准仓库 `external_latest/mstar` 复制（含完整历史） | M\* 最新版实验：`harness/run_all.py` 通过 `PYTHONPATH` 导入；`mstar_rust` 源码哈希与本目录的 `external/mstar/rust` 一致 |
| vLLM-Omni | https://github.com/vllm-project/vllm-omni.git | `main` | `2e3ad0a4ff6e7cd143b3d6e979ba66f6009cb8d5` | 2026-11-18 | 从基准仓库 `external_latest/vllm-omni` 复制（含完整历史） | 只读源码（OpenPI 的 `entrypoints/openpi`、Pi0.5 配方）。服务实际运行的是 `envs/vllm-omni` 指向的 campaign 版本 5917730 |
| SGLang-Omni | https://github.com/sgl-project/sglang-omni.git | `main` | `92a4c132be3588d501bf5b209db78ea6aad98cf2` | 2026-11-19 | 从基准仓库 `external_latest/sglang-omni` 复制（含完整历史） | 只读源码。SGLang core 的 Pi0.5 实验运行的是 `envs/sglang-omni` 中的 pip 包（sglang 0.5.22） |
| LeRobot | https://github.com/huggingface/lerobot.git | tag `v0.7.0` | `410c095eacc0011d998a73383eb94abed2715af0` | 2026-11-05 | 浅克隆（`--depth 1`，无历史），迁移自基准仓库 `external_latest/lerobot` | 只读源码（remote inference、ActionQueue、RTC、Pi0.5 状态分箱） |
| OpenPI | https://github.com/Physical-Intelligence/openpi.git | `main` | `215abfb217dbac7d5f1273282331b9b1866c0479` | 2026-08-25 | 浅克隆（`--depth 1`），迁移自基准仓库 `external_latest/openpi` | 只读源码（WebSocket 协议、参考客户端、pi05_droid 配置）。与 vLLM-Omni CI 固定的版本相同 |
| ROS 2 message_filters | https://github.com/ros2/message_filters.git | tag `8.0.5` | `995b9b20e55e641197d7d69059e5066b3ca2d7b2` | 2026-10-23 | 浅克隆（`--depth 1`），迁移自基准仓库 `external_latest/message_filters` | 只读源码（Cache、ApproximateTime、ExactTime）。未安装 ROS 2，未运行 |

注意：

- LeRobot 的浅克隆是最新标签 v0.7.0，不含已移除的 `async_inference` 包；它的历史需要完整克隆才能核对（v0.6.1 之前的提交见 `agents_results/2026-11-20_lerobot-async-rtc.md`）。
- 快照只含源码。依赖、模型权重和运行环境没有在这里安装或验证。
- `mstar/rust/target`（构建缓存，约 197 MB）没有复制。`mstar_rust` 与 `mstar-server` 的编译产物在 `build/`，见 [build/README.md](../build/README.md)。

## 不在本目录的内容

- 基准仓库的 campaign 检出（`external/`，例如 `mstar` 9c6f1ad、`vllm-omni` 5917730、`sglang`、`sglang-omni`）未复制。它们的版本记录在基准仓库的 `external/SOURCE-RECORD.md`。
- `envs/` 中的可编辑安装指向基准仓库的 campaign 检出（见 [README.md](../README.md) 的"与基准仓库的关系"）。
- 本目录的 `exps/` 继承了基准仓库的 `core/manifest.py`，其中的 pin 检查读的是基准仓库的 `external/SOURCE-RECORD.md`；本目录的 harness 不调用这些检查。

## 更新快照

快照是某一时刻的上游状态。更新时：在对应目录执行 `git fetch` 或重新浅克隆，并更新上表的提交号与日期；
同时检查 `build/README.md` 中的 `mstar_rust` 源码哈希是否还与 `external/mstar/rust` 一致（不一致时服务会在启动时报错）。
