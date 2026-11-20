# pinned：实验实际运行的 vLLM-Omni 与 SGLang-Omni 源码

`envs/vllm-omni` 与 `envs/sglang-omni` 以可编辑方式安装的，是这里的两个检出。它们固定到设计文档 §2.6 测量时的版本，
并保留了那时的本地 overlay，因此实验结果可以复现，与基准仓库报告使用的代码相同。

| 目录 | 上游 | 提交 | 版本 | 本地改动 |
|---|---|---|---|---|
| `vllm-omni/` | https://github.com/vllm-project/vllm-omni.git | `5917730a4b02aedb44d374173cdc1b10484ccd52` | v0.32.0 | 见下 |
| `sglang-omni/` | https://github.com/sgl-project/sglang-omni.git | `e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657` | v0.2.0 | 见下 |

两个检出都是从基准仓库的 campaign 检出 `external/vllm-omni`、`external/sglang-omni` 整体复制的（含 `.git`，2026-11-20）。
复制后 `git status` 与原检出逐行一致。

## 本地改动（overlay）

overlay 的登记与校验在基准仓库的 `exps/patches/applied.json` 中；这里列出的是工作区里的差异。

- `vllm-omni/`：8 个修改的文件（6 个运行时代码、2 个测试）+ 1 个新增测试 `tests/engine/test_stage_omni_kv_config.py`，对应补丁
  `vllm-01`（BAGEL 的 I2T 长度控制）、`vllm-02`（stage 级 `omni_kv_config` 合并，`recv_timeout` 不被 topology 覆盖）、
  `vllm-03`（worker 侧 connector 使用 `kv_transfer` 端口段）。另有 9 个 `.claude/skills/*` 的删除，是基准仓库的本地说明文件，不影响运行。
- `sglang-omni/`：6 个修改的文件（4 个运行时代码、2 个测试），对应补丁 `sglang-01`（Qwen 的长度控制）；另有 `CLAUDE.md` 从符号链接变为普通文件（同样是本地说明）。

## 用途与边界

- Pi0.5 的 vLLM-Omni 实验（`envs/vllm-omni/bin/vllm serve … --omni`）运行的是 `pinned/vllm-omni`。
- SGLang core 的 Pi0.5 实验运行的是 `envs/sglang-omni` 中的 pip 包 sglang 0.5.22，**不导入** `pinned/sglang-omni`。
  这个检出只为了让 `envs/sglang-omni` 的可编辑安装不再指向基准仓库。
- 要换用最新版（`external/vllm-omni`，2e3ad0a4）运行，需要改这里的检出并重新测量；不能把新版本的结果当作设计文档 §2.6 的结果。
- 这里的改动是运行时代码。修改时同时更新本文件和设计文档中对应的版本说明。
