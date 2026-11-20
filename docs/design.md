# 机器人 / VLA 实时运行时：Pi0.5 实测、上游源码与 Level 1 / Level 2 的边界

日期：2026-11-20。对象：Pi0.5 在 M\*、vLLM-Omni、SGLang core 三个框架中的实测与源码，以及 LeRobot、OpenPI、ROS 2 `message_filters` 三个上游的源码分析，用来修订 Level 1 / Level 2 的定义和落地计划。

版本（写明每一条结论依据的源码点）：

| 对象 | 实验所用版本 | 本次源码快照 |
|---|---|---|
| M\* | 9c6f1ad（main + patches，见 [report 09a](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters/09a_pi05_mech.tex)） | `external/mstar` 2c8ffde8 |
| vLLM-Omni | v0.32.0（5917730 + 本地 overlay，运行的是 `pinned/vllm-omni`） | `external/vllm-omni` 2e3ad0a4（只读，未用于运行） |
| SGLang core | pip sglang 0.5.22（`envs/sglang-omni`） | `../Inference_Engine_Investigation_Repo/external/sglang`（cookbook 来源，只读） |
| OpenPI | — | `external/openpi` 215abfb（与 vLLM-Omni CI 固定版本相同） |
| LeRobot | — | `external/lerobot` v0.7.0（410c095） |
| ROS 2 message_filters | — | `external/message_filters` 8.0.5（995b9b2） |

本次只读源码，未运行新的 GPU 任务。延迟数字区分两类：**本地实测**（来自 `exps/results/`，已经存在）和**上游自报**。

## 0. 结论

1. **M\* 能运行 Pi0.5，而且有完整实测。** 上一版文档写"M\* 没有运行过 Pi0.5""5880 没有干净基线"，这两条都是错的。我没有先读 `exps/results/` 和 report v2 的模型案例三就下了结论。
2. **5880 基线（DROID 46 行，pi05_base，10 步，E2E 中位数）：** M\* 97 ms，SGLang core 默认 162 ms，vLLM-Omni 211 ms。SGLang 换成分桶配置后为 93 ms（另一批，同批默认为 167 ms）。数据来自 [baseline-pi05_b2b_rtx5880.md](../exps/results/baseline-pi05_b2b_rtx5880.md) 和 [report 09b](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters/09b_pi05_results.tex)。
3. **步数会改变排序。** 1 步时 SGLang core（52 ms）快于 M\*（72 ms）；20 步时 M\*（122 ms）远快于 SGLang core（307 ms）。见 [pi05-steps_b2b_rtx5880.md](../exps/results/pi05-steps_b2b_rtx5880.md)。周期选型要按目标步数来比，不能只看 10 步。
4. **实验里三家的传输方式不对等。** M\* 走 HTTP `/generate`，把 PNG 作为 multipart 发送，计时包含服务端解码（[exps/tasks/pi05/transport.py:157–196](../exps/tasks/pi05/transport.py#L157-L196)）。SGLang core 和 vLLM-Omni 走 WebSocket，发送原始数组（[transport.py:254–267](../exps/tasks/pi05/transport.py#L254-L267)）。M\* 的 OpenPI WebSocket（[openpi.py:269](../external/mstar/mstar/api_server/openpi.py#L269)）存在，但 `exps/configs/mstar/pi05_bfloat16.yaml` 里没有 `openpi:` 块，所以实验没有测过它。**Level 1 的公平对比必须让三家走同一个 OpenPI WebSocket 协议重测。**
5. **三家都是"一条连接、一次在途调用"。** OpenPI 参考服务端（[websocket_policy_server.py:52–72](../external/openpi/src/openpi/serving/websocket_policy_server.py#L52-L72)）、M\*（[openpi.py:285–293](../external/mstar/mstar/api_server/openpi.py#L285-L293)）、vLLM-Omni（[connection.py:228–278](../external/vllm-omni/vllm_omni/entrypoints/openpi/connection.py#L228-L278)）、SGLang core（[ws_utils.py:32–45](../../Inference_Engine_Investigation_Repo/external/sglang/python/sglang/multimodal_gen/runtime/entrypoints/action/ws_utils.py#L32-L45)）都是 receive → infer → send 的串行循环。同一连接内无法抢占在途推理。
6. **上游没有一家真正取消在途 GPU 计算。** LeRobot v0.7.0 的取消只在客户端生效，服务端不会中断（[remote_inference/README.md:66](../external/lerobot/src/lerobot/remote_inference/README.md#L66)，由子代理读取，见 [日志](../agents_results/2026-11-20_lerobot-async-rtc.md)）。所以"抢占"的延迟上限是当前推理的剩余时间，约等于一次 E2E。（四个服务端（含 M\* 的两个前端）实测确认：断开连接后，在途推理仍然完成，B 等的是 A 的剩余时间，见第 2.6.4 节。）
7. **RTC 的修正。** 上一版说 RTC 需要重新训练模型，这不完整。LeRobot 的 guided 模式对 pi05 默认开启，不需要重训（[configuration_rtc.py:66–76](../external/lerobot/src/lerobot/policies/rtc/configuration_rtc.py#L66-L76)，[modeling_pi05.py:1072–1073](../external/lerobot/src/lerobot/policies/pi05/modeling_pi05.py#L1072-L1073)）。但它要求去噪步可求导（子代理在 `inference/prediction.py:58` 读到这一点），而 M\* 和 SGLang core 的去噪步都在 CUDA 图里，求导需要重新捕获。所以 RTC 是**引擎级**改动，不是模型改动，vLLM-Omni 明确拒绝 `rtc_config`（[config.py:433–436](../external/vllm-omni/vllm_omni/diffusion/models/pi05/config.py#L433-L436)）。
8. **Level 1 与 Level 2 的边界修订**（见第 4 节）。Level 1 = 运行时外部的全部调度、快照、队列、年龄检查。Level 2 = 引擎内部的同连接并发、真正的中止、guided RTC、按路增量更新。Pi0.5 的全注意力前缀和"state 写进 prompt"决定了增量输入对它几乎没有收益。
9. **LeRobot 的 `async_inference` 在 v0.6.1 是最后一个带有它的版本。** 移除提交是 `ae2b57f12`（#4937，引入 `remote_inference/`），v0.7.0 中已不存在（用完整历史核对，使用的是临时克隆，未保留）。另一位 AI 引用的 `robot_client.py`、`policy_server.py` 路径在 v0.6.1 存在，内容见第 3.2 节。
10. **另一位 AI 没有提到 SGLang core 的 Pi0.5。** `../Inference_Engine_Investigation_Repo/external/sglang` 的 cookbook 和 `multimodal_gen` 里有完整的 Pi0.5 路径（HTTP、`/v1/actions/realtime`、`/openpi/policy`、可选精确前缀缓存、CUDA 图）。`external/sglang-omni` 里没有 Pi0.5 代码（见 [third-case plan](../../Inference_Engine_Investigation_Repo/agents_results/2026-11-14_pi05-third-case-plan.md)）。
11. **2026-11-20 的同协议探测（OpenPI WebSocket，RTX 5880，GPU 0，第 2.6 节）。** E2E 中位数：M\* 默认 Rust 前端 66 ms（已弃用的 Python 前端 61 ms），SGLang core 默认配置 167 ms、分桶配置（`prompt_token_buckets: [200]`）97 ms，vLLM-Omni 211 ms。四家都是同一连接内串行处理。单连接最小可持续周期：M\* 在 50–75 ms 之间，SGLang core 默认在 150–200 ms、分桶后接近 100 ms，vLLM-Omni 在 200–300 ms。**版本注意：** 最新版 M\* 比活动对比版本（9c6f1ad，97 ms）快得多，不能把它和活动报告中的 M\* 数字混用。SGLang 和 vLLM 的数字与活动版本一致（167 对 162，211 对 211）。**SGLang 的默认配置不是它的最佳配置：** 默认的 CUDA 图按精确的提示词长度缓存，只有 4 个名额，满了以后新长度走 eager（第 2.6.3 节）。比较时 SGLang 应取分桶配置。
12. **Level 1 测量（2026-11-21，第 2.7 节）。** 基础模型 `lerobot/pi05_droid`（15 步动作块，DROID 的 15 Hz 控制），输入是 `lerobot/droid_100` 的多频回放（状态 100 Hz，相机 15 / 15 / 7.5 Hz，语言指令按 episode 到达），控制 15 Hz，七种触发策略。外部 Runtime 不改框架就能完成快照、触发、动作块替换与欠载计数。M\*（Rust）和 SGLang core（分桶）在所有触发策略下欠载为 0；vLLM-Omni 在周期和阈值触发下欠载 0.5–5%，在"执行 8 步后重规划"下 21%（最长连续欠载 467 ms）。同一触发下动作年龄的差别约等于 E2E 的差别。这些是回放加模拟控制器的结果，不是真机。

## 1. 修正记录

| 之前的说法 | 实际 | 依据 |
|---|---|---|
| M\* 没有运行过 Pi0.5 | 运行过，有完整实测（RTX 5880 与 H200） | [exps/results/baseline-pi05_b2b_rtx5880.md](../exps/results/baseline-pi05_b2b_rtx5880.md)；[report 09a–09c](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters) |
| 5880 没有干净的 Pi0.5 基线 | 有基线批（3 次启动，138 个请求/框架）。"clean"指标全部为 0%（时钟事件），比值只能当作相对量 | 同上，表头说明 |
| M\* 的前缀缓存未核对 | 对 Pi0.5 关闭：`kv_labels` 声明的是无键的 `KVLabel()`（实验版本 [Inference_Engine_Investigation_Repo/external/mstar …/pi05_model.py:580–581](../../Inference_Engine_Investigation_Repo/external/mstar/mstar/model/pi05/pi05_model.py#L580-L581)；最新快照 [:561–562](../external/mstar/mstar/model/pi05/pi05_model.py#L561-L562)） | [agents_results/2026-11-14_pi05-harness-implementation.md](../../Inference_Engine_Investigation_Repo/agents_results/2026-11-14_pi05-harness-implementation.md) |
| M\* 没有机器人入口 | 有 OpenPI WebSocket（根路径，需要 `openpi:` 块）和 Rust 端口；实验没有用它 | [auth.py:33–35](../external/mstar/mstar/api_server/auth.py#L33-L35)，[openpi.py:269](../external/mstar/mstar/api_server/openpi.py#L269) |
| RTC 需要重新训练模型 | guided 模式不需要重训，但需要去噪步可求导；trained 模式才需要重训 | [configuration_rtc.py:66–76](../external/lerobot/src/lerobot/policies/rtc/configuration_rtc.py#L66-L76)；子代理日志 Q2 |
| 100 ms 是默认周期 | 没有依据。5880 上 M\* 10 步的 p50 约 100 ms，周期只能取测得分布的高分位数 | 第 2 节 |
| `pi05_droid` 与我们实验的 Pi0.5 是同一个模型 | 不是。实验用 `lerobot/pi05_base`（3 相机，32 维，50×32）；OpenPI 的 `pi05_droid` 是 horizon 15、8 维、2 相机。两者 checkpoint 不同 | [config.py:629–634](../external/openpi/src/openpi/training/config.py#L629-L634)，[droid_policy.py:36–40](../external/openpi/src/openpi/policies/droid_policy.py#L36-L40)；实验用的是 `exps/tasks/pi05` 的 "DROID 46 行"数据夹具，名字容易混淆 |

## 2. Pi0.5 三框架的本地实测（RTX 5880）

### 2.1 基线（46 行 DROID 夹具，10 步，E2E 中位数，毫秒）

| 框架 | 默认配置 | p50 [各次启动范围] | 传输方式 | 来源 |
|---|---|---|---|---|
| M\* | 图开启，prefill 编译，action 用 max-autotune | 97.3 [93.8–100.6] | HTTP `/generate`，PNG，计时含服务端解码 | [baseline-pi05_b2b_rtx5880.md](../exps/results/baseline-pi05_b2b_rtx5880.md) |
| SGLang core | 前缀图与动作图开启，无编译 | 162 [160–167] | WebSocket，原始数组 | 同上 |
| SGLang core（分桶 `prompt_token_buckets: [200]`） | 同上，另加分桶 | 93（另一批，同批默认 167） | WebSocket，原始数组 | [report 09b](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters/09b_pi05_results.tex) |
| vLLM-Omni | eager，无图，无前缀缓存（配方默认） | 211 [209–215] | WebSocket，原始数组 | [baseline-pi05_b2b_rtx5880.md](../exps/results/baseline-pi05_b2b_rtx5880.md) |

补充的审计批（M\* 的开关）：关闭编译 107 ms，关闭图 281 ms，eager 258 ms（[third-case plan](../../Inference_Engine_Investigation_Repo/agents_results/2026-11-14_pi05-third-case-plan.md)）。这说明 M\* 的图和编译是它快的主要原因。

### 2.2 步数扫描（2 次启动，每格 24 个请求）

| 去噪步数 | M\* | SGLang core | vLLM-Omni |
|---:|---:|---:|---:|
| 1 | 71.6 | **52.2** | 90.7 |
| 5 | 116 [89–156] | 96.5 | 145 |
| 10 | 99.9 | 161 | 211 |
| 20 | 122 | 307 | 346 |

来源：[pi05-steps_b2b_rtx5880.md](../exps/results/pi05-steps_b2b_rtx5880.md)。M\* 的 5 步数据跨度大（89–156），需要重测。H200 上的步数实验（[report 09b](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters/09b_pi05_results.tex)）趋势相同：M\* 每步约 3.6 ms，SGLang core 约 11.5 ms。

### 2.3 前缀缓存与图拆分

- **同一样本重复 20 次**（[pi05-kv_b2b_rtx5880.md](../exps/results/pi05-kv_b2b_rtx5880.md)）：M\* 97.9 ms（无缓存）；SGLang core 默认 91.5 ms，开启精确全前缀缓存后 50.3 ms；vLLM-Omni 210 ms。缓存只对完全相同的观测有效。机器人的帧每个 tick 都在变，正式运行时几乎不会命中。
- **图拆分**（[pi05-graph-split_b2b_rtx5880.md](../exps/results/pi05-graph-split_b2b_rtx5880.md)）：在 46 行、10 步下，SGLang core 关闭前缀图或动作图后都是 162 ms，没有差别。

### 2.4 三框架的 Pi0.5 执行结构（源码，来自 report 09a）

- **M\***：2 个节点、2 条 Walk（prefill 为 ViT → LLM，action_gen 为 LLM 上的 Loop，最多 10 次）。每个请求 12 次 CUDA 图重放（ViT 1，prefill 1，action 10）。前缀 KV 在 prefill 时写入，去噪时只读。
- **SGLang core**：4 个流水线阶段；10 步循环在主机上，每步可选图重放（最多录 4 种形状）。
- **vLLM-Omni**：1 个阶段；10 步循环在主机上，每步 eager 前向。

三家都是"单请求内复用前缀 KV，跨请求不复用"。

### 2.5 上游自报（非本环境测量，只能说明量级）

| 框架 | 硬件 | 配置 | 数字 | 来源 |
|---|---|---|---|---|
| vLLM-Omni 配方 | H100 80 GB | `pi05_base`，eager，10 步，客户端 E2E（含 msgpack 与 WebSocket） | 1/2/3 相机：fp32 282/356/398 ms；bf16 290/301/312 ms | [Pi05.md:130–133](../external/vllm-omni/recipes/lerobot/Pi05.md#L130-L133) |
| vLLM-Omni 配方 | RTX PRO 5000 | 同上 | fp32 185/231/281 ms；bf16 156/177/203 ms | [Pi05.md:134–135](../external/vllm-omni/recipes/lerobot/Pi05.md#L134-L135) |
| SGLang cookbook | H100 | ALOHA 的 OpenPI WebSocket，持久连接，前缀缓存关闭 | 单样本 77.0 ms；batch=4 为 162.0 ms | [cookbook 验证表](../../Inference_Engine_Investigation_Repo/external/sglang/docs/cookbook/vla/OpenPI/Pi0.5.mdx#L470) |
| SGLang cookbook | H100 | ALOHA，Python grouped API，CUDA 图 | 单样本 52.4 ms；batch=4 为 91.9 ms | 同上 |

这些数字的 checkpoint、精度、eager/graph、测量边界各不相同，不能与第 2.1 节直接排名。

### 2.6 同协议探测（2026-11-20，RTX 5880，GPU 0）

**设置**：四个服务端都走 OpenPI WebSocket，同一份 46 行夹具，同样的输入（三路图像、归一化的 32 维 state、prompt、seed 42），10 步。三家都加载 `lerobot/pi05_base` 的缓存快照（M\* 的服务日志确认）。`scripts/run_probes.py` 起服务，`src/scenarios/protocol_probes.py` 发请求。

- **M\*（默认前端，Rust `mstar-server`）**：最新版 `external/mstar`（2c8ffde8）。`mstar-server` 在 `external/mstar/rust/server` 下用 `cargo build --release` 编译（1 分 22 秒），副本在 `build/bin/mstar-server`。`mstar_rust` 用最新源码编译，解包的副本在 `build/mstar_rust`；服务能启动，说明源码哈希检查（`BUILD_SOURCE_HASH` 5651ef713b0c03f5）通过。配置 `configs/mstar_pi05_openpi.yaml`。
- **M\*（Python 前端，`--python-frontend`）**：同一代码和配置。服务日志标明该前端已弃用，只作对照。
- **vLLM-Omni**：v0.32.0（5917730 + 本地 overlay，`pinned/vllm-omni`；`envs/vllm-omni` 的可编辑安装指向它），与活动报告相同。
- **SGLang core**：pip 0.5.22（`envs/sglang-omni`），与活动报告相同。

各模式顺序执行，模式之间没有额外等待。测量开始前 GPU 0 空闲（2 MiB）；GPU 3–7 在测量期间有其他任务（快照利用率 67–100%）。原始 JSON、服务日志和 GPU 快照在 `runs/2026-11-20_ws-probe/<框架>/`；取消探测的修正版在 `runs/2026-11-20_ws-probe-cancel-v2/`（见 2.6.4）。测量之后，运行环境改为本目录内的副本（`envs/`，见 [README](../README.md)），不再依赖基准仓库；上表数字不变，冒烟测试在 `runs/2026-11-20_env-independence-smoke/`。

**2.6.1 同协议端到端（单连接，138 个请求，中位数，毫秒）**

| 服务 | E2E p50 | p95 | 服务端 infer p50 | 备注 |
|---|---:|---:|---:|---|
| M\*（Rust 前端，WebSocket） | 66.4 | 68.7 | 64.1 | 同一服务里 HTTP `/generate` 为 68.5 |
| M\*（Python 前端，WebSocket） | 61.2 | 63.7 | 60.0 | 同一服务里 HTTP 为 70.3；已弃用 |
| SGLang core（WebSocket，默认配置） | 166.8 | 170.4 | 165.3 | 活动报告为 162；图缓存只有 4 个名额，见 2.6.3 |
| SGLang core（WebSocket，分桶 `prompt_token_buckets: [200]`） | 96.8 | 100.9 | — | 2026-11-21 同协议复测，`runs/2026-11-21_sglang-buckets-base/`；46 行全部 94–98 ms |
| vLLM-Omni（WebSocket） | 211.2 | 286.4 | — | 活动报告为 211 |

默认的 Rust 前端比 Python 前端的 WebSocket 端到端慢约 5 ms，服务端推理也慢约 4 ms，原因未查明。活动报告的 M\* 97 ms（9c6f1ad，HTTP）是另一个版本，不能与这里直接比较。

**2.6.2 开放式周期发送（单连接，每档 40 个请求，p50，毫秒）**

| 周期 | M\*（Rust） | M\*（Python） | SGLang core | vLLM-Omni |
|---:|---:|---:|---:|---:|
| 50 ms | 342.4 | 279.1 | 1866.5 | 3802.2 |
| 75 ms | 67.0 | 60.6 | 1368.5 | 2890.1 |
| 100 ms | 68.4 | 56.1 | 921.3 | 2628.6 |
| 150 ms | 63.0 | 56.9 | 203.3 | 1940.8 |
| 200 ms | 63.8 | 56.5 | 166.6 | 943.0 |
| 300 ms | 68.4 | 64.2 | 167.5 | 213.6 |

判据是 p50 不随请求数增长。M\*（两个前端）在 50 ms 时排队，75 ms 起可持续；SGLang core 在 150 ms 排队（p50 203 ms），200 ms 起可持续；vLLM-Omni 在 200 ms 排队（p50 943 ms），300 ms 起可持续。所以单连接的最小可持续周期：M\* 在 50–75 ms 之间，SGLang core 在 150–200 ms 之间，vLLM-Omni 在 200–300 ms 之间。100 ms 的控制周期只对 M\* 可行。

**2.6.3 同连接突发（每组 8 个请求，5 组）**

读数是相邻完成之间的间隔：

- M\*（Rust）：61–67 ms，五组一致。
- M\*（Python）：61–71 ms。
- vLLM-Omni：中位数约 211 ms。第 2、3、4 组各有 1–3 个间隔超过 250 ms（第 4 组的首个完成为 404 ms），最大 514 ms，原因未确定。
- SGLang core（默认配置）：前四个请求的完成间隔为 82–100 ms，后四个为 163–173 ms（第一个请求的数字是从突发开始算起的完成时间，82–99 ms）。单请求 E2E 也有同样的按行差异（第 0–3 行中位数 97–109 ms，第 4–7 行 167–169 ms，三轮一致），所以这不是突发排队造成的，也不是 prompt 字符数造成的（第 4 行只有 29 个字符但慢，第 1 行有 53 个字符但快）。**原因（2026-11-21 核对）：** SGLang 的去噪 CUDA 图按精确的前缀长度缓存，容量 4，`prompt_token_buckets` 默认为空；服务日志记录了对 4 个前缀长度（pi05_base 为 908、916、910 等）的图捕获，并在 pi05_droid 的日志里有 `VLA denoise CUDA graph capacity reached (4); new signatures run eagerly`。Pi0.5 的提示词带着离散化的 state 文本，长度逐请求变化，所以最先出现的 4 种长度走图（约 95 ms），其余走 eager（约 165 ms）。换成 `prompt_token_buckets: [200]` 后，46 行全部在 94–98 ms，突发里每个请求的完成间隔稳定在约 95 ms（默认配置：约 92 ms，之后 160–200 ms），数据在 `runs/2026-11-21_sglang-buckets-base/`。"新长度走 eager"这一点来自日志的明文，eager 路径更慢是由延迟差推断的，没有单独测。分桶会改变归约形状（SGLang 配置文件的注释），未核对数值差异。

**2.6.4 取消探测（修正版，每框架 15 组）**

方法：A 在 t=0 发出请求；B 在一条已打开的连接上，于 gap（= 0.4 × 对照中位数）时发出。`a_closed` 臂在 gap 时关闭 A 的连接，启动关闭握手但不等它完成就发送 B；`a_open` 臂保持 A 打开。若断开会丢弃在途推理，B 应接近对照值；若推理照常完成，B 应等于对照值加上 A 的剩余时间，即对照 +（对照 − gap）。

| 服务 | 对照 | gap | A 关闭后的 B | A 保持打开时的 B | 预测（不中止） |
|---|---:|---:|---:|---:|---:|
| M\*（Rust 前端） | 66.2 | 26.5 | 93.2 | 93.2 | 105.9 |
| M\*（Python 前端） | 62.6 | 25.0 | 90.9 | 90.6 | 100.1 |
| SGLang core | 164.7 | 65.9 | 258.5 | 257.1 | 263.5 |
| vLLM-Omni | 210.4 | 84.2 | 336.4 | 328.2 | 336.7 |

（毫秒，p50）

结论：四个服务端都没有在断开时丢弃在途推理。同一组内 A 关闭与保持打开两臂的配对差，中位数分别为：M\*（Rust）−0.2 ms，M\*（Python）+0.3 ms，SGLang core −1.6 ms，vLLM-Omni +7.1 ms。若断开会中止推理，关闭臂应比保持打开臂低约 27 ms（M\*）、92 ms（SGLang core）、118 ms（vLLM-Omni）。M\* 两个前端的 15 组配对差都在 −5.1 到 +5.0 ms 之间；SGLang core 和 vLLM-Omni 的单组差异较大（vLLM-Omni 为 −114 到 +55 ms），因为两臂不是同一次推理，但中位数都接近 0。B 与预测的偏差：SGLang core 和 vLLM-Omni 在 3% 以内，M\* 比预测小 9–12%，原因未查明。

Rust 服务端的接收循环在推理期间不读取套接字（[openpi.rs:566–579](../external/mstar/rust/server/src/ws/openpi.rs#L566-L579)），所以它不能在断开时中止推理，与数据一致。

第一版探测的 `a_closed` 臂在 `close()` 返回后才发送 B。Rust 前端的关闭要等推理完成，B 因此看起来"没有排队"（65.3 ms，等于对照），被误读为中止。修正后的探测不再等待关闭。第一版 vLLM-Omni 的关闭臂比保持打开臂高约 32 ms，修正后两臂的 p50 差为 8 ms；这与关闭握手的等待有关，但未完全确定。v2 的 `cancel.json` 里有一个 `a_close_handshake_ms` 字段，它测的是 B 完成之后的时间，不是关闭耗时；`probes.py` 已删除该字段，不要使用。

**2.6.5 多连接并发（每连接闭环，每连接 40 个请求）**

| 连接数 | M\*（Rust） 吞吐 / p50 | M\*（Python） | SGLang core | vLLM-Omni |
|---:|---:|---:|---:|---:|
| 1 | 15.01 req/s / 66 ms | 15.68 / 63 | 6.87 / 165 | 4.57 / 213 |
| 2 | 17.11 / 117 | 17.09 / 117 | 7.22 / 256 | 4.25 / 427 |
| 4 | 18.34 / 221 | 18.30 / 221 | 7.25 / 570 | 4.04 / 885 |

吞吐随连接数的变化：M\*（Rust）+22%，M\*（Python）+17%，SGLang core +6%，vLLM-Omni −12%；p50 随连接数线性增长。跨连接没有实质并行，每个服务端都在一个队列里串行推理。M\* 的 +17–22% 可能来自前后处理与推理的重叠，未查明。

**2.6.6 这一节确认了什么**

1. OpenPI 协议下四个服务端都是同一连接内串行处理。
2. 四个服务端都不会在断开时丢弃在途推理（修正版取消探测）。B 等的是 A 剩余的推理时间。第一版"Rust 前端会中止"的结果是探测错误造成的。
3. 同连接的排队延迟可以直接测到，并且和单请求 E2E 一致。
4. 在 5880 上，单连接的最小可持续周期：M\* 在 50–75 ms 之间，SGLang core 默认配置在 150–200 ms 之间（分桶后 E2E 为 97 ms，Level 1 里 100 ms 周期会丢约 5% 的触发），vLLM-Omni 在 200–300 ms 之间。100 ms 的控制周期对 M\* 可行，对分桶的 SGLang 接近饱和。
5. 同协议 E2E 的比值：用默认前端的 M\*（66.4 ms），SGLang core 默认配置慢 2.5 倍、分桶后慢 1.5 倍，vLLM-Omni 慢 3.2 倍。活动报告的比值是 1.67 倍和 2.17 倍（M\* 为 9c6f1ad 版，97 ms）。版本和前端不同，报告里要写明 M\* 的版本与前端。

## 2.7 Level 1 测量（2026-11-21，RTX 5880，GPU 0）

**问题：** 把调度全部放在框架之外（外部 Runtime，框架只看到完整请求），在多路输入以不同频率到来的情况下，三个框架各能支撑什么样的触发节奏，得到的动作有多新，会不会断档（欠载）。

**基础模型：`lerobot/pi05_droid`。** 与 `pi05_base` 同架构、同样的三个相机键、32 维 state，区别是动作块 15 步（DROID 的真实水平线）。选它的理由是队列行为取决于块长度：15 步在 15 Hz 下是 1 秒缓冲，欠载才有可测的余地；`pi05_base` 一次 50 步（3.3 秒）会掩盖欠载。三个框架都只需要改配置里的水平线（M\* 从检查点的 `config.json` 读取）。三家都已加载并返回 15×32 的动作块（`runs/2026-11-20_pi05-droid-check/`）。检查点在仓库中不带归一化统计，droid_100 的 state 是 7 维笛卡尔位姿加夹爪，而 pi05_droid 训练用的是关节角，所以**动作的数值没有语义，这里只测时间**。

**数据集：`lerobot/droid_100`。** 真实 DROID 遥操作，3 路相机（180×320）、state、每个 episode 一条语言指令，原生帧率只有一个，15 Hz。多频到达是回放时按固定种子模拟的，使用前 8 个有任务标签的 episode（约 127 秒；每个 cell 只用到前 4 个）：

| 输入 | 到达频率 | 实现 |
|---|---|---|
| state | 100 Hz | 由 15 Hz 样本线性插值（欧拉角先 unwrap 再插值，夹爪线性）；用来代替高频本体感知话题 |
| 相机 ext1 / wrist | 15 Hz | 原生帧，wrist 相位错开 22 ms |
| 相机 ext2 | 7.5 Hz | 每两帧取一帧（较慢的相机） |
| 语言指令 | 事件 | 每个 episode 开始时一次 |

每个样本有采集时间和到达时间（采集时间加固定延迟 2 / 35 / 25 / 45 ms，再加种子固定的均匀抖动 1 / 5 / 5 / 8 ms）。**这些延迟和抖动是假设，不是真机测量。** 三个框架用同一份时间表。另有 `lockstep` 对照：四路输入由同一个 15 Hz 时钟采样、没有延迟，相当于一行数据集记录。控制时钟与周期定时器比相机时钟慢 0.4%，并为每个 cell 设随机起始相位，否则 15 Hz 的控制时钟会与 15 Hz 的相机相位锁死，每次快照看到的输入年龄恒定（冒烟时发现）。

**Runtime（`src/runtime/`）：** 每路"≤ t 最新一条"的快照并记录各路年龄；三种触发：固定周期、OpenPI 式（执行 8 步后重规划，块用完也请求）、LeRobot 式回放阈值（队列剩余动作覆盖时长 ≤ `refill_s`，并不小于实测周转时间加一步）；动作块带代数号，旧代数丢弃；执行器按 15 Hz 取动作，对齐模式跳过已过期的动作（LeRobot 的 aligned），朴素模式从动作 0 开始（OpenPI 参考客户端）；动作用完即欠载。一条连接，最多一个在途请求。测试（`tests/`）用固定延迟的假服务器，预期值手算。

**指标：** 每个请求的 E2E、服务端时间、各路输入在快照时的年龄；每个控制 tick 的剩余动作数、是否欠载、**动作年龄**（自观测快照起到动作被执行）、**预定时刻滞后**（执行时刻减该动作被规划的时刻）、各路输入在动作被执行时的陈旧度。

**协议：** 十个 cell（七种触发策略的多频输入、朴素对照、两个 lockstep 对照），每个 cell 预热 3 秒后测 40 秒（600 个控制 tick）。三个框架交错，每一轮对每个框架各重新启动一次服务、跑一遍全部 cell，共 3 轮，框架顺序按轮轮换，cell 顺序每轮轮换 3 个位置。表中是各轮的中位数。主机 1 分钟 load average 在 28–169 之间（256 核，其他用户的作业同机运行）。原始数据和汇总在 `runs/2026-11-20_level1/`（含 `level1_summary.md`），补测在 `runs/2026-11-20_level1-openloop-fix/`、`runs/2026-11-20_level1-sglang-bucketed/`。

**请求速率（个/s）与 E2E p50 / p95（ms）**

| 触发 | M\*（Rust） | vLLM-Omni | SGLang core（默认） | SGLang core（分桶） |
|---|---|---|---|---|
| period-100ms | 10.0 · 60 / 66 | 3.1 · 220 / 432 | 7.2 · 94 / 172 | 9.4 · 94 / 100 |
| period-200ms | 5.0 · 61 / 66 | 2.4 · 227 / 509 | 5.0 · 110 / 173 | 5.0 · 95 / 100 |
| period-300ms | 3.3 · 62 / 68 | 2.8 · 216 / 418 | 3.3 · 119 / 174 | 3.3 · 95 / 100 |
| openloop-8 | 1.6 · 59 / 67 | 1.2 · 228 / 383 | 1.4 · 154 / 179 | 1.5 · 95 / 100 |
| threshold-0.3s | 1.5 · 60 / 67 | 1.6 · 222 / 495 | 1.5 · 156 / 179 | 1.5 · 94 / 98 |
| threshold-0.5s | 2.1 · 61 / 66 | 2.0 · 226 / 494 | 2.1 · 153 / 169 | 2.1 · 94 / 101 |
| threshold-0.8s | 5.4 · 61 / 67 | 2.9 · 229 / 527 | 5.2 · 98 / 175 | 5.4 · 94 / 100 |

**动作年龄 p50 / p95（ms，自观测快照起到动作被执行）**

| 触发 | M\*（Rust） | vLLM-Omni | SGLang core（默认） | SGLang core（分桶） |
|---|---|---|---|---|
| period-100ms | 101 / 134 | 403 / 668 | 168 / 335 | 134 / 201 |
| period-200ms | 134 / 202 | 469 / 802 | 266 / 336 | 201 / 268 |
| period-300ms | 201 / 335 | 436 / 771 | 268 / 436 | 234 / 368 |
| openloop-8 | 335 / 603 | 601 / 870 | 469 / 803 | 467 / 736 |
| threshold-0.3s | 401 / 670 | 601 / 870 | 469 / 801 | 467 / 736 |
| threshold-0.5s | 268 / 469 | 534 / 739 | 400 / 603 | 335 / 536 |
| threshold-0.8s | 134 / 201 | 468 / 804 | 202 / 336 | 201 / 268 |

**欠载 tick 占比（最长连续欠载，ms）**

| 触发 | M\*（Rust） | vLLM-Omni | SGLang core（默认） | SGLang core（分桶） |
|---|---|---|---|---|
| period-100ms | 0.0%（0） | 0.5%（133） | 0.0%（0） | 0.0%（0） |
| period-200ms | 0.0%（0） | 3.5%（267） | 0.0%（0） | 0.0%（0） |
| period-300ms | 0.0%（0） | 2.2%（200） | 0.0%（0） | 0.0%（0） |
| openloop-8 | 0.0%（0） | 21.4%（467） | 0.0%（0） | 0.0%（0） |
| threshold-0.3s | 0.0%（0） | 4.8%（333） | 0.0%（0） | 0.0%（0） |
| threshold-0.5s | 0.0%（0） | 2.3%（267） | 0.0%（0） | 0.0%（0） |
| threshold-0.8s | 0.0%（0） | 5.0%（333） | 0.0%（0） | 0.0%（0） |

各轮之间的范围（最小到最大）见 `level1_summary.md`。几个值得说明的点：

1. **M\*（Rust）和 SGLang core（分桶）在所有触发策略下欠载为 0。** 100 ms 周期下 M\* 每秒 10.0 个请求、没有跳过触发；分桶的 SGLang 为 9.4 个、跳过 21 个触发（约 5%）；默认配置的 SGLang 7.2 个；vLLM-Omni 3.1 个。
2. **vLLM-Omni 在周期和阈值触发下欠载 0.5–5%，在"执行 8 步后重规划"下 21%（最长连续 467 ms）。** 它的 E2E 中位数约 220 ms，各触发策略下的 p95（各轮中位数）在 383–527 ms 之间；15 步动作块只有 1 秒，一次慢回复就用掉大部分缓冲。尾部有多少来自主机负载没有分开（见局限）。
3. **同一触发下，动作年龄的差别约等于 E2E 的差别。** 例如 200 ms 周期：M\* 134 ms，SGLang（分桶）201 ms，vLLM-Omni 469 ms。动作有多新主要由触发有多快决定，框架决定触发能多快：重规划间隔不可能短于 E2E，更密的触发只会被跳过（vLLM 在 100 ms 周期下跳过 275 个触发）。
4. **多频输入的陈旧度由最慢的一路决定，与框架无关。** `threshold-0.5s` 下，快照时最旧输入的年龄 p50 / p95 为 114 / 174 ms（7.5 Hz 的 ext2 相机：延迟 45 ms 加上最长 133 ms 的采样间隔），lockstep 下为 34 / 63 ms；三个框架的这个数相同。动作被执行时最旧输入的陈旧度 p50：M\* 406 ms（lockstep 310 ms），200 ms 周期下 258 ms（lockstep 170 ms）。
5. **对齐执行与朴素执行的差别在"预定时刻滞后"，不在动作年龄。** 朴素执行下每个动作比它被规划的时刻晚约一个请求延迟：M\* +68 ms，SGLang（分桶）+135 ms，vLLM-Omni +337 ms；对齐执行下滞后在（−67, +2] ms 内（向上取整，动作最多提前一步，因此 p50 是 −64 ms，p95 是 +1 ms；这是定义的结果，不是误差）。M\* 和 SGLang（分桶）的动作年龄在两种模式下相同（335 / 335 ms，467 / 467 ms），因为每个块都是从它到达执行到下一个块到达。**vLLM-Omni 例外：** 动作年龄 601（对齐）对 803 ms（朴素），欠载 21.4% 对 2.3%。朴素执行把块"拉长"了一个延迟，用更旧的动作填住了缺口；对齐执行下块里剩余的动作更少，更容易用完。所以朴素执行的低欠载是用动作变旧换来的，不是更好。
6. **这些结果是在不改框架的前提下得到的。** 快照、触发、替换、欠载计数都在外部完成。外部 Runtime 做不到的是取消（四个服务端断开后都继续算完，第 2.6.4 节），所以更密的触发不能靠"取消旧的"来换新鲜度。

**测量中发现并改正的问题（以免被引用）：**

- 第一版 `open_loop` 触发有死锁：对齐执行会跳过已过期的动作，回复较慢时块里不足 8 个动作，"执行满 8 步"永远不满足，之后不再发请求。vLLM-Omni 的前两轮据此得到 0 个请求。已改为"执行满 8 步或块用完"（OpenPI 的参考客户端也在块用完时请求），加了回归测试，三个框架的 openloop-8 都用修复后的代码重测（`runs/2026-11-20_level1-openloop-fix/`）。M\* 和 SGLang 的数字与修复前一致。
- SGLang 默认配置的延迟与请求内容有关（第 2.6.3 节），各 cell 的请求内容不同，所以它的 E2E 随触发策略变化（94–156 ms）。分桶配置下全部为 94–95 ms，表中同时给出两种。
- "动作年龄"区分不了对齐与朴素执行，所以加了"预定时刻滞后"。

**局限：**

- 输入频率、延迟和抖动是模拟的；控制器是模拟的（15 Hz 取动作，没有机器人）；动作没有语义。因此结果说明的是"外部 Runtime 加这些框架的时间行为"，不是真机上的任务成功率。
- 只测了一个 GPU（5880 的 GPU 0），主机共享，其他用户的作业在同一台机器上。vLLM-Omni 是 eager 模式，对 Python 侧开销敏感，它的重尾（p95 383–527 ms；`pi05_droid` 的 e2e 检查里还出现过 p50 349 ms，同一次检查里最小值仍是 206 ms）里有多少来自主机负载，没有分开。
- M\* 只测了默认的 Rust 前端。SGLang 分桶会改变归约形状，数值差异未核对。
- 8 个 episode 里只用到前 4 个（cell 时长 40 秒），提示词和 state 的多样性有限。

**复现：**

```bash
envs/mstar/bin/python scripts/run_level1.py --run-dir runs/<日期>_level1 --repeats 3
envs/mstar/bin/python scripts/run_level1.py --run-dir runs/<日期>_level1-sglang-bucketed --repeats 3 --frameworks sglang-core-bucketed
envs/mstar/bin/python scripts/level1_report.py --run-dir runs/<日期>_level1 --extra-dir runs/<日期>_level1-sglang-bucketed
```

## 3. 上游源码分析

### 3.1 OpenPI（协议与参考客户端）

子代理日志：[2026-11-20_openpi-remote-pi05.md](../agents_results/2026-11-20_openpi-remote-pi05.md)（已抽查 5 处行号）。

- **线协议**：连接时服务端发一帧 msgpack 元数据（pi05_droid 为 `{}`，[websocket_policy_server.py:52](../external/openpi/src/openpi/serving/websocket_policy_server.py#L52)）。每次调用客户端发一个观测字典，服务端回 `actions` 加 `server_timing`（`infer_ms`、`prev_total_ms`，[:58–71](../external/openpi/src/openpi/serving/websocket_policy_server.py#L58-L71)）。
- **串行与排队**：每连接一次一个请求；`infer()` 是同一个 asyncio 循环里的同步调用，所以多个客户端会在服务端排队，并阻塞健康检查（推断，未运行）。
- **没有的东西**：没有 request 或 session id，没有 reset，没有 cancel，没有 RTC，没有跨请求状态（除每次调用的 RNG 拆分）。
- **变换在服务端**：`policy.py:70–71` 与 `policy_config.py:75–89` 完成全部输入变换。客户端只做 resize 到 224 和转 uint8（[docs/remote_inference.md:45–48](../external/openpi/docs/remote_inference.md#L45-L48)）。`api_key` 头会发送，但服务端不校验。
- **参考客户端的节奏**：DROID 示例在 chunk 下标达到 `open_loop_horizon=8` 时重新请求，先执行 8 步，然后阻塞在 infer 中（`examples/droid/main.py:40–42, 114, 131–133`）。LIBERO 用 `replan_steps=5`。`ActionChunkBroker` 只在 `_last_results is None` 时重新请求（`action_chunk_broker.py:28–30`，已核对），中间到达的观测被忽略。DROID 环境的控制循环是单线程的（observe → infer → step，`examples/droid/main.py:32–37, 133, 151`，已核对），阻塞的 infer 会卡住循环；机器人在停顿期间是否保持设定点未核对（DROID 环境 `droid-dataset/droid`，HEAD 33ae6a67）。这是 OpenPI 的 Runtime 级模式：固定的重规划间隔，没有年龄检查、取消或替换。
- **模型**：`pi0.py:222` 默认 `num_steps=10`；前缀（图像加 prompt）算一次写入 KV，去噪步复用（`pi0.py:233–267`）。前缀注意力是全双向（`ar_mask` 为 False，`pi0.py:125, 133`），所以任何前缀变化都会使全部前缀 KV 失效，没有增量追加。
- **Pi0.5 的 state**：没有 state token。state 经分位数归一化后，离散成 256 个 bin 的整数，写进 prompt 文本（`transforms.py:141–145`，`external/openpi/src/openpi/models/tokenizer.py:24–28`）。Pi0 则在后缀中保留连续 state token（`pi0.py:151–157`）。所以 **Pi0.5 的 state 一变，前缀 KV 就失效**。
- **配置**：`pi05_droid` 为 `action_horizon=15`、`action_dim=32`、`max_token_len=200`、离散 state（[config.py:629–642](../external/openpi/src/openpi/training/config.py#L629-L642)）。DROID 状态为 7 个关节加夹爪，共 8 维，补到 32，输出取前 8 维（`droid_policy.py:36–40, 81`）。DROID 示例断言输出为 (10, 8)，这与 `pi0_droid`（[config.py:603](../external/openpi/src/openpi/training/config.py#L603)）一致，而不是 `pi05_droid` 的 horizon 15。若用 `pi05_droid` 运行，回复为 (15, 8)，该断言会失败（代码核对，未运行）。这说明测试 harness 不能照抄 OpenPI 示例中的形状断言，应从服务端元数据或回复中读取 horizon。
- **计时注意**：`policy_timing.infer_ms` 在主机同步之前取，会偏低；应使用 `server_timing.infer_ms`（推断）。`prev_total_ms` 包含客户端的空闲时间。
- **依赖版本**（已核对）：`websockets` 在 `uv.lock:5160–5161` 固定为 15.0.1，`openpi-client` 下限 11.0（`packages/openpi-client/pyproject.toml:10`）；aloha 示例固定 14.1（`requirements.txt:131, 153`）。

### 3.2 LeRobot v0.7.0

子代理日志：[2026-11-20_lerobot-async-rtc.md](../agents_results/2026-11-20_lerobot-async-rtc.md)（已抽查 5 处行号）。

- **版本沿革（已用完整历史核对）**：`async_inference` 存在于 v0.6.1，在 `ae2b57f12`（#4937）中移除，由 `remote_inference/`（Zenoh 传输，`client.py`、`server.py`、`engine.py`）取代，v0.7.0 只有后者。v0.6.1 的客户端在 `chunk_size/qsize ≤ chunk_size_threshold`（默认 0.5，`configs.py:137`）时请求，每个 tick 检查一次（`robot_client.py:403–406`）；合并逐时间步进行，默认旧 0.3、新 0.7（`robot_client.py:224–267`）。
- **客户端决定何时请求，按"回放阈值"而不是固定周期**：当 `qsize × interval ≤ effective_refill` 时请求，`effective_refill = max(refill_seconds, turnaround + interval)`（`inference/execution.py:357–384, 228–230`）。默认值（[inference/factory.py:118–129](../external/lerobot/src/lerobot/inference/factory.py#L118-L129)）：`refill_seconds=0.5`，`action_timeout_s=5`，`max_observation_age_s=5`，`action_starvation_grace_s=1`，`chunk_merge="auto"`（解析为 aligned）。
- **观测为"只保留最新一条"的槽位**（`remote_inference/engine.py:521–528`），时间戳在读取机器人之前用 `time.monotonic()` 打（`rollout/robot_wrapper.py:52–59`）。一个时间戳覆盖所有传感器，没有按路的时间戳。
- **动作队列**：`ActionQueue`（`policies/rtc/action_queue.py:53`）在 `ChunkRuntime` 之下。它用一把非重入的 `threading.Lock`（`:88`），十个公开方法都持锁（`:109–260`），合并的辅助函数不重复加锁（`:266–274`），没有回调在锁内执行（已核对）。追加时用 merge；aligned 模式按游标修剪，并调用 `replace_future`（`execution.py:539–582`）。代数不对的结果被静默丢弃；超过 `max_age` 的结果是终止性故障（`execution.py:439, 454–455`）。
- **v0.7.0 的会话准入**：一个会话拥有部署，其他客户端收到 BUSY（`server.py:292–312`）。v0.6.1 的 `async_inference` 没有推理锁，gRPC 线程池 4 个（`policy_server.py:426`），可以并发（`policy_server.py:324`，已核对）。取消只在客户端（`engine.py:799–800`）。任何超时都是终止性的（`engine.py:815–846`）。
- **时间对齐**：没有。服务端把 `capture_time` 当作不透明字符串（`server.py:461–462`）。数据集的时间间隔检查只在离线时运行（`datasets/feature_utils.py:174–215`）。
- **Pi0.5 的服务端要求**：严格要求准入的特征集合，并检查形状，例如 `observation.state` float32 `[6]`、RGB uint8 HWC 图像、加 task（`server.py:463–476`，`docs/source/remote_inference.mdx:298–317`）。服务端归一化、分词并 resize 到 224（`deployment.py:158–165`，`modeling_pi05.py:1408–1412`）。
- **RTC 的两种模式**：
  - **guided**（默认）：`result = v − min(c·inv_r², max_guidance_weight)·correction`。autograd 的 correction 把 `x1 = x_t − t·v` 拉向前缀。权重在 delay 之前为 1，在 `execution_horizon` 处线性衰减到 0（`modeling_rtc.py:220–233, 255–274`）。这是软 inpainting，不是平均。
  - **trained**：需要训练时启用了 RTC 的 checkpoint（`rtc_training_max_delay > 0`），在每个 Euler 步硬性钳制前 delay 步（`modeling_pi05.py:81–128`，`flow_matching.py:370–375, 415–416`）。
  - 默认值（`configuration_rtc.py:66–76`）：mode `guided`，schedule `LINEAR`，`max_guidance_weight=10.0`，`execution_horizon=10`。文档建议 EXP（`docs/source/rtc.mdx:128`），代码默认是 LINEAR，两者不一致。
  - delay 是测得的：`ceil(turnaround / interval)`（`execution.py:132`）。
- **Pi0.5 的离散化**：QUANTILES 映射到 [−1, 1]，不做 clamp（`normalize_processor.py:404`），`np.digitize` 切成 256 个 bin，写成 `Task: ..., State: 12 200 ...;\nAction: `（`pi05/processor_pi05.py:79, 85–88`）。数值核对（用 numpy 复现 `processor_pi05.py:79` 的边界 −1 + i/128）：−1.5 → −1（哨兵，直接写进 prompt），−1.0 → 0，0.0 → 128，0.999 → 255，1.5 → 255。低端不截断，高端截断到 255。运行时应在发送前检查 state 是否在 [−1, 1] 内。

### 3.3 ROS 2 `message_filters` 8.0.5

子代理日志：[2026-11-20_ros2-message-filters.md](../agents_results/2026-11-20_ros2-message-filters.md)（已抽查 4 处行号）。本机没有安装 ROS 2（`/opt/ros` 不存在，没有 rclcpp 或 message_filters 头文件，没有 ROS 的 dpkg 包），因此本节全部是源码结论，没有编译或运行。

- **需要 ROS 节点吗**：`Subscriber` 需要节点（`subscriber.hpp:151–156`）。`Synchronizer` 可以不需要节点，直接 `add<i>(ptr)` 喂消息（`synchronizer.hpp:154–158`，单元测试就是这样写的）。库本身不建线程。
- **Cache**：按 `header.stamp` 排序、按数量上限截断（`cache.hpp:177–184, 357–360`）。`getElemBeforeTime(t)` 严格小于 t（`cache.hpp:275–291`）。"每路在 t 时刻或之前的最新一条"可以用 `getElemBeforeTime(t + 1ns)` 得到（推断）。没有最大年龄检查。Python 版的 Cache 两端都是闭区间，且追加时不排序，与 C++ 不一致（`__init__.py:174–207`）。
- **ApproximateTime**：没有 pivot 参数。`queue_size`、`age_penalty=0.1`、每路的消息间隔下界（只警告）、`max_interval` 实际上不限制。候选替换条件为 `(end − candEnd)·(1+p) < (start − candStart)`（`approximate_time.hpp:543–544`）。输出时每路各消耗一条，所以输出速率不超过最慢那一路（推断）。乱序消息被追加并告警一次，间隙不产生输出，也没有超时。丢弃是静默的。
- **ExactTime**：`std::map` 按纳秒时间戳索引，必须完全相等才匹配（`exact_time.hpp:99, 131–145`）。不同时钟的独立传感器几乎不会匹配（推断）。
- **时间来源**：header.stamp（`message_traits.hpp:96–119`）。没有 header 的类型静默得到 `Time(0,0)`。接收时间用系统时钟（`message_event.hpp:114–119`）。没有时钟偏差或延迟处理。
- **没有周期触发，没有 snapshot-at-tick**。`TimeSequencer` 和 `InputAligner` 有定时器但会转发每条消息。`LatestTime` 在最快那一路到达时发布，并无限期保留陈旧值（`latest_time.hpp:166–171`）。
- **可复用性**：约 65% 的 C++ 代码是通用的，只需替换 rclcpp 的 Time/Duration；约 33% 是可以直接复用的核心（近似、精确、Cache）。Python 包在导入时就需要 rclpy。

**对 Level 1 的含义**：每路取"≤ t 的最新一条"，并记录每路年龄，比用 `ApproximateTime` 更合适。`ApproximateTime` 适合相机帧之间的多传感器匹配，但输出速率跟着最慢的流走。

## 4. Level 1 与 Level 2 的修订定义

### 4.1 Level 1：运行时外部的周期快照（框架无状态）

框架只看到完整请求。全部调度、快照、队列、年龄检查都在 Runtime。

| 能力 | 上游参考 | 说明 |
|---|---|---|
| 触发 | LeRobot 回放阈值（`execution.py:357–384`）；OpenPI 的 `open_loop_horizon` / `replan_steps` | 固定周期和队列阈值两种都要测 |
| 观测槽位 | LeRobot 只保留最新一条，读取前打时间戳（`engine.py:521–528`） | 每路独立保存最新值和时间戳 |
| 快照 | message_filters Cache 的"≤ t 最新一条"语义 | 记录每路年龄 `age_i = t_snap − t_i` |
| 年龄上限 | LeRobot `max_observation_age_s=5`（过长） | 我们需要按传感器设定，相机和关节不同 |
| 动作队列与替换 | LeRobot `ActionQueue`，带代数号（`execution.py:539–582`） | 代数不对的结果静默丢弃 |
| 取消 | 只在客户端丢弃结果（`engine.py:799–800`） | 服务端不会中断 |
| 每连接并发 | 一个在途请求（OpenPI 协议的四个实现都串行） | 多客户端会排队，测到的是排队时间。LeRobot v0.6.1 的 `async_inference` 没有推理锁，可以并发 |

### 4.2 Level 2：引擎内部

| 能力 | 说明 | 三框架现状 | 结论 |
|---|---|---|---|
| L2-E1 同连接接收新观测而不阻塞 | 推理进行中读取下一条消息 | 三家都串行 | 需要改服务层 |
| L2-E2 真正中止在途推理 | 取消后 GPU 立即停止 | 上游都没有（LeRobot 只在客户端丢弃）；四个服务端实测均未中止（第 2.6.4 节） | 需要改引擎（当前推理不会被中断） |
| L2-E3 guided RTC | 去噪步可求导，把前缀拉向已执行动作 | LeRobot 有（pi05 默认 guided）；vLLM-Omni 拒绝；M\* 与 SGLang core 的图里未找到，且图需要重新捕获 | 引擎级改动；不需要重训 |
| L2-E4 按路增量输入 | 推理途中并入新帧或新 state | Pi0.5 全注意力，state 在 prompt 中，任何变化都使前缀失效 | 对 Pi0.5 收益很小，暂缓 |
| L2-E5 按路编码器缓存 | 相机没变时复用 SigLIP 输出 | 上游未见；vLLM-Omni 与 M\* 的图里 SigLIP 只有 bs=1 图 | 只可能省视觉部分，需实测（推断） |
| L2-E6 动作块内部流式输出 | 去噪中间结果或部分动作先行返回 | 三家都整块回复；OpenPI 也不流式（`openpi_api.md:111`） | 需要协议改动 |
| L2-E7 跨请求模型状态 | 上一次推理的状态影响下一次 | Pi0 有连续 state token，Pi0.5 没有；三家都不保留跨请求 KV | 结构上不成立，不需要测 |

**Level 2 的最小可验证项**：L2-E1 和 L2-E2。两者都可以在 5880 上用双连接实验测。

## 5. 对另一位 AI 分析的修订意见

| 观点 | 修订后的判断 | 依据 |
|---|---|---|
| Level 1 的主要工作在框架外 | 同意。observation 的离散化和相机映射在服务端，框架要接收原始 observation | §3.1，§3.2 |
| 框架能接收完整 Pi0.5 输入时可以不改框架 | 同意，前提是观测键名、相机数、checkpoint 对齐。三家的键名和 checkpoint 不同（§1 最后一行） | §1，§2 |
| Level 2 可在 Runtime 中做抢占和取消 | 部分同意。取消只能在客户端丢弃结果，四个服务端实测都会算完当前推理（§2.6.4）。抢占的延迟上限是当前推理的剩余时间 | §0 第 6 条，§4.2 |
| 增量进入模型需要改引擎 | 同意，对 Pi0.5 尤其如此。收益小于改动成本 | §3.1 全注意力，state 在 prompt |
| RTC 需要模型侧支持 | **修正**：guided RTC 不需要重训，但需要去噪步可求导，是引擎级改动 | §0 第 7 条，§3.2 |
| M\* 的 `StreamingGraphEdge` 与请求内的输入流无关 | 同意，是模型图内部的分区边（`mstar/README.md:158–163`；`qwen3_omni_model.py:540–696`） | §0 |
| LeRobot 的 `async_inference`、`robot_client.py`、`policy_server.py` | **修正**：在 v0.7.0 中不存在，已由 `remote_inference/` 取代 | §3.2 |
| LeRobot 的 RTC 与 `ActionQueue` | 存在，且比原文描述更具体（软 inpainting，LINEAR 默认） | §3.2 |
| ROS 2 `message_filters` 用于多路对齐 | 部分同意。有近似和精确对齐，没有周期触发，也没有 snapshot-at-tick。快照要自己写 | §3.3 |
| Level 1 每 100 ms 调用一次 | **修正**：不用固定 100 ms。5880 上 M\* 10 步的 p50 约 100 ms，需要测分布的高分位数 | §2.1 |
| 推理结果在模型执行中可以取消 | 上游都不支持，不能假设 | §0 第 6 条 |

## 6. 落地计划（修订）

| 步骤 | 内容 | 是否改框架 | 状态 |
|---|---|---|---|
| S0 | 确定 checkpoint。建议 `lerobot/pi05_base`（三家的实验都用它，3 相机，50×32）。确认 M\* 的 OpenPI WebSocket 能否为它配置 `openpi:` 块（需要 3 个相机键、32 维 state）。M\* 的 `pi05_droid.yaml` 只适用于 OpenPI 的 `pi05_droid` checkpoint | 否 | 已做（三家都加载 `lerobot/pi05_base`；M\* 的 `openpi:` 块已验证） |
| S1 | 统一的 Runtime harness，客户端讲 OpenPI WebSocket。快照用"每路 ≤ t 最新一条 + 年龄"。触发同时支持固定周期和队列阈值。动作队列带代数号。三家按键名映射接入；动作块形状从服务端读取，不照抄 OpenPI 示例的断言 | 否 | 已做（2026-11-20）：`src/runtime/`（观测快照、触发、动作管理、控制时钟）、`src/backends/`（三个框架的适配器）、`src/scenarios/level1_periodic.py`、`src/metrics/`；测试在 `tests/`；说明见 [README](../README.md)。基础模型 `lerobot/pi05_droid`（15 步），数据集 `lerobot/droid_100` 的多频回放 |
| S2 | Level 1 测量：三家走同一 WebSocket 协议，同一 checkpoint，单连接，扫周期。记录 E2E 分布、输入年龄、动作新鲜度、欠载次数。对照现有 HTTP+PNG 的 M\* 结果（§2.1）以量化传输差异 | 否 | 已做（2026-11-21，§2.7）：`pi05_droid`、droid_100 多频回放、七种触发策略、3 轮交错。传输差异已在 §2.6.1 量化；真机和 ROS 2 输入未测 |
| S3 | Level 1 并发：N 个客户端连到同一服务，测排队时间。现有 `exps/` 没有这一项（`pi05-conc` 尚未实现） | 否 | 已做（§2.6.5，N=1、2、4；用 `probes.py` 的 concurrency 模式） |
| S4 | L2-E1 和 L2-E2 双连接实验：第二连接在第一连接推理途中提交请求，测其等待时间是否等于第一条的剩余时间；测断开后 GPU 是否继续计算（看利用率或引擎日志） | 否 | 已做（§2.6.4，四个服务端；GPU 利用率未直接测） |
| S5 | 只有 S4 证明 Runtime 无法满足需求，或确有 RTC 需求时，才评估引擎改动（guided RTC 要求去噪步可求导；M\* 与 SGLang core 需要重新捕获图） | 是 | 暂缓 |

在 5880 上运行前，要确认显存余量和 GPU 占用。当前的 5880 基线全部是 clean 0%（时钟事件），比值只作相对参考。

## 7. 核对状态

**已核对（源码，子代理 + 主会话抽查）**：日志 [2026-11-20_verify-unverified-upstream.md](../agents_results/2026-11-20_verify-unverified-upstream.md)。

- LeRobot 历史与版本沿革（第 3.2 节）；`ActionQueue` 的锁与回调（第 3.2 节）。
- `websockets` 版本固定（第 3.1 节）。
- DROID 示例对 `pi05_droid` 的形状断言会失败（第 3.1 节，代码核对，未运行）。
- `ActionChunkBroker` 的重新请求条件；DROID 控制循环是单线程的（第 3.1 节）。
- `pi05_base` 没有归一化统计（`policy_preprocessor.json` 中 `features` 为空），每个相机 256 个图像 token（patch 14，224 像素）。
- state 分箱的边界行为（第 3.2 节，numpy 复现）。
- vLLM-Omni 的 Pi0.5 只读取 `robot_obs`，`session_id` 和 `reset` 对 Pi0.5 不产生状态（`pipeline_pi05.py:269–270`，`serving.py:169–170`）。

**未核对，原因说明**：

- ROS 2 `message_filters` 的运行时行为：本机没有 ROS 2，无法编译或运行（第 3.3 节）。
- DROID 机器人在 infer 阻塞期间是否保持设定点：DROID 的机器人层（`franka/robot.py`）没有读取，且不在本仓库（第 3.1 节）。

**实验核对（见第 2.6 节）**：

- 已完成：M\* 的 OpenPI WebSocket 能用 `pi05_base` 运行（最新版 2c8ffde8，配置见 `configs/mstar_pi05_openpi.yaml`），并与 HTTP 路径在同一服务中对照（第 2.6.1 节）。`../Inference_Engine_Investigation_Repo/external/mstar`（9c6f1ad，活动版本）没有 OpenPI socket。
- 已完成：四个服务端（含 M\* 默认 Rust 前端）的同协议端到端对比、周期上限、同连接排队、断开是否中止在途推理（修正版探测）、多连接并发（第 2.6 节）。

**仍未验证**：

- Level 1 的输入频率、延迟和抖动是模拟参数，没有对照真机；vLLM-Omni 重尾中主机负载的份额没有分开（第 2.7 节）。
- SigLIP 复用的收益：从 M\* 既有 nsys 数据估计，ViT 约占单请求 E2E 的 8%（插桩数据，只是上界，且只在相机图像完全不变时成立）。
- SGLang core 的 WebSocket 路径：已在本地复现（第 2.6、2.7 节）。SGLang 的"新长度走 eager"由日志明文和延迟差得到，eager 路径本身没有单独计时。
- report v2 第 09b 章的 H200 页曾把 SGLang core 写成 SGLang-Omni，已修正（见第 0 节之后的修订记录）。

## 8. 来源索引

- 实测：[exps/results/baseline-pi05_b2b_rtx5880.md](../exps/results/baseline-pi05_b2b_rtx5880.md)，[pi05-steps_b2b_rtx5880.md](../exps/results/pi05-steps_b2b_rtx5880.md)，[pi05-kv_b2b_rtx5880.md](../exps/results/pi05-kv_b2b_rtx5880.md)，[pi05-graph-split_b2b_rtx5880.md](../exps/results/pi05-graph-split_b2b_rtx5880.md)，[exps/tasks/pi05/transport.py](../exps/tasks/pi05/transport.py)
- 报告：[report v2 第 09a 章（机制）](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters/09a_pi05_mech.tex)，[09b 章（结果）](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters/09b_pi05_results.tex)，[09c 章（因素）](../../Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2/chapters/09c_pi05_factors.tex)
- 探测（2026-11-20，§2.6）：[src/scenarios/protocol_probes.py](../src/scenarios/protocol_probes.py)（客户端，五种模式），[scripts/run_probes.py](../scripts/run_probes.py)（起服务、跑探测、记录 GPU 快照），[configs/mstar_pi05_openpi.yaml](../configs/mstar_pi05_openpi.yaml)（M\* 最新版配置），结果在 [runs/2026-11-20_ws-probe/](../runs/2026-11-20_ws-probe)（每框架的 JSON、服务日志、GPU 快照；`mstar-rust/` 为默认前端），取消探测修正版在 [runs/2026-11-20_ws-probe-cancel-v2/](../runs/2026-11-20_ws-probe-cancel-v2)
- 子代理日志（本次）：[openpi-remote-pi05](../agents_results/2026-11-20_openpi-remote-pi05.md)，[lerobot-async-rtc](../agents_results/2026-11-20_lerobot-async-rtc.md)，[ros2-message-filters](../agents_results/2026-11-20_ros2-message-filters.md)
- 子代理日志（之前，位于基准仓库 `agents_results/`）：[pi05-third-case-plan](../../Inference_Engine_Investigation_Repo/agents_results/2026-11-14_pi05-third-case-plan.md)，[pi05-harness-implementation](../../Inference_Engine_Investigation_Repo/agents_results/2026-11-14_pi05-harness-implementation.md)，[realtime-mstar](../../Inference_Engine_Investigation_Repo/agents_results/2026-11-20_realtime-mstar.md)，[realtime-vllm-omni](../../Inference_Engine_Investigation_Repo/agents_results/2026-11-20_realtime-vllm-omni.md)
- 基准仓库（Inference_Engine_Investigation_Repo）的其他文档：[Inference_Engine_Investigation_Repo/share/pi0.5/README.md](../../Inference_Engine_Investigation_Repo/share/pi0.5/README.md)，[Inference_Engine_Investigation_Repo/share/realtime_streaming_multimodal/README.md](../../Inference_Engine_Investigation_Repo/share/realtime_streaming_multimodal/README.md)
