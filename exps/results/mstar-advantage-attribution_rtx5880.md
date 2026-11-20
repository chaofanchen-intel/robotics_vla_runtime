# M* 的性能优势来自哪里：RTX 5880 上的归因结论（等工作量，back-to-back）

数据：`exps/runs/*_01_B2B`，本机 RTX 5880，M* `9c6f1ad`，vLLM-Omni v0.32，SGLang-Omni 为仓库内 `external/sglang-omni`。旧 H200 结论用 M* v0.2.1 + vLLM-Omni v0.30，硬件和版本是混在一起的，本文不把两者的差异归因到硬件。
nsys 只用来看结构（它会放大主机侧时间，M* 受影响最大），不当作 benchmark 数字。证据强度分三档：**反事实**（关掉一个开关，数字按预测动了）、**结构**（nsys / 日志 / 源码一致指向，没有开关）、**未解释**。

> **公平性提示（先读）**：本文 §1–§8 的多数数字是在**各框架默认配置**下测的，其中缓存状态、SGLang 的 lookahead 门控、vLLM 的推理期 JIT、vLLM Talker 采样都不对等，逐项见 [fairness-audit_rtx5880.md](fairness-audit_rtx5880.md)。**公平档已跑（§9.5）：Qwen TXT / A2T 的基线领先在公平档下消失或缩小；T2S 的领先保留。** §2.4、§9 里引用基线数字的段落按 §9.5 读。

## 0. 一句话

在我们能做到等工作量的地方，**M* 的 kernel 没有更快**，也没有哪一个"只有 Graph + Walk 才做得到"的优势被证实。已经测到的领先，逐项拆开后是：

| 领先 | 来源 | 结论 | 证据 |
|---|---|---|---|
| BAGEL I2T 对 vLLM two-stage，TTFT 138 对 805 ms | prefix cache | **不是引擎能力，是 vLLM-Omni 上游配置默认关了 prefix caching**。vLLM 打开后 TTFT 86 ms，比 M* 的 135 ms 还低 | 反事实（两个方向都做了） |
| BAGEL I2I 对 vLLM two-stage（默认配置），5 步 1.48×（6,809 对 2,761 ms；50 步 1.19×，1024² 1.36×） | **都是 vLLM-Omni 部署拓扑的代价，三块**：默认 connector 56%、Thinker 不开 prefix cache 26%、剩余 19% | 5 步 4,048 ms 的差距 = **默认 SharedMemory connector 2,251 ms（换 Mooncake-TCP）+ Thinker 对两个 5k-token 分支各做一次 prefill、不开缓存 1,041 ms（开 prefix cache）+ 剩余 763 ms**（stage 间 hand-off 约 0.28 s、前端、其余未拆）。两者都配好之后 vLLM two-stage 3,523 ms，对 M* 2,760 ms 为 1.28×（5 步）/ 1.06×（50 步）；对 vLLM single-stage 为 1.02×。**M\* 自己的缓存开关在 I2I 上无影响（2,760 对 2,764 ms）**。DiT 循环三家相同（GPU busy 1.78 s，10,239 launch，每个 connector 一样）。输出图与默认 connector 逐字节相同 | 反事实（connector、缓存两个开关各自按预测动了）+ 结构（nsys：Thinker 两次 prefill 15.3k launch / 2.05 s busy，对 single-stage 6.4k / 0.85 s；Mooncake 的 KV 拷出到 DiT 第一次 H2D 仅 133–150 ms）。§7.7 |
| Cosmos3 256² 对 vLLM，2.46× | 实现选择 | 批量 CFG 1.54× × torch.compile 1.38× × vLLM 每个 kernel 之间的主机发射空闲 1.17× | 反事实 + 结构 |
| Qwen TXT / A2T 对 SGLang-Omni，基线 1.32× / 1.23× | 基线请求带 `seed` / `min_tokens`，把 SGLang 自己的一步 lookahead 解码关了 | **公平档下领先消失 / 缩小**：请求改成 lookahead 可用后，TXT（关缓存对关缓存）SGLang 1,596 对 M* 1,648 ms，**M* 慢 3.3%**；A2T SGLang 278 对 M* 255 ms，M* 领先 1.09×（全在解码：6.60 对 5.81 ms/token）。强制关 lookahead 的对照：TXT ITL 5.82 → 8.00 ms，E2E 2,135 ms，正好是基线里的 1.30× | **反事实**（§9.5） |
| Qwen T2S 对 SGLang-Omni / vLLM-Omni | M* 自己的发射链 1.24× + vLLM Talker 的额外开销 | **领先保留**：20 句预热后每音频秒 M* 96.9 / SGLang 120.2 / vLLM 152.8 ms，即 **1.24× / 1.58×**；vLLM 的 stall 在预热后为 0，Talker 改 greedy 也不变（160 对 153），所以不是 JIT 也不是采样器 | 反事实（M* 一侧的链）+ 结构（vLLM Talker，§9.5） |
| Qwen TXT / A2T 对 vLLM | 无 | **公平档下没有领先**：TXT M* 1,648 对 vLLM 1,610 ms（M* 慢 2.4%，TTFT 149 对 98 ms，解码一样）；A2T 255 对 251 ms（M* 慢 1.6%）。基线里 A2T 的 1.08× 是 vLLM 对新输入 shape 的推理期 JIT（预热同一批片段后消失） | **反事实**（§9.5） |
| **M\* 落后的地方：Qwen TXT 的 1,072-token 提示（TTFT 138–146 对 99 / 117 ms）** | **差距是真的；机制已定位：默认配置把 1,072 切成 1,024 + 48 两次 forward（多约 38 ms）；预算 4,096 时这一步被垫到 2,048 的图（本机只捕获了一张 2,048 图，bs = 32，bs = 1 的步也用它），多出的 token 数的 kernel 工作量翻倍；加一档 1,152 的图后一次 1,152 的图重放，TTFT 145 → 113 ms，去掉约 2/3 的差距；剩余约 15 ms 是 M\* 在 ≥1,024 token 的 kernel 级慢 10–15%** | nsys：M\* 一个请求 3 次 forward（1,024-token 块约 77 ms、48-token 块约 40 ms、第二个输出 token 约 6 ms），vLLM 1 次 77 ms。据此预测「预算改 4,096、提示变成一块，TTFT 降约 40 ms」，**`prefill-chunk` 的结果相反：TTFT 138 → 154 ms（慢 16 ms）**，`txt-tiny-1tok` 52.4 → 53.2 ms 不变（1 个 start，drift flat）。去掉第 2 块没有省时间，所以「第 2 块的 40 ms 就是差距」不成立。**第五轮（`prefill-len`，5 个设置 × 5 个长度）**：M\* 默认 TTFT 1,024 / 1,072 / 2,048 = 106.5 / 144.9 / 190.9 ms；预算 4,096 的 1,072 = 155.8 ms（更慢）；**预算 4,096 + 一档 1,152 的图 = 112.6 ms（−43 ms）**，vLLM 98.3 ms。`server.log`：非 nsys 的默认 / 预算 4,096 服务器只捕获了一张 2,048 的 prefill 图（bs = 32），源码里 bs = 1 的步选「bs ≥ 1、token ≥ 所需的最小已捕获桶」，所以 1,072 在这张图里重放，即**垫到 2,048（上一版的假设成立）**；nsys 的服务器因为 nsys 自己占显存，一张 2,048 图也没捕获，1,072 按实际大小 eager，所以 nsys 臂的 E2E（123 ms）比非 nsys（163–168 ms）快：**nsys 臂测的不是生产路径**（§9.6（1）） | **测量（5 臂 × 5 长度，1 个 start）+ 源码 + `server.log` 的捕获清单** |

**Graph + Walk 架构本身**：没有任何实验给出直接证据（没有开关能拆掉图抽象）。Walk 专属部分的上限是 W 臂相对 S 臂的 2–5%。最接近"架构"的证据是 BAGEL I2I 的 stage 边界：M* 在同一个引擎里传 GPU 张量，vLLM-Omni 的 two-stage 要过进程边界；但 vLLM single-stage 同样没有这个边界，所以它是"部署拓扑的代价"，不是 M* 独有的能力。

## 0.1 M* 的优势与劣势全景（公平档，本机 RTX 5880；每行的证据强度见最后一列）

**M\* 比对手快的地方（公平档之后还剩下的）**

| # | 场景 | 大小 | 来源 | 是不是 M\* 独有的能力 | 证据 |
|---|---|---|---|---|---|
| A1 | Qwen T2S 的 Talker / Code2Wav 帧循环，对 SGLang-Omni | 每音频秒 1.24× | M\* 自己的 CPU 侧发射链（关链后 M* 比 SGLang 慢 5%） | **不是**：vLLM 的异步调度、SGLang 的 lookahead 是同一种重叠，M* 的链只是在 T2S 这一路上实现得更紧 | 反事实（M* 一侧） |
| A2 | Qwen T2S，对 vLLM-Omni | 每音频秒 1.58× = 链 1.24× × 另外 1.27× | 另外的 1.27× 在 vLLM 帧循环（首个音频后每音频秒 100 对 54 ms）；JIT、Talker 采样器都已排除。**预热行的 nsys（`qwen-t2s-nsys-warm`）：GPU 2 上 vLLM 每音频秒发 15.4k 个 kernel，M\* 8.2k（1.9×）；GPU busy 60.6 对 47.0 ms（1.29×）；Talker span 221 对 130 ms（1.7×）；两边 busy / span 都只有约 27%——帧循环是 launch 受限的，时间随 kernel 个数走** | **不是** M\* 独有：vLLM 的 Talker 每帧发的 kernel 更多（没有融合 / 图），机制在 kernel 个数上，没有开关验证 | 结构（nsys，§9.6） |
| A3 | BAGEL I2I，对 vLLM-Omni **two-stage**（两个开关都配好后） | 5 步 1.28×，50 步 1.06×；对默认配置 1.48× / 1.19× | **部署拓扑**：Thinker→DiT 之间的 hand-off 约 0.28 s、前端、以及上面 763 ms 的其余部分；默认配置多出的 connector 2,251 ms 与不开缓存的 1,041 ms 是 vLLM-Omni 的默认值 | **不是**：同样的模型放进 vLLM **single-stage** 只慢 2% | 反事实 + 结构（§7.7） |
| A4 | Cosmos3 256²，对 vLLM-Omni | 2.46× | 批量 CFG 1.54× × compile 1.38× × vLLM 的 kernel 间主机空闲 1.17× | **不是**：都是实现选择，vLLM 可以做同样的事 | 反事实 + 结构 |
| A5 | Qwen A2T，对 SGLang-Omni | 1.09×（解码 5.81 对 6.60 ms/token） | 没查：SGLang 的 TXT 解码是 5.82，A2T 却是 6.60 | 未知 | 测量 |
| A6 | BAGEL I2T 解码 | 每 token 快 1.1%（16.91 对 17.10 ms） | 无意义的小差 | 否 | 测量 |

**M\* 比对手慢的地方**

| # | 场景 | 大小 | 来源 | 证据 |
|---|---|---|---|---|
| D1 | **Qwen TXT 的 1,072-token 提示**：TTFT 138–146 对 vLLM 99 / SGLang 117 ms | 差距 40–47 ms 仍在；**不是单独的 chunk 预算**（预算 4,096 后 TTFT 154 ms，更慢） | **机制（第五轮）**：默认的 1,024 + 48 是两次 forward（多一次约 38 ms 的地板）；预算 4,096 时 1,072 是一步，但本机只捕获了一张 2,048 的 prefill 图（bs = 32），这一步在它里面重放，**被垫到 2,048**（155.8 ms）；**加一档 1,152 的图后一次 1,152 的图重放 112.6 ms**。1,072 处 47 ms 的差距（M\* 145 对 vLLM 98）里约 32 ms（68%）是这个配置问题，约 15 ms 是 M\* 在 ≥1,024 token 的残留慢（1,024：107 对 96；512：71–79 对 72 持平；56 token：52 对 59，M\* 快）。1,072 恰好是 1,024 的预算和桶上方 48 token（正文 1,024 + 模板 48） | **测量**（`prefill-chunk`、`qwen-prefill-nsys`、`prefill-len` 5 臂 × 5 长度）+ 源码 + `server.log`；`prefill-chunk-nsys` 的预算 4,096 臂因为 nsys 占显存、没捕获 2,048 图，测的是 eager 路径，不是生产路径（§9.6（1）） |
| D2 | BAGEL I2T 的 TTFT | 两边缓存都开：M* 134 对 vLLM 85 ms（慢约 44–49 ms）；被每 token 快 1.1% 抵消，E2E vLLM 快 1% | **chunk 预算只占一小部分**：`bagel-chunk` 把 LLM 预算 1,024 → 8,192，I2T TTFT 139.6 → 132.2（r01）、130.0 → 123.8（r02），**约 −6 到 −7 ms（~5%）**；E2E 2,288 → 2,282（1.00×），5 步 I2I 2,739 → 2,734（1.00×）。剩下约 40 ms 不在 chunk 预算。稠密 7B 的 1,024-token 块是 compute 受限，每块额外成本小，与此一致 | **测量**（反事实，2 starts × 5） |
| D3 | Qwen A2T 的首 token | TTFT M* 93、vLLM 79 ms（E2E 255 对 251） | **不是 chunk 预算**：音频提示约 190 token，一块就够；剩下的 14 ms 里 nsys 看得到的只有 M\* 的第 1 块 forward 前有约 8.6 ms 的 kernel（TXT 的 nsys，原因没查）和单块 MoE / all-reduce 慢 5–10%；太小，不再查 | 测量 + 结构 |
| D4 | **TP=2 的 Thinker prefix cache 开了会崩** | 第一次缓存命中（第二个相同提示）触发 `TP ranks diverged: ... rank 0 leased 8 cached pages for 0/main, whose stream here already holds tokens or has no keys`；默认关，所以 M* 没有热前缀的 Qwen 数字 | **不是配置错**：`prefix_cache_tp: true` + `cpu_offload_pages: 0` 是官方文档写明的 opt-in，M\* 自己的加载期检查全部通过，两个 worker 都打出 `prefix cache open for main on Thinker`；崩的是 follower rank 的 `KVManager.adopt_cached_prefix` 里的 `RuntimeError`（`manager.py:1022`）。M\* 自带的 TP 前缀缓存单测（`test_kv_prefix_tp.py`，14 个，本机 CPU 通过）用桩，不走 Qwen3-Omni 的真实路径；官方的 TP 配置（`configs/qwen3omni_full_tp2.yaml`、`qwen3omni_thinker_tp2.yaml`）都不开这个选项。**这是 M\* 在 opt-in 路径上的缺陷**；触发条件（8 页 = 1,024 token = 恰好第一个 chunk）指向分块行。**`tp-cache-probe` 已跑**：预算改 4,096（1,072-token 提示一个 chunk）后，TP=2 + 前缀缓存 13 个请求全部成功，server.log 里 0 条 `TP ranks diverged`，所以**崩溃与「提示被切成多个 chunk」有关**，整段一个 chunk 就绕得过；缓存是否真的命中没有直接日志，只有 TTFT 间接证据（见 §9.6（4）） | 日志，两个 repeat 一致；源码核对（§9.6） |
| D5 | `max_output_tokens=1` 返回 2 个 token | 影响 1-token 用例的 E2E（多一步解码），256 的上限是精确的 | M* 的行为 | 日志 |

**没有证据 / 排除**：Graph + Walk 本身（没有开关能拆）；kernel 更快（BAGEL 每步 GPU busy 三家相同，Qwen 解码三家 5.8–6.0 ms/token）；torch.compile（M* 在 sm_89 能开，TXT / A2T 慢 3–4%，T2S 持平）；注意力后端；硬件时钟（M* 在更低的时钟下仍持平）。

**一句话**：M* 在本机剩下的领先主要是 **T2S 流式帧循环的发射链**（vLLM 在同一路上每音频秒多发 1.9× 的 kernel），其次是 **BAGEL I2I 避开了 two-stage 进程边界**（一半多是 vLLM-Omni 的默认值）；对 vLLM single-stage 和 SGLang（lookahead 可用）几乎没有领先。**M\* 落后的 Qwen TXT TTFT（1,072-token 提示，慢 40–47 ms）已定位：默认配置下它是 1,024 + 48 两次 forward，预算 4,096 虽然只剩一次 forward，但这一步被垫到本机唯一的 2,048 图里（更慢）；加一档 1,152 的图后 TTFT 145 → 113 ms（vLLM 98），约 2/3 的差距是这个配置问题，剩下约 15 ms 是 M\* 在 ≥1,024 token 的 kernel 级慢 10–15%；BAGEL 的 chunk 预算只占 I2T TTFT 差距的约 15%；`prefill-chunk-nsys` 的预算 4,096 臂因为 nsys 占显存、没捕获 2,048 图，测的是 eager 路径，不能解释生产的 156 ms**；M\* 在 TP 下的前缀缓存是 opt-in 路径上的缺陷，不是我们的配置问题。

## 1. 按 workload 的总表

| workload | 对手 | M* 领先 | 来源 | 证据 |
|---|---|---:|---|---|
| BAGEL T2I（nocfg / cfg3） | vLLM 两种部署 | 1.00–1.03× | 无 | 反事实+结构：每步 busy 相同 |
| BAGEL I2T | vLLM two-stage（默认配置） | 1.30× | prefix cache 639 ms + 1.5% | **反事实** |
| BAGEL I2T | vLLM two-stage（Thinker 开 prefix cache） | **0.99×（vLLM 略快）** | 无 | **反事实** |
| BAGEL I2T | vLLM single-stage | 2.6× | vLLM single-stage 的串行文本路径（47 ms/token，B=16 线性放大） | 结构，机制未验 |
| BAGEL I2I | vLLM single-stage | 1.02× | 未解释（小） | - |
| BAGEL I2I | vLLM two-stage（默认 SharedMemory） | 1.48×（512，5 步）/ 1.19×（50 步）；1024² 的旧基线为 1.36× | connector 56% + Thinker 不开缓存 26% + 剩余 19% | 结构 + 拟合 + 反事实（§7、§7.7） |
| BAGEL I2I | vLLM two-stage（Mooncake-TCP） | 1.10×（512，50 步）；5 步 1.65× | Thinker 不开缓存占固定差的 1,041 ms，剩余 763 ms | **反事实**（§7.4、§7.7） |
| BAGEL I2I | vLLM two-stage（Mooncake-TCP + Thinker prefix cache，两个开关都配好） | **1.28×（5 步）/ 1.06×（50 步）** | 只剩 stage hand-off 约 0.28 s、前端和其余 763 ms | **反事实**（§7.7） |
| BAGEL I2I | vLLM two-stage（NIXL-UCX） | 1.13×（512，50 步） | 同 Mooncake，且接收侧比 Mooncake 慢约 0.8–1.2 s（UCX 传输路径未查） | **反事实**（§7.4、§7.6） |
| Qwen TXT | vLLM（关缓存） | **0.98×（M* 慢 2.4%）** | 无；M* TTFT 慢 51 ms，解码相同 | 反事实（公平档，§9.5） |
| Qwen TXT / A2T | SGLang-Omni（lookahead 可用） | **0.97× / 1.09×** | TXT 无；A2T 是解码 6.60 对 5.81 ms/token | **反事实**（§9.5） |
| Qwen A2T | vLLM（预热同批片段） | **0.98×** | 无；基线 1.08× 是 vLLM 对新 shape 的 JIT | **反事实**（§9.5） |
| Qwen T2S | SGLang-Omni | **1.24×**（每音频秒） | M* 自己的链 1.24×；M* 关链后比 SGLang 慢 5% | 反事实（M* 一侧） |
| Qwen T2S | vLLM | **1.58×**（每音频秒） | 链 1.24×，剩 1.27× 在 vLLM 的 Talker / Code2Wav 帧循环（首个音频之后每音频秒 100 对 54 ms）；JIT 与 Talker 采样都已排除 | 结构（nsys launch 3×），机制未验 |
| Cosmos3 256² | vLLM | 2.46× | 批量 CFG 1.54× × compile 1.38× × 主机空闲 1.17× | 反事实 + 结构 |

被排除的假设：flash-attn / 注意力后端（拦掉后 M* 不变，vLLM 换 SDPA 也不变）、「M* launch 数少」（不转化成速度）、CUDA graph（Cosmos3 256² 不在捕获档位，手动加上也没变）。

## 2. "发射链"到底指什么

### 2.1 一句话

自回归解码每一步的 GPU 工作很短（Qwen3-Omni 的 Thinker 约 6 ms），而主机每步要做：调度（谁进下一批）、准备输入、为注意力 kernel 做 plan、提交 kernel、读回采样的 token、检查 EOS、把结果发出去。如果这些都等上一步的 GPU 结果出来再做，GPU 每步都要等主机，步时间 = GPU 时间 + 主机时间。**发射链的做法是：不等第 N 步的结果，就先推测第 N+1 步的批组成、规划并把它的 kernel 提交给 GPU，让主机的工作和 GPU 的第 N 步重叠。** 采样出的 token 一直留在 GPU 上，直接作为下一步的输入，主机只在"事后"才读它。

```
无重叠（M* 链关）:  主机 [调度 N][提交 N]          [读 N,查EOS][调度 N+1][提交 N+1]       [读 N+1 ...
                    GPU               [  forward N  ]                                   [ forward N+1 ]
                                      <---- GPU 在两个 forward 之间空等主机 ---->

有重叠（链开）:     主机 [提交 N][推测+规划 N+1][提交 N+1][读 N,查EOS][推测+规划 N+2][提交 N+2][读 N+1 ...
                    GPU       [  forward N  ][  forward N+1  ][  forward N+2  ]       （kernel 首尾相接）
```

代价：EOS 要晚一步发现，多跑的那一步被丢弃（M* 文档里叫 wasted last step；AR 解码本来就不知道自己的长度，所以吸收得掉）。

### 2.2 M* 的具体实现（`external/mstar/mstar/worker/worker.py`，说明见 `.claude/skills/async-worker/SKILL.md`）

- **三个线程**：主线程（收消息、调度、推测、后处理）、`gpu_executor`（声明→准入→规划→forward 提交→commit）、plan 线程（在自己的 CUDA stream 上为 N+1 预规划 FlashInfer，只等 N 的 commit 事件）。
- **一轮迭代**（batch N 正在 GPU 上跑）：
  1. 主线程在 GPU(N) 运行期间收消息、读 tensor、清理（全程不做 `torch.cuda.synchronize`，否则就把重叠杀掉了）；
  2. **推测 N+1 的批组成**（`_try_speculate_next`）：假设 N 在 M* 图里声明的所有输出都会产生，算出下一个就绪的节点和请求，并把能一起跑的请求并进来；
  3. **预规划**（`_arm_speculation`）：把 N+1 交给 plan 线程，它在主线程 `await_gpu`（释放 GIL）的同时为 N+1 做 FlashInfer plan，用双缓冲的 wrapper，plan(N+1) 和 replay(N) 落在不相交的 slot；
  4. 等 GPU(N) 的 **CPU 侧**完成（kernel 都已提交，GPU 还在跑）；
  5. 对账：把 N 的输出接到推测的批里，失败则回滚；
  6. **提交 N+1**；
  7. 此时才等 GPU(N) 真正完成，D2H、`check_stop`、发出 N 的结果。
- **device loopback**：采样 token 这类"回环输入"留在 GPU 上直接喂下一步，主机拷到 CPU 的那份只用于 EOS 检查和下发。
- **链长**：`MSTAR_MAX_CONSECUTIVE_SPEC_STEPS`（默认 1024）连续推测多少步后让出一次，让别的 (node, walk) 有机会。默认下 Thinker 的解码一直在链上。
- **本文用到的开关**：
  - **S 臂** `MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=0`：每步都让出，不推测下一步（链关，其余不变）。
  - **W 臂** = S 臂 + `MSTAR_PRE_PLAN_SPEC=0`（plan 回到 GPU 线程内联）+ `MSTAR_ENGINE_STEP_SYNC=1`（每步 forward 后同步，把 2 步的 launch 上限收到 1 步）。W 比 S 多慢 2–5%，是"读 Walk 前瞻"能贡献的上限。
- **和 Graph + Walk 的关系**：推测"N+1 是什么"依赖 M* 图里声明的输出和 Walk 的前瞻，所以实现上确实用到了图抽象；但它**产生的效果**（launch-ahead + 设备上回环 + 事后读结果）不依赖图，vLLM 和 SGLang-Omni 都有各自的版本（2.3、2.4）。

### 2.3 vLLM-Omni 有没有

有。vLLM v1 的 `AsyncScheduler`（vLLM-Omni 里是 `OmniARAsyncScheduler`，`async_scheduling: true` 时用）：调度器在第 N 步还在 GPU 上跑时就调度第 N+1 步，每个请求带 `num_output_placeholders` 占位，model runner 把上一步在 GPU 上采样的 token 直接拷进下一步的 `input_ids`，第 N 步的输出晚一步才在主机上处理。M* 的链和它是同一类机制。

我们的数据（§5）：两边开着重叠时每 token 延迟相同（TXT 5.85 对 5.92 ms，差 ≤2%）；vLLM 关异步（换成 `OmniARScheduler`）后变慢 1.3 ms/token，M* 关链变慢 2.0 ms/token。**所以相对 vLLM，发射链不是优势，是追平**：M* 不带链时主机循环比 vLLM 的同步路径更重（慢 0.6–1.2 ms/token）。

### 2.4 SGLang-Omni 有没有：有，而且我们的请求把它关了

SGLang-Omni 自己的调度循环 `event_loop_async_decode`（`external/sglang-omni/sglang_omni/scheduling/omni_scheduler.py`）是"一步 lookahead"：每轮 **先启动当前步**（forward + GPU 上采样，把下一步要用的结果发布出去，不等 GPU），**再处理上一步的主机侧收尾**，主机收尾和当前步的 GPU forward 重叠。单 stream + CUDA event。Qwen3-Omni 的 Thinker 默认开（`qwen3_omni/config.py`：`FactoryArgs(enable_async_decode=True)`，可用 `--thinker.factory.enable_async_decode false` 关）。注意它**不是**上游 SGLang 的 overlap 调度（`OmniScheduler._event_loop_overlap` 在 Omni 里直接抛错，所以我之前写的"SGLang 的 Omni scheduler 拒绝 overlap 循环"只对了这一半，漏了它有自己的 lookahead）。

但进入 lookahead 要通过 `ThinkerModelRunner.lookahead_eligible`（`sglang_omni/model_runner/thinker_model_runner.py`）：批里只要有请求满足下列任意一条就退回同步解码，原因是 lookahead 用的是"晚一步"的 output 历史，对依赖历史的采样项会和同步结果分叉：

- 请求要音频输出（Talker 同机时 lookahead 会抢 Talker 的 GPU 时间，**SGLang 自己的设计**）；
- `repetition_penalty ≠ 1` / presence / frequency penalty；
- `min_new_tokens > 0`；
- **`sampling_seed is not None`**；
- logit_bias、custom_params、要 logprob。

我们的请求（`exps/tasks/qwen_txt/client.py`、`qwen_a2t/client.py`）：

| 任务 | seed | min_tokens | 结果 |
|---|---|---|---|
| Qwen TXT | 42 | 256（配 `ignore_eos`，为了把长度钉在 256） | 两条都命中 → SGLang Thinker **同步解码** |
| Qwen A2T | 42 | 无 | 命中 seed → **同步解码** |
| Qwen T2S | 有 | - | 音频输出 → 同步解码（SGLang 自己的设计） |

这是源码推断，不是运行时观测（SGLang 没有日志打印走了哪条循环），所以要做反事实。greedy（temperature 0）解码不需要 seed，`ignore_eos + max_tokens` 已经能把 TXT 钉在 256 token，seed、min_tokens 对"等工作量"不是必需的。这也和§5 的数据吻合：**SGLang 的每 token 延迟（TXT 8.10、A2T 7.84）恰好等于 M* 关链的值（7.86）**，像是同步循环。

**这意味着**：TXT / A2T 上"M* 对 SGLang 1.32× / 1.23× = 发射链"的结论，测的可能是"M* 的重叠 vs SGLang 被请求关掉的重叠"，**不能当作 M* 引擎优于 SGLang 引擎**。是否还有剩余差距，取决于 SGLang 的 lookahead 打开后每步主机开销有没有追上 M*。已加反事实实验（`qwen-fair-txt` / `qwen-fair-a2t`，命令见 §12）。其中和 lookahead 直接相关的 arm：

- `sglang-on`（缓存开）/ `sglang-off`（缓存关）：请求去掉 seed / min_tokens，lookahead 留在默认的开；
- `sglang-on-async-off`：同样的请求，但配置强制 `enable_async_decode: false`（对照：证明 eligible 臂的变化确实来自 lookahead）；
- `vllm-pinned` / `mstar-pinned` 对 `vllm-off` / `mstar-off`：证明去掉 seed 对它们没有影响（预期没有）。

预先写下的读法：
- `sglang-on` / `sglang-off` 的 ITL 接近 M*（≈5.9）：**M* 对 SGLang 的解码领先是请求参数造成的**，TXT / A2T 对 SGLang 的行要改写成"无领先"；
- 比 SGLang 的原请求（8.10）快，但仍明显慢于 M*：lookahead 起作用了，剩下的是每步主机开销，**是实现差异**（再用 nsys 看每步 GPU 空闲，才能说是不是 Walk / 图带来的）；
- 与原请求相同，`sglang-on-async-off` 也相同：lookahead 没有被触发（检查 manifest 的 `sampling` 与 `request_sampling_mode`，以及是否有别的门控条件），需要重新查源码；
- 比原请求还慢：不预期，说明这个配置下 lookahead 本身是负收益。

T2S 不在 `qwen-fair-*` 里（用 `qwen-t2s-fit`）：T2S 的 Thinker lookahead 是 SGLang 有意关闭的，Talker 侧的调度器另外 `disable_overlap_schedule`，M* 对 SGLang 的 1.18× 在 T2S 上仍然是"M* 有重叠 vs SGLang 没有"，但这是 SGLang 的取舍，不是请求造成的。Talker 侧是否有可开的 lookahead 我没有逐行验证，在 §10 列为开放。

### 2.5 三者对比

| | M* | vLLM-Omni | SGLang-Omni |
|---|---|---|---|
| 机制 | 主线程推测下一批 + plan 线程预规划 + GPU 线程提交，设备上回环 | `AsyncScheduler`：占位 token + 设备上回环 | `event_loop_async_decode`：先 launch 再 resolve 上一步，单 stream |
| 重叠深度 | 2 步 launch 上限，可连续 1024 步 | 1 步 | 1 步 |
| 注意力元数据 plan 与 GPU 重叠 | 是（独立 plan 线程，双缓冲） | 未逐行核对 | resolve 与 launch 重叠 |
| 默认 | 开 | 上游 `async_scheduling` 默认开（反事实用 `OmniARScheduler` 关掉） | Thinker 默认开 |
| 资格门控 | 无（AR 解码总可推测；节点可 `enable_async_scheduling=False`） | 有（如 structured output 要等 token），未枚举 | **seed、min_tokens、penalty、logprob、音频输出** |
| 我们的请求下是否生效 | 是 | 是（数据证实） | **TXT/A2T 推断为否（seed），T2S 为否（音频输出）** |

## 3. prefix cache：三个框架各自的情况

"prefix cache 是不是基础功能，vLLM-Omni 和 SGLang-Omni 没有吗"——**有**，不是 M* 独有。

| | 引擎能力 | 我们 benchmark 里的状态 |
|---|---|---|
| M* | KV `prefix_cache` 默认开（`mstar/engine/resources/kv/config.py`：`prefix_cache = True`）；`prefix_cache_tp` 默认关，所以 Qwen Thinker TP=2 这套配置里按默认值它不工作（未在日志核对） | BAGEL 默认开，请求参数 `prefix_cache=false` 可关 |
| vLLM-Omni | 引擎（vLLM v1）有 automatic prefix caching；BAGEL 的 Thinker 开了之后能命中图像输入，输出前 403 个字符与冷启动逐字相同 | **上游 `deploy/bagel.yaml` 和 `deploy/qwen3_omni_moe.yaml` 把所有 stage 的 `enable_prefix_caching` 都设成 false**，我们的配置沿用上游默认，所以 baseline 没有缓存 |
| SGLang-Omni | radix cache，Qwen3-Omni Thinker 默认开（server.log 有 `Unified Radix Cache`）；Talker 关（`disable_radix_cache: True`）；没有 BAGEL 模型 | Qwen Thinker 开着。TXT 用同一条 1k-token prompt，SGLang 的 TXT TTFT 可能已经享受命中（未单独验证） |

所以 BAGEL I2T 的"M* TTFT 领先"是**各框架默认配置不同**造成的，不是引擎能力差异。对 vLLM 公平的比较是 cache on 对 cache on。

### vLLM-Omni 打开 prefix caching 的反事实（`bagel-vllm-prefix-cache`，2 starts × 5 请求）

| arm | I2T E2E ms | TTFT ms | ITL ms/token | I2I E2E ms |
|---|---:|---:|---:|---:|
| vLLM two-stage，cache off（默认） | 2,979 | 806 | 17.17 | 28,075 |
| **vLLM two-stage，Thinker cache on** | **2,256** | **86** | 17.10 | 27,152 |
| M* cache on（默认） | 2,289 | 135 | 16.94 | 23,447 |
| M* cache off（`bagel-prefix-cache`） | 2,934 | 777 | 16.94 | 23,583 |

- cache on 后 vLLM 的 TTFT 从 806 降到 86 ms，**比 M* 的 135 ms 低 49 ms**，I2T E2E 比 M* 低 1.4%。I2T 对 vLLM two-stage 的领先**整个消失，方向还反过来**。
- 预注册的预测是"接近 M* 的 ~140 ms"（= 开关）。实际更低：M* 的命中路径里还有约 100 ms 不在 LLM prefill 上（图像预处理 / 编码或首 token 路径，未拆）。**这是 M* 的一个弱点，不是优势**。
- 输出核对：vLLM cache on 的冷启动请求与 cache off 的输出逐字相同；命中请求前 403 个字符与冷启动相同，之后在近似并列的 token 上分叉（两个 repeat 都在字符 403 处分叉，分叉后是语义相同的另一种措辞）。这是命中路径的 bf16 数值差，不是脏上下文；cache off 的两个 repeat 之间本身也不完全逐字一致。
- I2I 在 vLLM cache on 下快 0.9 s（Thinker 的 4,935-token prefill 被命中），但仍比 M* 慢 3.7 s，所以 I2I 的差距不是缓存（§7）。
- 公平性结论：BAGEL I2T 的 baseline 对比必须要么两边都开缓存，要么两边都关；原来"M* 1.30×"这个数字不能再作为 M* 的优势引用。

## 4. BAGEL T2I：等工作量下没有每步优势

512² 的每步 GPU busy（M* / vLLM，`bagel-nsys-step-kernels_rtx5880.md`）：

| | M* | vLLM-Omni |
|---|---:|---:|
| 无 CFG，ms/步 | 143.5 | 145.0 |
| 三分支，ms/步 | 361 | 365 |
| 最大 GEMM kernel，ms/步 | 94.58 | 94.61 |

- E2E 拟合（`bagel-steps_fit`）：M* 262.2 ms/步，single-stage 260.8，two-stage 268.5；固定成本差在 ±150 ms 以内。
- vLLM 多出的是 `mot_unified_gemm` 和更多 launch（1,416→2,239，M* 恒约 1,340），但步总时间没变，因为两边 GPU 都几乎满载。
- 旧 H200 的「M* 每步 12.35 vs 17.53 ms」在本机不复现。是硬件还是版本造成，需要 H200 上同版本重跑。

## 5. 解码循环：M* 链与 vLLM 异步调度的数据

`analyze_decode_itl.py` 的每 token 延迟（ITL，ms/token）。`*-async-sched` 是 2 starts × 5 请求；vLLM 的 Thinker 在 `async-off` 臂里确实换成了 `OmniARScheduler`（server.log 可见）。

| | M* 链 on | M* 链 off | vLLM 异步 on | vLLM 异步 off | SGLang-Omni（请求带 seed，推断为同步循环） |
|---|---:|---:|---:|---:|---:|
| Qwen TXT（256 token） | 5.85 | 7.86 | 5.92 | 7.25 | 8.10 |
| Qwen A2T（基线臂） | 5.81 | 7.86 | 5.90 | - | 7.84 |
| BAGEL I2T（128 token） | 16.94 | 19.42 | 17.16 | 18.24 | 不流式 |

- **两边开着重叠时每 token 一样**（差 ≤2%）。
- vLLM 关异步也变慢（+1.3 / +1.1 ms/token），M* 关链多花 2.0 / 2.5：重叠在两个框架里都有价值。
- 预注册的读法（vLLM 关异步 7.9–8.2）只对一半，实际 7.25。
- **SGLang 的列要加星号**：它等于 M* 关链的值，推断原因见 §2.4（请求关掉了它的 lookahead），不是"SGLang 没有这个机制"。
- Walk 专属部分：W 臂相对 S 臂只多 2–5%。
- 注意：这几组 vLLM Qwen 臂是 3 卡 Thinker TP=2，主机开销比单卡大，链的绝对大小不可外推到单卡。

## 6. BAGEL I2T 的 TTFT：prefix cache，两个方向都已反事实证实

`bagel-prefix-cache`（M* 请求带 `prefix_cache=false` 关缓存，只作用于 M*）：

| arm | I2T E2E ms | TTFT ms | ITL ms/token | I2I E2E ms |
|---|---:|---:|---:|---:|
| M* cache on（默认） | 2,293 | 138 | 16.93 | 23,528 |
| **M\* cache off** | **2,934** | **777** | 16.94 | 23,583 |
| vLLM-Omni two-stage（默认，cache off） | 2,979 | 805 | 17.15 | 28,230 |

- 两个方向的结论一致：缓存状态相同时，M* 与 vLLM 的 I2T 差 1.5%（都关）或 vLLM 快 1.4%（都开）。**I2T 的 1.30× 领先 = 配置默认值 639 ms + 1.5%**。
- 结构证据同向：vLLM 默认配置的第一步是 736 ms 的 4,935-token prefill，GPU 从第一个 kernel 起就是满的；M* 命中时整个 LLM prefill 只有约 28 ms 的 GPU 工作。每次请求发同一张图，M* 的 key 是图像内容摘要，不会串图。
- **I2I 不受缓存影响**（M* 23,528 对 23,583，0.2%）。

## 7. BAGEL I2I：对 two-stage 的 1.18× / 1.36×，已经定位

### 7.1 拟合：差距 90% 是每请求固定成本

`bagel-i2i-steps`，三种部署，E2E = a + b × steps（2 starts × 5 请求，R² = 1.000）：

| 部署 | 5 步 | 25 步 | 50 步 | a（固定，ms） | b（ms/步） |
|---|---:|---:|---:|---:|---:|
| M* | 2,761 | 11,784 | 23,643 | 344 | 464.5 |
| vLLM single-stage | 2,909 | 12,030 | 23,926 | 491 | 467.4 |
| vLLM two-stage | 6,865 | 16,059 | 28,161 | **4,397** | 473.7 |

- two-stage 对 M* 的 4,515 ms 差距 = **固定 4,053 ms（90%）+ 每步 462 ms（10%，b 只差 2%）**。single-stage 的差距 295 ms 是固定 147 + 每步 148，各一半。
- 每步成本三家相同（b 在 2% 内），**DiT 循环本身没有差距**。（表中 M* 的 5 / 25 / 50 步取自"服务端步数"的 `srv-steps` 档；M* 的请求级步数档不改变步数，这是 M* 的步数属于服务端配置，已知。）

### 7.2 nsys 与日志：固定成本是 stage 之间的 KV 交接

`bagel-i2i-nsys`，一个请求的时间线（nsys 把 two-stage 放大了一点，只看结构）：

| | DiT 循环 GPU busy | 备注 |
|---|---:|---|
| M* | 22,747 ms | 69,650 launch，一个 burst |
| vLLM single-stage | 22,849 ms | 127,710 launch |
| vLLM two-stage（DiT 侧） | 22,369 ms | 121,344 launch |

- **三家的 DiT 循环 busy 在 2% 内相同**，和拟合的 b 一致。
- two-stage 的一个请求在 GPU 上是：Thinker 的 prefill（1.13 s、7,237 launch，期间 D2H 拷出约 340 MB 的 KV）→ **GPU 上没有任何 kernel 或拷贝，持续 3.7 s**（Thinker burst 结束于 152.94 s，DiT burst 开始于 156.62 s；最后一次 D2H 在 151.9 s，DiT 侧第一次 H2D 在 156.1 s，两次各 340 MB）→ DiT 的 22.4 s 循环。
- 日志（`server.log`，`kv_transfer_manager.py`）逐请求记录：主分支 KV 340,457,838 字节，**序列化 467 ms + 传输 0.359 s（903 MB/s）**；`cfg_text` 分支 339,998,985 字节，**序列化 464 ms + 传输 0.357 s**；`cfg_img` 只有 463 KB。发送侧合计约 1.65 s，**串行**。接收侧的反序列化和请求建立 vLLM 没有记录，剩下的约 2 s 空转在这一侧：我只能说"GPU 空闲 3.7–4.1 s，其中日志能解释 1.65 s"，不能把剩下的 2 s 归到具体函数。
- 固定成本差 4.05 s ≈ 交接空转 3.7 s + Thinker 单独 prefill 1.1 s − M* 自己的 0.34 s 固定成本，量级吻合（nsys 对 two-stage 有 6–18% 放大）。
- **更正（§7.7）**：上面的 "Thinker 的 prefill 1.13 s、7,237 launch" 只是 Thinker 的**一半**。two-stage 的 Thinker 对两个 5k-token 分支各做一次 prefill，完整的一个请求是约 15.3k launch、2.05 s GPU busy（SharedMemory 的 serialize 把两次 prefill 隔开，所以被 40 ms 的间隙切成两个 burst，当时只数到一个；那次的 nsys 里 Thinker 的轨迹还在最后被截断）。"GPU 空转 3.7 s" 是 SharedMemory 的序列化 + 传输 + 反序列化，换 Mooncake 后同一段只有约 0.28 s。

### 7.3 结论与边界

- 原因：vLLM-Omni 的 stage 是进程边界，two-stage 部署里 Thinker 的 KV 要序列化→共享内存→反序列化→H2D（`shared_memory_connector`，约 0.9 GB/s，BAGEL 一个请求要传两份 340 MB）。M* 在同一个引擎里把 KV 当 GPU 张量传，没有这笔开销。
- vLLM **single-stage**（Thinker 和 DiT 同进程）没有这笔开销，只比 M* 慢 1.2%，所以这不是 M* 独有的能力，是 two-stage 拓扑的代价。
- 1024² 时差距涨到 1.36×（固定成本随 KV 大小增加）与此一致。
- **默认 connector 不是唯一选择**：vLLM-Omni 还有 Mooncake Transfer Engine 和 NIXL（`supports_raw_data=True`，KV 以 GPU 张量直接搬，不走字节序列化）。已在本机装到用户目录并测过，结果见 §7.4：**换 connector 去掉了约 58% 的固定成本差**，所以上面“两阶段拓扑的代价”要改读成“默认 connector 加两阶段拓扑的代价”。
- 没有测的：M* 自己的跨 GPU 交接（`baseline-bagel-cfgpar` 的 3 卡 CFG-parallel）有自己的传输成本，不能说 M* 的 stage 边界是免费的。
- 排除：prefix cache、注意力后端、硬件时钟（M* 在 1,245 MHz，two-stage DiT 在 1,610–1,863 MHz，M* 在更低时钟下还更快）。

### 7.4 换 connector 的反事实（`bagel-connector`，已跑）

Mooncake 与 NIXL 装在 `~/.local/opt/omni-connectors`（`exps/tools/install_omni_connectors.sh`，不动基线环境）；每个 arm 只用 `--env vllm-omni:PYTHONPATH=…` 引入，别的 arm 看不到。本机无 RDMA 网卡、无 NVLink，所以 Mooncake 走 TCP，日志有 `Using TCP transport (no HCA detected)`。同一对卡、C=1、back-to-back、每格 10 个测量请求（2 starts × 5），标准差 21–37 ms（5 步）/ 约 300 ms（50 步）。

| arm | 5 步 E2E（ms） | 对 shm | 50 步 E2E（ms） | 对 shm | 发送侧序列化 + 传输（ms） |
|---|---:|---:|---:|---:|---:|
| shm（默认） | 6,809 | - | 28,190 | - | 948 + 725 = 1,673 |
| Mooncake-TCP | **4,558** | **−2,251** | **25,943** | **−2,247** | 2 + 26 = 28 |
| NIXL-UCX | 5,308 | −1,501 | 26,832 | −1,358 | 2 + 0（异步，见下） |
| vLLM single-stage（无交接） | 2,936 | −3,873 | 24,122 | −4,068 | - |
| M*（`bagel-i2i-steps`，另一个 campaign） | 2,761 | - | 23,643 | - | - |

- **三种 connector 的输出逐字节相同**：每个 fixture 序号的返回图 sha256，shm / Mooncake / NIXL 在 5 步和 50 步两个 case 上 5/5 一致，所以 KV 完整到达，速度数字是在做同样的事。（single-stage 的图与它们不同：它是另一条代码路径，返回图在 5 个请求里逐字节相同；它日志里有 `ref_images=1` 和 `img2img`，输入图被使用了，`input_tokens: 7` 只是 usage 只统计文本 token。）
- **节省与步数无关**：Mooncake 在 5 步省 2,251 ms，在 50 步省 2,247 ms，正是固定成本。发送侧日志只解释 1,645 ms，另约 0.6 s 是接收侧省下的（默认路径要在接收侧把字节反序列化回张量）。
- **NIXL 的发送侧“0 ms”不是真实传输时间**：发送侧只是提交异步拉取（日志里 `197,223 MB/s` 这类数字没有意义，分析表对应列留空），真正的拉取在接收侧，所以它的 E2E 才是比较口径。NIXL 比 Mooncake 慢 0.75–0.9 s，原因没查（UCX 是否用了 `cuda_ipc`：`nvidia-smi topo -p2p r` 显示任意两卡 P2P 可用，所以是可能的）。
- 距离 M* 的剩余差：5 步时 shm +4,048、Mooncake +1,797、NIXL +2,547、single-stage +175 ms；50 步时 +4,547 / +2,300 / +3,189 / +479 ms。其中约 4 ms/步是 two-stage DiT 每步稍慢（50 步差 1,821 ms 对 5 步差 1,622 ms），其余是固定的约 1.6 s。
- **结论的修改**：之前的“I2I 的 1.18× 是两阶段拓扑的代价”，应改成“**默认 connector 的字节序列化 + 接收侧反序列化占固定成本差的约 58%**（Mooncake 省的 2,251 ms 对 shm 到 single-stage 的 3,873 ms；对 M* 的 4,048 ms 则是 56%）；剩下的约 1.6 s（Mooncake）还没拆开”。用 Mooncake 后 M* 对 two-stage 的 I2I 领先从 1.19× 降到 1.10×。这个领先仍然**不是引擎 kernel 能力**，但“M* 没有 stage 边界”现在只剩下一半的解释力。
- **没有测的**：RDMA / NVLink 机器上 Mooncake 或 NIXL 会更快，本机结果是 TCP / PCIe；M* 自己的跨 GPU 交接（CFG-parallel）有自己的传输成本。
- 剩下约 1.6 s 的拆解实验：`bagel-connector-timeline`（毫秒级日志时间戳把一个请求拆成 client→engine / Thinker / 发送 / 编排间隙 / DiT 侧，并加 NIXL 的 `UCX_TLS=cuda_ipc` 与 `tcp` 两个传输对照），§12。

### 7.5 M* 在 Qwen 上开 torch.compile（`qwen-mstar-compile-*`，已跑）

M* 自己 A1（默认关）/ compile-on / A2（关），同一台机器、同一批输入，A2T / T2S 用预热过的同批输入。compile-on 的命令里没有 `TORCHDYNAMO_DISABLE`，就绪时间 178 s（A1 / A2 是 106 s，多出的 72 s 是编译），日志有 Dynamo 警告，没有 `OutOfResources`；compile 确实生效了。

| workload | A1 | compile-on | A2 | compile-on 对 A1 |
|---|---:|---:|---:|---|
| TXT-1k ITL（ms/token） | 5.86 | **6.10** | 5.85 | **慢 4.1%**（A1 与 A2 差 0.2%） |
| TXT-1k E2E（ms）/ TTFT | 1,647 / 149 | 1,715 / 153 | 1,649 / 151 | 慢 4.1% / +4 ms |
| A2T-en ITL（ms/token） | 5.81 | **6.00** | 5.81 | **慢 3.3%** |
| A2T-en E2E / TTFT（ms） | 261 / 92 | 264 / 92 | 258 / 91 | 持平 / 持平 |
| T2S 每秒音频耗时（ms） | 94.9 | 93.4 | 94.8 | −1.6%（持平） |
| T2S 稳态拟合 b（ms/音频秒） | 70.4 | 65.8 | 70.3 | −6.5%；a 94 → 104 |

- **本机 compile 没有收益**：Thinker 的解码在 TXT / A2T 上慢 3–4%，T2S 每秒音频持平（拟合 b 低 6.5%，但固定项 a 高 10 ms，E2E 口径是平的）。机制没有测，只能说在 C=1、这几个 workload、sm_89 上没有好处。
- **T2S 在 compile-on 下不是等工作量**：A1 与 A2 对同一句话的音频时长逐句完全相同（如 5.02 / 5.02 s），compile-on 对同一句话两次的时长不同（5.34 / 4.78 s），10 句全部与 A1 不同，平均 3.74 对 3.86 s。也就是 compile-on 的 Talker 输出不可复现，原因没有查。所以 T2S 只能用“每秒音频耗时”比较，不能用 E2E（349 对 367 ms 完全是音频短了）。
- **对公平档的结论**：M* 默认关 compile 不是给对手让路，反而对 M* 略有利；公平档不用 `ATTRIB_MSTAR_COMPILE=1` 重跑。fairness audit 第 7 行的偏向从“vLLM”改成“无”。
- **没测的**：H200 上旧结果里 T2S compile +20%，本机没复现，不能说成硬件差异，也可能是 M* 版本差异，H200 空了要用同版本重跑。

### 7.6 毫秒时间线与 1024²（`bagel-connector-timeline`，已跑）

日志换成毫秒时间戳（`VLLM_LOGGING_CONFIG_PATH`，格式在 `exps/configs/hardware/rtx5880/vllm_omni/logging/ms_timestamps.json`），`analyze_i2i_timeline.py` 把每个 5 步请求按日志标记切成五段。每格 10 个请求（2 starts × 5），中位数，ms：

| arm | E2E | pre（client→engine） | stage 0（到第一个 KV 分支出来） | send（三个分支交给 connector） | gap（到 DiT 收到请求） | DiT 侧（接收 KV + 去噪 + VAE） |
|---|---:|---:|---:|---:|---:|---:|
| shm | 6,862 | 292 | 2,666 | 1,235 | 25 | 2,626 |
| Mooncake-TCP | **4,542** | 274 | 2,188 | 56 | 25 | **1,998** |
| NIXL（默认 UCX） | 5,701 | 283 | 2,186 | 31 | 25 | 3,163 |
| NIXL `UCX_TLS=cuda_copy,cuda_ipc,tcp` | 5,486 | 287 | 2,199 | 32 | 25 | 2,960 |
| NIXL `UCX_TLS=cuda_copy,tcp` | 5,276 | 290 | 2,191 | 31 | 25 | 2,745 |

- **shm → Mooncake 省的 2,320 ms 的去向**：stage 0 −478（第一个分支的序列化在 stage 0 之内，约 0.47 s）、send −1,179、DiT 侧 −628（接收侧不再反序列化）、pre −18。Mooncake 发送侧一个 340 MB 的分支序列化 0.8 ms、传输 13 ms（24 GB/s，是写进 pinned pool 的拷贝），三个分支合计 56 ms。**gap 在所有 arm 都是 25 ms**，编排不是成本。
- **所以剩下约 1.6 s（Mooncake 对 single-stage）不在发送侧**：send + gap 只有 81 ms。stage 0 到第一个 KV 就要 2,188 ms，DiT 侧 1,998 ms，两者相加 4.19 s，而 single-stage 整个请求只要 2.94 s。**这不能靠 connector 解释**，两个假设：（a）stage 0 是 Thinker 对两个约 5k token 分支（KV 340 MB 各一份，`cfg_img` 只有 0.46 MB）的 prefill，I2T nsys 里一个分支约 0.74 s，两个就是 ~1.5 s，而 single-stage / M* 可能因为 prefix cache 命中而跳过；（b）DiT 侧含接收侧把 680 MB 从 pinned pool 经 loopback TCP 拉回 GPU。**（a）成立，（b）被否定，见 §7.7**（`bagel-i2i-fair` 与 `bagel-connector-nsys` 已跑）。
- **接收侧没有日志**：`kv_transfer_manager.py:1554` 有 INFO 级的 `Successfully received KV cache ... wait=..s`，但 DiT 进程（spawn 出来的 `StageDiffusionProc` 的 worker）在所有 arm 的 server.log 里一行都没有，所以接收时间不能直接读，只能比较 DiT 侧在各 connector 之间的差。
- **NIXL 比 Mooncake 慢在接收侧**：DiT 侧 3,163 对 1,998，慢 1.17 s。`UCX_TLS` 限定为 tcp 省 418 ms、限定为 cuda_ipc 省 203 ms，所以 UCX 默认的传输选择占一部分，但 NIXL 仍比 Mooncake 慢约 0.75 s。**限制**：没有日志证明 `UCX_TLS` 生效，`cuda_ipc` 两个 start 差 424 ms（5,704 对 5,280），这一臂噪声大；不要把 200 ms 级的差当结论。
- **1024²（`bagel-connector-1024`，已补全）**：Thinker 的 KV 每个大分支 516 MB，是 512² 的 340 MB 的 **1.52×**（我之前按 4× 预期是错的：KV 主要是 ViT / 文本 token，不随 VAE latent 面积线性增长）。

  | arm | 1024² / 5 步 E2E（ms，n=10） | 对 shm | 发送侧（一个请求两个分支合计 1,033 MB） |
  |---|---:|---:|---|
  | shm | 16,178 | - | 序列化 1,408 + 传输 1,108 ms |
  | Mooncake-TCP | 12,764 | **−3,414** | 2 + 42 ms |
  | NIXL | 14,085 | −2,093 | 2 + 1 ms（异步提交） |
  | single-stage | **9,897** | −6,281 | - |

  三种 connector 在 5 个 fixture 上返回图的 sha256 逐一相同（single-stage 是另一条代码路径，自己稳定）。Mooncake 省 3,414 ms，是 512² 的 2,251 ms 的 **1.52×，与 KV 字节数的 1.52× 同比例**，节省按字节随 KV 增长；**connector 占 shm → single-stage 差距的 54%**（512²：58%）。Mooncake 离 single-stage 还差 2,867 ms（512²：1,622 ms，**1.77×**）：剩余部分增长比 KV 字节更快，与"重复的 Thinker prefill（注意力随 token 数超线性）+ hand-off"一致，但这里没有 1024² 的 nsys，只算一致，不算证据。

### 7.7 剩下的 1.6 s 拆开了（`bagel-i2i-fair`、`bagel-connector-nsys`，已跑）

**（1）`bagel-i2i-fair`：两个开关都配好后还剩多少**（每格 10 个请求，2 starts × 5，中位数 ms）

| arm | 5 步 | 50 步 |
|---|---:|---:|
| M\* 缓存开（默认） | 2,760 | 23,506 |
| M\* 缓存关（每个请求带 `prefix_cache=false`） | 2,764 | 23,550 |
| vLLM two-stage + Mooncake，Thinker 缓存关 | 4,564 | 25,964 |
| vLLM two-stage + Mooncake，**Thinker 缓存开** | **3,523** | **24,959** |
| （参考）vLLM two-stage 默认 SharedMemory，缓存关 | 6,809（`bagel-connector`） | 28,190 |

- **M\* 的缓存开关在 I2I 上没有任何作用**（5 步差 4 ms，50 步差 44 ms，都在 start 间波动内）。日志里没有 I2I 的命中行，所以分不出"开关在 I2I 路径无效"还是"缓存永远不命中"；对比的结论相同：M* 没有靠缓存跳过 prefill，**之前担心的"M* 的 I2I 领先里有一块是缓存"不成立**。
- **vLLM 打开 Thinker 的 prefix cache 省 1,041 ms（5 步）/ 1,005 ms（50 步）**：与步数无关，是每请求固定成本。
- 三个开关叠起来，M* 的 5 步领先 4,048 ms（对默认 vLLM two-stage）的构成：**默认 connector 2,251 ms（56%）+ Thinker 不开缓存 1,041 ms（26%）+ 剩余 763 ms（19%）**。50 步 4,547 ms：2,226（49%，Mooncake）+ 1,005（22%）+ 1,316–1,453（29–32%）。
- 剩余 763 ms（5 步）= 1,453 − 690：50 步的剩余里有 ≈690 ms 是每步差（vLLM 476 对 M* 461 ms/步，3%）。**注意时钟**：50 步 M* 的格子在 64 °C、1,249 MHz，vLLM 在 34 °C、1,834 MHz，M* 的时钟吃亏仍然每步更快，所以这 3% 不是时钟造成的有利于 M* 的偏差；但它是不是 kernel 差异没有证据（上面 §7.1 的拟合 b 只差 2%），不要读成结论。

**（2）`bagel-connector-nsys`：一个请求在两张卡上的时间线**（5 步，nsys 下 Mooncake 的 E2E 4,572 对不开 nsys 的 4,542 ms，nsys 对 vLLM 几乎没放大）

先说一个**我的分析工具的错误**：vLLM-Omni 的两个 stage 进程各自用 `CUDA_VISIBLE_DEVICES` 起，nsys 里 Thinker 和 DiT 的 `deviceId` 都是 0，旧版 `analyze_nsys_burst_gaps.py` 按 `deviceId` 分表，把两个进程的 burst 并成了一张表，第一次读出来的"GPU 0"是混合的。现在按（deviceId，进程号）分表，并新增 `--handoff` 把每个请求的 Thinker / hand-off / DiT 列在一行（§13）。**§7.2 里用旧版读 two-stage 的数字要按下表修正。**

| 一个请求（ms） | Thinker 进程 GPU busy | Thinker launch | Thinker 最后一个 kernel → DiT 第一个 kernel | DiT span / busy | DiT launch |
|---|---:|---:|---:|---:|---:|
| Mooncake，precondition 请求 | 2,116（span 2,444） | 15,876 | 259 | 1,875 / 1,746 | 10,239 |
| Mooncake，measured 1（完整） | **2,055**（span 2,283） | **15,280** | **234**（其中 KV 拷出完成→DiT 第一次 H2D **133**，H2D→第一个 kernel **145**） | 1,825 / 1,773 | 10,239 |
| single-stage（一个进程，整个 burst） | 2,627（含 DiT） | 16,605 | - | 其中 DiT ≈ 1,775，10,239 launch；DiT 之前的部分 **6,366 launch，≈ 0.85 s busy** | |
| SharedMemory | Thinker 轨迹在最后被截断（工具会打印警告），这一行不用 | | | 1,827 / 1,774 | 10,239 |

- **DiT 去噪在所有 arm 里完全相同**：10,239 launch、span 1.83 s、busy 1.77 s，connector 不影响它。
- **接收侧很便宜，假设（b）被否定**：Mooncake 下 Thinker 把 2 × 340 MB 拷出 pinned pool（每份 13 ms）到 DiT 第一次 H2D 只有 133–150 ms，112 个 6.1 MB 的 H2D 拷贝合计 28 ms，从第一次 H2D 到第一个 DiT kernel 再 126–154 ms。**整个 hand-off 约 0.28 s**，不是 1.6 s。
- **假设（a）在结构上成立**：two-stage 的 Thinker 一个请求要 **15.3k launch / 2.05 s busy**（两个 5k-token 分支各一次 prefill），而 single-stage 在 DiT 之前一共只有 **6.4k launch / 0.85 s**，是 Thinker 侧 launch 的 2.4×、busy 多 1.2 s；**打开 Thinker prefix cache 恰好省掉 1,041 ms**（上表），与这段重复工作吻合。所以 Mooncake 对 single-stage 的 1,622 ms ≈ 重复的分支 prefill 1,041 + hand-off 280 + 前端及其余 ≈ 300。
- **Thinker 的 nsys 轨迹被截断**：Thinker 的最后一个 kernel 比 DiT 的最后一个早 3–12 s，原因没查（`--cuda-flush-interval=1000` 已经在用），所以只信后面还跟着完整一行的那些请求。为什么 single-stage 的 Thinker 侧只做了一次的工作量（是否共享分支的 KV）没有证实。

**（3）新增的两个臂与 `mooncake-tcp-prefix` 时间线**（`connector-nsys` 的这两臂跑完了，但同一个实验里三个旧臂只有 2 个测量请求，suite 的 `analyze` 因此拒绝出汇总表；这两个臂我用 `analyze_nsys_burst_gaps.py` 直接读，结果文件 `bagel-connector-nsys_mooncake-prefix_*.md`、`bagel-connector-nsys_mstar_*.md`，旧三臂补跑见 §12）

| 一个请求 | Thinker busy / launch | hand-off | DiT busy | 单进程总 busy / launch | E2E（不开 nsys） |
|---|---:|---:|---:|---:|---:|
| vLLM two-stage，Mooncake，缓存关 | 2,055 / 15.3k | 234 | 1,773 | - | 4,564 |
| vLLM two-stage，Mooncake，**缓存开** | **1,116–1,123 / 8.9k**（span 1,311–1,319） | 222–241 | 1,750 | - | **3,523** |
| vLLM single-stage | - | - | ≈ 1,775 | 2,627 / 16.6k（DiT 之前 ≈ 0.85 s、6.4k） | 2,869–2,936 |
| **M\***（`i2i-512-srv-steps5`） | - | 0 | - | **2,649–2,700 / 8,990**（span 2,730–2,779） | 2,760 |

- **Thinker 缓存把 Thinker busy 从 2.05 s 降到 1.12 s（−0.94 s，launch 15.3k → 8.9k）**，和 E2E 省的 1,041 ms 对得上：缓存确实抵掉了第二个 5k-token 分支的 prefill。但缓存开了之后 Thinker 还要 1.12 s，比 single-stage 在 DiT 之前的 0.85 s 多 0.27 s。
- **M\* 的单进程总 busy 2.65 s ≈ single-stage 的 2.63 s ≈ vLLM two-stage + 缓存的 Thinker 1.12 + DiT 1.75 = 2.87 s**：M\* 做的是**一次 prefill 的工作量**，和 single-stage 一样；上一版预测的「launch 数 16.6k = 一次 prefill、25k = 两次」**判据无效**（M\* 只有 8,990 次 launch，比 single-stage 的 16,605 少一半，busy 却相同，说明 M\* 每个 kernel 更长 / 更融合，不是少做事）。M\* 的 BAGEL prefill attention 是 168 次 launch = 6 块 × 28 层（1,024-token 的 chunk），所以 M\* 的 5k-token 分支在分块下做完。
- **时间线（`mooncake-tcp-prefix`）**：pre 292、stage 0 **1,222**（Mooncake 缓存关 2,188，省 966 ms）、send 55、gap 23、DiT 侧 1,939，合计 E2E 3,528。**缓存开之后 stage 0 仍有 1.2 s**，不全是 prefill：图像的 ViT / VAE 编码和未命中的尾部在这里。
- **剩余 763 ms（vLLM Mooncake + 缓存 3,523 对 M\* 2,760）的拆分**：GPU 侧 Thinker 1.12 s 对 M\* 约 0.9 s（M\* 总 busy 2.65 − DiT 1.75）≈ **0.2 s**；stage 间 hand-off **0.22–0.26 s**（M\* 没有）；其余约 **0.3 s** 是前端 / 编排（vLLM 的 pre 292 ms；M\* 的前端没有单独测）。三项加起来约 0.7 s，与 763 ms 吻合，但每一项的精度在 ±0.1 s。


**结论**：I2I 对 vLLM two-stage 的领先**不是引擎能力**，而是 vLLM-Omni **部署上的三个默认值 + 一个固有成本**：默认 SharedMemory connector（56%）、Thinker 不开 prefix cache 使 CFG 的第二个大分支被重复 prefill（26%）、固有的 stage 边界（hand-off 约 0.28 s + 前端，19%）。两个默认值改掉后 5 步领先 1.28×、50 步 1.06×，对 single-stage 为 1.02×。

## 8. Cosmos3 256²：2.46× 的拆解

`cosmos3-decompose`，3 starts × 10 请求，M* 的实现开关逐个关掉（hook 的 `applied: true, before: true, after: false` 已记录）：

| arm | E2E ms | 相对 m-control |
|---|---:|---:|
| m-control（M* 基线） | 1,268 | - |
| m-nograph（关 CUDA graph） | 1,279 | 1.01× |
| m-capture256（给 256² 加 graph） | 1,270 | 1.00× |
| m-seqcfg（顺序 CFG，两次 forward） | 1,952 | 1.54× |
| m-seqcfg-nograph | 1,945 | 1.53× |
| m-all-off（顺序 CFG + 无 graph + 无 compile） | 2,676 | 2.11× |
| vLLM-Omni | 3,118 | 2.46× |

- **graph 与此无关**：256² 不在 M* 的捕获档位（日志只捕获 120/780 token 两档），默认就是 eager；手动加上也没变。
- 批量 CFG 1.54×（约 680 ms）；torch.compile 在顺序 CFG 下 1.38×；乘起来 2.11×；vLLM / m-all-off = 3,118 / 2,676 = **1.17×（约 440 ms）**。

**1.17× 的 nsys 结构证据**（`cosmos3-nsys_*_rtx5880.md`，nsys 下的 span 不能和上表的 E2E 比）：

| | M* m-all-off | vLLM-Omni |
|---|---:|---:|
| 三个最大 GEMM（ms） | 870 + 553 + 296 = 1,719 | 831 + 537 + 287 = 1,655（-3.7%） |
| launch 数 | 212,300 | 83,590 |
| GPU busy（ms） | 2,150 | **1,815** |
| burst 内 GPU 空闲（ms） | 1,756 | **1,988** |
| 每个 launch 前的平均空闲（µs） | 8.3 | **23.8** |

- GEMM 工作相同；vLLM 因 compile 融合出更少的 kernel，GPU busy 反而少 335 ms，但墙钟更长：GPU 在每个 GEMM 前等主机发射。**440 ms 的残差是 vLLM-Omni 每个 kernel 之间的主机发射开销**。
- 这是结构证据：nsys 把 M* 放大得更多，所以只能说方向，不能说 440 ms 的归属。没有反事实的原因：要关掉 vLLM 侧的主机开销需要改 vLLM-Omni。
- 结论：2.46× 是实现选择（打包 CFG 分支、对 DiT compile，外加 vLLM 的主机开销），vLLM-Omni 可以做同样的事，与 Graph + Walk 无关。这一项对 SGLang-Omni 没测过：SGLang-Omni 也有 `cosmos3` 模型，是一个还没做的对照（§10）。

## 9. Qwen3-Omni：还剩什么，以及要跑的实验

**Qwen TXT**：E2E = TTFT + 255 × ITL。M* 149 + 255×5.85 = 1,641 ms；vLLM 99 + 255×5.93 = 1,611 ms。M* 的 TTFT 慢 50 ms（1k-token 的 prefill），解码快 20 ms，净慢 30 ms。**M* 在这里相对 vLLM 没有优势**。对 SGLang 的 534 ms 差距 = 解码 574 ms − TTFT 41 ms，解码部分是 §2.4 的"SGLang 疑似同步循环"。

**Qwen A2T**：
- 对 SGLang：TTFT 相同（90 对 91），解码差 (7.84−5.81)×27 = 55 ms，实测 E2E 差 58 ms。差距全在解码，同样要看 lookahead 实验。
- 对 vLLM（en）：TTFT 多 19 ms（109 对 90），解码差 2.4 ms，预测 21 ms，实测 19 ms。vLLM 多出的是首 token 之前的 20 ms，**机制未验**。
- 对 vLLM（zh）：**上一版写的"剩 34 ms 在最后一个 token 之后"不成立**。逐 chunk 看（`itl_ms` 是每个 chunk 的间隔）：最后一个 chunk 之后的尾部 vLLM 6.9 ms 对 M* 7.4 ms，解码间隔总和 108 对 109 ms，TTFT 多 20 ms。54 ms 的 E2E 中位数差是把不同 fixture 的中位数相减的假象，加上 vLLM 的两个编译 stall 请求（fixture 6 和 11，TTFT 507 / 629 ms，常态 ~100）。同一 fixture 逐对比较，vLLM 只比 M* 慢 10–25 ms（+4–11%）。
- **vLLM 的 stall 是推理期 JIT**：server.log 有 `Triton kernel JIT compilation during inference: fused_moe_kernel. This causes a latency spike; consider extending warmup to cover this shape/config`。我们的 precondition 只有 6 个请求，覆盖不了测量窗口里首次出现的输入 shape，所以 JIT 落进了测量。M* 无此现象。这是 vLLM 的真实冷 shape 行为，但不属于稳态；`qwen.rows=cycle` 把它挪进 precondition（§12）。
- M* 自己的 MoE 调优表：built-in 254 ms，tuned 236 ms（1.08×）：M* 的 A2T 数字取决于调优表。

**Qwen T2S**（ms/音频秒）：
- M* 发射链臂：A1 103，S（关链）128（1.24×），W 134，C（compile）103（1.00×；5880 上 M* 默认没开 compile，所以这一臂什么都没改）。
- 对 SGLang（121）：M* 关链后 128，比 SGLang 慢 5%；链 1.24× 过补偿，净领先 1.18×。T2S 上 SGLang 的 Thinker lookahead 是它自己因音频输出而关的，所以这一行不受 seed 影响。
- 对 vLLM（155）：关链后 M* 仍快 1.21×。**现有数据可以拆一部分**（`analyze_t2s_fit.py` 读 `BASELINE_QWEN_T2S_MAIN_01_B2B`，每个语言 5 句 × 3 starts，按 TTFA > 2× 中位数记为 stall）：

  | | stall 请求 | 稳态 TTFA | 稳态每音频秒 b（ms，en / zh） | 首个音频之后每音频秒 |
  |---|---:|---:|---:|---:|
  | M* | 0 / 15 | 153 / 155 | 73 / 67 | 51 / 56 |
  | SGLang | 0 / 15 | 170 / 173 | 95 / 89 | 63 / 70 |
  | vLLM | **4 / 15（每个语言）** | 173 / 182 | **110 / 124** | 98 / 104 |

  也就是说对 vLLM 的差距由三块组成：① 约 1/4 的请求有约 400 ms 的 TTFA stall（JIT，见 A2T 同一机制），② 首个音频晚 20–27 ms，③ **稳态的 Talker 帧循环每音频秒慢 1.5–1.8×**。③ 与 nsys 的 Talker launch 数（57,299 对 18,226）同向，但**那是结构证据，不是反事实**。样本只有 5 句，所以设计了 `qwen-t2s-fit`（20 句，预热掉 JIT，加 vLLM Talker greedy 对照；§12）。
- 这里还有一个不对等：vLLM Talker 实际用 yaml 的 `temperature 0.9 / top_k 50 / rep 1.05`，M* 和 SGLang 是 greedy，同一句子音频长度不同（M* 2.86 s，SGLang 2.70 s，vLLM 2.54–3.02 s），所以"每音频秒"不是完全等工作量。vLLM 的数字强烈依赖配置（plain 206，best 157）。
- nsys（1 个 start，3 个请求，结构证据）：Talker 的 GPU busy，M* 145 ms，SGLang 142 ms，vLLM 217 ms（launch 数 57,299 对 18,226）。M* 的 Talker 没有比 SGLang 少做 GPU 工作；相对 vLLM（Talker 的 launch 多 3 倍）有可能做得少，但这条 nsys 只有一个 start，Thinker 的工作量在三家的 trace 里也不等价。

**TXT 对 vLLM 的首 token 与 A2T 对 vLLM 的首 token**：现有数据只能说"M* 的 TXT TTFT 多 50 ms，A2T 的 vLLM TTFT 多 15–25 ms"，原因的实验在 §12（`qwen-ttft-*`，把首 token 拆成固定成本 / 1K prefill / 音频路径）。

### 9.5 公平档结果（`qwen-fair-*`、`qwen-t2s-fit`、`qwen-ttft-*`、`bagel-sampling-parity`，已跑）

**先说跑出来的问题**（都已修）：① 公平档的 4 个 Qwen 组的 `analyze` 全部拒绝出数，因为我写的 SGLang 关缓存检查是"日志里没有 `Init Unified Radix Cache`"，而 SGLang 即使 `disable_radix_cache: true` 也会构造这棵树（源码：被禁用的树"什么都不存"），这一行在所有 arm 里都有。cache 状态的真实证据是调度器的 `#cached-token`：关的 arm 的 6 次重复 1k prompt 全是 0，开的 arm 是 1071 命中。检查已改成按这个计数（`re:#cached-token: [1-9]`）。② M* 开 Thinker 缓存（TP=2）**崩溃**：第 2 个相同 prompt 触发 `TP ranks diverged ... rank 0 leased 8 cached pages`，worker_1 退出（两个 repeat 一致），公平档的 M* on 臂不存在，它还让后面的 pinned 臂被跳过。③ M* 对 `max_output_tokens=1` 返回 2 个 token（256 是精确的），`qwen-ttft-txt` 的 M* 4 个 cell 因此被判 void。

**Qwen TXT（1k-token prompt，256 token，E2E ms；SGLang 的请求已改成 lookahead 可用）**

| arm | E2E | TTFT | ITL ms/token |
|---|---:|---:|---:|
| SGLang 关缓存 | 1,596 | 120 | 5.81 |
| SGLang 开缓存 | 1,588 | 110 | 5.82 |
| SGLang 开缓存、lookahead 强制关（对照） | 2,135 | 107 | **8.00** |
| vLLM 关缓存 | 1,610 | 98 | 5.97 |
| vLLM 开缓存 | 1,637 | 76 | 6.06 |
| **M\* 关缓存** | **1,648** | **149** | 5.85 |

- **M\* 在 TXT 上没有领先，反而慢 2.4%（对 vLLM）/ 3.3%（对 SGLang）**。差在首 token（149 对 98 / 120 ms），解码三家一样（5.8–6.0 ms/token）。M* 的解码链（§2）与 vLLM 的异步调度、SGLang 的 lookahead 是同一种重叠，各自把 GPU 喂满。
- **基线里对 SGLang 的 1.32× 就是请求带 `seed` / `min_tokens` 把 SGLang 的 lookahead 关了**：对照臂（lookahead 强制关）ITL 8.00，E2E 2,135，2,135 / 1,648 = 1.30×。lookahead 开时 SGLang 比 M* 快。
- 开缓存对 E2E 帮不上忙（SGLang 1,596 → 1,588，vLLM 1,610 → 1,637）：1,071 个 token 命中后 TTFT 只降 10–22 ms，因为 TTFT 里 prefill 只占一小部分（固定成本约 56 ms）。M* 没有开缓存的 TXT 数字（崩溃）。
- **pinned 两臂已补跑**（基线请求，带 `seed` / `min_tokens`）：vLLM 1,614 对去掉后的 1,610 ms，M* 1,648 对 1,648 ms，所以**去掉 seed / min_tokens 对 vLLM 和 M* 没有影响**，只有 SGLang 受这个门控影响，对照实验的设计成立。

**Qwen TXT 的首 token 拆分（`qwen-ttft-txt`，一个输出 token，关缓存，每格 n=20）**

| 框架 | 固定成本（tiny 提示，服务端 56 token = 8 + 48 模板，TTFT ms） | 1,072 token 提示 TTFT | prefill 每 1k token |
|---|---:|---:|---:|
| SGLang | 55 | 117 | 62 |
| vLLM | 57 | 99 | 42 |
| **M\*** | 54 | 146 | ~~86~~（（146−54）/ 1.072；**这个斜率是假象**，见下与 §9.6（1））|

**更正（第三轮 nsys 之后）**：上一版在这里写「M\* 慢在 prefill 本身，每 1k token 比 vLLM 慢 2.1×」，并把它列为 M\* 唯一明显的落后。**这个结论撤回**：两点直线用的 1,072-token 提示正好跨过 M\* 的 1,024-token chunk 预算，M\* 的第 2 个 chunk（48 token）单独要约 40 ms，直线把它摊成了斜率。固定成本三家相同这一点仍然成立（M\* 的 15-token 请求 = 一块 forward）。M\* 的 1-token 请求返回 2 个 token，TTFT 取首个 token 到达的时间，不受影响（E2E 要多一步，所以这里不用 E2E）。**第四轮反事实之后的状态**：原先的「真正的结论」（≤1,024 token 的提示与 vLLM 持平，每跨一个 1,024 边界多约 40 ms，由 chunk 预算造成）**也被推翻**：`prefill-chunk` 把预算改成 4,096 后 TTFT 反而从 138 变成 154 ms，所以第 2 块的 40 ms 不是靠取消分块能省掉的，「持平」的预测没有证据。现在只能说：1,072-token 这一个长度上 M\* 的 TTFT 比 vLLM 高 40–47 ms，机制未定位（§9.6（1）），其他长度未测。固定成本三家相同这一点仍成立，但 tiny 提示在服务端是 56 token（8 + 48 模板），不是「约 15 token」。

**Qwen A2T（FLEURS-en 10 条，同一批片段预热，E2E ms）**

| arm | E2E | TTFT | ITL |
|---|---:|---:|---:|
| SGLang 关 / 开 / 开 + lookahead 关 | 278 / 308 / 353 | 88 / 119 / 119 | 6.60 / 6.53 / 8.00 |
| vLLM 关 / 开 / 基线请求（pinned） | 251 / **665** / 245 | 77 / **481** / 79 | 5.89 / 5.94 / 5.90 |
| M\* 关 / 开 / 基线请求（pinned） | 255 / 252 / 262 | 89 / 86 / 89 | 5.81 / 5.81 / 5.81 |

- **A2T 对 vLLM：没有领先（255 对 251，M* 慢 1.6%）**。基线里的 1.08× 是 vLLM 的推理期 JIT：关缓存的 vLLM 在 precondition 里首见每个音频长度时是 540 / 535 / 487 ms，预热同一批片段之后测量请求只有 64–94 ms。对 SGLang 是 1.09×，全部来自解码（6.60 对 5.81 ms/token，13%），首 token 相同（88 对 89）。
- **vLLM 开缓存反而慢 414 ms**（TTFT 481 对 77）：第二遍（测量，10 条都是第一遍见过的片段，应全命中）有 7 / 10 个请求 450–600 ms：其中 3 条（第 3、7、9 条）在第一遍只有 95–107 ms，另 3 条（第 1、2、6 条）两遍都是约 500 ms，也就是关缓存时只付一次的首见 shape 代价，开缓存后每次重付。这个量级的慢不是缓存命中该有的行为。server.log 里没有额外的 JIT 警告（每个 arm 都只有 1 条，在预热阶段），原因没查。**它不影响公平档的关缓存对比，但说明 vLLM-Omni 这一版开 Thinker prefix cache 处理音频输入有问题**。SGLang 开缓存也慢 30 ms（TTFT 88 → 119），原因也没查。 **（更新：三遍预热后回到 77 ms，是预热不足，见 §9.6（2）；SGLang 的 +30 ms 保留。）**
- M* 开缓存无变化（A2T 提示短，每请求只有 1 页可命中）。pinned 臂：vLLM 245 对 251，M* 262 对 255（+2.7%，在 start 间波动之内）。

**Qwen T2S（20 句预热，每个 arm 2 starts × 20）**

| arm | 音频 s | E2E ms | 每音频秒 ms | 稳态 TTFA ms | 首个音频之后每音频秒 | 拟合 a ms | 拟合 b ms/音频秒 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **M\*** | 3.54 | 342 | **96.9** | **154** | **53.7** | **100** | **69.0** |
| SGLang | 3.22 | 392 | 120.2 | 182 | 64.8 | 112 | 85.5 |
| vLLM | 3.50 | 527 | 152.8 | 180 | 100.3 | 135 | 113.5 |
| vLLM Talker greedy | 3.02 | 481 | 160.1 | 176 | 101.9 | 152 | 109.7 |

- **T2S 的领先保留**：每音频秒 M* 对 SGLang 1.24×、对 vLLM 1.58×。预热后三家 stall 都是 0，所以基线里 "vLLM 约 1/4 的请求有 400 ms stall" 已被预热排除；Talker 改 greedy 后每音频秒还略升（160 对 153），所以**采样器不是原因**。差距的主体在首个音频之后的帧循环（M* 54、SGLang 65、vLLM 100 ms/音频秒）：对 SGLang 的 1.21× 与 M* 自己关链的 1.24× 相当（M* 关链后 128 对 SGLang 121，慢 5%），**能归因到发射链**；对 vLLM 的额外 1.27× 只有 nsys 的结构证据（Talker launch 多 3 倍），机制未验。
- 固定成本 a（M* 100 对 SGLang 112 / vLLM 135）和首音频 TTFA（154 对 182 / 180）M* 都有 12–35 ms 的优势，来源没查。

**BAGEL I2T 采样器 / 缓存对等（`bagel-sampling-parity`，E2E ms / TTFT ms / ITL）**：vLLM 默认 2,976 / 802 / 17.15；vLLM 中性采样器（去掉重复惩罚）2,978 / 804 / 17.14，**重复惩罚对延迟没有影响**；vLLM 中性 + 开 Thinker 缓存 **2,256 / 85 / 17.10**；M* 默认 2,288 / 134 / 16.93，M* 中性 2,280 / 129 / 16.91。**最干净的对比（两边缓存都开、采样器相同）：vLLM two-stage 2,256 对 M* 2,280 ms，vLLM 快 1%**，M* 的 TTFT 慢 44 ms、每 token 快 1.1%，与 §6 一致。

**这些结果对 M\* 优势的整体含义**：M* 在 Qwen 文本路径（TXT、A2T）与 BAGEL I2T 上**没有可证实的领先**，基线里的 1.08–1.32× 全部来自对手配置（SGLang 的 lookahead 被关、vLLM 的 JIT、vLLM 的 prefix cache 关）；保留下来的领先是 **T2S 的流式 Talker 帧循环（1.24× / 1.58×）**、**BAGEL I2I 对 vLLM two-stage 的 1.10–1.19×（默认 connector 占一半多，剩余待 §7.6 的两个实验）**，以及 Cosmos3（实现选择）。

**仍然没有的数据**（第四轮补测之后）：被垫到 2,048 的那一次 forward 的 kernel 级时间（nsys 下 2,048 图被挤掉，没测到；§12 的第 1 项）；SGLang 开缓存慢 31 ms 的原因（不影响公平档）。**已经有答案的**：TXT 1,072 的 TTFT 差距的机制（第五轮：eager / 图档，§9.6（1））、BAGEL I2T 的 chunk 预算只占约 6–7 ms（`bagel-chunk`，§9.7）、`connector-nsys` 旧三臂已补跑（§9.7）、vLLM 开缓存音频慢的原因（预热，§9.6（2））、vLLM T2S 帧循环慢在 kernel 个数（§9.6（3））、TP 崩溃是 M\* 的缺陷且需要「命中 + 多 chunk」才触发（§9.6（4））。

### 9.6 第三轮补测的结果（`qwen-prefill-nsys`、`qwen-a2t-cache-warm`、`qwen-t2s-nsys-warm`，已跑）

**（1）M\* 的 TXT 1,072-token 提示：nsys 看到的结构，与「取消分块也没有变快」的反事实（`qwen-prefill-nsys`、`qwen-prefill-chunk`）**

| 一个请求（1,072-token 提示，1 个输出 token） | M\*（每个 TP rank） | vLLM（每个 TP rank） |
|---|---|---|
| forward 次数 | **3**：1,024 token 的块、48 token 的块、第二个输出 token | **1** |
| 时间线（ms，从 burst 起点） | 第 1 块：8.6 ms 的 pre-layer kernel，然后第 1 层 → 第 48 层 **8.6 → 85**（≈ 77 ms）；第 2 块 **87 → 125**（≈ 38–40 ms）；第 3 次（解码）130 → 136（≈ 6 ms） | 48 层一次 forward，span **77**、busy 75 |
| 总 span / busy / launch | 136 / 124–128 / 2,812 | 77 / 75 / 1,030 |
| 融合 MoE（全请求合计） | 65 ms，192 次 launch（第 1 块 0.52 + 0.27 ms/层） | 34 ms，96 次（0.71 ms/层） |
| all-reduce（全请求合计） | NCCL 35–39 ms，288 次（第 1 块每次约 300 µs；第 2 块约 56 µs） | 自定义 all-reduce 25 ms，98 次 |

- 第 1 块（1,024 token）≈ vLLM 整个 1,072-token 的 forward：每层 MoE 慢 11%、all-reduce 慢约 15%，**kernel 级别的差距在 5–10%**，不是 2×。M\* 的第 1 块比 vLLM 的单块少处理 48 个 token，所以按每 token 算 M\* 约慢 5–10%。
- **第 2 块（48 token）要 38–40 ms**：每层仍要读专家权重（MoE 带宽受限）、做两次 all-reduce、一次注意力。它的成本接近 15-token 请求的 forward（固定成本里的约 40 ms），这也解释了固定成本三家相同、而「每 1k token 86 ms」是两点直线跨过边界造成的假象。
- **预测（已被 `prefill-chunk` 推翻）**：把 Thinker 预算改成 4,096，1,072-token 的提示变成一块，TTFT 降约 40 ms 到约 105 ms。**实测 TTFT 138.3 → 153.8 ms（E2E 147.7 → 162.8，+15 ms，更慢）**，`txt-tiny-1tok` TTFT 52.4 → 53.2 ms（不动，说明预算只影响分块）；各 10 个测量请求，drift flat，每臂 1 个 start。预算 4,096 的 1,072-token 一块用了约 142 ms 的 forward（减去 tiny 的约 12 ms 前端开销），高于默认的 1,024 + 48 的约 126 ms，更高于 vLLM 的 77 ms。
- **图桶垫高（第五轮：成立）**：上一版的假设是对的：预算 4,096 的 1,072 token 一步被垫到 2,048 的图。细节：本机只捕获了**一张** 2,048 的 text prefill 图（bs = 32），bs = 1 的步按源码选「bs ≥ 1 且 token ≥ 所需的最小已捕获桶」，所以用它（见下）。「第 2 块 48 token 要约 40 ms」「差距不是 chunk 预算」仍然成立。
- **第五轮：`prefill-len`（M\* 三个设置 + vLLM + 默认重复）与 `prefill-chunk-nsys`，已跑完（每格 1 个 start，各 10 个测量请求；nsys 臂 5 个）**

  TTFT p50 ms（括号内为最小–最大）：

  | 服务端提示长度（名义） | M\* 默认（预算 1,024） | M\* 默认（重复） | M\* 预算 4,096 | M\* 预算 4,096 + 1,152 图档 | vLLM-Omni |
  |---|---|---|---|---|---|
  | 56（tiny） | 53.0（51–54） | 52.8（52–53） | 52.2（51–61） | 52.1（52–54） | 58.2（50–72） |
  | 512 | 71.3（70–81） | 78.7（76–84） | 79.8（78–85） | 75.6（74–81） | 71.5（69–85） |
  | 1,024 | 106.5（104–114） | 106.7（103–108） | 103.7（97–105） | 106.7（99–111） | 96.0（85–104） |
  | **1,072（原夹具）** | **144.9**（140–149） | **143.6**（142–146） | **155.8**（148–159） | **112.6**（104–123） | **98.3**（97–115） |
  | 2,048 | 190.9（182–198） | 187.0（183–196） | 164.6（156–168） | 166.4（155–168） | 151.6（147–155） |

  - **重复性**：默认臂跑两次，56 / 1,024 / 1,072 / 2,048 的 p50 差 0.2–3.9 ms，**512 差 7.4 ms（71.3 对 78.7，原因未查）**；所以小于约 5 ms 的差别不读，512 处四个 M\* 臂（71–80）不能排序。vLLM 有 4 个格被标 `suspect_intra_start_drift`，看每个请求是孤立的尖峰（如 1,072 第一个请求 114.9 ms，其余 97–106），斜率只有 −0.4 ~ −0.8 ms/请求，不是趋势，p50 仍可用。
  - **机制（`server.log` 的捕获清单 + 源码，已核对）**：非 nsys 的默认和预算 4,096 服务器，`prefill_text` 捕获了 128 / 256 / 512 / 1,024 各档（bs = 1、2、4、8、16、32）以及**一张** 2,048 的图（bs = 32）；bs = 1 ~ 16 的 2,048 图没捕获（日志：`captured 53 of 74 buckets; these run eagerly`）。`accelerator_graph_runner.py`（约 1373–1392）选桶的规则是「bs ≥ 请求 bs、token ≥ 所需的最小已捕获桶，没有就 eager」，所以 **bs = 1 的 1,072 token 一步在 bs = 32、2,048 的图里重放，即垫到 2,048**。因此：默认臂 1,072 = 1,024 的图 + 48 的图（两次 forward，多约 38 ms，与第三轮 nsys 的 38–40 ms 一致）；预算 4,096 的 1,072 = 一步 2,048 的图（155.8 ms，比默认还慢 11 ms）；加 1,152 档的两个臂，1,152 的图在所有 bs 上都捕获了（`captured 58 of 80`），**同时那张 2,048 图被挤掉了**（显存），1,072 在 1,152 的图里重放（112.6 ms，比预算 4,096 快 43 ms，比默认快 32 ms）。**事先写的预测（1,072 与 2,048 在预算 4,096 下几乎相同约 154 ms；加 1,152 档后 1,072 降到 110–115 ms）都成立**（155.8 对 164.6；112.6）。2,048 在预算 4,096 的臂用那张 2,048 图（164.6 ms），在 1,152 臂没有 2,048 图、走 eager（166.4 ms）：两者几乎一样，说明单请求下**图与 eager 差别很小，起决定作用的是垫到多少个 token**（1,024 → 107、1,152 → 113、2,048 → 165、1,072 垫到 2,048 → 156）。为什么 bs = 1 ~ 16 的 2,048 图捕获不了、只剩 bs = 32 一张，日志没写，`CapturePlan` 里 `cuda:0` 捕获前只剩 6,180 MiB 空闲，「显存预算」是推断。
  - **nsys 臂测的不是生产路径（矛盾已解开）**：nsys 的服务器**一张 2,048 的图也没捕获**（`captured 52 of 74`；nsys 自己占显存），所以预算 4,096 的 1,072 在 nsys 里按实际大小 eager，E2E 123 ms，与加 1,152 档的 126 ms 一样，而非 nsys 的同配置是 163–168 ms（垫到 2,048 的图）。默认臂两边一致（153 对 154），因为 1,024 + 48 用的是更小的图，不受 2,048 图有没有的影响。`prefill-chunk-nsys` 因此确认的是：预算 4,096 的 prompt 只有一次 forward（prefill 注意力 48 次 launch，默认 96；MoE 96 次，默认 192）；**eager 的 1,072 一次 forward 在 kernel 上约 span 105–112 ms、GPU busy 85–93 ms，MoE 合计 34 ms，与 vLLM 的 34 ms 相同**（默认 135–138 / 123–126 / 65 ms）。**它没有测到被垫到 2,048 的那次 forward**：事先登记的判据「GEMM / 注意力约 2× = 垫到 2,048」没有被检验（nsys 没有 2,048 图），§12 的第 1 项补这个。
  - **对 vLLM 的拆分**：1,072 处 M\* 默认 144.9 对 vLLM 98.3，差 46.6 ms。**约 32 ms（68%）是这个配置问题**（默认 → 加 1,152 档），**剩下 14 ms 是 M\* 的残留**，与其他长度一致：1,024 慢 10.5 ms（106.5 对 96.0），2,048 慢约 13–15 ms（164.6 对 151.6），56 token 反而 M\* 快 6 ms（52.2 对 58.2），512 持平（71–80 对 71.5）。kernel 级残留与第三轮 nsys 的「第 1 块比 vLLM 慢 5–10%」同向。**M\* 不是在所有长度上落后**：短提示领先或持平，≥1,024 才落后 10–15%。
  - **加 1,152 档的代价与适用范围**：它在每个测过的长度上都不比默认差（56 / 512 / 1,024：52 / 76 / 107 对 53 / 71–79 / 107；1,072：113 对 145；2,048：166 对 190），代价是多一档图的显存，把 2,048 图挤掉了（2,048 的 prompt 改走 eager，速度几乎不变）。**没测**：并发（本批全是 C = 1，这个档在多请求下是否仍好未知，2,048 图被挤掉对 bs > 1 的 prefill 有什么影响也未知）、1,024–2,048 之间的其他长度、1,152 以外的档位。
  - **局限**：每格 1 个 start，没有 A/B/A（默认臂的重复是唯一的一次顺序检查）；提示长度是名义值（正文 + 48 token 模板，harness 不断言服务端长度）；**捕获清单是这块 48 GB 卡的状态，H200 / 80 GB 卡上默认梯度可能是完整的，不能把这个机制外推过去**（H200 当前被占，没验证）；运行时 GPU 0 / 1 被别人的训练占着，M\* 用 2、3、4 号卡（`allowed_devices` 已限定为 2–7，§13），主机 256 核、负载约 5.5，没有证据显示有影响但没有隔离过；这批 cell 的 `all_requests_started_at_full_clock` 与此前的批次一样都是 False，温度 44–51 °C。

- **这是否是缺陷、是否能靠配置修好，现在都不知道**：之前写的「小预算是并发下的权衡」没有证据支持，也没有被反驳——C=1 测试看不到并发。能说的是：1,072 这个夹具长度比较不利（正文 1,024 token 恰好顶着预算，再加 48 token 的服务端模板就跨了过去），不能把 +40–47 ms 当作 M\* prefill 的通用速度差。
- SGLang 的两个 rank 的 busy 列互相矛盾（同形状的 burst，14 与 103 ms），这一轮的 SGLang 数据不作结论。

**（2）vLLM 开缓存处理音频「慢 414 ms」是预热不足，SGLang 开缓存慢 30 ms 是真的（`qwen-a2t-cache-warm`，三遍预热后第四遍测量，每格 2 starts × 10）**

| arm | TTFT（E2E，1 个输出 token）ms |
|---|---:|
| vLLM 关缓存 | 75.9 |
| **vLLM 开缓存** | **77.4**（上一版 481） |
| SGLang 关缓存 | 89.7 |
| SGLang 开缓存 | 121（20 个请求全部 115–130，稳定地慢 31 ms） |

- vLLM 开缓存 481 → 77 ms：**是首见 shape 的预热问题**（缓存路径上尾部 token 数每条不同，第一遍和第二遍都没覆盖到），三遍预热后与关缓存相同（+1.5 ms）。**上一版写的「这一版 vLLM-Omni 开 Thinker prefix cache 处理音频有问题」撤回**。两个 vLLM arm 的 server.log 里 JIT 警告都是 5 行（启动阶段）。
- SGLang 开缓存在预热充分的情况下稳定慢 31 ms（比 10 个请求的分布窄得多），不是预热问题；日志里没有线索，原因没查。它不影响公平档（公平档里 SGLang 关 radix）。

**（3）vLLM 的 T2S 帧循环为什么每音频秒慢（`qwen-t2s-nsys-warm`，5 个预热后的请求，GPU 2 = Talker + Code2Wav）**

| | 音频 s | GPU 2 busy ms | 每音频秒 | launch | 每音频秒 | Talker span ms | 每音频秒 | busy / span |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| M\*（Talker 127 + Code2Wav 5 + outside-node 40） | 3.66 | 172 | **47.0** | 29,899 | **8.2k** | 475 | **130** | 27% |
| vLLM（`talker` 一个进程） | 3.10 | 188 | **60.6** | 47,881 | **15.4k** | 685 | **221** | 27% |

- **vLLM 每音频秒多发 1.9× 的 kernel、busy 多 1.29×、span 长 1.7×**，两边 busy 都只占 span 的约 27%：**帧循环是 launch 受限的，墙钟随 kernel 个数走**。这与 `t2s-fit` 里首个音频之后每音频秒 100 对 54 ms（1.85×）的比例一致，所以那「另外的 1.27×」的机制是 **vLLM 的 Talker / Code2Wav 每帧发的 kernel 多近一倍**（没有融合或没有图），而不是单个 kernel 慢。预热行下 vLLM 的 Talker launch 从冷行的 57.3k 降到 47.9k，仍然最多。
- 限制：nsys 把 T2S 的主机时间放大 1.6–2.2×（M\* 最多），所以只读结构，不读绝对值；vLLM 的 Code2Wav 在 nsys 里没有单独记录（并在 `talker` 进程里）；SGLang 的 `talker_ar` 单独 146 ms / 25,408 launch，它的 Code2Wav 没有被记录，**不与上表同口径比较**。每个 kernel 属于帧循环的哪一步没有拆（要逐 kernel 名对比，未做）。

**（4）TP=2 前缀缓存崩溃：配置还是 bug（核对过的证据）**

- 配置通过了 M\* 自己的全部加载期检查（`prefix_cache_tp` 不能与 `cpu_offload_pages` 同用、缓存的节点必须在同一个 TP 组、无副本）：两个 worker 日志都有 `KV thinker_kv: prefix cache open for main on Thinker`。
- 第一个请求正常（无命中），**第二个相同提示是第一次缓存命中**，随即在 follower（worker_1）上 `RuntimeError`（`mstar/engine/resources/kv/manager.py` 的 `adopt_cached_prefix`：`chain is None or stream.stored_len`），被 worker 当作 `TP ranks diverged` 退出（`micro_scheduler.py:862`），整个服务关闭。rank 0 租了 8 页 = 8 × 128 = 1,024 token = **恰好第一个 chunk 的长度**。
- M\* 的 `test_kv_prefix_tp.py`（14 个，包含「分块行从 rank 0 测得的前缀之后开始」）本机通过；它用 CPU 桩，不经过 Qwen3-Omni 的 API 侧前缀键计算和 `thinker_mixed` 组合 walk。官方 TP 配置（`configs/qwen3omni_full_tp2.yaml`、`qwen3omni_thinker_tp2.yaml`）都不开 `prefix_cache_tp`；M\* 默认在 TP 下关缓存（日志 `prefix cache off at world size 2`）。
- 结论：**不是我们的配置错误，是 M\* 在 opt-in 路径上的缺陷**；官方「多卡并行」的 TP 路径本身没问题（我们的 TP=2 Thinker 全程正常），只是 TP + 前缀缓存这一组合没有被端到端覆盖。我们没有读 M\* 里 follower 的 chain / stored_len 为什么不对，所以**根因没有定位**。该向上游报，附 server.log 的那三行。
- **`tp-cache-probe` 结果（`MSTAR_TP_CACHE_PROBE_TP_CACHE_CHUNK4096_01_B2B`，TP=2 + `prefix_cache_tp` + offload 0 + Thinker 预算 4,096，其余同崩溃配置）**：3 个预热 + 10 个测量请求全部成功，server.log 0 条 `TP ranks diverged`，两个 worker 都打出 `prefix cache open`。**读法：崩溃的触发条件是「前缀命中 + 提示被切成多个 chunk」；整段放进一个 chunk 就不崩。** 范围缩小到 follower 上分块行与前缀采用（`adopt_cached_prefix`）的交互，与单测里「分块行在 rank 0 测得的前缀之后开始」那一类路径最相关（假设，未读代码证实）。
- **这个结果的限制**：①M\* 没有逐请求的命中日志，缓存命中是**间接推断**的：同一份 TP=2 配置、同样预算 4,096，缓存关（`prefill-chunk` 的 chunk-4096 臂）TTFT 153.8 ms，缓存开（本探针）约 70 ms，省约 84 ms，与「命中 1,024 token、只算 48 token」一致；但这是两个分别跑的实验，不是同一次 A/B，探针只有 1 个 start。②只有 1 个 start，drift 标记 `suspect_intra_start_drift`（最后两个请求降到 60 ms），`clean` 0%（SM 时钟 2,463 MHz 低于标称 98%），所以 70 ms 只用作「大约在这个量级」，不当精确数。③不崩不等于 TP 前缀缓存可用：提示超过一个 chunk 就会崩，真实请求常常超过 1,024 token。

### 9.7 第四轮补测的结果（`qwen-prefill-chunk`、`bagel-prefill-chunk`、`bagel-connector-nsys` 旧三臂、`mstar-tp-cache-probe`，已跑）

补跑前三组因为 GPU 被别人占用（launcher 日志里的 `needs N free GPU(s) but only 0 are free`、`prestart idle gate failed`）失败过一次，`--resume` 之后全部完成，`analyze` 状态 0。

**（1）`qwen-prefill-chunk`：预算 4,096 没有让 1,072-token 的 TTFT 变快，反而变慢**

| case | 臂 | TTFT p50 (ms) | E2E p50 (ms) | 备注 |
|---|---|---:|---:|---|
| txt-1k-1tok（服务端 1,072 token） | chunk-1024（默认） | 138.3 | 147.7 | 1 start，10 个测量请求，drift flat |
| txt-1k-1tok | chunk-4096 | **153.8** | 162.8 | **慢 15.5 ms（1.10×）** |
| txt-tiny-1tok（服务端 56 token） | chunk-1024 | 52.4 | 61.7 | |
| txt-tiny-1tok | chunk-4096 | 53.2 | 62.5 | 不动（1.01×），预算只影响分块 |

预测是 TTFT 降约 40 ms。**预测被推翻**：见 §9.6（1）、§0。限制：每臂 1 个 start、A/B 各一次，没有 A2，所以 15 ms 的差距没有排除顺序漂移（两个 tiny 臂只差 0.8 ms，说明机器在两个臂之间是稳的，但这不是对 txt-1k 的直接检验）。没有 chunk-4096 的 nsys，不知道它是不是一次 prompt forward、多长。

**（2）`bagel-prefill-chunk`：BAGEL 的 LLM 预算 1,024 → 8,192，几乎没有效果**

| case | 臂 | 起点数 | TTFT p50（r01 / r02，ms） | E2E p50 (ms) |
|---|---|---:|---|---:|
| i2t-512 | chunk-1024 | 2 | 139.6 / 130.0 | 2,288 |
| i2t-512 | chunk-8192 | 2 | 132.2 / 123.8 | 2,282（1.00×） |
| i2i-512-srv-steps5 | chunk-1024 | 2 | - | 2,739 |
| i2i-512-srv-steps5 | chunk-8192 | 2 | - | 2,734（1.00×） |

I2T TTFT 两个起点各降约 6–7 ms（~5%），占 M\* 对 vLLM 约 44–49 ms 差距的 15% 以内；I2I 不动。**D2 剩下的约 40 ms 不在 chunk 预算。** 四个 cell 全部在 GPU 2 上跑（r02 先跑完，r01 因为 GPU 被占而补跑，时间上晚了几个小时）；两个臂里 r01 都比 r02 高约 9 ms，所以 r01 / r02 之间有约 9 ms 的起点效应，比较要在同一个 repeat 内做（两个 repeat 内部的差都是 −6 到 −7 ms）。

**（3）`bagel-connector-nsys` 旧三臂补跑**：E2E 复现了之前的 `connector` 实验（nsys 没有扰动）：

| 臂 | 本次 E2E (ms)，n=4 | 之前（`connector` / `connector-nsys` 前）|
|---|---:|---:|
| single-stage | 2,913 | 2,936 |
| two-stage SharedMemory | 6,872 | 6,809 |
| two-stage Mooncake-TCP | 4,578 | 4,558 |
| two-stage Mooncake-TCP + Thinker 前缀缓存 | 3,533 | （新臂）|
| M\* | 2,828 | 2,761（`i2i-fair`） |

DiT 一侧（flash attention + `mot_unified_gemm`，10,239 次 launch）在所有 two-stage 臂里 busy 约 1.78 s，不随 connector 变；single-stage 单进程 busy 2.64 s（16,605 次 launch），M\* 单进程 2.65–2.70 s（8,990 次）。**Thinker 一侧的 burst 在这几个臂里不是「一个请求一个 burst」**（5 个 burst 的 busy 在 0.76–2.0 s 之间跳动），所以我没有新增 Thinker 的逐 burst 结论；§7.7（3）里「缓存命中让 Thinker busy 从约 2.0 降到约 1.1 s」沿用，本次旧三臂没有推翻它，也没有提供更精的数。

**（4）`mstar-tp-cache-probe`**：见 §9.6（4）。TP=2 + 前缀缓存、提示一个 chunk：13 个请求成功，0 条 `TP ranks diverged`。

## 10. 仍然开放（哪些能用实验关，哪些关不了）

**公平档实验覆盖（已跑；"还缺什么"一列对应 §12 的补测）：**

| 问题 | 实验 | 答案 | 还缺什么 |
|---|---|---|---|
| Qwen TXT / A2T 在缓存状态相同、SGLang lookahead 可用时，M* 还领先吗？ | `qwen-fair-txt` / `qwen-fair-a2t` | **不领先**（TXT M* 慢 2–3%；A2T 对 SGLang 1.09×、对 vLLM 0.98×）；SGLang 的 lookahead 对照 ITL 5.82 → 8.00；pinned 两臂：去掉 seed / min_tokens 对 vLLM、M* 无影响 | M* TXT 开缓存崩溃，无 on 数字（M* 的 bug，没有实验能补） |
| Qwen TXT 的 M* TTFT 慢 40–47 ms 在哪 | `qwen-ttft-txt` + `qwen-prefill-nsys` + `qwen-prefill-chunk` | **约 2/3（约 32 ms）是默认梯度对 1,072 不合适：默认是 1,024 + 48 两次 forward，预算 4,096 是一步但被垫到 2,048 的图，加一档 1,152 的图后 TTFT 145 → 113 ms；约 15 ms 是 M\* 在 ≥1,024 token 的 kernel 级慢 10–15%**（§9.6（1）） | 剩下的实验见 §12 | 
| A2T 对 vLLM 的首 token | `qwen-ttft-a2t` | TTFT：M* 93 / SGLang 92 / vLLM 79 ms；音频提示约 190 token 只有一块，**不是 chunk 预算**；M\* 的第 1 块比 vLLM 慢 5–10%（MoE、all-reduce），加上 forward 前约 8.6 ms 的 kernel（原因没查），合计和 14 ms 同量级 | 不查（太小） |
| T2S 对 vLLM 的 1.27× | `qwen-t2s-fit` + `qwen-t2s-nsys-warm` | **领先保留**（1.58× 每音频秒）；stall 预热后为 0，greedy Talker 不改变；**nsys：vLLM 每音频秒多发 1.9× 的 kernel（15.4k 对 8.2k），两边 busy 都只有 span 的约 27%，launch 受限**（§9.6（3）） | 每个 kernel 属于帧循环的哪一步没拆（要逐 kernel 名对比；见 §12 的可选项） |
| BAGEL I2T 的采样器不对等 | `bagel-sampling-parity` | 重复惩罚无影响；两边缓存都开 vLLM 快 1% | - |
| BAGEL I2I 剩下的约 1.6 s | `bagel-connector-timeline` / `bagel-connector-nsys` / `bagel-i2i-fair` | **拆开了（§7.7）**：hand-off 约 0.28 s、接收侧很快；Thinker 的第二次 prefill 1,041 ms 被 vLLM 的 prefix cache 抵掉（Thinker busy 2.05 → 1.12 s）；M\* 的缓存开关无影响，M\* 的 busy 2.65 s ≈ single-stage，**是一次 prefill 的工作量**；缓存开后剩下的 763 ms ≈ Thinker GPU 0.2 s + hand-off 0.22–0.26 s + 前端约 0.3 s | `connector-nsys` 的旧三臂只有 2 个测量请求，需补跑（§12）；M\* 前端没有单独测 |
| BAGEL I2I 的 1024² | `bagel-connector-1024` | 节省与 KV 字节同比例（1.52×）；connector 占 54%；Mooncake 剩余 2,867 ms 增长比 KV 更快 | 没有 1024² 的 nsys，不做 |
| vLLM 开 Thinker 缓存处理音频反而慢 414 ms | `qwen-fair-a2t` 的 vllm-on → `qwen-a2t-cache-warm` | **是预热不足**：三遍预热后 vLLM 开缓存 77.4 对关 75.9 ms；SGLang 开缓存稳定慢 31 ms（89.7 → 121）是真的（§9.6（2）） | SGLang 的原因没查，不影响公平档 |
| NIXL 为什么比 Mooncake 慢 | `bagel-connector-timeline` | DiT 侧慢 0.75–1.2 s；`UCX_TLS` 只解释 0.2–0.4 s；1024² 下 NIXL 省 2,093 对 Mooncake 3,414 | 未安排（NIXL 不是默认选择，优先级低；需要 UCX 日志级别才能读） |

**需要改别人的代码才能做反事实，本轮没做：**
- Cosmos3 的 vLLM 主机发射开销（1.17×）：结构证据已经指向，要证明需要给 vLLM-Omni 的 DiT 加 CUDA graph 或 launch 合并。
- Cosmos3 对 SGLang-Omni：SGLang-Omni 有 cosmos3，没跑过。
- T2S Talker 逐帧 kernel 表：`qwen-t2s-fit` 出来后若 vLLM 的 `b` 仍高，再用 `t2s-stage-nsys`（把 measured 提到 10、用预热行）做逐帧对比。

**本机条件限制：**
- H200 上的旧结论（§11）需要同版本重跑，现在 H200 被占。
- Pi0.5 全部关掉后仍领先 SGLang core 约 1.4×，原因未查，也没有审计过它的公平性。
- torch.compile：vLLM 开、SGLang 无、M* 在 5880 上默认关。**关它不是 M* 的硬限制**：当初的失败是 M* 自己的 `_cp_linear_kernel`（H200 调的 tile，要 147456 B 共享内存，sm_89 上限 101376 B），现在 pin 的 M* 有回退，**compile-on 能跑**，见 §7.5：Qwen 在本机 compile 没有收益（TXT / A2T 慢 3–4%，T2S 每秒音频持平），所以公平档不需要为 compile 重跑。H200 上 compile 的收益没有在本机验证，不能外推。

**不可能被"实验"证明或证伪的：**
- Graph + Walk 的独立贡献：没有开关能拆掉图抽象。现在能说的只有：发射链的**实现**用了图的声明做推测（§2.2），但它**产生的效果**vLLM、SGLang 都有；BAGEL I2I 的 stage 边界是一个"stage 框架付、M* 不付"的代价，但 vLLM single-stage 同样不付。

## 11. 对旧 H200 结论的更新

| 旧结论 | 本机 | 处理 |
|---|---|---|
| M* BAGEL 每步 GPU busy 少 8–20% | 无差异（≈1%） | 在 H200 上同版本重跑 |
| T2I 512² M* 1.18× / 1.10× | 1.01–1.03× | 同上 |
| 「launch 数少」是杠杆 | 不转化成速度 | 不再作为解释 |
| A2T M* 慢 | M* 快 1.08–1.26×（H200 NVL 不在 M* 的 MoE 调优表里，旧结论可能是没调优） | 重验 |
| T2S compile ≈ +20% | 本机 5880 上 compile-on 对 T2S 每秒音频持平（−1.6%），TXT / A2T 慢 3–4%（§7.5）；H200 的 +20% 本机没复现 | H200 空了同版本重跑 |
| BAGEL I2T M* 领先 | **TTFT 领先是 vLLM-Omni 上游默认关 prefix caching；开了之后 vLLM 更快** | 用同缓存状态重比 |
| BAGEL I2I M* 领先 | stage 边界的 KV 交接；默认 connector 占固定成本差的约 58%，Mooncake 后 1.10×（§7.4）；single-stage 只有 1.2% | 区分部署和 connector |
| 「M* 解码比 SGLang 快 = 发射链」 | **SGLang 的 lookahead 疑似被我们请求里的 seed / min_tokens 关了** | 待 `qwen-fair-*` 实验 |
| 「只有 T2S 发射链有直接证据；graph+walk 无对照」 | 仍成立；发射链是通用的重叠机制，vLLM 与 SGLang-Omni 都有 | 保持 |

H200 空出来后，同一份 M* 和 vLLM 版本：
```bash
envs/mstar/bin/python -m exps.suite run bagel-nsys-equal-work mstar-ablation-qwen-t2s baseline-bagel --hardware h200nvl --protocol back-to-back --execute
```
H200 上跑 Qwen TXT / A2T 的对 SGLang 比较时，也要用 `qwen.sampling=lookahead-eligible`，否则同样的门控问题会再出现。

## 12. 已做 / 待做的命令

已完成（本机结果已读入本文）：`prefix`、`cosmos-nsys`、`i2i-nsys`、`vllm-prefix`、`i2i-steps`、`connector`、`connector-timeline`（含 `mooncake-tcp-prefix` 臂与 1024²）、`mstar-compile`、`bagel-parity`、`fair-txt`（含 pinned）、`ttft`、`t2s-fit`、`fair-a2t`、`i2i-fair`、`prefill-nsys`、`a2t-cache`、`t2s-nsys`、`connector-nsys` 的 `mooncake-prefix` 与 `mstar` 两臂。

**第四轮补跑已完成**：`connector-nsys` 旧三臂、`prefill-chunk`、`bagel-chunk`、`tp-cache-probe` 都跑完并分析（§9.7、§9.6（4））。其中 `prefill-chunk` 的结果推翻了预测，所以又加了两个实验判定机制。

**第五轮（`prefill-len`、`prefill-chunk-nsys`）已跑完并分析**（结果与解读在 §9.6（1））。第一次启动因为别人的训练占着 GPU，大部分 cell 没跑成（`run_attrib.sh` 默认只等 30 分钟，且调度器在 0 / 1 号卡刚好空出来的瞬间选中它们，随后 idle gate 一直等一块别人在用的卡）；之后把 rtx5880 的 `allowed_devices` 限定为 2–7，用 `EXPS_GPU_WAIT_S=14400` 重跑，两个实验各 5 个 / 3 个臂全部完成（`prefill-len` 25 个 cell，`prefill-chunk-nsys` 3 个）。

**第六轮补测：7 个实验已定义（`catalogue.py`）、`plan` 能解析、309 个测试通过，还没跑。** 第五轮的 nsys 矛盾已经解开（§9.6（1）：nsys 服务器没有 2,048 图），所以原来的「重跑 nsys A/B/A」换成了「让 nsys 下也有 2,048 图」。

| 组（`run_attrib.sh`） | 实验 | 要回答什么 | 判据 | 约 |
|---|---|---|---|---|
| `nsys-room` | `qwen-prefill-nsys-room` | 被垫到 2,048 的那次 forward 在 GPU 上多长（预算 4,096，Thinker KV 池减半让 nsys 下也捕获 2,048 图；nsys / 无 nsys 各两次） | 日志里必须有 `Captured ... prefill_text[bs=32,tokens=2048]`（Expect 强制，没有则 cell 被拒）；无 nsys 臂 E2E 应 ≈ 160–168 ms（仍被垫）；nsys 臂 E2E ≈ 无 nsys 臂 = nsys 不再改路径；nsys 的 GEMM / 注意力约 2× 对比 eager 1,072 的（span 105–112、busy 85–93、MoE 34 ms）= 垫桶是 GPU 上的真工作 | 35 分钟 |
| `fair-txt-b1152` | `qwen-fair-txt-b1152` | 公平档 TXT（缓存关）里 M\* 慢 2.4% 的标题数字，在 M\* + 1,152 档下还剩多少（256 与 1 个输出 token，2 starts，SGLang / vLLM / M\* 默认 / M\* + 1,152 档） | `mstar-b1152` E2E 在 vLLM 的 ±1% 内或更低 = TXT 落后是配置；仍慢约 2% = kernel 残留；ITL 应与 `mstar-off` 相同 | 1.5 小时 |
| `len-repeat` | `qwen-prefill-len-repeat` | 512 / 1,072 token 的 start 间离散（默认与 +1,152 档各 3 个 start） | 512 的离散 ≥5 ms = 长度扫描里 <8 ms 的差别不读；1,072 的 +1,152 档在 3 个 start 里都低于默认 = 32 ms 不是一次运气 | 1 小时 |
| `len-dense` | `qwen-prefill-len-dense` | 1,100 / 1,300 / 1,536 / 1,800 token 的 TTFT：默认 / 预算 4,096 / +1,152 / +1,152 +1,536 / vLLM；1,536 档能不能被捕获（日志） | 默认应是阶梯（两次 forward）；预算 4,096 平在约 155–165 ms（垫到 2,048）；+1,536 档的 1,300 / 1,536 降下来 = 第二档有效；日志没有 `tokens=1536` = 这块卡放不下 | 1.5 小时 |
| `len-sglang` | `qwen-prefill-len-sglang` | SGLang-Omni 在 56 … 2,048 的 TTFT（缓存关），和 M\* / vLLM 同一组长度 | SGLang 是直线（像 vLLM）还是有阶梯；1,072 处应在 117 ms 附近 | 35 分钟 |
| `pi05-nsys` | `pi05-nsys` | Pi0.5：M\* 104 / SGLang core 164 / vLLM eager 220 ms 的差在哪（GPU busy 对 span） | busy 相同、span 更长 = launch / host 间隙；busy 不同 = 看 top kernel（视觉编码器 / VLM prefill / 10 步 action expert） | 30 分钟 |
| `orpheus-nsys` | `orpheus-nsys` | Orpheus b1、b16：M\* 对 VoxServe 的 1.09× → 1.38× 差在哪（每步 busy 与间隙，随 batch 的变化） | 同上；b1 对 b16 看间隙是否随 batch 增长 | 30 分钟 |

```bash
export EXPS_GPU_WAIT_S=14400      # 等 4 小时空卡；先 nvidia-smi 看 2–7 号卡里有 3 张空闲
# 第一批：决定 TXT 结论（约 3 小时）
./run_attrib.sh nsys-room fair-txt-b1152 len-repeat
# 第二批：梯度形状（约 2 小时）
./run_attrib.sh len-dense len-sglang
# 第三批：其他模型的归因（约 1 小时）
./run_attrib.sh pi05-nsys orpheus-nsys
```

**第七轮补测：Pi0.5 与 Cosmos3 的代码和实验已加好（`plan` 能解析，CPU 上验证了两个 hook 和 SGLang-Omni 配置，310 + 9 个测试通过，未上 GPU）。**

Pi0.5 的源码审计先给出了三个「不是引擎」的差异：①vLLM-Omni 的 `modeling_pi05._attend` 是手写的 matmul + softmax（fp32 upcast），与 `--enforce-eager` 无关，是模型的写法；②M\* 把 SigLIP（bs 1 / 2 / 4）、prefill、10 步 action expert 都放进 CUDA graph，并 `torch.compile`（`max-autotune-no-cudagraphs`）；③SGLang core 有 prefix / action 两个独立的 CUDA graph 开关，基线里都开着。dtype 并不对齐：vLLM-Omni 把 PaliGemma 之外的参数（action_in/out_proj、time MLP）全部放在 fp32（`pipeline_pi05.py:82-108`），nsys 里的 `ampere_sgemm` 就是这部分；M\* 只把 norm 保持 fp32（`submodules.py:301-310`）；SGLang core 的 dtype 没有逐层核对。该说法在 2026-11-14 的复核中更正。

| 组 | 实验 | 内容 | 判据 | 约 |
|---|---|---|---|---|
| `pi05-audit` | `pi05-audit` | 10 个臂 × 2 starts：M\* 默认 / `nocompile` / `nograph`（hook，不捕获任何图）/ `eager`（两者都关）；SGLang core 默认 / `nograph`（两个图开关都关）；vLLM-Omni 默认 / `sdpa`（hook，注意力走 `F.scaled_dot_product_attention`）/ `graph`（enforce_eager 关）/ `sdpa-graph` | 框架内：每个开关各占多少；框架间：`mstar-eager` 对 `vllm` 是「都不用图和 compile」的同口径比较，`vllm-sdpa` 对 `sglang` 是「vLLM 换成现代注意力内核」的比较。<5% 的开关不算差距的一部分。hook 臂必须有 `hook_proof.jsonl`（`applied: true`），否则 `analyze` 拒绝出数 | 1.5 小时 |
| `cosmos3-sglang` | `cosmos3-sglang` | 256² 三家：M\* / vLLM-Omni / **SGLang-Omni**（新加进 harness：`Cosmos3PipelineConfig` 直接生成流水线，原生 `/v1/images/generations`，guardrails 关，与 vLLM 的 `guardrails: false` 对等） | SGLang 在两者之间 = 结构解释在第三个实现里也成立；接近 vLLM = 差距是 M\* 做的事（批量 CFG、compile）；不高于 M\* = vLLM-Omni 的 Cosmos3 流水线本身慢 | 30 分钟 |
| `cosmos3-size` | `cosmos3-size` | 256² / 512² / 1,024²，三家（新加 `t2i-512`、`t2i-1024`，不进标准 profile） | 对 vLLM、对 SGLang 的比值随尺寸怎么走：文档预期差距随 GEMM 变成算力受限而缩小；1,024² 仍约 2× = 领先不是 launch 受限。M\* 的 denoise 图有 `graph_max_latent_area: 2000` 的上限，大尺寸在 M\* 里 eager | 2 小时 |
| `cosmos3-sglang-nsys` | `cosmos3-sglang-nsys` | M\* 默认 / vLLM-Omni / SGLang-Omni 的 nsys | 每请求的 DiT forward 数（≈50 = 批量 CFG，≈100 = 两个分支顺序跑）、GPU busy 对 span | 30 分钟 |
| `bagel-conc-fair` | `bagel-concurrency-fair` | BAGEL I2T B = 1 / 4 / 16，M\*（缓存默认开）对 vLLM two-stage（Thinker 开 prefix cache） | `analyze_concurrency` 的 wave 时间与吞吐扩展曲线，两边缓存状态一致 | 1 小时 |
| `t2s-conc-fair` | `t2s-concurrency-fair` | T2S B = 1 / 4 / 16，行预热（`qwen.rows=cycle`，10 个 precondition wave），三家 | 与冷行的 `t2s-concurrency` 对照：B = 16 的数字掉了多少 = 之前量到的 JIT | 2 小时 |

```bash
export EXPS_GPU_WAIT_S=14400      # 先 nvidia-smi：2–7 号卡里要有 3 张空闲（Qwen）；Pi0.5 / Cosmos3 / BAGEL 用 1–2 张
# 第一批：Pi0.5 与 Cosmos3 的归因（约 2.5 小时）
./run_attrib.sh pi05-audit cosmos3-sglang cosmos3-sglang-nsys
# 第二批：Cosmos3 的尺寸（约 2 小时）
./run_attrib.sh cosmos3-size
# 第三批：BAGEL / T2S 的公平档并发（约 3 小时）
./run_attrib.sh bagel-conc-fair t2s-conc-fair
```

**这批没有做、以及原因**：V-JEPA2-AC 的 Meta native `torch.compile` / CUDA graph 臂（要给 `meta_native_client.py` 加编译分支，编译行为必须上 GPU 才能验证，没法保证一次改对）；Orpheus 对 VoxServe 的 serve 参数审计（只有 `orpheus-nsys`）；BAGEL 的 M\* 前端计时（要改 M\* 的代码）；T2S 逐 kernel 对比（`analyze_nsys_t2s_stages.py` 的窗口问题，上轮试过与旧表不一致）；Pi0.5 三家的输出等价检查（噪声是各框架自己随机采样的，不固定种子不能比 action，三家的采样接口不一致，不能在不改框架的情况下对齐）。

**其他模型还缺什么（Orpheus / V-JEPA2-AC / Pi0.5 / Cosmos3 / BAGEL）**——这几个模型的领先没有对手侧的归因，§1 的总表里甚至没有 Orpheus、V-JEPA2-AC、Pi0.5 的行：

| 模型 | 现状 | 缺什么 | 现在能跑吗 |
|---|---|---|---|
| Pi0.5 | M\* 104 / SGLang core 164 / vLLM eager 220 ms；M\* 全关消融仍 120 ms（领先 SGLang core 1.4×，原因未查）；公平性**没审计过** | ①nsys（`pi05-nsys`）；②**对手配置审计已做（第七轮）**：源码审计的三个差异与对应的 `pi05-audit` 10 臂；③三家 denoise 步数与 dtype 已对等；输出等价检查没做（见上） | ①②能 |
| Orpheus | b1 / b4 / b16：M\* 领先 VoxServe 1.09 / 1.30 / 1.38×，随 batch 增大；采样已对等（temperature / top_p / repetition_penalty / max_tokens） | ①nsys（`orpheus-nsys`）；②VoxServe serve 参数审计：CUDA graph、`max_batch_size`（现取 max(batch, 8)）、dtype；③传输不对称（M\* base64 NDJSON，VoxServe WAV）的客户端开销没有单独量化 | ①能；②③要读代码 / 加探针 |
| V-JEPA2-AC | M\* 对 Meta native 1.7× / 2.9× / 4.3×（H = 4 / 15 / 30）；两边都是 bf16 autocast；M\* 计 HTTP + multipart + base64，Meta 进程内；Meta 是未编译的 eager 脚本 | ①Meta native 的 `torch.compile` / CUDA graph 臂（`meta_native_client.py` 没有这个选项，**要加代码**）：判断倍数里多少是「eager 对编译」；②M\* 的 HTTP 开销已有 transport probe，量级未核 | **不能**：先给 `meta_native_client.py` 加 `--compile` |
| Cosmos3 | 256²：对 vLLM 2.46× 已拆开（批量 CFG 1.54× × compile 1.38× × 主机空闲 1.17×）；对 SGLang-Omni 没跑 | ①对 SGLang-Omni：**适配已加（第七轮）**，`cosmos3-sglang`、`cosmos3-sglang-nsys`；②更大尺寸：**`t2i-512`、`t2i-1024` 已加**，`cosmos3-size` | 能（SGLang 臂首次在 harness 里跑，起不来先读 server.log） |
| BAGEL | 已拆得最细（§4 – §7）；I2I 剩 763 ms 里 M\* 前端约 0.3 s 没单独测 | ①M\* 前端（图像预处理 / tokenizer）的单独计时；②`bagel-concurrency`（B = 1–16）是缺省配置，没有在「两边缓存都开」的公平档下重跑：**已加 `bagel-concurrency-fair`（第七轮）** | ①要改 M\* 的代码，暂缓；②能 |
| Qwen T2S / A2T | T2S 领先保留（1.24× / 1.58× 每音频秒）；A2T 与 vLLM 持平 | ①T2S 每个 kernel 属于帧循环的哪一步（逐 kernel 名对比，CPU 分析，要把 `analyze_nsys_t2s_stages.py` 的窗口定在单请求内）；②`t2s-concurrency` 在公平档（预热行）下重跑：**已加 `t2s-concurrency-fair`（第七轮）** | ①窗口问题未解，暂缓；②能 |
| 全部（H200） | §11 的旧结论需要同版本重跑；H200 当前被占 | 需要 H200 版本的 M\* 配置（`config.mstar=` 现在指向 rtx5880 的 yaml），再跑 `qwen-prefill-len`：看 H200 / 80 GB 上默认梯度的 2,048 图是不是完整捕获（本节的机制是 48 GB 卡的状态） | H200 空出来后；命令见 §11 |

**不是实验、但要做的**：上游（M\*）值得知道两点：①48 GB 卡上默认只捕获一张 bs = 32 的 2,048 text prefill 图（bs = 1 ~ 16 的没捕获，只有一行 ERROR 日志），bs = 1 的步按规则借用它，被垫到 2,048 个 token，1,072 这类长度在预算 4,096 下比不垫的 1,152 图慢 43 ms；`MSTAR_REQUIRE_ACCELERATOR_GRAPHS=1` 能让缺图的启动直接失败，而不是静默走慢路径。②默认 ladder 没有 1,152 / 1,536 这类中间档，且 `prefill_token_buckets` 增加一档会把那张 2,048 图挤掉（显存）。

下面是第五轮之前写的预先判据，保留作记录：

| 实验 | 结果 → 结论（预先登记，已读完，见 §9.6（1）） |
|---|---|
| `prefill-chunk-nsys` | 预算 4,096 的臂：**attention launch 数**判断是不是一次 prompt forward（**是，48 次**）；forward 的 GEMM / 注意力 kernel 毫秒与默认第 1 块比：约 2× = 垫到 2,048 桶（**没有被检验：nsys 下没有 2,048 图，测到的是 eager 的 1,072；`nsys-room` 补**）；整体仍慢 = 慢在别处（**nsys 下不慢，因为 nsys 服务器没有 2,048 图、按 eager 跑；见 `nsys-room`**）。 |
| `prefill-len` | M\* 在 512 / 1,024 与 vLLM 持平而 1,072 落后 = 边界位置的现象（**部分成立：512 持平，1,024 慢 10 ms，1,072 慢 47 ms**）；`chunk-4096` 在 1,072 比默认慢、在 2,048 更好 = 垫桶解释（**数字符合，机制就是垫到 2,048 的图**）；`chunk-4096-b1152` 在 1,072 追上 vLLM（**追上约 2/3：113 对 98 ms**）；`chunk-1024-again` 对默认看顺序漂移（**1,024 / 1,072 / 2,048 差 0.2–3.9 ms，512 差 7.4 ms**）。 |

提示长度由夹具文本裁剪 / 循环得到，正文 = 目标 − 48 个模板 token（来自 SGLang 日志里 `#new-token: 1072` / `56`）；harness 不断言服务端长度，所以 `prefill-len` 的「512 / 1,024 / 2,048」是名义值。

**可选的 CPU 分析（不占 GPU，没做）**：T2S 的 vLLM 对 M\* 逐 kernel 名对比，看每帧多出来的 1.9× launch 是哪一步（Talker 主干、code predictor 还是 Code2Wav）；需要把 `analyze_nsys_t2s_stages.py` 的窗口定在单个请求内，我这轮拿 burst 窗口试过，窗口和之前的表不一致，所以没有写进结论。

**不安排实验的**：M\* TP=2 开 Thinker 缓存的根因（`tp-cache-probe` 已缩小到「命中 + 提示多于一个 chunk」）（需要在 M\* 里加日志，是上游的事；我们已经把触发条件、日志、源码位置整理成 §9.6（4），应当向上游报）；NIXL 为什么慢（不是默认选择，需要 UCX 日志级别才读得出）；SGLang 开缓存音频慢 31 ms（不影响公平档）；1024² 的 nsys；H200（被占）。**还没办法关的混杂**：50 步 I2I 里 M\* 在 64 °C / 1,249 MHz、vLLM 在 34 °C / 1,834 MHz（不同 GPU 对），要锁时钟才能排除，需要 root；5 步的格子（M\* 49–53 °C、1,610 MHz，vLLM 37–40 °C、2,147–2,430 MHz）同样有这个差，所以 3% 以内的每步差异不下结论。

**两个新增组的前置与读法：**
- `connector`：先 `exps/tools/install_omni_connectors.sh`（本机已装好，幂等）。arm 通过 `--env vllm-omni:PYTHONPATH=~/.local/opt/omni-connectors/site` 才导入它们，基线环境不受影响。读 `bagel-connector_rtx5880.md`：mooncake / nixl 的 E2E 接近 single-stage = 差距是默认 connector，不是两阶段架构；接近 shm = 差距在别处（接收侧、编排、Thinker 对三个分支各做一次 prefill）。connector 起不来是结果，看 server.log。
- `mstar-compile`（已跑，§7.5）：compile-on 能跑且没有收益，公平档不用为 compile 重跑。
- `connector-timeline`：用 `exps/tools/analyze_i2i_timeline.py`（`analyze` 会自动调用）。读法：`stage 0` 与 `pre` 在所有 arm 里应相同；`send` / `DiT side` 的差是 connector 的代价；`nixl-cuda-ipc` 快于 `nixl` = UCX 默认传输不是最优，相同 = 默认已经用了它。

各组都做了什么，一句话：
- **prefix cache 状态**：三家都显式关或开。SGLang 关 = `engine.disable_radix_cache: true`；vLLM 开 = Thinker `enable_prefix_caching: true`；M* 开 = `prefix_cache_tp: true` 且 `cpu_offload_pages: 0`（TP 下不能和 offload 同时用）。每个 arm 都有日志断言（SGLang 调度器日志里的 `#cached-token` 是否大于 0、vLLM 日志里 `enable_prefix_caching=True/False`、M* 日志里有无 `prefix cache off at world size`），不满足 `analyze` 拒绝出数。
- **请求**：`qwen.sampling=lookahead-eligible`（去掉 seed、min_tokens；`ignore_eos` + `max_tokens` 仍然钉长度）。`vllm-pinned` / `mstar-pinned` 保留原请求，验证去掉 seed 对它们没有影响。
- **预热**：`qwen.rows=cycle`，precondition = measured = 10（A2T）/ 20（T2S），所以测量用到的每个输入 shape 都已经出现过。
- 如果 M* 在 TP=2 开 cache 起不来，那是一个结果，不重试。

## 13. 工具修复与改动（影响复现）

- `analyze_nsys_burst_gaps.py`：空的 `.sqlite`（失败的导出）现在会被重导；导出总是带 `-o`（2026 版 nsys 不带 `-o` 时静默不写文件）；按 `$NSYS`、`~/.local/opt/nsys-*`、PATH 的顺序找能读该报告的 nsys（发行版自带的 2022.4 读不了 2026 的报告）；多进程的报告按（deviceId，进程）分表（**第一版按 deviceId 分是错的**：见 §13 末尾的更正）。
- `run_attrib.sh`：nsys 路径兼容 `bin/` 与 `target-linux-x64/` 两种布局；新增 `bagel-parity`、`fair-txt`、`ttft`、`t2s-fit`、`fair-a2t` 组。
- 新选项（默认行为都不变，已有 baseline 数字不受影响，manifest 里记录实际值）：`--opt qwen.sampling=lookahead-eligible`、`--opt qwen.rows=cycle`（T2S / A2T 客户端读 `EXPS_QWEN_SAMPLING` / `EXPS_QWEN_ROWS`）、`--opt bagel.sampling=neutral`（BAGEL I2T 请求显式 `repetition_penalty 1.0 / top_k 1 / top_p 1.0`）。
- 新增用例：TXT 的 `txt-tiny-1tok`、`txt-1k-1tok`，A2T 的 `a2t-en-1tok`（只生成 1 个 token，E2E 就是首 token）。
- 新增反事实配置（只经 `--opt config.<family>=` 使用，和各自默认只差一个开关，测试里逐行核对）：vLLM 的 Qwen Thinker prefix cache 开、vLLM Talker greedy、SGLang Thinker radix cache 关、SGLang lookahead 关、M* TP prefix cache 开。
- 新增 `exps/tools/install_omni_connectors.sh`（Mooncake / NIXL 装到用户目录）、`analyze_kv_handoff.py`（从 `server.log` 读每个请求的 KV 序列化 / 传输耗时）、`--opt qwen.mstar_compile=on` / `EXPS_QWEN_MSTAR_COMPILE=on`、`run_attrib.sh` 的 `ATTRIB_RUN_ID` / `ATTRIB_MSTAR_COMPILE`。
- 新增分析工具：`analyze_t2s_fit.py`（E2E / TTFA 对音频秒的拟合 + stall 计数）、`analyze_ttft_parts.py`（首 token 拆成固定 / 每 1k token / 音频额外）。
- 公平档第一轮跑完后的修正：①SGLang 的 cache 检查从"日志里有无 `Init Unified Radix Cache`"改成 `#cached-token` 计数（`Expect` 的 `log_has` / `log_lacks` 现在接受 `re:` 开头的正则）；②`qwen-fair-txt` 删掉 M* on 臂（TP=2 开缓存崩溃，§9.5）；③TXT 的 1-token 用例允许 cap+1（M* 返回 2 个 token），`analyze_ttft_parts.py` 改读 TTFT；④新增实验 `bagel-connector-nsys`、`bagel-i2i-fair` 与配置 `bagel_two_stage_2gpu_mooncake_tcp_prefix_cache.yaml`；`run_attrib.sh` 新增 `connector-nsys`、`i2i-fair` 组。
- **第二轮补测后的修正**：①`analyze_nsys_burst_gaps.py` 原来按 `deviceId` 分表，但 vLLM-Omni 的每个 stage 进程各自设了 `CUDA_VISIBLE_DEVICES`，nsys 里 Thinker 和 DiT 的 `deviceId` 都是 0，两个进程的 burst 被并成一张表（第一次读 `bagel-connector-nsys` 的"GPU 0"是混合的）；现在按（deviceId，进程号）分表，新增 `--handoff`（每个 DiT burst 一行：Thinker 窗口、Thinker 最后 kernel → DiT 第一个 kernel、KV 拷出 → 第一次 H2D、H2D → 第一个 kernel；Thinker 轨迹被截断时打印警告）。②`bagel-connector-nsys` 加 `mooncake-prefix`（Mooncake + Thinker 缓存开）与 `mstar` 两臂，measured 2 → 4；`bagel-connector-timeline` 加 `mooncake-tcp-prefix` 臂。③新增实验 `qwen-prefill-nsys`、`qwen-a2t-cache-warm`（30 个 precondition = 三遍预热）、`qwen-t2s-nsys-warm`（`t2s-stage-nsys` 加预热行）；`run_attrib.sh` 新增 `prefill-nsys` / `a2t-cache` / `t2s-nsys` 三组。④没有改任何已有实验的请求或配置，已有 campaign 的数字不受影响。
- **第七轮新增（已跑，结果见 §14）**：实验 `pi05-audit`、`cosmos3-sglang`、`cosmos3-size`、`cosmos3-sglang-nsys`、`bagel-concurrency-fair`、`t2s-concurrency-fair`，`run_attrib.sh` 组 `pi05-audit`、`cosmos3-sglang`、`cosmos3-size`、`cosmos3-sglang-nsys`、`bagel-conc-fair`、`t2s-conc-fair`；hook `exps/tools/mstar_pi05_nograph_hook`（M\* Pi0.5 不捕获任何图）、`exps/tools/vllm_pi05_sdpa_hook`（vLLM-Omni Pi0.5 的 `_attend` 换成 SDPA，CPU 上与原实现比：fp32 5e-7，bf16 一个 ulp）；配置 `sglang_core/pi05_pipeline_graph_off.json`、`sglang_omni/cosmos3_direct.yaml`（用 SGLang-Omni 自己的 `ConfigManager.from_file` 解析通过）；Cosmos3 任务新增框架 `sglang-omni`（端口 19363，`SGLANG_DISABLE_COSMOS3_GUARDRAILS=1`）与 case `t2i-512`、`t2i-1024`，客户端按 case 校验尺寸；测试 `exps/tests/test_round7.py`；`test_cosmos_cell_plan` 因 `COSMOS.frameworks` 多了 `sglang-omni` 自动覆盖新框架。
- **第六轮新增（已跑，结果见 §14）**：实验 `qwen-prefill-nsys-room`、`qwen-fair-txt-b1152`、`qwen-prefill-len-dense`、`qwen-prefill-len-sglang`、`qwen-prefill-len-repeat`、`pi05-nsys`、`orpheus-nsys`（`catalogue.py`），`run_attrib.sh` 组 `nsys-room`、`fair-txt-b1152`、`len-dense`、`len-sglang`、`len-repeat`、`pi05-nsys`、`orpheus-nsys`；配置 `qwen3omni_3gpu_thinker_tp2_chunk4096_bucket1152_1536.yaml`（多 1,536 档）、`qwen3omni_3gpu_thinker_tp2_chunk4096_nsys_room.yaml`（Thinker KV 池 512 → 256 页，只为让 nsys 下也捕获 2,048 图）；TXT 的长度用例新增 `txt-1100/1300/1536/1800-1tok`（`PROMPT_LENGTHS`，CPU 上已核对能构造出精确的服务端 token 数）；`..._bucket1152.yaml` 的头部注释更正为已证实的机制。
- **第五轮补测后的修正与新增**：①`qwen-prefill-chunk` 的「垫到 2,048 桶」假设被 `server.log` 的捕获清单和源码的选桶规则**证实**（本机只捕获一张 bs = 32 的 2,048 图，bs = 1 的步用它）；中间一版曾写成「2,048 没捕获，走 eager」，是只看了 eager 列表、漏看 bs = 32 的错误，已更正。nsys 的服务器因为占显存没有 2,048 图，所以 `prefill-chunk-nsys` 的预算 4,096 臂测的是 eager 路径，不是生产路径。改写了 §0、§0.1 D1、§9.6（1）、§10、§12。②**rtx5880 profile 新增 `allowed_devices = [2,3,4,5,6,7]`**：0 / 1 号卡是别人长期的训练任务，会在 epoch 之间空出来一瞬，调度器按最低序号选中后 idle gate 要等一块别人在用的卡（上限 1800 s），第一次启动因此卡住；`test_suite.py` / `test_task_cells.py` 里依赖 0 / 1 号卡的断言已改（309 个测试通过）。如果别人的任务结束了想恢复 8 张卡，删掉 profile 里这一行并把 `test_hardware_profiles_load_and_describe_the_two_machines` 与 `test_auto_select_takes_exactly_the_free_gpus_a_deployment_needs` 改回去。
- **第四轮补测后的修正与新增**：①`qwen-prefill-chunk` 推翻了第三轮的「TXT 差距 = chunk 预算」：预算 4,096 后 TTFT 138 → 154 ms，改写了 §0、§0.1 D1、§9.5 更正段、§9.6（1）、§10。②`bagel-prefill-chunk`：BAGEL 预算 1,024 → 8,192 只降 I2T TTFT 约 6–7 ms（D2 改写）。③`tp-cache-probe`：预算 4,096（提示一个 chunk）时 TP=2 + 前缀缓存不崩，13 个请求成功（§9.6（4））。④tiny 提示在服务端是 56 token（8 + 48 模板），不是「约 15 token」；1k 提示 1,072 = 1,024 + 48，两个数都来自 SGLang 日志的 `#new-token`（`analyze_ttft_parts.py` / catalogue / cases 里的措辞已改）。⑤新增 CPU 可验的提示长度夹具（`txt-512-1tok` / `txt-1024-1tok` / `txt-2048-1tok`，`client.load_length_prompt`）、配置 `qwen3omni_3gpu_thinker_tp2_chunk4096_bucket1152.yaml`（只多一个 `model_kwargs.prefill_token_buckets` 1,152 桶），实验 `qwen-prefill-chunk-nsys`、`qwen-prefill-len`，`run_attrib.sh` 新增 `prefill-chunk-nsys` / `prefill-len`；测试 309 个通过。
- **第三轮补测后的修正与新增**：①`qwen-prefill-nsys` 的结论推翻了 `qwen-ttft-txt` 上一版的读法：「M\* prefill 每 1k token 86 ms」是两点直线跨过 1,024 chunk 边界的假象（§0、§9.5 已改，§9.6（1））。②`bagel-connector-nsys` 的 `mstar` 臂原来的判据（launch 数 16.6k = 一次 prefill、25k = 两次）无效：M\* 的 launch 只有 8,990、busy 与 single-stage 相同；改用 busy 判断（§7.7（3））。③`bagel-connector-nsys` 的旧三臂需要把旧 cell 挪开后补跑（§12）。④新增实验 `qwen-prefill-chunk`（M\* Thinker 预算 1,024 → 4,096）、`bagel-prefill-chunk`（M\* LLM 预算 1,024 → 8,192）、`mstar-tp-cache-probe`（TP 前缀缓存 + 4,096 预算）与配置 `qwen3omni_3gpu_thinker_tp2_chunk4096.yaml`、`qwen3omni_3gpu_thinker_tp2_prefix_cache_chunk4096.yaml`、`bagel_single_gpu_kv512_chunk8192.yaml`（只经 `--opt config.mstar=` 使用，各自与默认只差 `max_batch_tokens`，`prefix_cache_chunk4096` 另差 `prefix_cache_tp`）；`run_attrib.sh` 新增 `prefill-chunk` / `bagel-chunk` / `tp-cache-probe` 三组。⑤`qwen-a2t-cache-warm` 撤回了「vLLM 开缓存处理音频有问题」。⑥没有改任何已有实验的请求或配置。

## 14. 第六、七轮补测结果（2026-11-12，RTX 5880，已跑）

报告的对应页在 `share/omni_runtime_report_v2/`。以下数字都是每格 1–2 个 start 的 p50，除非另注。

### 14.1 Qwen TXT 的 TTFT（`qwen-prefill-nsys-room`、`qwen-fair-txt-b1152`、`qwen-prefill-len-repeat`、`qwen-prefill-len-dense`、`qwen-prefill-len-sglang`）

- `nsys-room`（预算 4,096，KV 池减半让 nsys 下也能捕获 2,048 的图）：无 nsys / nsys 的 E2E 161 / 165 ms，重测 165 / 164 ms，nsys 不再改变路径；垫到 2,048 的 forward span 143、busy 130、MoE 45.7、all-reduce 56 ms；eager 的 1,072 是 span 105–112、busy 85–93、MoE 34 ms。结论：垫到 2,048 在 GPU 上多约 40 ms busy。
- `fair-txt-b1152`（缓存关，256 token）：vLLM 1,613、SGLang 1,596、M* 默认 1,645、M* 加 1,152 档 1,612 ms；1 token E2E 157 → 123 ms（vLLM 101、SGLang 120），ITL 不变。TXT 的 2.4% 差距来自这个配置。
- `len-repeat`（3 个 start）：1,072 token 默认 156（153–160）→ 加 1,152 档 123（121–123）；512 token 88.1 对 87.6。
- `len-dense`（1 个 start，1 token E2E ms）：

| 长度 | 默认 | 预算 4,096 | +1,152 | +1,152 +1,536 | vLLM |
|---|---:|---:|---:|---:|---:|
| 1,100 | 154 | 165 | 121 | 121 | 101 |
| 1,300 | 172 | 167 | 140 | 134 | 111 |
| 1,536 | 175 | 168 | 146 | 143 | 121 |
| 1,800 | 196 | 171 | 164 | 165 | 133 |

  **1,536 档没有被捕获**（`captured 55 of 86 buckets`，日志里没有 `tokens=1536` 的 capture 行），所以最后一臂与 +1,152 是同一配置，差别是噪声。预算 4,096 的曲线是平的（都垫到 2,048）；+1,152 只在 1,100–1,536 起作用，1,800 回到 164。默认比 vLLM 在 1,100–1,800 一直慢 50–60 ms。
- `len-sglang`（缓存关）：56 / 512 / 1,024 / 1,072 / 2,048 token = 56 / 83 / 108 / 120 / 157 ms，近似线性，没有阶梯，1,072 处在 vLLM（98）与 M*（145）之间。

### 14.2 BAGEL 并发（`bagel-concurrency-fair`，两边缓存都开，2 start × 10 请求）

| E2E ms | B=1 | B=4 | B=16 |
|---|---:|---:|---:|
| M*（缓存开） | 2,293 | 2,665 | 3,647 |
| vLLM two-stage（Thinker 缓存开） | 2,258 | 4,545 | 13,647 |

B=16 相对 B=1 的扩展倍数：M* 10.1（吞吐 0.44 / 1.50 / 4.38 req/s）、vLLM 2.6。缓存对等后差别仍在，机制没测（可能是 M* 统一引擎的批处理，也可能是 vLLM two-stage 的 stage 排队）。

### 14.3 T2S 并发（`t2s-concurrency-fair`，**失败：run=1 analyze=2**）

7/9 个格完成，2 个格 `void_contract_violation`，analyze 拒绝出表，后续 arm 因 A/B/A 顺序被跳过：

- M* B=16：第 2 个请求返回 20.5 s 的音频，目标句 64 字符，合格范围 2.1–10.7 s。
- vLLM B=16：第 10 个请求返回 1.4 s 的音频，目标句 43 字符，合格范围 1.4–7.2 s 的下限以下。
- SGLang B=16 全部合格（p50 1,182 ms）。

从各格 `summary.json` 直接读到的 p50（1 个 start、6 个请求、`suspect_intra_start_drift`，**不是正式结果**）：B=1 M* 299 / SGLang 403 / vLLM 502 ms；B=4 647 / 688 / 924 ms。B=16 的作废是需要先查的问题：音频不是目标句，是 Talker 采样随机性还是并发下的状态串扰，没有查。

### 14.4 Pi0.5（`pi05-audit`、`pi05-nsys`）

- audit 数字（2 start × 12 请求，E2E p50）：M* 101、M* 不 compile 107、M* 不捕获图（hook）281、M* 不捕获图不 compile 258、SGLang core 165、SGLang core 两个图开关关 160、vLLM 214、vLLM 注意力改 SDPA 204、vLLM 关 eager 216、SDPA + 关 eager 202 ms。关图慢 2.77 倍；vLLM 的 `_attend` 是手写 matmul + softmax，SDPA 快 5%，关 `enforce_eager` 无变化（这条 pipeline 本就不走图）；SGLang core 关图开关无变化，原因没查。输出等价和 HTTP/WebSocket 传输差异没有审计。
- nsys（1 start、4 请求，取最后一个 burst）：M* E2E 113 ms，span 80、busy 60、launch 4.9k；vLLM 274 ms，span 260、busy 100、launch 14.8k，kernel 里有 `ampere_sgemm`（fp32 GEMM）约 25 ms；SGLang core 180 ms，一个请求分成 1.7k / 7.9k / 1.7k launch 的几段，每段 busy 约 40 ms，没有求和。读法：vLLM 的差距是 GPU 工作量加 launch 间隙（busy 占 span 38%，M* 75%），SGLang core 的时间在 GPU 之外，原因没查。

### 14.5 Cosmos3（`cosmos3-sglang`、`cosmos3-sglang-nsys`、`cosmos3-size`）

- 256²，50 步，E2E p50：M* 1,268、M* 关图 1,279、M* 给 256² 加图 1,270、M* 顺序 CFG 1,952、M* 顺序 CFG + 无图 + 无 compile 2,676、vLLM 3,118、SGLang-Omni 4,743 ms。graph 对 256² 无关（不在捕获档位）。
- nsys 一个请求：launch M* 40k / vLLM 83.6k / SGLang 113k；GPU busy 1,033 / 1,815 / 1,868 ms；DiT forward 数（注意力 launch 除以 36 层）约 50 / 约 91 / 约 80（91 是推断：50 加 41 个 CFG 步各多一个分支）。SGLang 请求总时间里约 2.7 s 在 DiT 之外，没拆。
- 尺寸（2 start × 3 请求）：256² / 512² / 1024²：M* 1,315 / 3,672 / 13,517；vLLM 3,092 / 4,056 / 13,465；SGLang 4,753 / 6,536 / 16,512 ms。vLLM / M* = 2.35 / 1.10 / 1.00；SGLang / M* = 3.61 / 1.78 / 1.22。领先是小尺寸现象；M* 的 denoise 图有 `graph_max_latent_area: 2000` 上限。

### 14.6 Orpheus nsys（`orpheus-nsys`，1 start、2 请求）

| | span ms | busy ms | busy/span |
|---|---|---|---:|
| M* b1 | 2,522 / 4,959 | 2,480 / 4,853 | 98% |
| VoxServe b1 | 2,638 / 5,585 | 2,211 / 4,690 | 84% |
| M* b16 | 4,306–5,589 | 4,182–5,426 | 97% |
| VoxServe b16 | 5,986 / 8,730 | 4,979 / 7,377 | 83–85% |

请求输出长度不同，没有按 token 归一。b1 最大的 kernel：VoxServe `gemvx` 约 4.1 s，M* Triton mm 约 4.3 s，矩阵乘总量相近。b1 的 1.09× 可由 VoxServe 15–17% 的 GPU 空闲解释；b16 的 1.38× 里 VoxServe 的 busy 也更多（7.4 对 5.1 s），原因没拆。VoxServe 的 CUDA graph / max_batch_size 配置仍未审计。

### 14.7 BAGEL 影响性能的因素汇总（RTX 5880，已有结果的整理）

| 开关 | 对象 | 影响 |
|---|---|---|
| connector：SharedMemory → Mooncake-TCP | vLLM two-stage I2I | 5 步 6,809 → 4,558（−33%）；50 步 28,190 → 25,943（−8%）；1024² 5 步 16,178 → 12,764（−21%），节省与 KV 字节同比例（1.52×） |
| connector：→ NIXL-UCX | 同上 | 5 步 5,308；50 步 26,832；1024² 14,085；比 Mooncake 慢在接收侧 0.75–1.2 s，`UCX_TLS=tcp/cuda_ipc` 只回收 0.2–0.4 s |
| Thinker prefix cache 开 | vLLM I2T / I2I 5 步（Mooncake） | 2,979 → 2,256（0.76×）/ 4,564 → 3,523 |
| Thinker prefix cache 开 | M* I2T / I2I | 2,934 → 2,293 / 无影响 |
| 异步调度 / 发射链关 | vLLM / M* I2T | vLLM 2,982 → 3,123（+5%）；M* 2,284 → 2,599（+14%） |
| attention 后端 | M* 去 flash-attn / vLLM SDPA | M* 无变化；vLLM T2I、I2T 无变化，I2I 50 步 28,121 → 31,412（+12%） |
| MoT GEMM 调优 | vLLM T2I | 不调优 +2%（无 CFG）/ +4%（三分支） |
| LLM chunk 预算 1,024 / 8,192、vLLM 重复惩罚 | M* / vLLM | 无变化 |

connector 毫秒时间线（512²，5 步，ms，前置 / Thinker / 发送 / 间隔 / DiT 侧 / E2E）：SharedMemory 292 / 2,666 / 1,235 / 25 / 2,626 / 6,862；Mooncake 274 / 2,188 / 56 / 25 / 1,998 / 4,542；Mooncake + 缓存 292 / 1,222 / 55 / 23 / 1,939 / 3,528；NIXL 默认 283 / 2,186 / 31 / 25 / 3,163 / 5,701。nsys 下 DiT 去噪前的 GPU 空闲：SharedMemory 5.25 s，Mooncake 3.05 s，Mooncake + 缓存 1.97 s（nsys 放大主机时间，只看顺序）。

### 14.8 对 §0 / §10 的更新

- 仍然成立：TXT / A2T / I2T 公平档下无领先；Pi0.5、Cosmos3 的领先来自实现选择；BAGEL I2I 对配好的 two-stage 5 步 1.28×、50 步 1.06×。
- 新增：1,536 档图在 48 GB 卡上放不下（没有被捕获）；SGLang 的 TTFT 随长度线性；Orpheus 的差距在 nsys 里表现为 VoxServe 的 GPU 空闲加 busy 差别；Pi0.5 对 vLLM 是 GPU 工作量加 launch 间隙。
- 未解决：T2S 并发公平档的 B=16 作废（需先查输出不合格）；BAGEL 并发机制；VoxServe 配置审计；Pi0.5 SGLang core 的 GPU 之外时间。

## 15. Round 8 results (2026-11-12): BAGEL concurrency cap, T2S B=16 outliers

**15.1 BAGEL I2T concurrency = vLLM Thinker `max_num_seqs`** (`bagel-concurrency-seqs`, both caches on, 2 starts x 10 requests; E2E ms p50 at B = 1 / 4 / 16)

| arm | B=1 | B=4 | B=16 |
|---|---:|---:|---:|
| M* | 2,297 | 2,670 | 3,654 |
| vLLM two-stage, max_num_seqs 3 (upstream) | 2,253 | 4,537 | 13,673 |
| vLLM two-stage, 8 | 2,254 | 2,359 | 4,893 |
| vLLM two-stage, 16 | 2,257 | 2,349 | 3,221 |

Pre-registered prediction held: seqs3 = ceil(B/3) rounds (2.0x / 6.1x B=1), seqs8 two rounds at B=16, seqs16 one round. With the cap lifted vLLM is 12% faster than M* at B=4 and B=16 (I2T, hot prefix). The 3.7x lead of earlier rounds is the upstream recipe's cap, not engine batching.

**15.2 T2S B=16 outliers** (`t2s-conc-outliers`, record mode, 30 measured + 10 precondition waves per cell, ms per audio second p50)

| framework | B=1 | B=4 | B=16 | B=16 outliers |
|---|---:|---:|---:|---:|
| M* | 97.1 | 46.2 | 24.6 | 2/640 (row 66, waves 1 and 26) |
| SGLang-Omni | 121 | 48.5 | 20.8 | 2/640 (row 91, waves 1 and 21 of measured) |
| vLLM-Omni | 151 | 66.5 | 29.6 | 0/640 |

Outliers are ~20.5 s of audio for 19-64 character targets (Talker did not stop), only at B=16, always the same row per framework. B=1/B=4 (1,000 responses) are clean. (Superseded by 15.3 for the row attribution.) The fair-profile `t2s-concurrency-fair` would void these cells under enforce mode, so the report uses this campaign for T2S concurrency.

**15.3 T2S B=16 outliers with shuffled row order** (`t2s-conc-shuffle`, record mode, seeded permutation per pass, 100 measured + 10 precondition waves per cell)

| framework | B=1 | B=4 | B=16 outliers | rows |
|---|---:|---:|---:|---|
| M* | 0/110 | 0/440 | 13/1,760 | seedtts_en_0590 (64 chars, 20.46 s): 13 of 17 servings at B=16 |
| SGLang-Omni | 0/110 | 0/440 | 3/1,760 | seedtts_en_0139 (60 chars, 20.48 s): 3 of 18 servings at B=16 |
| vLLM-Omni | 0/110 | 0/440 | 0/1,760 | - |

With row order randomised the outliers stay on one sentence per framework (not on the wave position), and those sentences are clean at B=1 and B=4 (4 servings each at B=4). So the trigger is the sentence inside a B=16 batch; whether it fires depends on the rest of the batch. Why the Talker does not stop is not investigated. t2s-conc-fair stays void under enforce.

**15.4 BAGEL I2I concurrency** (`bagel-i2i-concurrency`, 512x512, 5 steps, Thinker cache on, vLLM Mooncake-TCP; wave E2E ms p50, 2 starts x 10 requests)

| arm | B=1 | B=4 | B=16 |
|---|---:|---:|---:|
| M* (server steps 5) | 2,757 | 11,156 | 46,427 |
| vLLM two-stage, Thinker max_num_seqs 3 | 3,521 | 10,975 | 97,106 |
| vLLM two-stage, Thinker max_num_seqs 16 | 3,505 | 13,480 | 226,336 |

M* throughput is flat at 0.34-0.36 req/s: I2I is serial there too. vLLM matches it at B=4 and degrades at B=16 (0.16 and 0.07 req/s). B=16 server.log has "Timeout waiting for KV cache ... after 30.0s" (10 lines with cap 3, 37 with cap 16): the DiT stage (max_num_seqs 1) cannot consume the KV as fast as the Thinker produces it, and raising the Thinker cap makes it worse. Root cause (KV buffer lifetime vs DiT queue) not isolated.
