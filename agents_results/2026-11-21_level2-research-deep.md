# Level 2 开源参考：深入调研（2026-11-21，第二轮）

角色：只读研究（主会话完成，未启动子代理，未写实现代码）。
取代：[2026-11-21_level2-open-source-survey.md](2026-11-21_level2-open-source-survey.md) 中的"未验证"项，其中的结论大部分保留。

## 问题

用户要求：换一种方式获取上一轮未能取得的资料（git 克隆、直接下载），充分调研 Level 2 的开源参考，**在用户准许前不进入 Level 2 实现**。
条目沿用设计文档 §4.2：

- L2-E1 同连接上推理期间接收新观测。
- L2-E2 真正中止在途推理。
- L2-E3 引导式 RTC（需要可微去噪）。
- L2-E4 每路增量输入（Pi0.5 上降级）。
- L2-E5 每路编码器缓存。
- L2-E6 分块内部流式输出。
- L2-E7 跨请求状态（结构上不适用）。

## 范围与版本

所有下载的源码都在 `external/l2-research/`（被 `.gitignore` 忽略，与 `external/README.md` 的约定一致）。提交号取自本地 `git log -1`。

| 资料 | 来源 | 版本 / 提交 | 许可证（读过 LICENSE 文件） |
|---|---|---|---|
| Kinetix RTC（PI 官方） | github.com/Physical-Intelligence/real-time-chunking-kinetix | 9296f31（2025-12-07） | MIT（Physical Intelligence） |
| rtc-pi05（pi0.5 的推理期 RTC 适配器） | huggingface.co/Daniel-F/rtc-pi05（模型仓库，下载了 rtc/*.py、examples、tests） | 未记录（HF 文件直取，2026-11-21） | HF 页面元数据写 `License: mit`；文件中没有 LICENSE |
| openpi_rlt（JAX openpi 分叉，含 `Pi0RTC`） | github.com/Ziyeeee/openpi_rlt | d42e408（2026-10-21） | 未核对（仓库中未读） |
| openpi05-bsp（pi0.5 的 B 样条 + 异步客户端，含 RTC 客户端） | github.com/Siyuan-Xue/openpi05-bsp | dfb7444（2026-10-10） | Apache-2.0（LICENSE 首行） |
| A2C2 Kinetix | github.com/k1000dai/a2c2-kinetix | 0e21d0e（2025-07-15） | MIT |
| A2C2 LIBERO（LeRobot 分叉） | github.com/k1000dai/a2c2-libero | 54dd088（2025-09-25） | 版权头为 Hugging Face；许可类型未在本轮核对 |
| EagleVLA | github.com/PKU-SEC-Lab/EagleVLA | 852ef5b（2026-10-02） | Apache-2.0 |
| π0 + LIBERO RTC 实验记录（Aryaman） | github.com/AryamanMMahajan/VLA-Action-Chunking-and-RTC | 30a6f76（2026-10-19） | **无 LICENSE 文件**，不可复用代码，只作参考 |
| edge-vla（PR #2 的仓库） | github.com/hungho77/edge-vla | f6620df（2026-10-23） | 未核对；PR 正文 API 返回空，未读到 |
| LeRobot 文档 RTC（main） | huggingface.co/docs/lerobot/main/en/rtc | 页面是 main 版（需从源码安装） | — |
| Intel OpenEdge pi05_with_rtc 文档 | docs.openedgeplatform.intel.com/2026.0/.../pi05_with_rtc.html | 2026.0 页面 | 只读了文档 |
| 论文 | arXiv 2607.12659（EagleVLA）、2610.05062（VLA 工作负载表征，ASPLOS 2027）、2509.23224（A2C2）、2604.25050（DiscreteRTC） | PDF 经 `pdftotext` 抽取（`external/l2-research/arxiv/`） | — |
| vLLM PR #55521、#63952；SGLang PR #39461、issue #45002 | GitHub API（未认证） | 2026-11-21 查询 | — |

本地 Python 环境（只读检查）：`envs/sglang-omni`（sglang 包）、`envs/vllm-omni`（vllm 包）、`pinned/vllm-omni`。

## 结果

### A. 引导式 RTC（L2-E3）

1. **数学形式（已验证，读过代码）。** Kinetix `src/model.py:236–251` 的 `realtime_action`：对去噪器做 VJP，前缀误差乘以权重，再由 `guidance_weight = min(c·(t²+(1−t)²)/(1−t)², β)`、`c=(1−t)/t` 缩放修正量并加到速度场上。rtc-pi05 的 `rtc/masking.py:122–139` 写的是同一个公式，并注明 `t=0` 时无穷大、由 `β` 截断。把两式展开：`c·inv_r2 = (t²+(1−t)²)/(t(1−t))`，与 PiGDM 的 `(1−τ)/(τ·r²_τ)` 代数相同（手算，未用数值验证）。
2. **rtc-pi05（已验证，读过代码）。** 这是一个独立的、面向 openpi pi0.5 的推理期 RTC 实现。
   - `rtc/sampler.py:154–156`：`torch.autograd.grad(x_hat, x, grad_outputs=error)` 做 VJP。
   - `rtc/controller.py`：推理在后台线程进行；延迟估计取最近 `b` 次观测延迟的最大值（默认 `b=10`，`controller.py:146–160`）；等待 `t ≥ max(s_min, d)` 再发起推理（`:157`）；推理结束后 `t -= s`，实际消耗的动作数即观测延迟（`:180–183`）。
   - 有 `pi0.py`（263 行）适配 openpi `PI0Pytorch`。本轮只看了文件列表和 `masking`/`sampler`/`controller`，未逐行读 `pi0.py`。
   - `stop()`（`controller.py:103–111`）只置停止标志并 `join` 线程，**没有中止推理**：它等待当前推理完成。
   - GitHub 上的对应仓库没有找到（搜索只返回 HF 页面），本次取得的是 HF 模型仓库中的文件。
3. **Intel OpenEdge 文档（已验证文档，未读代码）。** 文档说明 pi0.5 + RTC 的 OpenVINO 导出需要两个额外输入 `prev_chunk_left_over` 与 `prefix_weights`，并用 LeRobot 的检查点（默认 `lerobot/pi05_base`）。它的 RTC 是推理期引导，不需要训练。
4. **LeRobot v0.7.0（已验证，见上一轮）。** `guided` 与 `trained` 两种模式都在源码中；`trained` 需要 Pi05 检查点满足 `rtc_training_max_delay > 0`（`configuration_rtc.py:41–52`，`modeling_rtc.py:46–48`）。
5. **Aryaman 的 π0 + LIBERO 记录（已验证 README，代码未读）。** 同样是推理期引导，`d=6`，10 步去噪，`~0.29 s` 每次调用；它的 README 描述了 `torch.autograd.grad` 的 VJP 路径。无 LICENSE，不能复用代码。

**结论（A）**：推理期引导式 RTC 有三个独立的开源实现（Kinetix JAX、rtc-pi05 PyTorch、LeRobot），数学形式相同，可以作为 L2-E3 的对照。rtc-pi05 是唯一面向 pi0.5 的独立实现，MIT 许可，最适合直接读代码作为参考。

### B. 训练期动作条件化（替代 L2-E3 的推理开销）

1. **Kinetix 训练与推理（已验证，读过代码）。** `src/model.py:267–288` 的 `loss`：从权重 `∝ exp` 的分布采样延迟 `d`，前 `d` 个动作的时间设为 1（视为干净动作）且不计入损失；`:253–258` 的推理：对前 `inference_delay` 个位置直接替换为上一块动作并把时间设为 1（**硬前缀**，无反向传播）。
2. **openpi_rlt 的 `Pi0RTC`（已验证，读过代码）。** `src/openpi/models/pi0_rtc.py:78–90` 说明它与 `Pi0` 结构相同，但用前缀训练与采样；`:235–237` 训练时对每个样本采样 `delay ∈ [0, max_delay)`；`sample_actions`（`:273–302`）接受 `action_prefix` 与 `delay`。它是 JAX 版本，并且含 agilex 机器人专用代码。
3. **EagleVLA（已验证 README 与配置，代码未深读）。** 它是 JAX openpi 分叉，提供 π0.5-LIBERO 检查点，并训练了一个**未来修正模块**（future correction module）：预测未来的环境表示，让动作专家从未来时间步预测动作（论文摘要）。这需要训练，不是推理期改动。训练阶段表见 `README`（约 215–216 行）。

**结论（B）**：训练期条件化在开源里有 Kinetix（小模型）和 openpi_rlt（JAX）两个实现，但都需要重新训练检查点。LeRobot 官方的 `trained` 模式也依赖这种检查点。对 pi0.5 基础检查点（`pi05_base`/`pi05_droid`）而言，没有已发布的、经过前缀训练的版本（未核对 LeRobot 的 Hub 列表）。

### C. 异步执行与延迟容忍（与 L2-E1 相关）

1. **openpi05-bsp 的异步客户端（已验证，读过代码）。** `packages/openpi-client/src/openpi_client/async_inference.py`：
   - 专用拥有者线程处理推理请求；`reset_generation`（`:293–313`）作废排队与在途的任务，并把连接置空，即"退役"套接字（`:306–313`）。
   - **客户端取消 = 丢弃套接字。** 这不会让服务端停止在途推理（服务端 openpi 串行处理，见下）。
   - `rtc.py` 有 RTC 计划状态机（`RtcPlanState`、`RtcPlan.begin_guided`、`forecast_delay`、`RtcRequestInFlightError` 等，`:22–60`、`:98–192`），是客户端侧的 RTC 调度，与 rtc-pi05 的控制器思路类似。
   - README 声明的结果"coming soon"，**没有已验证的数字**，不能引用其性能结论。
2. **BSP 分叉的服务端（已验证，grep）。** `src/openpi/serving` 中对 `disconnect|cancel|abort` 的搜索没有结果。服务端仍是串行处理（与 M\* 和 OpenPI 参考一致）。HEAD 是 dfb7444。
3. **A2C2（已验证摘要与代码结构）。** `2509.23224`：一个轻量的逐步修正头，加在任意现成 VLA 的动作块之上；不需要重新训练基础策略；与 RTC 正交。摘要声称在 Kinetix 动态任务上比 RTC 高 23 个百分点（在 LIBERO Spatial 上高 7 个百分点）。代码（`a2c2-kinetix`，MIT）中有残差策略训练流程（README 第 32–35 行）。**摘要中的数字是论文自报，未复现。**
4. **EagleVLA（论文摘要，已验证）。** 在 Jetson Orin 上，控制频率比 naive PyTorch 高 14.71 倍，成功率比 VLASH 在 LIBERO 上高 14.8%。这是论文自报数字。代码 Apache-2.0。
5. **DiscreteRTC（2604.25050）。** 针对离散扩散策略，不适用于 pi0.5 的连续流匹配（摘要明确说 RTC 对流匹配是次优的）。论文文中未给出代码链接。
6. **2610.05062（ASPLOS 2027 工作负载表征，已验证摘要与正文）。** 测量了 4 个 VLA 在边缘 GPU 与 SoC 上的表现，共 43,200 个闭环回合；指出"更激进的重叠（overlap）换来更快的控制，但迫使机器人基于更陈旧的观测行动"（正文约 219–224 行）。这个新鲜度与频率的权衡与我们的 Level 1 框架一致，可作为对照引用，但不是 Level 2 的实现参考。

**结论（C）**：开源异步参考都在客户端做调度，没有服务端改动。服务端仍然串行，所以 L2-E1 的收益需要服务端配合，只靠客户端无法消除"推理期间不接收"。EagleVLA 与 A2C2 的数字是自报，需要自己复现才能采信。

### D. 中止在途推理（L2-E2）

1. **vLLM（已验证，读过代码）。** `envs/vllm-omni/.../vllm/v1/engine/async_llm.py:911–925`：`AsyncLLM.abort` 同时在 `output_processor` 与 `engine_core` 上中止请求，即中止的是引擎中的请求；是否能在一次前向中途停止 GPU 计算，需要看 `core` 的调度循环（本轮未深入）。
   - PR #55521（OPEN，2026-09-06 创建）：新增显式取消 HTTP 端点。它的动机正文写明：目前取消的唯一方式是关闭 TCP 连接。
   - PR #63952（OPEN，2026-10-27）：流式输入在第一块到达前中止。
   - pinned vllm-omni 的协作式取消只在 MiniMax-H3 与通用扩散运行器中有调用点（上一轮已确认，pi05 无）。
2. **SGLang 调度器（已验证，读过代码）。** `envs/sglang-omni/.../managers/scheduler.py` 的 `abort_request`：
   - 从 `waiting_queue` 中移除匹配请求，并给 tokenizer 发送中止消息；
   - 对分块中的请求设置待中止标志（`_pending_chunked_abort_req`）；
   - 匹配方式是 `req.rid.startswith(recv_req.rid)`，即**前缀匹配**。
   - 这一段代码只覆盖排队中与分块中的请求。正在运行的一批前向是否被中途打断，本轮没有读到对应代码（**未验证**）。
   - 对外的 `/abort_request` 端点在 `srt/entrypoints/http_server.py:1686–1691`。
3. **SGLang router PR #39461（MERGED 2026-09-22，PR 正文）。** 客户端断开时对 engine 做尽力而为的取消，仅限于"普通、单样本的 chat 请求"；`n>1` 与 PD 请求被排除。这是 router 层的功能，不覆盖 OpenPI 端点。
4. **SGLang issue #45002（OPEN）。** 客户端可控的 RID 前缀让一个请求能通过断开清理中止另一个请求（安全问题，报告者声称 GHSA-c7p3-8wqm-7cq3）。这与上面的前缀匹配直接相关：任何中止实现都要避免把客户端提供的 ID 当作前缀使用。
5. **pi0.5 在 SGLang 中的 OpenPI 端点不在 `sglang/srt` 包中**（`grep openpi` 无结果）。它在 sglang-omni 中，上一轮已确认其 denoising 阶段没有 abort/cancel（`denoising.py`，无命中）。

**结论（D）**：
- 四个框架中，没有一个能中止 Pi0.5 正在进行的去噪。
- SGLang 的调度器和 vLLM 的引擎都能中止**排队中**的请求，这在 Pi0.5 的串行服务里几乎没用（服务端一次只处理一个请求）。
- 可借鉴的模式是 vLLM-Omni 的 `cancellation.py`：在去噪步之间检查取消标志。它需要在 Pi0.5 的去噪循环里加检查点，属于 Level 2 的实际改动。
- 中止 ID 必须精确匹配，不能用前缀（SGLang #45002 的教训）。

### E. 每路编码器缓存（L2-E5）

- vLLM v1 中有 `EncoderCache`（`envs/vllm-omni/.../vllm/v1/engine/core.py` 与 `executor/abstract.py` 引用），按多模态输入哈希缓存编码器输出。这说明机制存在。
- **未验证**：Pi0.5 在 vLLM-Omni 的 OpenPI 路径（`pinned/vllm-omni/vllm_omni/entrypoints/openpi/serving.py`）是否使用该缓存，以及机器人多路输入（同一台相机的未变图像）能否命中。这需要读 `serving.py` 与 Pi0.5 模型的编码器调用，本轮未做。

### F. 分块内部流式输出（L2-E6）

- 本轮没有找到开源参考。M\* 的 `openpi.py:247` 的 `_infer` 使用 `streaming=True` 并逐块读取（已在上一轮读过），但它是引擎内部的分块流，不是动作块内部的流式输出。
- **未研究到足以给结论的程度。**

## 主会话核对与修正

- 上一轮的"未验证"项中：Kinetix 许可证（MIT）、Intel OpenEdge 文档（已取得）、四篇论文（已抽取，摘要已读）、vLLM/SGLang PR 与 issue（已取得状态）已经解决。
- **修正**：上一轮写 SGLang PR #39461 "已关闭"（来自一次错位的 API 输出）。实际状态是 **MERGED 2026-09-22**，但范围仅限 router 的普通 chat 请求。
- 上一轮的"SGLang scheduler.py:1379 有 abortion 文档字符串"：现在读到的 `abort_request` 实现是本轮新增的证据，覆盖排队中与分块中的请求。
- `openpi.py:250` 的注释（"提前离开会中止提交，由迭代器负责"）仍然只说明提交与迭代器的关系，与 §2.6.4 的实测（四个服务器断开后都不中止 GPU 工作）一致。

## 未验证的部分

- rtc-pi05 的 `pi0.py`（263 行）与 `examples/pi05_realtime.py`：未逐行读，因此没有核对它如何调用 openpi 的 pi0.5 权重。
- openpi_rlt 与 EagleVLA 的训练代码：只读了模型或 README，没有验证训练数据与损失是否与 Kinetix 一致。
- A2C2 与 EagleVLA 的数字（A2C2 +23 点，EagleVLA 14.71×、+14.8%）：论文自报，未复现。
- openpi05-bsp 的结果：README 写"coming soon"，没有数字。
- Intel OpenEdge 的 RTC 实现代码：只看了文档，没有取得源码。
- PI 官方博客（pi.website/research/real_time_chunking）：curl 返回的是反爬页面，未读到正文。
- vLLM 的 `core` 调度循环是否能在一次前向中途中止：未读。
- SGLang `pi05` 的 OpenPI 端点在 sglang-omni 中的 abort 行为：只确认了 denoising 阶段无 abort 调用，未确认调度器对正在运行批次的处理。
- L2-E5 与 L2-E6：仅有背景，未得到结论。
- PI 官方 RTC 的 JAX 实现（Kinetix）与 LeRobot 的 `guided` 模式是否逐行一致：只核对了公式层面。

## 对后续 Level 2 设计的输入（不构成实现决定）

1. L2-E3 候选参考：Kinetix（JAX，MIT）与 rtc-pi05（PyTorch，MIT）。rtc-pi05 的控制器（延迟缓冲、`s = max(d, s_min)`）可直接对照我们 Level 1 的 action manager。
2. L2-E2 候选：vLLM-Omni `cancellation.py` 的去噪步检查点模式；不要用客户端前缀匹配的 ID（SGLang #45002）。
3. L2-E1：开源参考都在客户端，服务端必须改。
4. 替代路线（训练期条件化、A2C2 残差头、EagleVLA 未来修正模块）都需要训练或额外模型，不属于"只改运行时"的范围，需要用户决定是否接受。
