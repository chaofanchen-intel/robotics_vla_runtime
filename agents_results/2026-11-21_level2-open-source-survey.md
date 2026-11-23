# Level 2 开源参考调研（2026-11-21）

> 取代说明：本文件中"未验证"的条目已在 [2026-11-21_level2-research-deep.md](2026-11-21_level2-research-deep.md) 中解决（许可证、PR 状态、Intel 文档、论文摘要、RTC 实现）。SGLang PR #39461 的状态在本文件中写的"已关闭"是错误的，实际为已合并（2026-09-22，范围仅限 router 的 chat 请求）。以深入版为准。

角色：只读研究（主会话直接完成，未启动子代理）。未写任何实现代码。

## 问题

设计文档 §4.2 的 Level 2 条目，逐条找开源框架、论文或已有实现可参考，避免从零造轮子：

- L2-E1：同一连接上，推理进行中仍能接收新观测（不阻塞接收）。
- L2-E2：真正中止在途推理。
- L2-E3：引导式 RTC（需要可微的去噪过程）。
- L2-E4：每路增量输入（Pi0.5 上已降级）。
- L2-E5：每路编码器缓存。
- L2-E6：分块内部的流式输出。
- L2-E7：跨请求状态（结构上不适用）。

## 范围

- 本地源码（只读）：
  - `external/mstar`（2c8ffde8）：`mstar/api_server/openpi.py`（240–302、286、304 行）。
  - `pinned/vllm-omni`（5917730 + overlay）：`vllm_omni/diffusion/cancellation.py`，以及它的全部调用点。
  - `external/lerobot`（v0.7.0，410c095）：`src/lerobot/policies/rtc/configuration_rtc.py`、`modeling_rtc.py`、`src/lerobot/policies/pi05/configuration_pi05.py`。
  - `envs/sglang-omni/.../multimodal_gen/runtime/pipelines_core/stages/denoising.py`（只查 abort/cancel）。
- 远程读取（2026-11-21，经代理 proxy-igk.intel.com:912 用 curl 取得）：
  - LeRobot 文档 `huggingface.co/docs/lerobot/main/en/rtc`（页面标注为 main 版，需从源码安装，比 v0.7.0 新）。
  - Kinetix `src/model.py`（JAX 参考实现，取得时未记录提交号）。
  - `AryamanMMahajan/VLA-Action-Chunking-and-RTC` README（π0 + LIBERO 的 RTC 实验记录；只读了 README，未读代码）。
- 未取得：Intel OpenEdge `pi05_with_rtc` 文档（猜测的 URL 返回 404，未找到正确地址）；两个仓库的 LICENSE（`main` 分支的 raw 地址都是 404）。

## 结果

### 1. RTC 的数学与参考实现（L2-E3）

- 已验证（读过该行）：Kinetix `src/model.py` 236–251 行的 `realtime_action`：
  - `x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)`：对去噪器做 VJP。
  - `error = (y - x_1) * weights[:, None]`：前缀误差，权重来自 `get_prefix_weights(inference_delay, prefix_attention_horizon, chunk, schedule)`。
  - `pinv_correction = vjp_fun(error)[0]`。
  - `guidance_weight = min(c * (t²+(1-t)²)/(1-t)², max_guidance_weight)`，其中 `c = (1-t)/t`（`t→0` 时用 `posinf=max_guidance_weight` 替代）。
  - 返回 `v_t + guidance_weight * pinv_correction`。
- 已验证（读过该文档段落）：LeRobot 文档页说明 RTC 是在 flow matching 去噪中加入引导项，使重叠的时间步贴近已执行的上一块动作，使用软过渡掩码。`mode` 有两种：`guided`（默认，适用于普通 flow-matching 检查点，需要反向传播）和 `trained`（对 Pi05 检查点用 `rtc_training_max_delay > 0` 训练过的权重，硬性覆盖前缀，不做反向传播）。
- 已验证（读过该行）：LeRobot v0.7.0 源码里也有这两种模式（`configuration_rtc.py:41–52, 84–85`；`modeling_rtc.py:46–48`；`configuration_pi05.py:79, 133–136`）。trained 模式的准入规则（`rtc_training_max_delay ≤ execution_horizon ≤ prediction_steps − rtc_training_max_delay`，Pi05 的 `prediction_steps` 为 `chunk_size`）也在文档中给出。
- 已验证（读过该段落）：AryamanMMahajan 的 README 写了 π0 + LIBERO 上的 RTC，前缀权重为 `get_prefix_weights(inference_delay, execution_horizon, chunk_size)`，修正量用 `torch.autograd.grad` 对 `x1_t` 做 VJP，再从速度场中减去 `guidance_weight * correction`。它的配置是 ~0.29 s/次 π0 调用，`d = 6` 步，10 步去噪。
- 推断：Kinetix 的 JAX 实现与 LeRobot 的 `guided` 模式在数学上是同一套（都是对 x₁ 做 VJP 再乘以时间相关的引导权重）。未逐行核对 LeRobot 的 `modeling_rtc.py`，这一点是推断。

### 2. 中止在途推理（L2-E2）

- 已验证（读过该行）：vLLM-Omni 的协作式取消（`cancellation.py` 头部注释）只在"已选择加入（opt-in）"的流水线中生效，在安全边界调用 `check_request_cancellation`。
- 已验证（`grep` 全部调用点）：调用点只有 4 个文件：
  - `diffusion/worker/diffusion_model_runner.py:881`（通用扩散运行器）；
  - `diffusion/models/minimax_h3/denoise_loop.py:373, 456`；
  - `diffusion/models/minimax_h3/pipeline_minimax_h3.py:3239, 3244, 3253, 3269`。
  - **没有任何 pi05 流水线调用它**。因此 vLLM-Omni 的 Pi0.5 路径不能取消在途推理。
- 已验证（读过该行）：SGLang 的 `multimodal_gen/.../denoising.py` 里没有 abort/cancel 相关代码。`managers/scheduler.py:1379` 有一个 abortion 的文档字符串，但没有对 Pi0.5 去噪路径的检查（未深入）。
- 已验证（读过该行）：M\* `openpi.py:250` 注释说"handler 被取消时，提前离开会中止这次提交，由迭代器负责"。注释只说明了提交与迭代器的关系，没有说明 GPU 上的去噪是否会停止。`design.md` §2.6.4 的实测（四个服务器断开后都不中止在途 GPU 工作）优先于这条注释。
- 结论：四个框架里，没有一个对 Pi0.5 的在途推理提供可用的中止。最接近的参考是 vLLM-Omni 的 `cancellation.py`（在去噪步之间检查取消标志，选择加入）。它的机制可借鉴，代码不能直接用于 Pi0.5 的路径。

### 3. 同连接接收与推理并行（L2-E1）

- 已验证（读过该行）：M\* `openpi.py` 的服务循环是串行的：`receive` 在循环顶部，推理在循环内同步执行，推理期间不接收。
- 已验证（读过该行，见 [agents_results/2026-11-20_openpi-remote-pi05.md](2026-11-20_openpi-remote-pi05.md) 与 [2026-11-20_lerobot-async-rtc.md](2026-11-20_lerobot-async-rtc.md)）：OpenPI 参考客户端与 LeRobot 的 remote inference 都是"每次一个请求"。LeRobot 的做法是客户端用"最新观测槽"（只保留最新一条），而服务端仍然串行处理。
- 结论：开源参考里没有服务端"推理中接收并保留最新观测"的实现。L2-E1 要做，只能照 LeRobot 的客户端做法，服务端仍需改动。

### 4. 训练期动作条件化（替代方案，不是 Level 2 的必选项）

- 已验证（读过该文档段落）：LeRobot trained 模式用的是训练期动作条件化，不增加推理开销，但需要重新训练 Pi05 检查点。
- 未验证：LeRobot 官方是否发布了已用 `rtc_training_max_delay > 0` 训练的 Pi05 检查点。文档只说"compatible Pi05 checkpoints"，没有列出具体的权重。

### 5. 未研究的条目

- L2-E4：已降级，未调研。
- L2-E5（每路编码器缓存）、L2-E6（分块内部流式输出）：本轮没有查开源参考，不能据此给结论。

## 主会话核对与修正

- 上一步里 M\* 的 `cancel|abort` grep 结果很多，多数与请求取消无关（如 `journal.cancel` 是预留账本）。收窄到 `openpi.py` 与 `engine.py` 后，只有 `openpi.py:250` 的注释与请求取消有关。
- 上一步 vLLM-Omni 的调用点由 `grep -l` 确认：4 个文件，没有 pi05。
- 上一步 `head` 截断后 `exit 0` 不代表没有命中，已用收窄的 grep 重新确认。

## 未验证的部分

- Kinetix 与 AryamanMMahajan 仓库的许可证：未取得（`main` 的 LICENSE 为 404）。在引用代码前必须先确认许可证。
- Intel OpenEdge `pi05_with_rtc` 文档：未取得。
- arXiv 候选（2607.12659 EagleVLA、2610.05062 overlap vs staleness、2509.23224 A2C2、2604.25050 DiscreteRTC）：未读。
- vLLM 的 abort 相关 PR（#55521、#63952）与 SGLang 的 issue（#45002）、PR（#39461）：未读。
- LeRobot 的 trained 模式在 v0.7.0 中已有，但 main 文档描述的 Pi05 兼容检查点列表未核对。
- SGLang `scheduler.py:1379` 的 abortion 路径是否影响 Pi0.5 的去噪：未深入。
