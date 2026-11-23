# agents_results

机器人 / VLA 研究的子代理记录。每个子代理一个文件，记录给它的问题、它的结果、以及主会话事后核对的结论。
在启动新的研究子代理之前，先搜索这个目录（`grep -ril "<关键词>" agents_results/`），已有答案的问题不要重复研究。

命名：`YYYY-MM-DD_<主题-短横线>.md`。每个文件包含：

- 日期与子代理的角色（只读研究、实现等）；
- 问题：给出的完整问题列表；
- 范围：读过的路径与版本（提交号或标签）；
- 结果：每条结论标注"已验证（读过该行）"或"推断"，并给出 `path:line`；
- 主会话核对与修正；
- 未验证的部分。

基准仓库的研究记录（模型集成、realtime、Pi0.5 第三案例）仍在 `../Inference_Engine_Investigation_Repo/agents_results/`。

## 文件

- [2026-11-20_openpi-remote-pi05.md](2026-11-20_openpi-remote-pi05.md)：OpenPI 远程推理的线协议、服务端循环（每连接一个请求）、参考客户端的请求节奏、Pi0/Pi0.5 模型与 pi05_droid 配置；没有 RTC、会话或取消。只读源码（HEAD 215abfb），含主会话的抽查。
- [2026-11-20_lerobot-async-rtc.md](2026-11-20_lerobot-async-rtc.md)：LeRobot v0.7.0 的远程推理（async_inference 已被 remote_inference 替代）、回放阈值调度、最新观测槽、带代数号的 ActionQueue、RTC 的 guided 与 trained 模式及默认值、Pi0.5 状态分箱。只读源码（tag v0.7.0，410c095），含主会话的抽查。
- [2026-11-20_ros2-message-filters.md](2026-11-20_ros2-message-filters.md)：ROS 2 message_filters 8.0.5 的 Subscriber、Cache 查询语义、ApproximateTime 与 ExactTime 算法、时间源、没有周期触发器，以及复用估计。只读源码（995b9b2），含主会话的抽查。
- [2026-11-20_verify-unverified-upstream.md](2026-11-20_verify-unverified-upstream.md)：设计文档 §7 中未验证项 A–J 的核对（LeRobot 的 async 历史、ActionQueue 锁、websockets 版本、DROID 断言、ActionChunkBroker、pi05_base 归一化与图像 token、状态分箱边界、vLLM-Omni 会话处理、ROS 2 未安装）。只读，未运行。

- [2026-11-21_level2-open-source-survey.md](2026-11-21_level2-open-source-survey.md)：Level 2（§4.2 的 L2-E1 至 E7）的开源参考。RTC 的数学与参考实现（Kinetix JAX、LeRobot guided 与 trained、π0 + LIBERO 的 RTC 实验记录）；对 Pi0.5 在途推理中止：四个框架都没有可用实现（vLLM-Omni 的协作式取消只在 MiniMax-H3 与通用扩散运行器中，没有 pi05 调用点）；同连接接收：没有服务端参考。L2-E5、E6 未调研。主会话核对了调用点与 M\* `openpi.py:250`。未验证许可证与若干论文/PR。

- [2026-11-21_level2-research-deep.md](2026-11-21_level2-research-deep.md)：Level 2 的深入调研（第二轮，取代上一份的"未验证"项）。用 git 克隆与直接下载取得的源码都在 `external/l2-research/`（不进 git）。RTC 三个独立实现（Kinetix MIT、rtc-pi05 MIT、LeRobot）；训练期条件化（Kinetix、openpi_rlt 的 `Pi0RTC`、EagleVLA）；异步客户端（openpi05-bsp，服务端仍串行）；中止（vLLM `AsyncLLM.abort`、SGLang 调度器 `abort_request` 与前缀匹配、PR #55521/#63952/#39461、issue #45002）；E5、E6 仅有背景。主会话核对了代码行与 PR 状态，修正了上一份中 SGLang PR 的状态。论文数字为自报，未复现。

- [2026-11-23_mstar-sglang-capability-audit.md](2026-11-23_mstar-sglang-capability-audit.md)：M\* 与 SGLang-Omni 的能力审计（只读，静态阅读）。M\* 的 `/generate/ws` 追加路径与取消链；Qwen3-Omni 在 M\* 中未声明客户端输入流；OpenPI 端点无追加路径；SGLang-Omni 的 Qwen3-Omni 会话入口（PCM16、摄像头、VAD、INTERRUPT）按轮次工作。M\* 的 Rust 前端是否为实验所用的前端未确认。主会话核对了汇报稿引用的行号。

- [2026-11-23_vllm-omni-capability-audit.md](2026-11-23_vllm-omni-capability-audit.md)：vLLM-Omni 的能力审计（只读，pinned 5917730 与 latest 2e3ad0a4 对照）。`/v1/video/chat/stream` 的事件与缓冲；查询抢占；增量 prefill 在 pinned 默认关、latest 默认开；双工端点的 barge-in 与纪元；async_chunk 只覆盖阶段间输出；中止到 vLLM 调度器。文档与代码不一致：`full_duplex_api.md` 写 duplex=1 回退轮次式，代码返回错误并以 1008 关闭。GPU 停止时机为推断。

这四个文件于 2026-11-20 从基准仓库迁入。文件内的路径已改为本目录的 `external/` 与 `docs/design.md`；`/tmp` 路径是当时的临时克隆，未保留。
