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

这四个文件于 2026-11-20 从基准仓库迁入。文件内的路径已改为本目录的 `external/` 与 `docs/design.md`；`/tmp` 路径是当时的临时克隆，未保留。
