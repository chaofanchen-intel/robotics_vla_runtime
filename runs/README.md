# runs：实验输出

每次实验一个目录，名字是 `<日期>_<名称>`。目录内按框架分子目录，每个框架包含：

- `<模式>.json`：探测结果（`e2e`、`period`、`burst`、`cancel`、`concurrency`），含原始样本与统计；
- `server.log`、`serve_command.txt`：服务日志与实际执行的启动命令；
- `gpu_before.txt`、`gpu_after_<模式>.txt`、`gpu_after_stop.txt`：`nvidia-smi` 快照；
- `ready_seconds.txt`：服务健康检查通过用时。

`run_<框架>.log` 是 `run_all.py` 的标准输出。

## 目录

| 目录 | 内容 | 状态 |
|---|---|---|
| `2026-11-20_ws-probe/` | 四个框架的 `e2e`、`period`、`burst`、`concurrency` 探测；M\* 的 Python 前端与 Rust 前端（`mstar`、`mstar-rust`） | 有效。`cancel` 模式的结果已移除（见下） |
| `2026-11-20_ws-probe-cancel-v2/` | 修正版 `cancel` 探测，四个框架 | 有效，是断开是否中止在途推理的唯一依据 |
| `2026-11-20_migration-smoke/` | 目录迁移后的冒烟测试：三个框架各跑一次 `e2e` | 只用于确认迁移后的路径与源码哈希检查 |

## 已移除的结果

`2026-11-20_ws-probe/` 原先的 `cancel.json`（四个框架）已删除。第一版取消探测在 `close()` 返回后才发送 B，
关闭握手要等在途推理完成，因此 M\* Rust 前端的 B 看起来"没有排队"，被误读为"会中止"。
修正版（`2026-11-20_ws-probe-cancel-v2/`）让 B 在固定间隔发送，结论是四个框架都不中止。
设计文档 §2.6.4 记录了这一过程。

## 注意

- 日志与 `serve_command.txt` 中的路径是迁移前的绝对路径（`/home/chaofan/workspaces/Inference_Engine_Investigation_Repo/...`），没有改写。
- 运行时的 socket 与上传目录在 `/tmp/exps-robot-probe`，不保存在这里。
