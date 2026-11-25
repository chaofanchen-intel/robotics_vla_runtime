# 机器人 VLA 实时运行时：汇报稿（2026-11-23）

对象：两个领域的实时多模态推理。
- Pi0.5（机器人 VLA）：实时输入、推理、测量，以及 Level 1 / Level 2 的现状与路线；
- Qwen3-Omni（全模态对话）：输入与输出、三层路线、评价指标。
- 框架能力（M\*、vLLM-Omni、SGLang-Omni）按六项能力判据逐页比较：只写能做什么、不能做什么、要改哪一层，不再重复框架介绍。

读者：项目负责人与同组同事。页面以图表和流程图为主；数字都能在 `docs/design.md`、`runs/` 或上游源码行号中找到来源。

| 文件 | 内容 |
|---|---|
| `main.tex` → `main.pdf` | 汇报稿（16:9 幻灯片，共 44 页） |
| `chapters/01_summary.tex` | 主要结论、汇报结构 |
| `chapters/10_fw_criteria.tex` | 六项能力的定义与判据；两个领域分别需要什么 |
| `chapters/11_fw_mstar.tex` | M\*：输入路径、输出与取消、Qwen3-Omni 的图 |
| `chapters/12_fw_vllm.tex` | vLLM-Omni：流式视频会话、查询抢占与增量 prefill、双工端点、异步输出与取消 |
| `chapters/13_fw_sglang.tex` | SGLang-Omni：Qwen3-Omni 会话入口、打断、取消 |
| `chapters/14_fw_matrix.tex` | 能力对照（输入与输出；打断、取消与增量）；各框架要改哪一层 |
| `chapters/pi05_io.tex` | Pi0.5：输入、输出流程，多路输入时间线，三种读取方式，运行时流程 |
| `chapters/pi05_realtime.tex` | Pi0.5：控制与推理解耦，动作块衔接，主流方案，RTC |
| `chapters/pi05_results.tex` | Pi0.5：端到端与周期上限，断开实验，Level 1 欠载与动作年龄 |
| `chapters/pi05_gap.tex` | Pi0.5：Level 2 的缺口（E1–E7）与路线 |
| `chapters/qwen_io.tex` | Qwen3-Omni：两种流式，结构，各模态速率，窗口对齐，已有基准数字 |
| `chapters/qwen_levels.tex` | Qwen3-Omni：三层路线，路线 A 与 B，评价指标 |
| `chapters/40_plan.tex` | 计划（P0–P7）与待确定事项 |
| `chapters/90_appendix.tex` | 版本与来源；局限与未验证项 |
| `common.tex`、`deck.tex`、`build.sh` | 样式与编译脚本，与 `Inference_Engine_Investigation_Repo/share/omni_runtime_report_v2` 相同（封面日期 2026-11-23） |

已被移出的旧章节（框架通用介绍、带审批语句的进展页）不在本目录中。

## 数字与判断的来源

- 同协议探测、断开实验：`docs/design.md` §2.6；原始结果在 `runs/2026-11-20_ws-probe/`、`runs/2026-11-20_ws-probe-cancel-v2/`、`runs/2026-11-21_sglang-buckets-base/`。
- Level 1：`docs/design.md` §2.7；汇总在 `runs/2026-11-20_level1/level1_summary.md`。
- 框架能力：上游源码行号写在每页的脚注中；审计记录在 `agents_results/2026-11-23_*.md`。
- Qwen3-Omni 的基准数字来自基准仓库的 report v2（08a 章），本仓库未重测。
- 上游源码与论文：`external/`、`external/l2-research/`（不进 git）；结论记录在 `agents_results/`。

## 编译

```bash
bash build.sh main      # 生成 main.pdf（需要 Tectonic，或 latexmk + XeLaTeX）
```

## 更新

- 新的 Level 1 或探测结果：改 `pi05_results.tex` 的对应页，并同步 `01_summary.tex` 的结论。
- 版本变化（M\*、vLLM-Omni、SGLang）：改 `90_appendix.tex` 的版本表，同时更新 `docs/design.md` 开头的版本表，并复核框架页中带版本号的行号。
- Level 2 进展：改 `pi05_gap.tex`；计划状态改 `40_plan.tex`。Level 2 的实现按用户的约定，在用户准许后才开始。
