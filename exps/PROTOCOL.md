# 测量协议：冷却门控 还是 请求连续发出

> `exps/` 的每个任务、每个实验都能选这两种协议（以及一种固定空闲间隔的变体）。选哪种、怎么选、各自能支持什么结论，写在这里。命令见 [README](README.md)。

## 三种协议

| 协议 | 开关 | 一个 cell 的顺序 | 它回答什么 |
|---|---|---|---|
| **gated（默认）** | 不加参数，或 `--protocol gated` | 启服务 → loaded-idle gate → 不计分预热 → 第二次 loaded-idle gate → **每个测量请求前独立的冷却门** → 测量 | 每个请求都在「卡是凉的、满时钟、无降频原因」时开始。回答「引擎在好条件下每请求要多久」。**代价**：请求之间空闲 ≥30 s（门要求连续 30 s 满足） |
| **back-to-back** | `--protocol back-to-back`（别名 `--back-to-back`） | 启服务 → loaded-idle gate → 预热 → 测量请求**连续发出**，没有任何门 | 卡一直在忙。回答「持续发请求时每请求要多久」，更接近线上连续流量；热状态只记录、不强制，所以会受降频影响 |
| **fixed-gap** | `--protocol back-to-back --inter-request-gap-s N` | 同 back-to-back，但每个测量请求前固定 `sleep N` 秒，不附加温度条件 | 把「请求间卡空闲了一会儿」和「门要求卡凉 + 满时钟」分开 |

所有任务共用 `core/measure.py` 的同一个测量循环，所以这个选项对**每个模型**都有效（测试 `test_every_task_forwards_the_protocol_to_its_client` 逐任务验证开关能传到客户端）。协议名、间隔、门的次数都写进每个 cell 的 `summary.json`；`suite analyze` 会核对 cell 实际跑的协议是否与该 arm 声明的一致。**换协议必须换 campaign**（suite 自动加 `_B2B` / `_GAP30` 后缀），不能让 `--resume` 混用。

## 已有证据：协议会改变结论，方向因任务而异

| 任务 | 观察 | 来源 |
|---|---|---|
| **Pi0.5** | 背靠背**比门控快**，且框架受影响不同：vLLM-Omni 333.0 → 221.5 ms（0.67×），SGLang core 168.6 → 144.3（0.86×），M\* 95.6 → 90.3（0.94×）。请求前温度（57–61 °C）与 SM 时钟（1,785 MHz）两臂相同，**不是 GPU 热效应**；空闲后 eager 发射为何变慢（host 侧？）未隔离。结果：M\*/vLLM 3.48×（门控）vs 2.45×（背靠背），M\*/SGLang 1.76× vs 1.60× | 旧 `ATTR_PI05_{GATED,B2B}_01`（2 starts × 12，同 fixture 行、同 5 个预热） |
| **BAGEL ≥384²** | 不加门会**变差**：功率限制 / 热降频，M\* 在 ≥384² 平均 SM 时钟 1,405–1,614 MHz（vLLM 1,621–1,785）。门控下 256² 也有轻微 `sw_power_cap` | 热审计（4,488 个请求）；`thermal_history` |
| **T2S / A2T** | 门控下 100% 热洁净（1,785 MHz）；背靠背的影响**未测** | — |
| 其余（V-JEPA、Orpheus、Cosmos3、Qwen TXT） | **未测** | — |

「其他模型是不是和 Pi0.5 一样」现在没有答案；`protocol-ab-<任务>` 就是为此准备的（每个任务一个，arm = gated / b2b / gap30）。

## 硬件决定默认协议

协议的默认值来自硬件档案（`profiles/hardware/<名>.json` 的 `default_protocol`，`--hardware` 选档案，不选则按 GPU 名检测）：

| 档案 | 散热 | 默认协议 | 原因 |
|---|---|---|---|
| `h200nvl`（当前机器） | 约 80 °C 掉频，BAGEL ≥384² 会降频 | gated | 不加门，热状态会成为最大的混杂 |
| `rtx5880`（8 卡机器） | 稳定，不降频 | back-to-back | 冷却门没有要排除的东西，只会引入请求间空闲；用 `thermal-probe` 确认后再信这条档案 |

显式写 `--protocol gated|back-to-back` 永远优先；换默认协议的 campaign 名自动带 `_B2B` 后缀。门的显存阈值（空载 / 载入）也来自档案。稳定散热的机器上 gated 仍可跑，用来回答「空闲间隔本身有没有影响」（见 `protocol-ab-*` 的 gap30 臂）。

## 怎么选：由 `decide-protocol` 实现

规则已经写成代码（[protocols.py](protocols.py)），流程见 [GUIDE](GUIDE.md) §0：每个任务先跑 `protocol-ab-<任务>`（门控 vs 背靠背，2 次 start × 5 个请求），再 `python -m exps.suite decide-protocol`，决定写入 `profiles/protocol/<档案>.json`，`baseline-*` 按它运行（「两种」则两种都跑、两张表）；没有决定就拒绝规划。下面是同一套规则的文字版：

## 怎么选（规则的文字版）

1. **headline 默认用 gated**：它排除了热状态这一最大的混杂，且对 BAGEL ≥384² 这类热受限任务是唯一能比较的方式。
2. **每个任务至少跑一次 `protocol-ab-<任务>`**，用 `exps/results/protocol-ab-<任务>[_b2b]_<硬件档案>.md` 的比值判断：
   - gated 与 b2b 对**所有框架**的比值相同（差别在 A1/A2 自身漂移 ≈ ±4% 内）→ 该任务与协议无关，gated 的数就是结论。
   - 比值**随框架不同**（Pi0.5 这种）→ 该任务**对协议敏感**，任何跨框架倍数必须**同时给出两种协议下的值**，不能只报其中一个。
   - gap30 ≈ b2b 且都 ≠ gated → 是「门要求凉的满时钟卡」在起作用；gap30 ≈ gated ≠ b2b → 是「空闲了一会儿」在起作用（host 侧冷启动、频率恢复等）。
3. 同一个倍数引用时写明协议。门控和背靠背**不是**同一件事的两个精度，是两个不同的量：前者是「好条件下的单请求延迟」，后者是「持续负载下的单请求延迟」。

## 门的参数与标定

`profiles/defaults.json` 是**未实测**的保守参数（冷却目标 70 °C，比本机约 80 °C 的掉频拐点低 10 °C；连续 30 s；超时 900 s；loaded-idle 60 s / 1,800 s）。`calibrate.py --execute` 在不同目标温度下测每请求温升与冷却等待。参数优先级：命令行 ＞ `profiles/calibrated.json` ＞ `profiles/defaults.json`，来源写进 manifest 的 `calibration_source`。

## 统计口径

- **headline p50 = 该 cell 所有 start 的所有测量请求合在一起（pooled）的中位数**；旁边的 `[min–max]` 是各 start p50 的范围。旧报告用的是「各 start p50 的中位数」，两者在小样本下差几个百分点，不可混比。
- 冷却超时的请求仍保留该行，但排除出 headline；门控 cell 里只要有超时，`suite analyze` 直接报错而不是悄悄少算。
- 运行中的功率 / 频率 / 温度只作诊断（`clean` 列、`T °C`、`SM MHz`），不用来剔除成功请求；倍数要结合这几列读。
- 起跑温度在框架间相差 >3 °C、只剩一个有效 start 的比值，不能无标签排序。
- 音频任务（T2S）的长度不等，比「每音频秒」：`e2e_ms / audio_duration_s` 的逐请求值再取中位数。
- 模型输出违反契约（Pi0.5 action 形状、图像尺寸不符、I2T 长度不符）写入 `VOID.json`，不参与比较。

## 等工作量与最优配置（headline 的两条硬规则）

- **等工作量**：任务包只把等工作量的 case 放进 profile。BAGEL T2I 用 `-nocfg`（一分支，输出语义也相同）和 `-cfg3`（vLLM-Omni 设 `cfg_img_scale=2.0` 也算三分支）；默认请求 `t2i-<res>-default` 里 M\* 多算一个被丢弃的分支，只作参考、不进任何 profile 和 baseline。
- **各自最优配置**：cell 默认用每个框架的最优配方：vLLM-Omni BAGEL 加载调优 MoT 配置（没有就报错，不静默用未调优）；vLLM-Omni T2S 用 `async_chunk: true` + Code2Wav `enforce_eager: false`；SGLang-Omni T2S 用 partial start；V-JEPA 的 Meta 基线只有 BF16（M\* 本身是 BF16 autocast，FP32 不测）。偏离最优的 arm 是命名的反事实（`note` 里写明）。BAGEL 在 256² 对 M\* 额外跑一个 `mstar-nocompile` 候选 arm，取每个 case 里更快的作为 M\* 的最优。
- 跨框架倍数只引用 `baseline-*` 实验输出的「最强基线」那一行。
