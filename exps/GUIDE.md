# 测试指南（全量任务）

> 目标：**(1) 复现论文的基础对比**（等工作量、各自最优配置），**(2) 用消融找出 M\* 的优势来自哪里**。下面是**全部**实验的一份指南；已有结果主要在 H200 NVL 上，8×RTX 5880 等其他机器上选做哪些见 §9。命令在仓库根目录；`$HW` 是机器相关参数（本机省略，自动检测；5880 上使用 `--hardware rtx5880`，共享机器，每个 cell 自动取空闲的物理 GPU、只取部署所需张数；也可显式写 `--gpus 2,3,5`）。**时长是 harness 的保守估计**（含每个 cell 约 2–3 分钟的服务启动与 gate），背靠背实际更短，门控在会降频的机器上更长；是单次通过，不含重跑。

## 0. 流程：先定协议，再跑基线

同一个任务里，不同框架对测量协议的反应不一样（Pi0.5：背靠背相对门控，vLLM 0.67×、SGLang 0.86×、M\* 0.94×，跨框架倍数从 3.48× 变成 2.45×）。所以**协议必须先由证据决定，baseline 才能跑**：

```
环境自检 ─► smoke ─► 调优 ─► protocol-ab-<任务> ─► decide-protocol ─► baseline-* ─► 消融（advantage / fairness / profile）
```

- `protocol-ab-*` **每个任务都有**（8 个）：每个只测一个代表 case，**2 次 fresh start × 5 个测量请求**（Pi0.5 是 12 个）。目的是判断「协议改变不改变结论」，不是测精确延迟；2×5 的 pooled p50 噪声约 ±3–4%，只能分辨 >5% 的协议效应（Pi0.5 是 14–33%），更小的效应对 headline 倍数无影响，按「不敏感」处理。
- `python -m exps.suite decide-protocol` 读完成的 protocol-ab，按每个框架的 `背靠背 / 门控` 比值（pooled p50）决定，规则在 [protocols.py](protocols.py)，容差 5%（A1/A2 漂移）：

  | 情况 | 决定 | 含义 |
  |---|---|---|
  | 所有框架比值都在 1±5% 内，**且**背靠背的 SM 时钟没掉（≥门控的 97%） | **背靠背** | 协议无影响且没降频，用便宜的那个 |
  | 同上但时钟掉了 / 没有时钟证据 | 硬件默认协议 | 背靠背可能降频，保守 |
  | 各框架偏移量不同（max/min−1 > 5%） | **两种协议都跑**，两张表都报 | 协议改变了框架间比值，没有单一诚实的数 |
  | 各框架一起偏移同样的量 | 硬件默认协议 | 框架间比值不变，只是绝对值随协议变 |

- 决定写进 `profiles/protocol/<硬件档案>.json`（每台机器各自一份，证据只对该机器有效），报告在 `results/protocol-decision_<档案>.md`。**没有决定的任务，`baseline-*` 拒绝规划**（`--protocol` 显式指定可绕过，须在报告里说明理由）。决定为「两种」时，`baseline-*` 自动对每个协议各建一组 campaign（后缀 `_B2B`）并各出一张表。消融（M\* 对自己）不需要两种，用单一决定或硬件默认。

## 1. 全部任务（38 个条目）

`python -m exps.suite list` 列同样的内容，`plan <名字>` 给出当前的 cell 数、估算和每个 arm 的 `exps.run` 命令。下表的估算是**本机（H200 NVL）**的 harness 估计：门控列按每请求约 3 分钟冷却算（会降频的机器上的保守值），背靠背列按连续发请求算；`baseline-*` 与消融实际用哪一列由 §0 的协议决定。


**baseline（论文复现）**

| 实验 | cells | 门控 | 背靠背 | 内容 |
|---|---:|---:|---:|---|
| `baseline-bagel` | 36 | 15.6 h | 5.1 h | BAGEL 512²：T2I 一分支 / 三分支、I2I、I2T；M\* vs vLLM two-stage 与 single-stage（取更快的） |
| `baseline-bagel-1024` | 24 | 8.0 h | 3.1 h | BAGEL 1024²（论文的分辨率）；会降频的机器上一个请求就跨过降频拐点，仅稳定散热时有意义 |
| `baseline-qwen-t2s` | 18 | 7.8 h | 2.5 h | Qwen3-Omni T2S（EN / ZH）；三框架，各自最优配方 |
| `baseline-qwen-txt` | 9 | 3.9 h | 1.3 h | Qwen3-Omni TXT（固定 1K prompt） |
| `baseline-qwen-a2t` | 18 | 16.8 h | 3.5 h | Qwen3-Omni A2T（EN / ZH） |
| `baseline-orpheus` | 18 | 7.8 h | 2.5 h | Orpheus B=1/4/16；M\* vs VoxServe |
| `baseline-vjepa-ac` | 18 | 7.8 h | 2.5 h | V-JEPA 2-AC H4/H15/H30；M\* vs Meta-native（都是 BF16） |
| `baseline-cosmos3` | 6 | 2.6 h | 0.8 h | Cosmos3-Nano 256²（本地能力对照，不是论文基准） |
| `baseline-pi05` | 9 | 24.1 h | 3.5 h | Pi0.5 全 46 行 fixture；M\* / SGLang core / vLLM eager |

**protocol（先于 baseline）**

| 实验 | cells | 门控 | 背靠背 | 内容 |
|---|---:|---:|---:|---|
| `protocol-ab-pi05` | 18 | 9.2 h | 9.2 h | 门控 vs 背靠背 vs 固定 30 s 间隔（把空闲和热分开） |
| `protocol-ab-bagel` | 8 | 2.3 h | 2.3 h | 门控 vs 背靠背（t2i-512-nocfg） |
| `protocol-ab-vjepa-ac` | 8 | 2.3 h | 2.3 h | 门控 vs 背靠背（h15） |
| `protocol-ab-orpheus` | 8 | 2.3 h | 2.3 h | 门控 vs 背靠背（b1） |
| `protocol-ab-cosmos3` | 8 | 2.3 h | 2.3 h | 门控 vs 背靠背 |
| `protocol-ab-qwen-t2s` | 12 | 3.4 h | 3.4 h | 门控 vs 背靠背（verbatim-en） |
| `protocol-ab-qwen-txt` | 12 | 3.4 h | 3.4 h | 门控 vs 背靠背 |
| `protocol-ab-qwen-a2t` | 12 | 6.1 h | 6.1 h | 门控 vs 背靠背（a2t-en） |

**advantage（M\* 的优势来自哪里）**

| 实验 | cells | 门控 | 背靠背 | 内容 |
|---|---:|---:|---:|---|
| `mstar-ablation-qwen-t2s` | 14 | 6.3 h | 2.1 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：qwen-t2s |
| `mstar-ablation-bagel` | 28 | 11.9 h | 4.2 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：bagel |
| `mstar-ablation-vjepa-ac` | 14 | 6.3 h | 2.1 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：vjepa-ac |
| `mstar-ablation-pi05` | 14 | 14.0 h | 2.8 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：pi05 |
| `mstar-ablation-orpheus` | 14 | 6.3 h | 2.1 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：orpheus |
| `mstar-ablation-cosmos3` | 14 | 6.3 h | 2.1 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：cosmos3 |
| `mstar-ablation-qwen-a2t` | 14 | 11.9 h | 2.8 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：qwen-a2t |
| `mstar-ablation-qwen-txt` | 14 | 6.3 h | 2.1 h | M\* 组件矩阵（A1/C/S/SC/W/WC/A2，每臂 2 次 start）：qwen-txt |
| `bagel-attn-backend` | 24 | 10.4 h | 3.2 h | flash-attn 假设：调用计数探针 + 屏蔽 flash_attn + 后端互换 |
| `bagel-cfg-ladder` | 18 | 7.8 h | 2.5 h | E2E 随 CFG 分支数（1 / 默认 / 3），每额外分支每步多少 ms |
| `bagel-steps` | 22 | 9.5 h | 3.1 h | E2E = a + b·steps：差距在每请求固定开销还是每步 |
| `bagel-concurrency` | 18 | 7.8 h | 2.5 h | I2T 的 B=1/4/16（论文的批量主张） |
| `t2s-concurrency` | 18 | 13.2 h | 3.1 h | T2S 的 B=1/4/16（论文的 B=16 吞吐主张） |

**fairness（基线配置 / 调优值多少）**

| 实验 | cells | 门控 | 背靠背 | 内容 |
|---|---:|---:|---:|---|
| `bagel-mot-tuned-vs-default` | 8 | 3.4 h | 1.2 h | vLLM 的 MoT 调优在 E2E 上值多少 |
| `t2s-deploy-config` | 12 | 5.2 h | 1.6 h | vLLM async-chunk + Code2Wav graph、SGLang partial start 各值多少 |
| `a2t-moe-tuned-vs-default` | 4 | 3.4 h | 0.8 h | M\* 自己的 MoE 调优在 E2E 上值多少 |

**profile（nsys）**

| 实验 | cells | 门控 | 背靠背 | 内容 |
|---|---:|---:|---:|---|
| `bagel-nsys-equal-work` | 4 | 1.1 h | 0.5 h | 等工作量下每步 GPU busy / kernel（nsys，只看结构） |
| `t2s-stage-nsys` | 3 | 1.3 h | 0.4 h | 三框架 T2S 每 stage 的 GPU 时间线（nsys，只看结构） |

**tool（调优与探针）**

| 实验 | cells | 门控 | 背靠背 | 内容 |
|---|---:|---:|---:|---|
| `bagel-mot-tune` | — | — | — | 生成 vLLM-Omni 的 BAGEL MoT 调优文件（约 5 分钟） |
| `mstar-moe-tune` | — | — | — | 扫 M\* 的 Qwen3-Omni MoE tile 表（Thinker + Talker） |
| `vllm-moe-tune` | — | — | — | 调 vLLM 的 Qwen3-Omni fused-MoE tile（H200 系列 wheel 里已带，不用调） |
| `sglang-moe-tune` | — | — | — | 调 SGLang 的 Qwen3-Omni fused-MoE tile（在安装的 Triton 版本下） |
| `thermal-probe` | — | — | — | GPU 持续负载探针，约 10 分钟 |


全部跑一遍的保守上限：全门控约 258 h，全背靠背约 95 h（含 protocol-ab 两种协议都跑的部分）；实际取决于协议决定，通常远低于全门控。

**取舍**：BAGEL **256² 不需要**（host 受限、跨 campaign 漂移 3–12%，不是论文工作点），工作点是 **512²（主）+ 1024²（论文对齐）**，`--cases t2i-256-nocfg` 仍可选。**轮次已按任务压缩**，`--repeats N --measured M` 可对任何实验整体缩放：基线 2–3 次 fresh start × 3–10 个请求；消融矩阵每臂 2 次 start（A1 与 A2 已把漂移夹住）；并发只测 B=1/4/16；BAGEL steps 只测 5/25 步；Orpheus 基线只测 B=1/4/16。删掉的旧实验见 [README](README.md)。

## 2. 环境自检（约 0.5 小时）

```bash
envs/mstar/bin/python -m exps.hardware $HW          # 检测到的 GPU、档案、默认协议、显存门限
envs/mstar/bin/python -m exps.doctor                # 源码 ref / 补丁 / fixture / 各 env 的 nvcc 与 CUDA 头文件版本 / GPU 遥测，必须全 ok
bash exps/patches/bootstrap.sh --verify
envs/mstar/bin/python -m pytest exps/tests -q       # CPU 测试全绿
```

## 3. smoke —— 每个框架 × 每个模型起得来（1–2 小时）

三框架刚升级，先证明服务能起。smoke = 每 cell 1 次 fresh start、1 个预热、1 个测量请求。

```bash
envs/mstar/bin/python -m exps.run sweep --profile smoke --campaign SMOKE_01 $HW
# 一次只想验证几个任务：--tasks bagel vjepa_ac orpheus cosmos3 pi05 qwen_t2s qwen_txt qwen_a2t
```

每个 cell 看 `exps/runs/<campaign>/<TASK>/<framework>/<case>/r01/`：`manifest.json` 的 `status: completed`、`hardware`、`service_command`（`CUDA_VISIBLE_DEVICES`、配置路径、`VLLM_TUNED_CONFIG_FOLDER` / `MSTAR_KERNEL_TABLES`）；`summary.json` 的 `protocol.name`；起不来读 `server.log`，**失败 cell 换 `--campaign` 名重跑**。smoke 的第一个 BAGEL case 是 `t2i-512-nocfg`。顺手 grep 日志里的调优警告（这就是下一步的待办）：

```bash
grep -l  "No tuned MoT config"              exps/runs/SMOKE_01/BAGEL/vllm-*/*/r01/server.log
grep -rl "Using default MoE config"         exps/runs/SMOKE_01/QWEN_*/vllm-omni/*/r01/server.log
grep -rl "Using default MoE kernel config"  exps/runs/SMOKE_01/QWEN_*/sglang-omni/*/r01/server.log
grep -rl "kernel table: no moe_gemm"        exps/runs/SMOKE_01/QWEN_*/mstar/*/r01/server.log
```

验收：每个（任务 × 框架）至少一个 completed 的 smoke cell；起不来的有明确原因。

## 4. 调优 —— 能调优的框架都要调优

基线用默认 tile 而 M\* 调优过（或反过来），比的就是配置不是引擎。**换 GPU、升级 vLLM-Omni / SGLang / Triton 后都要重做**：调优文件按 GPU 名（SGLang 还按 Triton 版本）匹配，不匹配时框架只在日志里警告然后用默认值。四个调优器的源码都在本地 `external/`（`mstar`、`vllm-omni`、**`vllm` v0.32.0、`sglang` v0.5.22**，后两个是为调优器 checkout 的 core 源码，版本与 env 一致，记在 [SOURCE-RECORD](../external/SOURCE-RECORD.md)）。先看清单：

```bash
envs/mstar/bin/python -m exps.tuning status $HW             # OK / STALE / MISSING，缺了怎么生成
envs/mstar/bin/python -m exps.tuning status $HW --require   # 全部就绪才返回 0（正式测量前的闸门）
```

| 项 | 谁 | 命令（均需要空闲 GPU） | 说明 |
|---|---|---|---|
| BAGEL MoT GEMM | vLLM-Omni | `python -m exps.suite run bagel-mot-tune --execute` | 约 5 分钟；需要 `envs/vllm-omni/bin/pip install ray`（一次）；batch 1 3 4 12 16 48 覆盖 256–1024、一 / 三分支。现有文件是旧版本生成的，升级后建议重做 |
| MoE tile（Thinker `E=128,N=1536,K=2048,top_k=8`、Talker `E=128,N=768,K=1024,top_k=6`） | **M\*** | `python -m exps.suite run mstar-moe-tune --execute` | M\* 按**精确 torch 设备名**查表：包里带 H100 / H200 / GB200 / B300 的表，**H200 NVL（`NVIDIA H200 NVL`）和 RTX 5880 都匹配不上**，所以本机也缺。harness 对每个 M\* cell 自动设 `MSTAR_KERNEL_TABLES=exps/configs/mstar/kernel_tables` |
| fused-MoE（同样两个形状：N=768 / N=384） | vLLM-Omni | `python -m exps.suite run vllm-moe-tune --execute` | 需要 `ray`。vLLM 把 H200 系列都映射到 `NVIDIA_H200`，wheel 里带这张表，**本机不用调**；5880 没有，要调。输出到 `exps/configs/vllm_omni/tuned/`，harness 自动设 `VLLM_TUNED_CONFIG_FOLDER` |
| fused-MoE Triton | SGLang-Omni | `python -m exps.suite run sglang-moe-tune --execute` | **本机现有的表是 triton 3.7.1 下调的，env 已是 3.8.0：SGLang 回退到旧表并警告，`tuning status` 标 STALE，应在 3.8.0 下重调**。输出到 `exps/configs/sglang_omni/moe_<档案名>/configs/triton_<版本>/`；搜索空间大，耗时较长 |
| 其他 | FlashInfer JIT / autotune、torch.compile、CUDA graph 捕获 | — | 都在服务预热里自动完成（每个 cell 至少 1 个不计分预热；A2T 用 6 个） |

两个 MoE 调优器的输入是 `exps/configs/tuner_inputs/{thinker,talker}/config.json`（Qwen3Moe 形状，已核对两个调优器解析出的形状分别是 Thinker `E=128,top_k=8,N=768,K=2048`、Talker `E=128,top_k=6,N=384,K=1024`）。调优命令本身（batch 列表、搜索空间）用各脚本的默认值，**脚本实际运行还没在本机验证过**，第一次跑时盯一下输出文件名是否与 `tuning status` 期望的一致。

**验证**：`tuning status --require` 通过；再重跑 §3 的 grep，几类警告都应消失；`baseline-qwen-*` 的分析会逐个服务日志检查（vLLM / SGLang 的 “Using default MoE …”、M\* 的 “kernel table: no moe_gemm” 都判失败），`baseline-bagel` 检查 “MoT config loaded from”。`a2t-moe-tuned-vs-default` 量出 M\* 自己的 MoE 调优在 E2E 上值多少。

## 5. 定协议（先于所有基线）

```bash
envs/mstar/bin/python -m exps.suite run 'protocol-ab-*' $HW --execute
envs/mstar/bin/python -m exps.suite decide-protocol $HW      # 写 profiles/protocol/<档案>.json 和 results/protocol-decision_<档案>.md
```
**Pi0.5 的 `gap30` 臂**把「空闲间隔」和「门要求凉卡」分开：热稳定的机器上若门控 / 背靠背的差依旧存在，就是空闲间隔（host 侧冷启动 / 频率恢复），不是热。

## 6. 论文复现基线（`baseline-*`，用决定的协议）

等工作量、各自最优、调优已加载；分析表给出「每个框架在各自最优 arm 下」和「最强基线 / M\*」，**跨框架倍数只引用最后一行**，并写明协议。

```bash
envs/mstar/bin/python -m exps.suite run 'baseline-*' $HW --execute      # 或逐个：baseline-bagel baseline-vjepa-ac ...
envs/mstar/bin/python -m exps.suite analyze 'baseline-*' $HW
```

读结果：先看 `clean` / `T °C` / `SM MHz` 列；起跑温度差 >3 °C 的比值要打标签；任何 cell 缺失 / 协议不符 / 调优警告，`analyze` 直接报错不出表。

## 7. 消融 —— M\* 的优势来自哪里

**主力：`mstar-ablation-<任务>`**（A1 / C / S / SC / W / WC / A2，7 臂，每臂 2 次 start，每任务一个代表 case）。只用 M\* 源码里已有的环境变量：

| 臂 | 关掉什么 |
|---|---|
| C | torch.compile |
| S | 推测发射链（Worker 在第 N 步返回前按 Walk 发射第 N+1 步） |
| W | walk-blind：S + 预规划 + 开每步同步，Worker 只发射被告知的这一步（保留 Walk / Loop 结构和 kernel，**不是**手写循环 / stage 框架的对照） |
| SC / WC | 叠加；WC 对比 `baseline-*` 的最强基线 = 没有前瞻也没有 compile 时 M\* 还领先多少 |

判读：A1 与 A2 的差是漂移，臂间比值要超过它才算；SC ≈ C×S 说明相乘；W ≈ S 说明前瞻的价值都在推测链里；比值接近 1.0 是**空结果**，照实写。没有 CUDA graph / 分页 KV / FlashInfer 的开关，这几项不声称，靠 nsys 的 GPU busy 对比（`bagel-nsys-equal-work`）。

| 实验 | 读法 |
|---|---|
| `bagel-attn-backend` | 先看 proof（T2I 的 ViT 调用数 = 0、LLM 注意力是 FlashInfer、屏蔽 `flash_attn` 后走 SDPA 回退）；再看 vLLM / M\* 两两配对的比值是否都不变。**在 sm_89（5880）上先核对 vLLM 日志里实际解析到的注意力后端** |
| `bagel-cfg-ladder` | 每额外分支每步的 ms；默认请求下的落后是工作量差，不是效率差 |
| `bagel-steps` | E2E = a + b·steps 的 a、b；`srv-steps` 只有 M\*（它的步数是服务端配置） |
| `bagel-concurrency` / `t2s-concurrency` | B=16 的框架间比值（论文的批量吞吐主张） |
| `bagel-mot-tuned-vs-default`、`t2s-deploy-config`、`a2t-moe-tuned-vs-default` | 公平性：untuned / tuned，plain / best；`untuned`、`builtin`、`plain` 臂不是最优配置，只用来量调优 / 配置的价值 |
| `bagel-nsys-equal-work`、`t2s-stage-nsys` | 需要 `nsys` 在 PATH；**只看结构，不看 E2E**（nsys 会把时间线拉长 1.6–2.2×） |

## 8. 按顺序的命令（全量）

```bash
P="envs/mstar/bin/python -m exps"
$P.hardware && $P.doctor && bash exps/patches/bootstrap.sh --verify
$P.run sweep --profile smoke --campaign SMOKE_01                       # §3
# §4 调优（缺什么调什么；每条都需要空闲 GPU）
$P.suite run bagel-mot-tune mstar-moe-tune sglang-moe-tune --execute   # vllm-moe-tune 在 H200 系列上可省
$P.tuning status --require
# §5 协议
$P.suite run 'protocol-ab-*' --execute && $P.suite decide-protocol
# §6 基线
$P.suite run 'baseline-*' --execute
# §7 消融
$P.suite run 'mstar-ablation-*' --execute
$P.suite run bagel-attn-backend bagel-cfg-ladder bagel-steps bagel-concurrency t2s-concurrency --execute
$P.suite run bagel-mot-tuned-vs-default t2s-deploy-config a2t-moe-tuned-vs-default bagel-nsys-equal-work t2s-stage-nsys --execute
# 分析（没跑的会报 no numbers reported，正常）
$P.suite analyze 'baseline-*' 'mstar-ablation-*' 'protocol-ab-*' bagel-attn-backend bagel-cfg-ladder bagel-steps bagel-mot-tuned-vs-default
```
任何一个实验失败会跳过它的后续 arm，不影响下一个实验；失败的 cell 用 `--run-id 02` 新开 campaign 重跑。`bash run.sh --plan` 列出主要实验的计划。

## 9. 在 8×RTX 5880 等其他机器上选做

整份指南在另一台机器上同样适用（`--hardware <档案> --gpus ...`）：每台机器有自己的协议决定和调优文件（按 GPU 名匹配），数据不与本机混表。**选做建议**：

| 值得做 | 原因 |
|---|---|
| `thermal-probe`（先做） | 确认真的不降频，再保留档案里的 `thermal_stable: true` 与默认背靠背 |
| 全部单卡 `baseline-*`、`mstar-ablation-*` | 无降频的数据更干净；换 GPU 架构（Ada sm_89）也检验结论是否依赖 Hopper |
| `baseline-bagel-1024` | 论文的分辨率，只有不降频的机器才有意义 |
| `protocol-ab-pi05`（含 gap30） | 热稳定时门控 / 背靠背的差若依旧存在，就是空闲间隔而不是热 |
| `bagel-attn-backend` | sm_89 没有 FlashAttention-3 的 kernel，先核对 vLLM 实际解析到的注意力后端 |
| 调优：`bagel-mot-tune`、`mstar-moe-tune`、`vllm-moe-tune`（5880 没有 wheel 表）、`sglang-moe-tune` | 每块 GPU 都要重做 |

**需要额外工作的**：

- **Qwen3-Omni（T2S / TXT / A2T 与相关消融）单卡放不下**（30B BF16 ≈ 61 GB > 48 GB）；5880 档案标记 `needs_overlay_for_tasks`，本仓库的硬件 overlay 用 3 张卡（能跑的最少张数；论文是 2×H200，Thinker 与 Talker 分卡）：Thinker TP=2 占两张，Talker + Code2Wav 占第三张，SGLang-Omni 切到非 colocated speech pipeline（详见 [configs/hardware/README.md](configs/hardware/README.md)）。M* 在 sm_89 上还需 `TORCHDYNAMO_DISABLE=1`，并且 Talker 的 code-predictor Triton 内核（按 H200 调，要 144 KiB shared memory，sm_89 每 block 上限 99 KiB）靠补丁 `mstar-05-code-predictor-smem-fit` 在启动失败时自动减少 stages / 缩小 K 步长。VoxServe 的 FlashInfer JIT 在 nvcc 12.0 + sm_89 上编不过 FP8 转换，`envs/setup.sh voxserve` 会去掉两个 FP8 编译宏（doctor 检查）。这些是新的多卡 / 硬件被测条件，不与 H200 单卡数据混合。
- **vLLM two-stage BAGEL 在一张 48 GB 卡上放不下**（两个 stage 各存一份完整 LLM，约 2×28 GB）；5880 overlay 把 Thinker 与 DiT 放在两张卡（`gpus_per_task`），harness 挑当时空闲的卡。
- **M\* BAGEL 单卡要缩 KV 池**：默认 2048 页×128 token（约 15 GB）加 29 GB 权重在 48 GiB 卡上放不下（512² 三分支、I2I 和全部 1024² 都 OOM）；overlay `bagel_single_gpu_kv512.yaml` 把池限到 512 页（约 3.75 GB，足够一个 1024² 三分支请求约 13k token）。池只是容量，B=1 时不改变每步工作量。
- **论文的多卡拓扑**：Qwen 2 卡分 Thinker / Talker（本机 3 卡）；BAGEL T2I/I2I 是 3 卡 CFG 并行（每个 CFG 分支一个 rank，I2T 和附录 E 才是单卡）。后者由 `baseline-bagel-cfgpar` 复现（`--opt bagel.topology=cfg-parallel`）：M\* 3 卡，vLLM-Omni single-stage 3 卡（`cfg_parallel_size: 3`），vLLM-Omni two-stage 在 48 GiB 卡上要 4 卡（Thinker 不能和 DiT rank 共卡）。5880 之间没有 NVLink，分支交换走 PCIe，绝对值和比值都不等同于论文的 NVLink H100。单卡机器（H200）上该实验被跳过，不算失败。
- 5880 档案里 `empty_gpu_max_memory_mib: 64`、`service_gate_continuous_s: 15` 是估计值，用 `exps.hardware` 打印的空载显存校正。`loaded_gpu_max_memory_mib: 48500` 根据 GPU 4 上 M* BAGEL 图捕获后的实测空闲占用（47,638 MiB）设置。
- **多卡门控范围**：温度 / 时钟 / 利用率门只读主 GPU；其余卡要求起停时无计算进程。8 张卡同机并行跑不同实验不建议（GPU 锁是全局的，且会引入 PCIe / CPU / 内存带宽争用）。

## 10. 汇报规则与已知风险

- 只在「等工作量 + 各自最优 + 调优已加载 + 协议已决定」时给 headline；倍数引用 `baseline-*` 的「最强基线」行，写明协议、机器与硬件档案；headline 是 pooled p50。
- 所有新 cell 都没在升级后的源码上跑过：`configs/` 里的 yaml 是旧版本的副本，hook 与 M\* 源码绑定（采样 hook 已在新源码上 CPU 验证）。第一次 smoke 暴露的问题（flag 改名、配置字段变化、端口 / 健康检查路径）要记下并修，再往后。
- 本机上 SGLang 的 MoE 表是 STALE（triton 3.7.1）、vLLM 的 MoT 表是旧版本生成的、M\* 的 MoE 表缺失：这几项做完前，Qwen / BAGEL 的新数据都不是各自最优。
- 第一次 smoke 已暴露两个问题并修好：①`vllm-omni` / `sglang-omni` env 里的 `nvidia-cuda-nvcc` 被升到 13.4.92，而 CUDA 头文件是 13.0，FlashInfer 启动时 JIT 编译 sampling 内核报 “CUDA compiler and CUDA toolkit headers are incompatible”（这两个框架的所有 cell 都起不来）；按 `envs/setup.sh` 的 pin 重装 `nvidia-cuda-nvcc / nvidia-nvvm / nvidia-cuda-crt==13.0.88`，`doctor` 现在会检查这一项。②Qwen T2S 客户端发 `code_predictor_top_k=-1`，M\* 只接受 ≥ 0（0 表示关），每个请求都是 HTTP 400，改成 0。③第二轮 smoke 又暴露两个：M\* 的 `_text_chunks` 是 bytes、其他框架是 str，Qwen T2S / A2T 的验证器按 str 拼接而 `TypeError`（现统一用 `adapters.response_text`，BAGEL 的同类写法也一并改了）；Qwen TXT 给 M\* 的上限曾按“prefill 首 token 不计入”减 1，实测请求 255 就只返回 255（conductor 把 prefill 采样的首 token 计入上限），改成所有框架都发 256。**失败的 cell 一律新 campaign id，不要 `--resume` 同一个 id**。
- 论文里依赖多卡或大批量的结论只能在多卡阶段复现；单卡结果只回答单卡的问题。
