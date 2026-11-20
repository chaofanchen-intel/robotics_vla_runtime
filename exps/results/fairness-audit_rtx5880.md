# 公平性审计：M* / vLLM-Omni / SGLang-Omni 在本机基准里哪些设置不对等（RTX 5880）

目的：之前的"M* 领先 X 倍"是在**各框架默认配置**下测的。这份审计逐项列出每个会影响延迟的设置在三个框架里的实际取值、证据来源、不对等时偏向谁，以及对应的修正实验。"等工作量"不只是输出长度相同，**缓存状态、采样器、预热状态、重叠机制是否可用**也必须相同。

证据来源标记：`日志` = 本机 `server.log`；`配置` = 仓库内 yaml / 上游 deploy yaml；`源码` = `external/` 下读到的代码；`数据` = `requests.jsonl`。

## 1. 已发现的不对等（按影响排序）

| # | 设置 | 任务 | M* | vLLM-Omni | SGLang-Omni | 偏向 | 证据 | 已测到的影响 | 处理 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Thinker prefix cache** | BAGEL I2T / I2I | **开**（`KV kv: prefix cache open for main on LLM`） | **关**（上游 `deploy/bagel.yaml` 的 `enable_prefix_caching: false`） | 无 BAGEL | **M\*** | 日志、配置 | I2T TTFT 806 → 86 ms（vLLM 打开后）；M* 的 TTFT 反而是 135 ms | 已测（`bagel-vllm-prefix-cache`）：两边都开则 vLLM 略快 1.4%。**但 I2I 里 M* 是否真的命中缓存未知**（M* 日志没有 I2I 的命中行；50 步 cache on / off 为 23,528 / 23,583 ms，看不出差别），见第 12 行 |
| 2 | **Thinker prefix cache** | Qwen TXT / A2T / T2S | **关**（TP=2：`KV thinker_kv: prefix cache off at world size 2`） | **关**（上游 `qwen3_omni_moe.yaml` 三个 stage 都 false；`enable_prefix_caching=False` 在日志里） | **开**（`Init Unified Radix Cache`） | **无**（E2E 上） | 日志、配置 | **已测（`qwen-fair-txt` / `qwen-fair-a2t`）**：TXT 关对关 SGLang 1,596 / vLLM 1,610 / M* 1,648 ms，SGLang 开 1,588，vLLM 开 1,637；A2T 关 SGLang 278 / vLLM 251 / M* 255，SGLang 开 308（TTFT 慢 31 ms）、**vLLM 开 665（TTFT 481 对 77 ms，开缓存后重复请求反而慢约 400 ms，原因未查）**、M* 开 252。M* TXT 开缓存在 TP=2 **崩溃**（第 11 行） | 已测；M* TXT 没有 on 臂，其余三家结论：缓存状态对 E2E 无利于 M* 的偏向
| 3 | **SGLang 自带的 lookahead 解码门控** | Qwen TXT / A2T | 无门控 | 无对应门控 | 请求带 `seed` 或 `min_tokens>0` 时 **退回同步解码** | **M\*、vLLM**（SGLang 被请求关掉了自己的重叠） | 源码 `thinker_model_runner.py: lookahead_eligible`；我们的请求带 `seed: 42`（TXT 还有 `min_tokens: 256`）；SGLang 的 ITL 8.10/7.84 恰等于 M* 关链的 7.86 | **已测**：SGLang 的 `enable_async_decode` 关掉后 TXT ITL 5.82 → 8.00 ms（E2E 1,588 → 2,135），所以 lookahead 在去掉 seed / min_tokens 后确实生效。基线里 M* 对 SGLang 的 1.32× 就是这一项（2,135 / 1,648 = 1.30×） | 已测；对 TXT / A2T 的 SGLang 领先应作废（见 attribution §9.5）
| 4 | **推理期 JIT（新输入 shape）** | Qwen A2T / T2S | 无 | **有**：`Triton kernel JIT compilation during inference: fused_moe_kernel ... latency spike; consider extending warmup`；A2T-zh 10 个测量请求里 2 个 TTFT 507 / 629 ms（常态 ~100），T2S 15 个里 4 个 TTFA ~570 ms（常态 ~175） | 无 stall 迹象 | **M\*、SGLang** | 日志、数据 | **已测**：把 A2T 的 10 条片段放进 precondition 后，vLLM-Omni 的测量请求不再有 ~500 ms 的首见 shape 请求（precondition 里仍有 540 / 535 / 487 ms），A2T 关缓存 vLLM 251 ms 对 M* 255 ms，**基线里 A2T 对 vLLM 的 1.08× 领先是这个 JIT**；T2S 20 句预热后三家 stall 都是 0 | `qwen.rows=cycle`：precondition 与测量用同一批输入，JIT 全部落在 precondition |
| 5 | **T2S Talker 采样** | Qwen T2S | greedy（请求的 `talker_temperature: 0`） | **温度 0.9 / top_k 50 / 重复惩罚 1.05**（yaml 默认；请求的 `talker_*` 不到 Talker，`base.py` 注释） | greedy | 无（实测） | 配置、源码 | **已测**（`qwen-t2s-fit`）：vLLM Talker 改 greedy 后每音频秒 160.1 对默认 152.8 ms，稳态斜率 b 109.7 对 113.5 ms，**Talker 采样不是 vLLM T2S 慢的原因** | `qwen-t2s-fit` 的 `vllm-talker-greedy` arm（温度 0 + 保留 1.05；greedy 加 1.0 会死循环，`diag_talker_greedy_rep105.yaml` 的注释） |
| 6 | **BAGEL Thinker 采样器** | BAGEL I2T | greedy，无重复惩罚 | yaml 默认 `repetition_penalty 1.05`、`top_p 0.9`、`top_k 1`，请求只设 temperature | - | 无（实测） | 配置；两边同一张图的 caption 不一致 | **已测**（`bagel-sampling-parity`）：vLLM 去掉重复惩罚后 E2E 2,978 对 2,976 ms、ITL 17.14 对 17.15，**没有影响** | `bagel-sampling-parity`：请求显式 `repetition_penalty 1.0 / top_k 1 / top_p 1.0` |
| 7 | **torch.compile** | Qwen | **默认关**（5880 上 `TORCHDYNAMO_DISABLE=1`）。这个开关是过时的 workaround，当初失败的是 M* 自己的 `_cp_linear_kernel`（`OutOfResources: shared memory 147456 > 101376`），现在 pin 的 M* 有 fewer-stages 回退，**compile-on 能跑** | 开（日志：`torch.compile took 88.6 s`，piecewise graph） | 无 compile，只有 CUDA graph | **无**（实测 compile 对 M* 没收益） | 日志、`qwen_common.py`、`code_predictor_kernels.py`、`qwen-mstar-compile-*` | **已测**：M* compile-on 对 A1：TXT ITL 5.86 → 6.10 ms（慢 4.1%，A1/A2 差 0.2%）、A2T 5.81 → 6.00（慢 3.3%）、T2S 每秒音频 94.9 → 93.4 ms（持平）；启动多 72 s。compile-on 的 T2S 输出不可复现（同一句两次时长不同，A1 / A2 逐句相同），所以 T2S 只能比每秒音频耗时 | 公平档**不**为 compile 重跑；H200 上的 +20% 没有在本机复现，不能外推 |
| 8 | **硬件**（BAGEL） | BAGEL | 单卡 | two-stage 用 2 张卡，single-stage 单卡 | - | vLLM two-stage 多一张卡 | 配置 | I2I：M* 在 1,245 MHz，two-stage 的 DiT 在 1,610–1,863 MHz（M* 单卡更热） | 对 M* 保守；等硬件对比看 single-stage |
| 9 | **TTFT 的口径** | Qwen TXT | 每 chunk 一个 token | 每 chunk 一个 token | **一个 chunk 多个 token**（`timing_boundary` 说明） | **SGLang**（首个 chunk 迟一点才到，TTFT 偏高） | `cases.py` | SGLang TXT TTFT 偏高的一部分是口径：SGLang 开缓存后 1,071 个 token 里 1,071 命中，TTFT 仍有 110 ms（关缓存 120 ms） | 用 `max_tokens=1` 的 `qwen-ttft-*` 避开：E2E 就是首 token |
| 10 | **1-token 上限的实际输出长度** | Qwen TXT / A2T 的 `*-1tok` | **返回 2 个 token**（`max_output_tokens=1`；256 的上限是精确的） | 1 个 | 1 个 | 仅影响 E2E（M* 多一步解码约 6–10 ms） | 日志（`QWEN_TTFT_TXT_MSTAR_01_B2B` 的 contract violation：`completion is 2 tokens against a requested 1`）、`requests.jsonl` | `qwen-ttft-txt` 的 M* 4 个 cell 全被判 void，TXT 首 token 拆分没有 M* 数据 | 1-token 用例允许 cap+1，`analyze_ttft_parts.py` 改读 TTFT 而不是 E2E；M* 的 TXT 需重跑 |
| 11 | **M* Thinker prefix cache 在 TP=2 的可用性** | Qwen TXT | 关（`prefix_cache_tp: false`）；**强行打开会崩**：第 1 个请求通过，第 2 个相同 prompt（第一次缓存命中）触发 `TP ranks diverged: KV thinker_kv: rank 0 leased 8 cached pages for 0/main, whose stream here already holds tokens or has no keys`，worker_1 退出 | 可开 | 可开 | 无法比较 | 日志（`QWEN_FAIR_TXT_MSTAR_ON_01_B2B`，两个 repeat 一致）；源码（`kv/manager.py` 的 `adopt_cached_prefix`）；配置（官方文档写明的 opt-in，加载期检查通过，两个 worker 都打出 `prefix cache open`） | M* 没有热前缀的 TXT 数字；A2T 的 M* on 臂能跑（提示短，每请求只有 1 页可命中） | **已核对：不是我们的配置问题，是 M\* 在 opt-in 路径上的缺陷**（官方 TP 配置都不开该选项；M\* 自带的 14 个 TP 前缀缓存单测本机通过但用桩）；根因未定位；`mstar-tp-cache-probe` 已跑：预算 4,096（提示一个 chunk）时 TP=2 + 前缀缓存不崩（13 个请求成功），所以触发条件是「命中 + 提示被切成多个 chunk」；应向上游报。`qwen-fair-txt` 删掉 M* on 臂（它会让后面的 pinned 臂被跳过） |
| 12 | **BAGEL I2I 的 prefill 是否被各框架跳过** | BAGEL I2I | prefix cache 默认开，但**开关在 I2I 上无效**：5 步 cache on / off 为 2,760 / 2,764 ms，50 步 23,506 / 23,550（`bagel-i2i-fair`）；日志没有命中行，分不出"路径无效"还是"永不命中" | cache 关：Thinker 对两个 5k-token 分支各做一次 prefill（nsys：15.3k launch、2.05 s busy，而 single-stage 在 DiT 之前只有 6.4k launch、0.85 s） | - | **vLLM-Omni 默认配置不利于它自己** | `bagel-i2i-fair`、`bagel-connector-nsys` | **已测**：vLLM 打开 Thinker 缓存后 5 步 4,564 → 3,523 ms（−1,041），50 步 25,964 → 24,959（−1,005）；M* 的缓存开关无影响，所以 I2I 领先里**没有 M\* 的缓存**，但有 vLLM 不开缓存的 1,041 ms | 已测；I2I 的对比应写成"两个开关都配好后 1.28×（5 步）/ 1.06×（50 步）"，见 attribution §7.7；M\* 的单进程 busy 2.65 s ≈ single-stage 2.63 s，做的是一次 prefill 的工作量 |
| 13 | **Thinker 的 chunked-prefill 预算** | Qwen TXT / A2T；BAGEL | **1,024 token**（`ThinkerSubmodule.MAX_BATCH_TOKENS`；BAGEL LLM 也是 1,024，Qwen3.5 是 2,048） | Thinker `max_num_batched_tokens: 8192`（BAGEL 32,768） | 默认（未设置） | **vLLM**（C=1 的首 token） | 源码、配置；`qwen-prefill-nsys`：M* 一个请求 3 次 forward（1,024 + 48 token + 第二个输出 token），vLLM 1 次 | 1,072-token 提示 TTFT：M* 146 对 vLLM 99 / SGLang 117；第 2 块（48 token）约 40 ms；第 1 块 77 ms 与 vLLM 的整个提示相同；上一版的「M* prefill 每 1k token 86 ms」是跨边界的两点直线造成的假象 | **这是一个被我们漏掉的不对等开关**（M* 的 `max_batch_tokens` 是官方 serving 配置项），**但反事实表明它不是 TXT TTFT 差距的原因**：`qwen-prefill-chunk` 预算 4,096 后 1,072-token 的 TTFT 138 → 154 ms（更慢）；`bagel-prefill-chunk` 预算 8,192 只让 I2T TTFT 降 6–7 ms、I2I 不动。开关仍不对等，所以保留这一行；两个预算都不是 M* 的默认，不改基线。TXT 差距的机制待 `prefill-chunk-nsys` / `prefill-len` |
| 14 | **时钟 / 温度**（BAGEL I2I） | BAGEL I2I | 单卡，5 步 49–53 °C、1,610 MHz；50 步 64 °C、1,249 MHz | 两张卡各半载，5 步 37–40 °C、2,147–2,430 MHz；50 步 34 °C、1,808–1,834 MHz | - | vLLM two-stage（时钟高 1.3–1.5×） | `requests.jsonl` 的 thermal_monitor | M* 在更低的时钟下仍持平，所以对 M* 保守；但 3% 以内的每步差异（476 对 461 ms/步）不能当 kernel 差异读 | 锁时钟需要 root，没有做；不下 3% 以内的结论 |

## 2. 已核对、基本对等的设置

| 设置 | 结论 | 证据 |
|---|---|---|
| 输出长度 | 对等。TXT / BAGEL I2T 用 `ignore_eos` + `max_tokens` 钉死并逐请求校验；A2T 三家 28 / 28 / 28 个 token | validator、`requests.jsonl` |
| MoE kernel 调优 | 对等且被强制检查：baseline arm 的 `MOE_TUNED` 要求日志里没有 "Using default MoE config"，M* 的 kernel table 命中。fair 实验的每个 Qwen arm 都带这个检查 | `catalogue.py: MOE_TUNED` |
| 注意力后端 | M* 的 FlashInfer 对 vLLM 的 FLASH_ATTN：互相换后端，M* 与 vLLM 都不变 | `bagel-attn-backend` |
| BAGEL 输入图像预处理 | 对等：`image_preprocess: vllm`；图像必须已是目标尺寸 | `bagel/client.py` |
| 采样 greedy / 并发 | Qwen Thinker greedy（temperature 0）；C=1 | 客户端 |
| BAGEL MoT GEMM | vLLM 用了 `bagel-mot-tune` 调过的表（M* 自带 kernel tables）；调不调对 E2E 的影响见 `bagel-mot-tuned-vs-default` | 结果文件 |
| 测量协议 | back-to-back、同一台机器、同一热状态检查、无请求间 gap | `protocol` |

## 3. 尚未核对、需要在结论里保留的项

- **Qwen 三家的 KV / 显存预算、`max_num_seqs`、graph 捕获档位**：C=1 下按常理不影响，但没有单独的对照。
- **vLLM-Omni 的 Thinker→DiT KV 交接用的是默认 `SharedMemoryConnector`**（序列化成 bytes，~0.9 GB/s）；它还有 Mooncake / NIXL 两个直接搬 GPU 张量的 connector，本机已装到用户目录（`bagel-connector` 实验）。**已测（`bagel-connector`）**：5 步固定成本 shm 6,809 → Mooncake-TCP 4,558（−2,251 ms）/ NIXL-UCX 5,308（−1,501），single-stage 2,936；50 步 28,190 → 25,943 / 26,832，single-stage 24,122；三种 connector 的输出图逐字节相同。所以 I2I 对 two-stage 的差距约 56–58% 是默认 connector，不是两阶段架构本身；剩下约 1.6 s（Mooncake）已拆开：约 1,041 ms 是 Thinker 不开缓存时的第二次分支 prefill（开缓存即抵掉，`bagel-i2i-fair`），hand-off 约 0.28 s（`bagel-connector-nsys`），其余约 300 ms 是前端等。先前用 `bagel-connector-timeline` 部分拆开（毫秒日志，5 步，n=10）：Mooncake 发送侧三个分支合计只有 56 ms（340 MB 的 KV 13 ms），gap 25 ms，所以这 1.6 s **不在发送侧**；stage 0 到第一个 KV 就要 2.19 s、DiT 侧 2.0 s，接收侧没有日志（DiT 进程不输出）。1024²（KV 516 MB，是 512² 的 1.52×）下 shm → Mooncake 省 3,414 ms，是 512² 的 1.52×，节省与 KV 字节数同比例，connector 占 shm → single-stage 差距的 54%。接收侧很快（KV 拷出 → DiT 第一次 H2D 133–150 ms），见第 12 行。结果只对本机的 TCP / PCIe 成立，没有 RDMA / NVLink。
- **vLLM 的 Cosmos3 `enforce_eager` / compile 档位**：日志显示 vLLM 已做 compile（`triton_poi_fused` kernel），所以"vLLM 可以做同样的事"里 compile 一项他们已经做了；批量 CFG 一项需要改 vLLM-Omni 的代码，没有办法做反事实。
- **Pi0.5**：没有审计。

## 4. 修正后的对比怎么做

"公平档"的定义（所有新实验遵循）：

1. **缓存状态显式且三家一致**：要么三家都关（冷 prefill），要么三家都开（热 prefix）。两种都跑，因为缓存对重复输入的 fixture 极其敏感。
2. **请求对三家都"可用 lookahead"**：不带 seed / min_tokens；用 `ignore_eos + max_tokens` 钉长度。
3. **预热覆盖测量用到的每个输入 shape**（`qwen.rows=cycle`）。
4. **采样器显式对等**：BAGEL 去掉 yaml 的重复惩罚；T2S 给 vLLM Talker 的 greedy 对照。
5. 每个 arm 带 `Expect`：cache 状态、config 文件、MoE 调优表都在日志里被检查，不满足 `analyze` 拒绝出数。

实验与命令见 `mstar-advantage-attribution_rtx5880.md` §12。
