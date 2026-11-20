# Pi0.5: what do each framework's own switches (CUDA graphs, torch.compile, attention kernel) account for? M* 104 / SGLang core 164 / vLLM-Omni 220 ms

**Question.** baseline-pi05: M* 104 ms, SGLang core 164, vLLM-Omni (eager) 220 for the same 10 flow-matching steps and [50,32] action. mstar-ablation-pi05 switched M*'s own chain / Walk / compile off and left 120 ms, still 1.4x ahead of SGLang core, and never touched the other two. Source reading found three differences that are not 'the engine': M* captures SigLIP, prefill and the 10-step action expert in CUDA graphs and torch.compiles them (max-autotune); SGLang core has two independent graph switches (prefix, action; both on); vLLM-Omni's `_attend` is a hand-written matmul + softmax, whatever --enforce-eager says. Which of those is the gap?

**Pre-registered reading.** Within a framework, each arm against its first (mstar / sglang / vllm): mstar-nocompile (TORCHDYNAMO_DISABLE=1) and mstar-nograph (hook: no accelerator graph at all) alone, and mstar-eager both: M* with the same 'no graph, no compile' as vLLM's eager recipe. sglang-nograph: both of SGLang core's CUDA graphs off. vllm-sdpa: the same attention math through F.scaled_dot_product_attention; vllm-graph: enforce_eager off (earlier: no change, the pipeline is not graph-captured); vllm-sdpa-graph both. Then across frameworks: mstar-eager against vllm (and sglang-nograph) is the like-for-like 'no graph, no compile' comparison; vllm-sdpa against sglang is vLLM with a modern attention kernel. A switch that moves a framework by <5% is not part of the gap. Hook arms need their proof file (applied=true) or the cell is rejected.

**Caveats.** Counterfactual arms are not baselines: hooks change how a framework runs (they are listed in the manifest, `external/` is untouched). The output actions are not compared; the SDPA hook changes the softmax from an explicit fp32 upcast to the kernel's own accumulation (rounding level, CPU-checked: fp32 5e-7, bf16 one ulp). Not audited: attention backends inside SGLang core and M* (both are their own kernels), image preprocessing (outside the timer on every path), the transport (HTTP against msgpack WebSocket).

Campaigns: `PI05_AUDIT_MSTAR_01_B2B` (mstar, back-to-back), `PI05_AUDIT_MSTAR_NOCOMPILE_01_B2B` (mstar-nocompile, back-to-back), `PI05_AUDIT_MSTAR_NOGRAPH_01_B2B` (mstar-nograph, back-to-back), `PI05_AUDIT_MSTAR_EAGER_01_B2B` (mstar-eager, back-to-back), `PI05_AUDIT_SGLANG_01_B2B` (sglang, back-to-back), `PI05_AUDIT_SGLANG_NOGRAPH_01_B2B` (sglang-nograph, back-to-back), `PI05_AUDIT_VLLM_01_B2B` (vllm, back-to-back), `PI05_AUDIT_VLLM_SDPA_01_B2B` (vllm-sdpa, back-to-back), `PI05_AUDIT_VLLM_GRAPH_01_B2B` (vllm-graph, back-to-back), `PI05_AUDIT_VLLM_SDPA_GRAPH_01_B2B` (vllm-sdpa-graph, back-to-back)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row | mstar | 2 | 24 | 101 [99–103] | 0% | 35 | 2,460 | - |
| mstar | droid-46row | mstar-nocompile | 2 | 24 | 107 [104–108] | 0% | 36 | 2,460 | 1.06x |
| mstar | droid-46row | mstar-nograph | 2 | 24 | 281 [281–281] | 0% | 35 | 2,460 | 2.77x |
| mstar | droid-46row | mstar-eager | 2 | 24 | 258 [257–259] | 0% | 34 | 2,460 | 2.54x |
| sglang-core | droid-46row | sglang | 2 | 24 | 165 [165–166] | 0% | 34 | 2,232 | - |
| sglang-core | droid-46row | sglang-nograph | 2 | 24 | 160 [160–160] | 0% | 34 | 2,075 | - |
| vllm-omni | droid-46row | vllm | 2 | 24 | 214 [210–216] | 0% | 35 | 2,487 | - |
| vllm-omni | droid-46row | vllm-sdpa | 2 | 24 | 204 [196–206] | 0% | 35 | 2,424 | - |
| vllm-omni | droid-46row | vllm-graph | 2 | 24 | 216 [212–219] | 0% | 35 | 2,457 | - |
| vllm-omni | droid-46row | vllm-sdpa-graph | 2 | 24 | 202 [202–203] | 0% | 35 | 2,414 | - |
