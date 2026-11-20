# Pi0.5: what do each framework's own switches (CUDA graphs, torch.compile, attention kernel) account for? M* 104 / SGLang core 164 / vLLM-Omni 220 ms

**Question.** baseline-pi05: M* 104 ms, SGLang core 164, vLLM-Omni (eager) 220 for the same 10 flow-matching steps and [50,32] action. mstar-ablation-pi05 switched M*'s own chain / Walk / compile off and left 120 ms, still 1.4x ahead of SGLang core, and never touched the other two. Source reading found three differences that are not 'the engine': M* captures SigLIP, prefill and the 10-step action expert in CUDA graphs and torch.compiles them (max-autotune); SGLang core has two independent graph switches (prefix, action; both on); vLLM-Omni's `_attend` is a hand-written matmul + softmax, whatever --enforce-eager says. Which of those is the gap?

**Pre-registered reading.** Within a framework, each arm against its first (mstar / sglang / vllm): mstar-nocompile (TORCHDYNAMO_DISABLE=1) and mstar-nograph (hook: no accelerator graph at all) alone, and mstar-eager both: M* with the same 'no graph, no compile' as vLLM's eager recipe. sglang-nograph: both of SGLang core's CUDA graphs off. vllm-sdpa: the same attention math through F.scaled_dot_product_attention; vllm-graph: enforce_eager off (earlier: no change, the pipeline is not graph-captured); vllm-sdpa-graph both. Then across frameworks: mstar-eager against vllm (and sglang-nograph) is the like-for-like 'no graph, no compile' comparison; vllm-sdpa against sglang is vLLM with a modern attention kernel. A switch that moves a framework by <5% is not part of the gap. Hook arms need their proof file (applied=true) or the cell is rejected.

**Caveats.** Counterfactual arms are not baselines: hooks change how a framework runs (they are listed in the manifest, `external/` is untouched). The output actions are not compared; the SDPA hook changes the softmax from an explicit fp32 upcast to the kernel's own accumulation (rounding level, CPU-checked: fp32 5e-7, bf16 one ulp). Not audited: attention backends inside SGLang core and M* (both are their own kernels), image preprocessing (outside the timer on every path), the transport (HTTP against msgpack WebSocket).

Campaigns: `PI05_AUDIT_MSTAR_01` (mstar, gated), `PI05_AUDIT_MSTAR_NOCOMPILE_01` (mstar-nocompile, gated), `PI05_AUDIT_MSTAR_NOGRAPH_01` (mstar-nograph, gated), `PI05_AUDIT_MSTAR_EAGER_01` (mstar-eager, gated), `PI05_AUDIT_SGLANG_01` (sglang, gated), `PI05_AUDIT_SGLANG_NOGRAPH_01` (sglang-nograph, gated), `PI05_AUDIT_VLLM_01` (vllm, gated), `PI05_AUDIT_VLLM_SDPA_01` (vllm-sdpa, gated), `PI05_AUDIT_VLLM_GRAPH_01` (vllm-graph, gated), `PI05_AUDIT_VLLM_SDPA_GRAPH_01` (vllm-sdpa-graph, gated)

Headline = pooled p50 of E2E ms over every measured request of every start; [min–max] = per-start medians. `clean` = share of requests with no clock-event (throttle) reason, SM clock held >= 98% of nominal and a clock spread <= 60 MHz during the request; it does not test temperature. T °C = mean GPU temperature at request start, SM MHz = mean clock during the request: read both beside `clean`.

| framework | case | arm | starts | n | E2E ms p50 [start range] | clean | T °C | SM MHz | vs mstar |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mstar | droid-46row | mstar | 2 | 24 | 96.4 [96.1–96.9] | 100% | 62 | 1,785 | - |
| mstar | droid-46row | mstar-nocompile | 2 | 24 | 99.6 [98.3–101.8] | 100% | 62 | 1,785 | 1.03x |
| mstar | droid-46row | mstar-nograph | 2 | 24 | 346 [340–348] | 100% | 62 | 1,785 | 3.59x |
| mstar | droid-46row | mstar-eager | 2 | 24 | 306 [305–307] | 100% | 62 | 1,785 | 3.18x |
| sglang-core | droid-46row | sglang | 2 | 24 | 147 [146–150] | 100% | 63 | 1,785 | - |
| sglang-core | droid-46row | sglang-nograph | 2 | 24 | 148 [144–148] | 100% | 64 | 1,785 | - |
| vllm-omni | droid-46row | vllm | 2 | 24 | 308 [308–309] | 100% | 63 | 1,785 | - |
| vllm-omni | droid-46row | vllm-sdpa | 2 | 24 | 293 [291–294] | 100% | 63 | 1,785 | - |
| vllm-omni | droid-46row | vllm-graph | 2 | 24 | 292 [266–309] | 100% | 63 | 1,785 | - |
| vllm-omni | droid-46row | vllm-sdpa-graph | 2 | 24 | 279 [275–286] | 100% | 63 | 1,785 | - |
