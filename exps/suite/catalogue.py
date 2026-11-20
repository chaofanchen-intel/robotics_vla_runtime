#!/usr/bin/env python3
"""Every comparison and ablation the harness can run, declared once.

Replaces the dozen `run_*.sh` scripts of the earlier harness (one per round) with data, pruned to what the project is for:
reproducing the paper's comparisons (`baseline-*`) and finding where M*'s advantage comes from (`mstar-ablation-*` and the
BAGEL / T2S specifics). Groups: baseline (paper reproduction), advantage (what M* components buy), fairness (what each
baseline's own configuration / tuning is worth, so baselines are not strawmen), protocol, profile, tool. Each entry says what
question it answers and how each outcome would be read *before* anything is run. Nothing here touches a GPU;
`python -m exps.suite list | plan | run | analyze NAME` does.

Rules every entry follows
-------------------------
* **Equal work.** Cases come from the task packages' headline sets only (BAGEL `-nocfg` / `-cfg3`, not the default
  request that gives M* a third CFG branch vLLM-Omni does not compute).
* **Best config per framework.** Task cells default to each framework's best recipe; an arm that departs from it is
  a named counterfactual and says so in `note`. A `compare` experiment may carry several candidate arms for a
  framework (BAGEL M* with and without compile at 256^2); the headline is the best candidate, per case.
* **Protocol is a parameter.** Every entry runs under the gated protocol by default and under any other with
  `--protocol back-to-back [--gap-s N]`; that makes new campaigns (`_B2B`, `_GAP30` suffixes), never mixed ones.
  `protocol-ab-*` fixes it per arm to measure what the protocol itself does.
* **Hooks are pinned to source.** `external/` is never edited. Hook-based arms (`{hooks}` paths) demand a proof file
  showing the hook applied in the worker, because a hook that silently refused tests nothing.
"""
from __future__ import annotations

from pathlib import Path

from exps.suite.spec import Arm, Experiment, Expect

MSTAR = "mstar"
VLLM_BAGEL = ("vllm-omni-two-stage", "vllm-omni-single-stage")
BAGEL_FW = (MSTAR, *VLLM_BAGEL)
QWEN_FW = ("mstar", "sglang-omni", "vllm-omni")

# --- hook environments ----------------------------------------------------------------------------------------------
SAMPLER_HOOK = ("mstar:PYTHONPATH={hooks}/mstar_bagel_sampler_hook", "mstar:MSTAR_BAGEL_SAMPLER_EAGER=2",
                "mstar:MSTAR_BAGEL_SAMPLER_LOG={campaign_dir}/hook_proof.jsonl")
ATTN_PROBE = ("mstar:PYTHONPATH={hooks}/mstar_attn_probe_hook",
              "mstar:MSTAR_ATTN_PROBE_LOG={campaign_dir}/attn_probe")

COSMOS3_SEQ_CFG = ("mstar:PYTHONPATH={hooks}/mstar_cosmos3_cfg_hook", "mstar:MSTAR_COSMOS3_BATCHED_CFG=0",
                   "mstar:MSTAR_COSMOS3_CFG_LOG={campaign_dir}/hook_proof.jsonl")
COSMOS3_NO_GRAPH = ("mstar:COSMOS3_DISABLE_ACCELERATOR_GRAPH=1", "mstar:COSMOS3_DISABLE_PREFILL_ACCELERATOR_GRAPH=1")
PI05_MSTAR_NO_GRAPH = ("mstar:PYTHONPATH={hooks}/mstar_pi05_nograph_hook", "mstar:MSTAR_PI05_NO_GRAPH=1",
                       "mstar:MSTAR_PI05_NO_GRAPH_LOG={campaign_dir}/hook_proof.jsonl")
PI05_VLLM_SDPA = ("vllm-omni:PYTHONPATH={hooks}/vllm_pi05_sdpa_hook", "vllm-omni:VLLM_PI05_SDPA=1",
                  "vllm-omni:VLLM_PI05_SDPA_LOG={campaign_dir}/hook_proof.jsonl")
PI05_SGLANG_GRAPH_OFF = "exps/configs/sglang_core/pi05_pipeline_graph_off.json"
PI05_SGLANG_PREFIX_CACHE = "exps/configs/sglang_core/pi05_pipeline_prefix_cache.json"
PI05_SGLANG_PREFIX_GRAPH_OFF = "exps/configs/sglang_core/pi05_pipeline_prefix_graph_off.json"
PI05_SGLANG_ACTION_GRAPH_OFF = "exps/configs/sglang_core/pi05_pipeline_action_graph_off.json"
PI05_SGLANG_BUCKETS = "exps/configs/sglang_core/pi05_pipeline_buckets200.json"
#: Counterfactual serve configs are named by a hardware-neutral key; `exps.hardware` maps it to the machine's own file (`config_variants` in
#: profiles/hardware/<name>.json: the rtx5880 multi-card overlays, the h200nvl one-card versions built by exps/tools/h200_variants.py).
#: `{variant:KEY}` is expanded when the arm is planned / analysed, so a machine without the file refuses the experiment instead of planning
#: another machine's topology.
def _variant(key: str) -> str:
    return "{variant:" + key + "}"


def _vfile(key: str) -> str:
    """The file name of the variant (a `command_has` check: the manifest's serve command must name it)."""
    return "{variant_file:" + key + "}"


QWEN_SYNC_SCHED = _variant("vllm_omni/qwen_thinker_sync_sched")
BAGEL_SYNC_SCHED = _variant("vllm_omni/bagel_two_stage_thinker_sync_sched")
QWEN_VLLM_PREFIX = _variant("vllm_omni/qwen_prefix_cache")
QWEN_VLLM_BATCH = _variant("vllm_omni/qwen_batch_async_c2wgraph")
QWEN_VLLM_TALKER_GREEDY = _variant("vllm_omni/qwen_talker_greedy")
#: vLLM-Omni's own Qwen3-Omni code-predictor switches (K/V cache, short-KV, fused sampling), and the same on MRv2 (qwen-t2s-vllm-opt)
QWEN_VLLM_CODEPRED = _variant("vllm_omni/qwen_codepred")
QWEN_VLLM_MRV2_CODEPRED = _variant("vllm_omni/qwen_mrv2_codepred")
#: transport counterfactuals (transport-qwen-t2s, transport-bagel-cfg): vLLM-Omni stage edges on Mooncake TCP instead of SharedMemory
QWEN_VLLM_MOONCAKE_TCP = _variant("vllm_omni/qwen_mooncake_tcp")
BAGEL_CFGPAR_MOONCAKE_TCP = _variant("vllm_omni/bagel_cfg_parallel_mooncake_tcp")
#: M*'s own tensor transport (`mstar serve` defaults to SHM): the serve arg and the command text it must appear in
MSTAR_TCP = "mstar:--tensor-comm-protocol=TCP"
MSTAR_TCP_CHECK = "--tensor-comm-protocol=TCP"
QWEN_SGLANG_NO_RADIX = _variant("sglang_omni/qwen_no_radix")
QWEN_MSTAR_PREFIX = _variant("mstar/qwen_prefix_cache")
BAGEL_MSTAR_CHUNK8192 = _variant("mstar/bagel_chunk8192")
#: M* BAGEL one-card KV pool at 1,024 pages (bagel-kv-pool): the shipped 512-page pool with the pool doubled, nothing else
BAGEL_MSTAR_KV1024 = _variant("mstar/bagel_kv1024")
QWEN_MSTAR_CHUNK4096 = _variant("mstar/qwen_chunk4096")
QWEN_MSTAR_PREFIX_CHUNK4096 = _variant("mstar/qwen_prefix_cache_chunk4096")
QWEN_MSTAR_CHUNK4096_B1152 = _variant("mstar/qwen_chunk4096_bucket1152")
QWEN_MSTAR_CHUNK4096_B1152_1536 = _variant("mstar/qwen_chunk4096_bucket1152_1536")
QWEN_MSTAR_CHUNK4096_ROOM = _variant("mstar/qwen_chunk4096_nsys_room")
#: M* Qwen with the Thinker prefix cache OFF, so every request is a cold prefill. rtx5880 gets that from TP=2 (M* shuts the cache at world size 2,
#: server.log "prefix cache off at world size 2"), so the token is empty there and the default config is already it. On one rank (h200nvl) M* opens the
#: cache by itself ("prefix cache open for main on Thinker", then "whole prompt already cached True" on the second identical prompt), so the token is the
#: `config.mstar=` option naming the cache-off recipe, and the logs say the opposite: no "open for" line. `_mstar_cold` carries both sides' evidence.
MSTAR_CACHE_OFF = "{hw:mstar_qwen_cache_off_opt}"
#: M* logs this when the bs=32 / 2,048 text prefill graph is captured: the one graph a budget-4,096 1,072-token step is replayed in (qwen-prefill-len, section 9.6).
G2048 = "Captured accelerator graph for Thinker: prefill_text[bs=32,tokens=2048]"
#: the request profile every fair Qwen comparison uses: no seed / min_tokens (SGLang-Omni's lookahead gate), explicit cache state
ELIGIBLE = "qwen.sampling=lookahead-eligible"
WARM_ROWS = "qwen.rows=cycle"
SGLANG_NO_ASYNC_DECODE = _variant("sglang_omni/qwen_no_async_decode")
CONNECTOR_SITE = str(Path.home() / ".local" / "opt" / "omni-connectors" / "site")
BAGEL_MOONCAKE_TCP = _variant("vllm_omni/bagel_two_stage_mooncake_tcp")
BAGEL_NIXL = _variant("vllm_omni/bagel_two_stage_nixl")
BAGEL_MOONCAKE_TCP_PREFIX = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache")
MS_LOG = "vllm-omni:VLLM_LOGGING_CONFIG_PATH={variant_abs:vllm_omni/ms_timestamps_log}"
BAGEL_VLLM_PREFIX_CACHE = _variant("vllm_omni/bagel_two_stage_prefix_cache")
BAGEL_VLLM_SEQS8 = _variant("vllm_omni/bagel_two_stage_prefix_cache_seqs8")
BAGEL_VLLM_SEQS16 = _variant("vllm_omni/bagel_two_stage_prefix_cache_seqs16")
BAGEL_MOONCAKE_TCP_PREFIX_KVT900 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_kvtimeout900")
BAGEL_MOONCAKE_TCP_PREFIX_SEQS16_KVT900 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16_kvtimeout900")
BAGEL_MOONCAKE_TCP_PREFIX_SEQS16 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16")
BAGEL_MOONCAKE_TCP_PREFIX_POOL16 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool16g")
BAGEL_MOONCAKE_TCP_PREFIX_POOL16_KVT900 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool16g_kvtimeout900")
BAGEL_MOONCAKE_TCP_PREFIX_SEQS16_POOL16 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16_pool16g")
BAGEL_MOONCAKE_TCP_PREFIX_SEQS16_POOL16_KVT900 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16_pool16g_kvtimeout900")
#: sender pool sizes between the 4 GiB default and the 16 GiB that removed the B = 16 collapse (bagel-i2i-pool-size)
BAGEL_MOONCAKE_TCP_PREFIX_POOL8 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool8g")
BAGEL_MOONCAKE_TCP_PREFIX_POOL12 = _variant("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool12g")

NO_COMPILE = ("mstar:TORCHDYNAMO_DISABLE=1",)
SPEC_OFF = ("mstar:MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=0",)
PREPLAN_OFF = ("mstar:MSTAR_PRE_PLAN_SPEC=0",)
STEP_SYNC = ("mstar:MSTAR_ENGINE_STEP_SYNC=1",)
WALK_BLIND = SPEC_OFF + PREPLAN_OFF + STEP_SYNC


#: Every framework logs when it falls back to default MoE tiles: vLLM "Using default MoE config", SGLang "Using default MoE
#: kernel config", M* "kernel table: no moe_gemm ... built-in configs". A Qwen3-Omni cell running default tiles on one side and
#: tuned tiles on the other would be a configuration artefact, so the analysis refuses it. `mstar@` scopes a check to M* logs.
MOE_TUNED = Expect(log_lacks=("Using default MoE config", "Using default MoE kernel config", "mstar@kernel table: no moe_gemm"),
                   log_has=("mstar@kernel table: moe_gemm",))


# SGLang builds its radix-cache object even when `disable_radix_cache: true` (a disabled tree "holds nothing"), so the line
# `Init Unified Radix Cache` is in every log; the evidence that the cache is on / off is the scheduler's own `#cached-token` counter.
SGLANG_PREFIX_HIT = "re:#cached-token: [1-9]"
SGLANG_NO_PREFIX_HIT = "re:#cached-token: [1-9]"


def _moe(**checks) -> Expect:
    """An arm's own checks plus the MOE_TUNED ones: every Qwen3-Omni arm of the fair profile runs tuned MoE tiles on all three frameworks."""
    return Expect(command_has=tuple(checks.get("command_has", ())),
                  log_has=tuple(checks.get("log_has", ())) + MOE_TUNED.log_has,
                  log_lacks=tuple(checks.get("log_lacks", ())) + MOE_TUNED.log_lacks)


def _mstar_cold(**checks) -> Expect:
    """`_moe` plus this machine's evidence that M*'s Thinker prefix cache is OFF for the arm (see MSTAR_CACHE_OFF): the TP=2 log line on rtx5880,
    on one rank the absence of the 'prefix cache open for' line. Each machine's token is empty on the other's side, so it drops out there."""
    return _moe(**{**checks, "log_has": ("{hw:mstar_qwen_cache_off_log_has}", *checks.get("log_has", ())),
                   "log_lacks": ("{hw:mstar_qwen_cache_off_log_lacks}", *checks.get("log_lacks", ()))})


def _mstar_warm(**checks) -> Expect:
    """`_moe` plus the evidence that M*'s Thinker prefix cache is ON: on one rank the 'open for' line (the default), on rtx5880 only with
    `prefix_cache_tp`, i.e. no 'off at world size' line."""
    return _moe(**{**checks, "log_has": ("{hw:mstar_qwen_cache_on_log_has}", *checks.get("log_has", ())),
                   "log_lacks": ("{hw:mstar_qwen_cache_on_log_lacks}", *checks.get("log_lacks", ()))})


def _cmd_has(*pairs: str) -> Expect:
    return Expect(command_has=tuple(p.split(":", 1)[1] for p in pairs))


def _aba(name_b: str, env: tuple[str, ...], **kwargs) -> tuple[Arm, ...]:
    """A1 / B / A2: the reference twice around the switch, so order drift shows as A1 != A2."""
    expect = _cmd_has(*env) if env else Expect()
    return (Arm("A1", **kwargs), Arm(name_b, env=env, expect=expect, **kwargs), Arm("A2", **kwargs))


def _baseline(task: str, title: str, frameworks, cases, **kwargs) -> Experiment:
    return Experiment(
        name=f"baseline-{task.replace('_', '-')}", title=title, task=task, group="baseline", kind="compare",
        question=f"{title}: how do the frameworks compare at equal work, each at its best configuration?",
        reading="Ratio = best baseline p50 / M* p50 per case (pooled over measured requests of all starts), with the "
                "per-start range beside it. A ratio is only a claim about the configurations named in the table.",
        frameworks=tuple(frameworks), cases=tuple(cases), **kwargs)


def _protocol_ab(task: str, frameworks, cases, *, gap30: bool = False, **kwargs) -> Experiment:
    kwargs.setdefault("repeats", 2)
    kwargs.setdefault("measured", 5)
    return Experiment(
        name=f"protocol-ab-{task.replace('_', '-')}", task=task, group="protocol", kind="ab",
        title=f"Measurement protocol A/B on {task}: gated vs back-to-back" + (" vs fixed 30 s gap" if gap30 else ""),
        question="Does the protocol itself change this task's latency, and for which framework? (Pi0.5: vLLM-Omni 333 ms "
                 "gated vs 222 ms back-to-back at identical temperature and clock; BAGEL throttles without the gate.)",
        reading="gated is the reference. back-to-back differs: the cooldown gate / idle matters for this task. gap30 matches "
                "gated: the idle time matters but not the thermal condition; gap30 matches back-to-back: the gate's cool, "
                "full-clock card does. Compare the per-request temperature and SM clock columns before reading any ratio.",
        frameworks=tuple(frameworks), cases=tuple(cases),
        arms=(Arm("gated", protocol="gated"), Arm("b2b", protocol="back-to-back"),
              *((Arm("gap30", protocol="back-to-back", gap_s=30.0),) if gap30 else ())),
        **kwargs)


# --- baselines: equal work, best config -------------------------------------------------------------------------------
BAGEL_HEADLINE = ("t2i-512-nocfg", "t2i-512-cfg3", "i2i-512", "i2t-512")
BAGEL_1024 = ("t2i-1024-nocfg", "t2i-1024-cfg3", "i2i-1024", "i2t-1024")
#: CFG-parallel cases: image generation with three branches on every engine, at the paper's size and at 512.
BAGEL_CFGPAR = ("t2i-1024-cfg3", "i2i-1024", "t2i-512-cfg3", "i2i-512")

BASELINES = (
    Experiment(
        name="baseline-bagel", title="BAGEL at 512x512: M* vs vLLM-Omni (two deployments), equal work", task="bagel",
        group="baseline", kind="compare", frameworks=BAGEL_FW, cases=BAGEL_HEADLINE,
        question="At equal work and with vLLM-Omni's MoT GEMM tuned for this GPU, how do M* and vLLM-Omni compare per task? "
                 "512x512 is the main operating point (256x256 is launch-bound and drift-prone; it is not run by default).",
        reading="Per case: best vLLM-Omni deployment / M*. T2I `-nocfg` is one branch on both (same output semantics); "
                "`-cfg3` is three on both. The tuned MoT config is passed to every vLLM-Omni server; only the single-stage server log shows it loading (the two-stage DiT subprocess logs nothing to the server log), so `No tuned MoT config` is checked absent.",
        arms=(Arm("main", expect=Expect(log_lacks=("No tuned MoT config",))),),
        repeats=3, caveats="vLLM-Omni's tuner covers the M values 258 x {1,3,4,12,16,48}: 256..1024 at one and three branches."),
    Experiment(
        name="baseline-bagel-1024", title="BAGEL at 1024x1024 (the paper's size), equal work", task="bagel", group="baseline",
        kind="compare", frameworks=BAGEL_FW, cases=BAGEL_1024,
        question="Same as baseline-bagel at the paper's resolution. Needs stable cooling: on the throttling H200 NVL one request "
                 "at this size crosses the thermal knee and measures the chassis.",
        reading="As baseline-bagel; read the clock column first: a clock that drops across starts invalidates the ratio.",
        arms=(Arm("main", expect=Expect(log_lacks=("No tuned MoT config",))),), repeats=2, measured=3),
    Experiment(
        name="baseline-bagel-cfgpar", title="BAGEL at the paper's topology: 3-GPU CFG-parallel, M* vs vLLM-Omni (two deployments)",
        task="bagel", group="baseline", kind="compare", frameworks=BAGEL_FW, cases=BAGEL_CFGPAR,
        question="The paper's headline BAGEL figure (T2I / I2I, B=1, 1024x1024) runs one rank per CFG branch on 3 GPUs (M* `Parallel` "
                 "placement vs vLLM-Omni's CFG-parallel plugin); baseline-bagel* are its single-GPU Appendix E counterpart. Does M*'s "
                 "lead survive the cross-GPU branch exchange?",
        reading="Per case: best vLLM-Omni deployment / M*, three branches on every engine. vLLM-Omni two-stage needs a 4th card on "
                "48 GiB GPUs (Thinker cannot share a card with a DiT rank). No NVLink on the RTX 5880: the branch exchange crosses "
                "PCIe, so absolute numbers and possibly the ratio are not the paper's (NVLink H100).",
        arms=(Arm("main", opts=("bagel.topology=cfg-parallel",), expect=Expect(log_lacks=("No tuned MoT config",))),),
        repeats=2, measured=3, min_gpus=3,
        caveats="Needs >= 3 GPUs (4 for vLLM-Omni two-stage on 48 GiB cards): skipped on a single-GPU machine."),
    Experiment(
        name="transport-bagel-cfg", title="BAGEL CFG-parallel, three branches: M* and vLLM-Omni on the same tensor transport (B=1, 512 and 1024)",
        task="bagel", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), cases=BAGEL_CFGPAR, repeats=2, measured=3, min_gpus=3,
        question="baseline-bagel-cfgpar pairs M*'s shared-memory tensor manager with vLLM-Omni's shared-memory Thinker -> DiT connector. Does M* "
                 "move when its tensors go over Mooncake TCP, and does vLLM-Omni move when its one stage edge does?",
        reading="Per case, E2E by arm. Same-transport pairs: mstar-shm with vllm-shm, and mstar-tcp with vllm-tcp. The branch exchange is not "
                "switched: vLLM-Omni all_gathers the branches with NCCL inside its DiT group, and M* hands branch tensors over its tensor manager. "
                "So the transport arms cover the Thinker -> DiT hand-off on vLLM-Omni and M*'s own tensor hand-off, not the branch exchange.",
        arms=(Arm("mstar-shm", frameworks=(MSTAR,), opts=("bagel.topology=cfg-parallel",),
                  expect=Expect(log_lacks=("No tuned MoT config",))),
              Arm("mstar-tcp", frameworks=(MSTAR,), opts=("bagel.topology=cfg-parallel",), serve_args=(MSTAR_TCP,),
                  expect=Expect(command_has=(MSTAR_TCP_CHECK,), log_lacks=("No tuned MoT config",)),
                  note="M* tensor manager over Mooncake TCP (loopback)"),
              Arm("vllm-shm", frameworks=("vllm-omni-two-stage",), opts=("bagel.topology=cfg-parallel",),
                  expect=Expect(log_lacks=("No tuned MoT config",))),
              Arm("vllm-tcp", frameworks=("vllm-omni-two-stage",), env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",),
                  opts=("bagel.topology=cfg-parallel", f"config.vllm-omni={BAGEL_CFGPAR_MOONCAKE_TCP}"),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_cfg_parallel_mooncake_tcp"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector",),
                                log_lacks=("Failed to create connector", "No tuned MoT config")),
                  note="Thinker -> DiT edge on Mooncake TCP")),
        caveats="Needs 3 cards for M* and 4 for vLLM-Omni two-stage on 48 GiB cards. No NVLink on this box: all branch traffic crosses PCIe."),
    _baseline("qwen_t2s", "Qwen3-Omni text-to-speech", QWEN_FW, ("verbatim-en", "verbatim-zh"),
              metric="ms_per_audio_s", repeats=3, arms=(Arm("main", expect=MOE_TUNED),)),
    _baseline("qwen_txt", "Qwen3-Omni text-to-text", QWEN_FW, ("txt-1k",), repeats=3, arms=(Arm("main", expect=MOE_TUNED),)),
    _baseline("qwen_a2t", "Qwen3-Omni audio-to-text", QWEN_FW, ("a2t-en", "a2t-zh"), repeats=3, preconditions=6,
              measured=10, arms=(Arm("main", expect=MOE_TUNED),),
              caveats="Steady-state numbers: 6 preconditioning requests are not scored. vLLM-Omni's first ~4 requests after a server "
                      "start take 0.6-1.5 s TTFB (DIAG_A2T_02/03/04: same with the default config, back-to-back, and the old eager "
                      "config), so the cold path is not represented. A request whose input length is new can also be slow once "
                      "(vLLM a2t-zh: 0.6-1.1 s TTFB on the 25 s clip in every start); the cause is not identified. The p50 is robust "
                      "to those requests, the start range is not."),
    _baseline("orpheus", "Orpheus TTS (M* vs VoxServe)", ("mstar", "voxserve"), ("b1", "b4", "b16"), repeats=3,
              caveats="E2E latency of the whole request (a wave of B requests) only: no RTF or tokens-per-second, so this table "
                      "does not answer a throughput or real-time-factor claim."),
    _baseline("vjepa_ac", "V-JEPA 2-AC rollout (M* vs Meta-native, both BF16)", ("mstar", "meta-native"),
              ("h4", "h15", "h30"), repeats=3,
              caveats="Both clients read frozen copies of the DROID videos (exps/inputs/VJ-AC/frozen_videos), so the source bytes are "
                      "equal across engines and starts; the `*_pairs_*` audit must show every start valid before a ratio is quoted. "
                      "Timing boundaries differ: M* is HTTP E2E, Meta-native is in-process, and equal decoded tensors are not proven.",
              post=("{py} exps/tools/analyze_vjepa_paired.py --campaign {campaign} --repeats 3 --measured 5 "
                    "--markdown {results}/baseline-vjepa-ac_pairs_{hardware}.md",)),
    _baseline("cosmos3", "Cosmos3-Nano T2I 256x256 (M* vs vLLM-Omni)", ("mstar", "vllm-omni"), ("t2i-256",), repeats=3,
              caveats="Equal work (50 steps, 256x256, guidance_interval [400,1000], same prompts). The ~2.4x gap is M*'s design at this "
                      "launch/weight-bound size, not unequal work (nsys, RTX 5880): M* packs cond+uncond into ONE forward per step "
                      "(denoise_step_batched_cfg), vLLM-Omni runs two sequential forwards (~90 passes vs ~50) at ~83k vs ~40k kernels per "
                      "request; GPU time per forward is about equal (~20 ms). Whether the 256x256 step is CUDA-graph replayed is NOT "
                      "established: M*'s captured tiers are 192x320 and 480x832 (server log), so 256x256 may run eager; "
                      "`cosmos3-decompose` measures graph / batched CFG / compile one by one. Expect the gap to shrink at larger "
                      "sizes, where the GEMMs are compute-bound."),
    Experiment(
        name="baseline-pi05", title="Pi0.5 on the 46-row DROID fixture (M* vs SGLang core vs vLLM-Omni eager)",
        task="pi05", group="baseline", kind="compare", frameworks=("mstar", "sglang-core", "vllm-omni"),
        cases=("droid-46row",), repeats=3, measured=46, preconditions=5,
        question="Pi0.5 end-to-end action latency at equal work (10 flow steps, [50,32] action).",
        reading="Per-case ratios under the protocol named in the campaign (`_B2B` = back-to-back). The protocol changes this "
                "task's ratios a lot (see protocol-ab-pi05), so always report both. vLLM-Omni is measured eager only.",
        caveats="vLLM-Omni's graph variant (`pi05-vllm-graph`) did not change latency; no better vLLM recipe is known."),
)

# --- protocol A/B (the option exists for every task; the catalogue runs it where it can matter) ------------------------------
PROTOCOL_AB = (
    _protocol_ab("pi05", ("mstar", "sglang-core", "vllm-omni"), ("droid-46row",), gap30=True, measured=12, preconditions=5),
    _protocol_ab("bagel", BAGEL_FW[:2], ("t2i-512-nocfg",)),
    _protocol_ab("vjepa_ac", ("mstar", "meta-native"), ("h15",)),
    _protocol_ab("orpheus", ("mstar", "voxserve"), ("b1",)),
    _protocol_ab("cosmos3", ("mstar", "vllm-omni"), ("t2i-256",)),
    _protocol_ab("qwen_t2s", QWEN_FW, ("verbatim-en",), metric="ms_per_audio_s"),
    _protocol_ab("qwen_txt", QWEN_FW, ("txt-1k",)),
    _protocol_ab("qwen_a2t", QWEN_FW, ("a2t-en",), measured=8, preconditions=6),
)

# --- advantage: what do M*'s components buy? One matrix, every task ----------------------------------------------------------
#: Components switchable without touching framework source (all are M* environment variables that the source already reads):
#:   C   torch.compile off                    TORCHDYNAMO_DISABLE=1
#:   S   speculative launch chain off         MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=0   (Worker no longer launches step N+1 from the Walk
#:                                            before step N returns)
#:   W   walk-blind: S + pre-plan off + step sync on    the Worker launches only the step it is told to; the Walk / Loop structure and
#:                                            kernels stay, so this is NOT a hand-written-loop / stage-framework control
#: There is no switch for M*'s CUDA graphs, paged KV or FlashInfer; those need a code change or a hook, and are not claimed.
def _mstar_ablation(task: str, title: str, cases, **kwargs) -> Experiment:
    kwargs.setdefault("repeats", 2)   # A1 and A2 already bracket the drift; two fresh starts per arm is the budget choice
    kwargs.setdefault("measured", 5)
    return Experiment(
        name=f"mstar-ablation-{task.replace('_', '-')}", task=task, group="advantage", kind="ab", frameworks=(MSTAR,),
        cases=tuple(cases), title=f"M* component ablation on {title}",
        question="Which of M*'s switchable components (torch.compile, speculative launch chain, Walk-reading lookaheads) "
                 "account for its advantage on this model, and what is left when they are all off?",
        reading="Each arm / A (A1 and A2 bracket the drift; trust a ratio only beyond their spread). SC vs C and S shows whether "
                "chain and compile combine multiplicatively; WC against the baselines' best (`baseline-*`) is M* with no "
                "lookahead and no compile: a lead that survives it is not those two components. Ratios near 1.0 are null "
                "results, not missing data.",
        arms=(Arm("A1"),
              Arm("C", env=NO_COMPILE, expect=_cmd_has(*NO_COMPILE)),
              Arm("S", env=SPEC_OFF, expect=_cmd_has(*SPEC_OFF)),
              Arm("SC", env=SPEC_OFF + NO_COMPILE, expect=_cmd_has(*(SPEC_OFF + NO_COMPILE))),
              Arm("W", env=WALK_BLIND, expect=_cmd_has(*WALK_BLIND)),
              Arm("WC", env=WALK_BLIND + NO_COMPILE, expect=_cmd_has(*(WALK_BLIND + NO_COMPILE))),
              Arm("A2")),
        **kwargs)


ABLATIONS_MSTAR = (
    _mstar_ablation("qwen_t2s", "Qwen3-Omni T2S", ("verbatim-en",), metric="ms_per_audio_s"),
    _mstar_ablation("bagel", "BAGEL (one-branch T2I, and the AR path)", ("t2i-512-nocfg", "i2t-512")),
    _mstar_ablation("vjepa_ac", "V-JEPA 2-AC", ("h15",)),
    _mstar_ablation("pi05", "Pi0.5", ("droid-46row",), measured=12, preconditions=5),
    _mstar_ablation("orpheus", "Orpheus", ("b1",)),
    _mstar_ablation("cosmos3", "Cosmos3", ("t2i-256",)),
    _mstar_ablation("qwen_a2t", "Qwen3-Omni A2T", ("a2t-en",), preconditions=6, measured=8),
    _mstar_ablation("qwen_txt", "Qwen3-Omni TXT", ("txt-1k",)),
)

B_ONE = dict(task="bagel", frameworks=(MSTAR,), kind="ab")

ADVANTAGE = (
    Experiment(
        name="bagel-attn-backend", task="bagel", group="advantage", kind="ab",
        title="BAGEL: is M*'s lead flash-attn? Attention call counts, flash_attn blocked, and the backends swapped",
        question="M*'s README recommends flash-attn for BAGEL. Does M* use the pip `flash_attn` for the attention that "
                 "matters, and does anyone else use flash attention? Source reading says the pip package is only in the "
                 "ViT (T2I never runs it), M* LLM attention is FlashInfer and vLLM-Omni's generation step is FLASH_ATTN.",
        reading="(1) `mstar-probe`: T2I cases show vit_flash_attn_calls == 0 and the LLM attention managers are "
                "FlashInfer; I2T/I2I show > 0. (2) `mstar-no-flash-attn`: the ViT takes the SDPA fallback "
                "(vit_sdpa_fallback_calls > 0, flash calls == 0); T2I E2E unchanged vs mstar-probe within drift; I2T/I2I "
                "slower only by the ViT's share. (3) `vllm-sdpa`: vLLM-Omni's generation step on TORCH_SDPA instead of "
                "FLASH_ATTN; if its E2E barely moves, attention is not where the gap lives (kernel data already say <=4%). "
                "The ratio block pairs every vLLM arm with every M* arm; a ratio that survives all four pairings is not a "
                "flash-attn effect. On a GPU with no FlashAttention-3 kernel (RTX 5880 Ada is sm_89) check the vLLM log line "
                "for which backend it actually resolved.",
        frameworks=(MSTAR,), cases=("t2i-512-nocfg", "i2i-512", "i2t-512"), repeats=2,
        arms=(
            Arm("mstar-probe", env=ATTN_PROBE + ("mstar:MSTAR_ATTN_PROBE=count",), frameworks=(MSTAR,),
                expect=Expect(proof="attn_probe")),
            Arm("mstar-no-flash-attn", env=ATTN_PROBE + ("mstar:MSTAR_ATTN_PROBE=count,block-flash-attn",),
                frameworks=(MSTAR,), expect=Expect(proof="attn_probe_blocked"),
                note="the ViT behaves as if flash_attn were not installed"),
            # The two-stage DiT subprocess does not forward its INFO log to the server log (only the single-stage deployment's
            # DiffusionWorker does), so the resolved-backend line cannot be required; the serve command is what is checked.
            Arm("vllm-default", frameworks=("vllm-omni-two-stage",),
                expect=Expect(command_lacks=("DIFFUSION_ATTENTION_BACKEND",))),
            Arm("vllm-sdpa", env=("vllm-omni:DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA",), frameworks=("vllm-omni-two-stage",),
                expect=Expect(command_has=("DIFFUSION_ATTENTION_BACKEND=TORCH_SDPA",)),
                note="generation-step attention swapped to PyTorch SDPA (not FlashAttention)"),
        ),
        post=("{py} exps/tools/analyze_attn_probe.py --campaigns {campaign:mstar-probe} {campaign:mstar-no-flash-attn} "
              "--vllm {campaign:vllm-default} {campaign:vllm-sdpa} --markdown {results}/bagel-attn-backend_{hardware}.md",),
        caveats="The probe patches imported M* modules in the worker (never external/); `proof` files show it applied."),
    Experiment(
        name="bagel-cfg-ladder", title="BAGEL: E2E against the number of CFG branches (1 / default / 3)", task="bagel",
        group="advantage", frameworks=BAGEL_FW, kind="ab",
        cases=("t2i-512-nocfg", "t2i-512-default", "t2i-512-cfg3"),
        question="Does equal work (three branches on both) give equal time, and what does one extra branch cost per step on each "
                 "framework? (Both engines compute the image-guidance branch only when cfg_img_scale > 1.)",
        reading="ms per extra branch per step per framework. `default` (cfg_img_scale=1.0) is two branches on both engines, `cfg3` "
                "three; the per-branch cost should agree across the two variants of one engine.",
        arms=(Arm("main"),), repeats=2,
        post=("{py} exps/tools/analyze_followups.py ladder {campaign:main} --markdown {results}/bagel-cfg-ladder_per_branch_{hardware}.md",)),
    Experiment(
        name="bagel-steps", title="BAGEL: fixed and per-step cost (request-level steps; M* server-side steps)", task="bagel",
        group="advantage", frameworks=BAGEL_FW, kind="ab",
        cases=("t2i-512-default", "t2i-512-steps5", "t2i-512-steps25", "t2i-512-srv-steps5", "t2i-512-srv-steps25"),
        repeats=2,
        question="E2E = a + b*steps: does a framework's gap live in the per-request fixed cost or in the per-step cost?",
        reading="Fit per framework; `srv-steps` are M*-only (its steps are server config). Steps cases use the default "
                "request, so compare a and b within a framework, not the absolute E2E across frameworks.",
        arms=(Arm("main"),), post=("{py} exps/tools/analyze_bagel_ablation.py --campaign {campaign:main} "
                                    "--markdown {results}/bagel-steps_fit_{hardware}.md",)),
    Experiment(
        name="bagel-concurrency", title="BAGEL I2T: B = 1..16 concurrent requests", task="bagel", group="advantage",
        frameworks=BAGEL_FW, kind="ab", cases=("i2t-512", "i2t-512-b4", "i2t-512-b16"), repeats=2,
        question="Throughput scaling of the AR path with batch (the paper's I2T batch claim).",
        reading="Wave time vs B per framework; scaling ~1.0 means serialised waves.", arms=(Arm("main"),),
        post=("{py} exps/tools/analyze_concurrency.py --campaign {campaign:main} --markdown {results}/bagel-concurrency_{hardware}.md",)),
    Experiment(
        name="qwen-async-sched", title="Qwen3-Omni TXT: is M*'s speculative launch chain just vLLM's async scheduling?",
        task="qwen_txt", group="advantage", kind="ab", frameworks=("vllm-omni", MSTAR), cases=("txt-1k",), repeats=2,
        question="On a pinned 256-token decode M* takes 5.85 ms/token with its launch chain and 7.9 without; vLLM-Omni takes 5.93 "
                 "(its Thinker runs OmniARAsyncScheduler) and SGLang-Omni 8.1 (its Omni scheduler refuses the overlap loop). Is the "
                 "chain the same CPU/GPU-overlap mechanism vLLM already has?",
        reading="ITL (analyze_decode_itl.py) per arm. If vllm-async-off lands near mstar-spec-off (~7.9-8.2 ms/token) and "
                "vllm-async-on near mstar-spec-on, the lever is generic overlap scheduling, not something only a Walk / graph "
                "runtime can do. If vLLM stays at ~5.9 with async off, the two mechanisms differ and this reading is wrong.",
        arms=(Arm("vllm-async-on", frameworks=("vllm-omni",), expect=Expect(log_lacks=("omni_ar_scheduler.OmniARScheduler",))),
              Arm("vllm-async-off", frameworks=("vllm-omni",), opts=(f"config.vllm-omni={QWEN_SYNC_SCHED}",),
                  expect=Expect(log_has=("omni_ar_scheduler.OmniARScheduler",)),
                  note="Thinker stage on the non-async scheduler: a counterfactual, not a baseline"),
              Arm("mstar-spec-on", frameworks=(MSTAR,)),
              Arm("mstar-spec-off", frameworks=(MSTAR,), env=SPEC_OFF, expect=_cmd_has(*SPEC_OFF))),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:vllm-async-on} exps/runs/{campaign:vllm-async-off} "
              "exps/runs/{campaign:mstar-spec-on} exps/runs/{campaign:mstar-spec-off} --markdown {results}/qwen-async-sched_{hardware}.md",),
        caveats="Machine-specific: rtx5880 = 3 GPUs, Thinker TP=2, M* without torch.compile on sm_89; h200nvl = one card, TP=1, M* compiled (the one-card config variants, exps/tools/h200_variants.py). Host overhead per token is larger with TP=2, "
                "so the size of the lever is not transferable to a single-GPU Thinker."),
    Experiment(
        name="qwen-async-sched-a2t", title="Qwen3-Omni A2T: is M*'s speculative launch chain just vLLM's async scheduling? (the TXT arms on audio input)",
        task="qwen_a2t", group="advantage", kind="ab", frameworks=("vllm-omni", MSTAR), cases=("a2t-en",), repeats=2,
        question="The same question as qwen-async-sched on audio input. The per-token table has M* chain on / off and vLLM async on, but the "
                 "vLLM async-off arm was only run for TXT, so A2T has no vLLM async-off number.",
        reading="ITL (analyze_decode_itl.py) per arm, same reading as qwen-async-sched: vllm-async-off near mstar-spec-off and vllm-async-on near "
                "mstar-spec-on means generic overlap scheduling; vLLM staying flat with async off means the mechanisms differ.",
        arms=(Arm("vllm-async-on", frameworks=("vllm-omni",), expect=Expect(log_lacks=("omni_ar_scheduler.OmniARScheduler",))),
              Arm("vllm-async-off", frameworks=("vllm-omni",), opts=(f"config.vllm-omni={QWEN_SYNC_SCHED}",),
                  expect=Expect(log_has=("omni_ar_scheduler.OmniARScheduler",)),
                  note="Thinker stage on the non-async scheduler: a counterfactual, not a baseline"),
              Arm("mstar-spec-on", frameworks=(MSTAR,)),
              Arm("mstar-spec-off", frameworks=(MSTAR,), env=SPEC_OFF, expect=_cmd_has(*SPEC_OFF))),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:vllm-async-on} exps/runs/{campaign:vllm-async-off} "
              "exps/runs/{campaign:mstar-spec-on} exps/runs/{campaign:mstar-spec-off} --markdown {results}/qwen-async-sched-a2t_{hardware}.md",),
        caveats="Machine-specific: rtx5880 = 3 GPUs, Thinker TP=2, M* without torch.compile on sm_89; h200nvl = one card, TP=1, M* compiled. Audio input adds the audio encoder to the prompt "
                "path; ITL is per output token and is not affected by it."),
    Experiment(
        name="qwen-fair-txt", title="Qwen3-Omni TXT, fair profile: prefix cache off/on in all three frameworks, SGLang lookahead eligible",
        task="qwen_txt", group="fair", kind="ab", frameworks=("sglang-omni", "vllm-omni", MSTAR), cases=("txt-1k",), repeats=2,
        question="The baseline TXT cells were not equal: SGLang-Omni's Thinker had its radix cache ON while M* (prefix_cache_tp off at TP=2) and "
                 "vLLM-Omni (upstream enable_prefix_caching: false) had none, the prompt repeats, and our seed / min_tokens switched SGLang's "
                 "own lookahead decode off. With the cache OFF in all three, and again ON in all three, and a request SGLang's lookahead accepts, "
                 "does M* still lead, and by how much?",
        reading="Compare within a cache state: off-vs-off (a cold prefill every request) and on-vs-on (a hot prefix). TTFT carries the cache, ITL the "
                "decode loop. sglang-on-async-off must be slower than sglang-on in ITL, or the lookahead was never engaged. vllm-pinned / "
                "mstar-pinned against vllm-off / mstar-off say whether dropping seed + min_tokens changes them (expected: no). Anything the "
                "arms leave unequal is listed in the fairness audit, not assumed away.",
        arms=(Arm("sglang-off", frameworks=("sglang-omni",), opts=(ELIGIBLE, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(command_has=(_vfile("sglang_omni/qwen_no_radix"),), log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("sglang-on", frameworks=("sglang-omni",), opts=(ELIGIBLE,), expect=_moe(log_has=(SGLANG_PREFIX_HIT,))),
              Arm("sglang-on-async-off", frameworks=("sglang-omni",), opts=(ELIGIBLE, f"config.sglang={SGLANG_NO_ASYNC_DECODE}"),
                  expect=_moe(command_has=(_vfile("sglang_omni/qwen_no_async_decode"),)),
                  note="control: SGLang's lookahead forced off, radix cache on"),
              Arm("vllm-off", frameworks=("vllm-omni",), opts=(ELIGIBLE,), expect=_moe(log_lacks=("enable_prefix_caching=True",))),
              Arm("vllm-on", frameworks=("vllm-omni",), opts=(ELIGIBLE, f"config.vllm-omni={QWEN_VLLM_PREFIX}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_prefix_cache"),), log_has=("enable_prefix_caching=True",))),
              Arm("mstar-off", frameworks=(MSTAR,), opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("vllm-pinned", frameworks=("vllm-omni",), note="the baseline request (seed 42, min_tokens 256): does the profile change vLLM?", expect=_moe()),
              Arm("mstar-pinned", frameworks=(MSTAR,), opts=(MSTAR_CACHE_OFF,), note="the baseline request: does the profile change M*?", expect=_moe())),
        post=("{py} exps/tools/analyze_decode_itl.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in (
              "sglang-off", "sglang-on", "sglang-on-async-off", "vllm-off", "vllm-on", "mstar-off", "vllm-pinned", "mstar-pinned"))
              + " --markdown {results}/qwen-fair-txt_{hardware}.md",),
        caveats="`on` = the same prompt repeated, i.e. a hot prefix (best case for every cache); `off` = a cold prefill each request. There is no M* `on` arm: "
                "with the Thinker's prefix cache enabled at TP=2 (`prefix_cache_tp: true`, offload 0) M* serves the first 1k-token request and the SECOND "
                "(identical prompt) kills worker_1 with `TP ranks diverged: KV thinker_kv: rank 0 leased 8 cached pages ...` "
                "(QWEN_FAIR_TXT_MSTAR_ON_01_B2B, both repeats). M* ships that switch off at world size 2, so M* has no hot-prefix TXT number on this box. "
                "On h200nvl (one rank) M* opens the Thinker cache by itself: its hot-prefix number is the M* cell of baseline-qwen-txt (default config, "
                "server.log 'whole prompt already cached True'), and `mstar-off` / `mstar-pinned` here run the cache-off recipe (resources.thinker_kv.prefix_cache: false)."),
    Experiment(
        name="qwen-fair-a2t", title="Qwen3-Omni A2T, fair profile: prefix cache off/on in all three frameworks, SGLang lookahead eligible, shapes pre-warmed",
        task="qwen_a2t", group="fair", kind="ab", frameworks=("sglang-omni", "vllm-omni", MSTAR), cases=("a2t-en",), repeats=2,
        measured=10, preconditions=10,
        question="Same as qwen-fair-txt for audio input, plus: the precondition phase serves the SAME ten clips the measured phase will send "
                 "(`qwen.rows=cycle`), because vLLM-Omni JIT-compiles Triton fused-MoE kernels at inference time for input shapes it has not "
                 "seen (server.log: 'Triton kernel JIT compilation during inference: fused_moe_kernel ... latency spike') and 2 of 10 measured "
                 "A2T-zh requests per start were such first-seen shapes (TTFT 507 / 629 ms against ~100).",
        reading="As qwen-fair-txt. The `vllm-pinned` / `mstar-pinned` arms use the baseline request but the same warm rows. A2T's cache `on` arms "
                "only help if the framework caches audio-token prefixes at all; no TTFT change is itself a finding (TTFT unchanged on vs off).",
        arms=(Arm("sglang-off", frameworks=("sglang-omni",), opts=(ELIGIBLE, WARM_ROWS, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(command_has=(_vfile("sglang_omni/qwen_no_radix"),), log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("sglang-on", frameworks=("sglang-omni",), opts=(ELIGIBLE, WARM_ROWS), expect=_moe(log_has=(SGLANG_PREFIX_HIT,))),
              Arm("sglang-on-async-off", frameworks=("sglang-omni",),
                  opts=(ELIGIBLE, WARM_ROWS, f"config.sglang={SGLANG_NO_ASYNC_DECODE}"),
                  expect=_moe(command_has=(_vfile("sglang_omni/qwen_no_async_decode"),))),
              Arm("vllm-off", frameworks=("vllm-omni",), opts=(ELIGIBLE, WARM_ROWS), expect=_moe(log_lacks=("enable_prefix_caching=True",))),
              Arm("vllm-on", frameworks=("vllm-omni",), opts=(ELIGIBLE, WARM_ROWS, f"config.vllm-omni={QWEN_VLLM_PREFIX}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_prefix_cache"),), log_has=("enable_prefix_caching=True",))),
              Arm("mstar-off", frameworks=(MSTAR,), opts=(ELIGIBLE, WARM_ROWS, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("mstar-on", frameworks=(MSTAR,), opts=(ELIGIBLE, WARM_ROWS, f"config.mstar={QWEN_MSTAR_PREFIX}"),
                  expect=_mstar_warm(command_has=(_vfile("mstar/qwen_prefix_cache"),))),
              Arm("vllm-pinned", frameworks=("vllm-omni",), opts=(WARM_ROWS,), expect=_moe()),
              Arm("mstar-pinned", frameworks=(MSTAR,), opts=(WARM_ROWS, MSTAR_CACHE_OFF), expect=_moe())),
        post=("{py} exps/tools/analyze_decode_itl.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in (
              "sglang-off", "sglang-on", "sglang-on-async-off", "vllm-off", "vllm-on", "mstar-off", "mstar-on", "vllm-pinned", "mstar-pinned"))
              + " --markdown {results}/qwen-fair-a2t_{hardware}.md",),
        caveats="The warm profile removes first-seen-shape JIT from the measured phase; the baseline numbers (6 preconditions, rows sequential) "
                "are the cold-shape view and stay valid as such."),
    Experiment(
        name="qwen-t2s-vllm-opt", title="Qwen3-Omni T2S: do vLLM-Omni's own code-predictor switches and MRv2 close its gap to M*? (20 sentences, warm)",
        task="qwen_t2s", group="fair", kind="ab", frameworks=("vllm-omni", MSTAR), cases=("verbatim-en",), repeats=2, measured=20, preconditions=20,
        question="The stock vLLM-Omni Qwen3-Omni recipe leaves three opt-in switches off: the code predictor's K/V cache, its short-KV attention and fused "
                 "sampling (read from the connector extra). The second-generation model runner (MRv2) replays the Talker's decode in full CUDA graphs. "
                 "Upstream's single-H200 profile turns all of these on; our 5880 and H200 vLLM arms never did. Per audio second vLLM-Omni costs 152.8 ms against "
                 "M*'s 96.0 (qwen-t2s-fit). How much of that moves with these switches?",
        reading="analyze_t2s_fit.py per arm: b (ms per audio second) for vllm-default, vllm-codepred and vllm-mrv2-codepred, against vllm-default-again, "
                "whose spread is the drift reference. Only a change larger than that spread counts. The codepred arms must log the K/V decoding line "
                "('code_predictor: torch.compile + incremental K/V decoding'); without it the switch did not reach the predictor and the arm is void, not a result. "
                "If b falls toward M*'s 96 ms, the vLLM gap was the recipe; if it stays near 153 ms, the cause lies elsewhere (the launch count is then the next thing to check).",
        arms=(Arm("vllm-default", frameworks=("vllm-omni",), opts=(WARM_ROWS,), expect=_moe()),
              Arm("vllm-codepred", frameworks=("vllm-omni",), opts=(WARM_ROWS, f"config.vllm-omni={QWEN_VLLM_CODEPRED}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_codepred"),),
                              log_has=("code_predictor: torch.compile + incremental K/V decoding",))),
              Arm("vllm-mrv2-codepred", frameworks=("vllm-omni",), opts=(WARM_ROWS, f"config.vllm-omni={QWEN_VLLM_MRV2_CODEPRED}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_mrv2_codepred"),),
                              log_has=("code_predictor: torch.compile + incremental K/V decoding",)),
                  note="stages 1 and 2 share card 2: run under NVIDIA MPS (see the config header)"),
              Arm("vllm-default-again", frameworks=("vllm-omni",), opts=(WARM_ROWS,), expect=_moe()),
              Arm("mstar", frameworks=(MSTAR,), opts=(WARM_ROWS,), expect=_moe())),
        post=("{py} exps/tools/analyze_t2s_fit.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in (
              "vllm-default", "vllm-codepred", "vllm-mrv2-codepred", "vllm-default-again", "mstar"))
              + " --markdown {results}/qwen-t2s-vllm-opt_{hardware}.md",),
        caveats="Audio length still differs a little across frameworks for the same sentence; the fit uses the realised duration. The MRv2 arm is new "
                "on this stack: a failed start is recorded, not retried. Read the codepred arms' log line before the numbers."),
    Experiment(
        name="transport-qwen-t2s", title="Qwen3-Omni T2S on 3 cards: the same tensor transport for every framework (20 sentences, warm)",
        task="qwen_t2s", group="fair", kind="ab", frameworks=QWEN_FW, cases=("verbatim-en",), repeats=2, measured=20, preconditions=20, min_gpus=3,
        question="The three frameworks move tensors between stages differently. M* writes one file per tensor under /dev/shm by default, "
                 "vLLM-Omni uses its shared-memory connector by default, and SGLang-Omni uses CUDA IPC between GPU stages. Does the per-audio-second "
                 "gap change when M* and vLLM-Omni use the same Mooncake TCP transport?",
        reading="analyze_t2s_fit.py per arm: b (ms per audio second). Pair the arms by transport: shm (mstar-shm with vllm-shm) and Mooncake TCP "
                "(mstar-tcp with vllm-tcp). mstar-shm-again is the drift reference: a transport effect counts only if it is larger than the gap "
                "between mstar-shm and mstar-shm-again. sglang-cuda-ipc has no switch for its same-node edges, so it stays on CUDA IPC.",
        arms=(Arm("mstar-shm", frameworks=(MSTAR,), opts=(WARM_ROWS,), expect=_moe()),
              Arm("mstar-tcp", frameworks=(MSTAR,), opts=(WARM_ROWS,), serve_args=(MSTAR_TCP,),
                  expect=_moe(command_has=(MSTAR_TCP_CHECK,)), note="M* tensor manager over Mooncake TCP (loopback)"),
              Arm("vllm-shm", frameworks=("vllm-omni",), opts=(WARM_ROWS,), expect=_moe()),
              Arm("vllm-tcp", frameworks=("vllm-omni",), env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",),
                  opts=(WARM_ROWS, f"config.vllm-omni={QWEN_VLLM_MOONCAKE_TCP}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_mooncake_tcp"), CONNECTOR_SITE),
                              log_has=("Created connector: MooncakeTransferEngineConnector",), log_lacks=("Failed to create connector",)),
                  note="all three stage edges on Mooncake TCP"),
              Arm("sglang-cuda-ipc", frameworks=("sglang-omni",), opts=(WARM_ROWS,), expect=_moe()),
              Arm("mstar-shm-again", frameworks=(MSTAR,), opts=(WARM_ROWS,), expect=_moe())),
        post=("{py} exps/tools/analyze_t2s_fit.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in (
              "mstar-shm", "mstar-tcp", "vllm-shm", "vllm-tcp", "sglang-cuda-ipc", "mstar-shm-again"))
              + " --markdown {results}/transport-qwen-t2s_{hardware}.md",),
        caveats="SGLang-Omni has no config switch for its same-node stage edges (CUDA IPC, SHM when peer access is missing), so its SHM and "
                "Mooncake paths would need a source change and are not in this experiment. Mooncake here is TCP over loopback (no RDMA NIC). "
                "The M* and vLLM-Omni Mooncake arms are first runs on this stack: read the connector lines in server.log before the numbers."),
    Experiment(
        name="qwen-t2s-fit", title="Qwen3-Omni T2S: E2E and time-to-first-audio against audio duration (20 sentences, shapes pre-warmed)",
        task="qwen_t2s", group="fair", kind="ab", frameworks=QWEN_FW, cases=("verbatim-en",), repeats=2, measured=20, preconditions=20,
        question="T2S: M* is 1.18x ahead of SGLang-Omni and ~1.5x ahead of vLLM-Omni per audio second, and M*'s own launch chain explains only 1.24x "
                 "of it. Existing data (5 sentences) say: vLLM-Omni has ~1 in 4 requests with a time-to-first-audio stall (~400 ms) and, off the "
                 "stalls, a per-audio-second cost of 110-124 ms against M*'s 67-73 and SGLang's 89-95; first audio is ~20 ms later. Which part "
                 "is fixed latency and which the Talker's frame loop, over a wider set of sentence lengths, with the stalls removed by warming?",
        reading="analyze_t2s_fit.py: `a` (fixed ms) and `b` (ms per audio second) per framework from E2E = a + b * audio_s over 20 distinct sentences; "
                "`stalls` should be 0 when the rows are pre-warmed; if vLLM still stalls it is not first-seen-shape JIT. vllm-talker-greedy: "
                "vLLM's Talker sampled at temperature 0.9 / top_k 50 in the baseline (the request's talker_* fields do not reach it); if b drops "
                "to M*'s with a greedy Talker, the gap was the sampler / longer audio, not the engine. sglang is run with the Thinker radix cache "
                "off, vLLM and M* have none, so no framework gets a cache hit on the shared instruction.",
        arms=(Arm("sglang", frameworks=("sglang-omni",), opts=(WARM_ROWS, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("vllm", frameworks=("vllm-omni",), opts=(WARM_ROWS,), expect=_moe()),
              Arm("mstar", frameworks=(MSTAR,), opts=(WARM_ROWS,), expect=_moe()),
              Arm("vllm-talker-greedy", frameworks=("vllm-omni",), opts=(WARM_ROWS, f"config.vllm-omni={QWEN_VLLM_TALKER_GREEDY}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_talker_greedy"),)),
                  note="Talker temperature 0 / top_k 1 with the yaml's repetition_penalty 1.05, the closest vLLM-Omni accepts to M*'s greedy Talker")),
        post=("{py} exps/tools/analyze_t2s_fit.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("sglang", "vllm", "mstar", "vllm-talker-greedy"))
              + " --markdown {results}/qwen-t2s-fit_{hardware}.md",),
        caveats="Audio length still differs a little across frameworks for the same sentence (the Talkers do not decode identically); the fit uses "
                "the realised duration. If the greedy-Talker vLLM server hangs (repetition loop) that is recorded, not retried."),
    Experiment(
        name="qwen-ttft-txt", title="Qwen3-Omni first-token latency, text: fixed cost vs prefill (tiny and 1K prompts, one output token, cache off)",
        task="qwen_txt", group="fair", kind="ab", frameworks=("sglang-omni", "vllm-omni", MSTAR), cases=("txt-tiny-1tok", "txt-1k-1tok"), repeats=2,
        measured=10, preconditions=3,
        question="TXT: M*'s TTFT is 149 ms against vLLM-Omni's 99 (both without a prefix cache) and SGLang-Omni's ~108 (cache on). Is the extra ~50 ms "
                 "a per-request fixed cost (front end, scheduling, TP sync, sampling, first chunk) or the 1K-token prefill (MoE tiles, attention)?",
        reading="analyze_ttft_parts.py: `fixed` = E2E of a 56-token (8 + the server's 48 template) prompt with one output token; `per 1k tokens` = the extra for the 1K prompt. "
                "M* ahead of vLLM in neither column = noise; M* behind in `fixed` = the per-request path (the BAGEL I2T cache-hit path also "
                "left ~100 ms unaccounted); M* behind in `per 1k tokens` = prefill kernels (MoE tile tables, attention).",
        arms=(Arm("sglang", frameworks=("sglang-omni",), opts=(ELIGIBLE, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("vllm", frameworks=("vllm-omni",), opts=(ELIGIBLE,), expect=_moe()),
              Arm("mstar", frameworks=(MSTAR,), opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_moe())),
        post=("{py} exps/tools/analyze_ttft_parts.py exps/runs/{campaign:sglang} exps/runs/{campaign:vllm} exps/runs/{campaign:mstar} "
              "--markdown {results}/qwen-ttft-txt_{hardware}.md",),
        caveats="Linear prefill model between ~15 and 1,024 tokens. One output token, so ITL says nothing here."),
    Experiment(
        name="qwen-ttft-a2t", title="Qwen3-Omni first-token latency, audio: A2T with one output token (the audio path's share), cache off, shapes pre-warmed",
        task="qwen_a2t", group="fair", kind="ab", frameworks=("sglang-omni", "vllm-omni", MSTAR), cases=("a2t-en-1tok",), repeats=2,
        measured=10, preconditions=10,
        question="A2T: vLLM-Omni's TTFT is ~19 ms above M*'s on the typical clip (109 vs 90 ms). Which part of the audio path: the encoder, the "
                 "prefill of the audio tokens, or the front end?",
        reading="Run with qwen-ttft-txt and read analyze_ttft_parts.py's `A2T excess over text prefill`: the A2T E2E minus what a text prompt of the "
                "same token count costs on that framework = the audio encoder and its hand-off. A gap in the excess is the audio path; a gap "
                "that is already in `fixed` / `per 1k tokens` is the text path.",
        arms=(Arm("sglang", frameworks=("sglang-omni",), opts=(ELIGIBLE, WARM_ROWS, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("vllm", frameworks=("vllm-omni",), opts=(ELIGIBLE, WARM_ROWS), expect=_moe()),
              Arm("mstar", frameworks=(MSTAR,), opts=(ELIGIBLE, WARM_ROWS, MSTAR_CACHE_OFF), expect=_moe())),
        post=("{py} exps/tools/analyze_ttft_parts.py exps/runs/{campaign:sglang} exps/runs/{campaign:vllm} exps/runs/{campaign:mstar} "
              "--markdown {results}/qwen-ttft-a2t_{hardware}.md",),
        caveats="The text columns of analyze_ttft_parts.py come from qwen-ttft-txt; run its analyze with both campaigns' directories to fill them."),
    Experiment(
        name="qwen-prefill-nsys", title="Qwen3-Omni TXT: what the 1k-token prefill is made of in each framework (nsys, one output token, cache off)",
        task="qwen_txt", group="profile", kind="ab", frameworks=("sglang-omni", "vllm-omni", MSTAR), cases=("txt-1k-1tok",), repeats=1, measured=5, preconditions=3,
        question="qwen-ttft-txt: the fixed cost is the same in all three (M* 54 / SGLang 55 / vLLM 57 ms, one 56-token (8 + the server's 48 template) prompt) but the 1k-token prefill is M* 86, "
                 "SGLang 62, vLLM 42 ms per 1k tokens (TTFT 146 / 117 / 99 ms). That is the only TXT / A2T place where M* is clearly behind. Is it the MoE "
                 "kernels (M*'s own fused-MoE tile tables against vLLM's tuned Triton ones), the attention kernel, the TP=2 all-reduce, or launch gaps?",
        reading="analyze_nsys_burst_gaps.py, one table per process (both TP ranks) and the top kernels of the last burst, per framework. Compare GPU busy against span "
                "(span >> busy = launch / sync gaps, not kernels), then the top kernels' ms: a MoE GEMM that is ~2x vLLM's = the tile table (`mstar-moe-tune` "
                "re-sweeps it); attention or NCCL dominating = a different lever. A framework whose burst has a CUDA-graph replay shows few launches: say so. "
                "nsys inflates host time, kernel ms are the evidence.",
        arms=(Arm("sglang", frameworks=("sglang-omni",), nsys=True, opts=(ELIGIBLE, f"config.sglang={QWEN_SGLANG_NO_RADIX}"), expect=_moe(log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("vllm", frameworks=("vllm-omni",), nsys=True, opts=(ELIGIBLE,), expect=_moe()),
              Arm("mstar", frameworks=(MSTAR,), nsys=True, opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_moe())),
        post=tuple("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:" + a + "}/nsys --min-launches 300 --gap-ms 15 --last 4 --top 14 "
                   "--markdown {results}/qwen-prefill-nsys_" + a + "_{hardware}.md" for a in ("sglang", "vllm", "mstar")),
        caveats="One prompt length (1,072 tokens); the per-1k-token figure came from a two-point line between ~15 and ~1,072 tokens."),
    Experiment(
        name="qwen-prefill-chunk", title="Qwen3-Omni TXT, one output token: M*'s 1,024-token chunked-prefill budget against 4,096 (the 1,072-token prompt as one pass)",
        task="qwen_txt", group="fair", kind="ab", frameworks=(MSTAR,), cases=("txt-1k-1tok", "txt-tiny-1tok"), repeats=1, measured=10, preconditions=3,
        question="qwen-prefill-nsys: M*'s request has three 48-layer forwards, not one. Thinker MAX_BATCH_TOKENS is 1,024, so the 1,072-token prompt is cut 1,024 + 48; the "
                 "first chunk takes ~77 ms (vLLM's whole 1,072-token pass takes 77 ms), the 48-token chunk another ~40 ms (every expert's weights are read for 48 tokens), "
                 "the third forward is the second output token (M* returns 2 tokens for max_output_tokens=1). vLLM's Thinker budget is 8,192. Is the whole TXT prefill gap "
                 "(M* TTFT 146 against vLLM 99 / SGLang 117) this one chunk boundary?",
        reading="E2E and TTFT by arm and case (analyze_decode_itl.py). chunk-4096 on txt-1k-1tok close to vLLM's ~100 ms (about 40 ms below chunk-1024) = the whole gap is the "
                "chunk budget, i.e. M*'s prefill kernels are not slower; unchanged = the cause is elsewhere. txt-tiny-1tok must not move (a 15-token prompt is one chunk "
                "either way); if it moves, the budget changes something besides chunking. A raised budget costs decode steps that share a step with a long prompt, which "
                "this C=1 test cannot see.",
        arms=(Arm("chunk-1024", opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("chunk-4096", opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096"),)))),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:chunk-1024} exps/runs/{campaign:chunk-4096} --markdown {results}/qwen-prefill-chunk_{hardware}.md",),
        caveats="One long prompt length (1,072 tokens, 48 past the budget). The prediction for other lengths is a step function of ceil(tokens / budget); the Thinker "
                "budget of 4,096 is a counterfactual, not M*'s shipped config."),
    Experiment(
        name="qwen-prefill-len", title="Qwen3-Omni TXT, one output token: TTFT against prompt length (56 / 512 / 1,024 / 1,072 / 2,048 tokens) for M* with three chunk / graph-ladder settings, and vLLM-Omni",
        task="qwen_txt", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni"),
        cases=("txt-tiny-1tok", "txt-512-1tok", "txt-1024-1tok", "txt-1k-1tok", "txt-2048-1tok"), repeats=1, measured=10, preconditions=3,
        question="qwen-prefill-chunk falsified the chunk-budget reading of the 1,072-token TTFT gap: budget 4,096 made M* SLOWER (154 against 138 ms), not 40 ms faster. "
                 "M*'s Thinker prefill replays CUDA graphs on a 128...2048 ladder and pads a step to the smallest bucket that fits, so a 1,072-token step goes to 2,048 "
                 "when it is not cut. Where does M*'s TTFT sit against vLLM's as the prompt length moves across the 1,024 budget and the ladder's rungs, and does "
                 "a ladder with a rung near the prompt (chunk-4096-b1152) remove the penalty at 1,072?",
        reading="analyze_decode_itl.py, TTFT (ms) per case and arm; read TTFT, not E2E (M* returns 2 tokens for a 1-token cap). Plot TTFT against server-side prompt tokens "
                "(tiny = 56, 512, 1,024, 1,072, 2,048). vLLM's line is the reference. If M* matches vLLM at 512 and 1,024 (both exact rungs and within one chunk) the 1,072 gap "
                "is a boundary position, not slow kernels; chunk-4096 slower than chunk-1024 at 1,072 but equal at 1,024 and 2,048 = the padding story; chunk-4096-b1152 near "
                "vLLM at 1,072 = the ladder is the lever. M* behind vLLM at 512 or 1,024 too = the gap is kernels / per-step cost after all and the padding story is wrong. "
                "chunk-1024-again against chunk-1024 = order drift.",
        arms=(Arm("chunk-1024", frameworks=(MSTAR,), opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("chunk-4096", frameworks=(MSTAR,), opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096"),))),
              Arm("chunk-4096-b1152", frameworks=(MSTAR,), opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_B1152}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_bucket1152"),))),
              Arm("vllm", frameworks=("vllm-omni",), opts=(ELIGIBLE,), expect=_moe()),
              Arm("chunk-1024-again", frameworks=(MSTAR,), opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold())),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:chunk-1024} exps/runs/{campaign:chunk-4096} exps/runs/{campaign:chunk-4096-b1152} "
              "exps/runs/{campaign:vllm} exps/runs/{campaign:chunk-1024-again} --markdown {results}/qwen-prefill-len_{hardware}.md",),
        caveats="The 512 / 1,024 / 2,048 prompts are the 1k prompt's words cut or cycled (body = target - 48 template tokens, the overhead read off SGLang-Omni's log); the "
                "server-side length is not asserted by the harness. One start per arm and case. 2,048 is the ladder's top rung: past it a step runs eagerly (not measured). "
                "C=1 only: the chunk budget's effect on decode rows sharing a step with a prompt is invisible here."),
    Experiment(
        name="qwen-prefill-chunk-nsys", title="Qwen3-Omni TXT 1,072-token prompt on M*: forwards per request and kernel time with chunk 1,024 / 4,096 / 4,096 + a 1,152 graph rung (nsys)",
        task="qwen_txt", group="profile", kind="ab", frameworks=(MSTAR,), cases=("txt-1k-1tok",), repeats=1, measured=5, preconditions=3,
        question="qwen-prefill-chunk: budget 4,096 did not remove the second forward's cost (TTFT 154 against 138 ms). Is the 1,072-token prompt really ONE forward there, and "
                 "how long is it: a graph padded to 2,048 tokens (about twice the dense / attention work), or something else? And does a 1,152 rung shorten it?",
        reading="analyze_nsys_burst_gaps.py per arm, both TP ranks. Count the attention launches in a burst (144 = three 48-layer forwards in the default; 96 = two). Chunk-4096 with "
                "one prompt forward + a decode forward whose prompt forward is ~100+ ms = padding (compare its GEMM / attention kernel ms against chunk-1024's first pass: "
                "~2x = a 2,048 graph); chunk-4096 still showing three forwards = the budget did not take effect. chunk-4096-b1152's prompt forward near 80 ms = the ladder is the lever.",
        arms=(Arm("chunk-1024", nsys=True, opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("chunk-4096", nsys=True, opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096"),))),
              Arm("chunk-4096-b1152", nsys=True, opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_B1152}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_bucket1152"),)))),
        post=tuple("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:" + a + "}/nsys --min-launches 300 --gap-ms 15 --last 4 --top 14 "
                   "--markdown {results}/qwen-prefill-chunk-nsys_" + a + "_{hardware}.md" for a in ("chunk-1024", "chunk-4096", "chunk-4096-b1152")),
        caveats="One prompt length (1,072 tokens). nsys inflates host time; kernel ms are the evidence."),
    Experiment(
        name="qwen-prefill-nsys-room", title="Qwen3-Omni TXT 1,072-token prompt on M*, budget 4,096: nsys WITH the 2,048 prefill graph (KV pool halved), against the same config without nsys",
        task="qwen_txt", group="profile", kind="ab", frameworks=(MSTAR,), cases=("txt-1k-1tok",), repeats=1, measured=10, preconditions=3,
        question="qwen-prefill-chunk-nsys measured the budget-4,096 arm with NO 2,048 prefill graph (nsys itself takes the memory; `captured 52 of 74`), so its 1,072-token step ran eagerly "
                 "(E2E 123 ms) instead of in the padded 2,048 graph the production server uses (E2E 163-168 ms). How long is the padded forward on the GPU, and is it ~2x the unpadded one "
                 "(GEMM / attention / MoE kernels at 2,048 tokens)? Halving the Thinker KV pool (512 -> 256 pages) is the attempt to leave room for that graph under nsys.",
        reading="First the capture line: the nsys arms' Expect needs `Captured accelerator graph for Thinker: prefill_text[bs=32,tokens=2048]` in server.log; without it the cell is rejected and "
                "the answer is `nsys under this config still drops the graph`. Then room-plain (no nsys) E2E ~160-168 = this config still pads to 2,048; room-nsys E2E ~ room-plain = nsys no longer "
                "changes the path. Kernel ms of room-nsys's one prompt forward against the eager 1,072 forward of qwen-prefill-chunk-nsys (span 105-112 / busy 85-93 / MoE 34 ms): dense GEMM + "
                "attention ~2x and MoE up = padding to 2,048 tokens is real GPU work. room-*-again are the A/B/A/B order check.",
        arms=(Arm("room-nsys", nsys=True, opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_ROOM}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_nsys_room"),), log_has=(G2048,))),
              Arm("room-plain", opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_ROOM}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_nsys_room"),), log_has=(G2048,))),
              Arm("room-nsys-again", nsys=True, opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_ROOM}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_nsys_room"),), log_has=(G2048,))),
              Arm("room-plain-again", opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_ROOM}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_nsys_room"),), log_has=(G2048,)))),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:room-plain} exps/runs/{campaign:room-nsys} exps/runs/{campaign:room-plain-again} "
              "exps/runs/{campaign:room-nsys-again} --markdown {results}/qwen-prefill-nsys-room_{hardware}.md",
              "{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:room-nsys}/nsys --min-launches 300 --gap-ms 15 --last 4 --top 14 "
              "--markdown {results}/qwen-prefill-nsys-room_room-nsys_{hardware}.md"),
        caveats="One prompt length (1,072 tokens). The smaller KV pool is a second change next to nsys; room-plain is the control for it (it must still show ~160 ms, not 123). nsys inflates host time; "
                "kernel ms are the evidence."),
    Experiment(
        name="qwen-fair-txt-b1152", title="Qwen3-Omni TXT, fair profile (cache off): does M*'s shipped ladder or the 1,152 rung decide the 'M* is 2.4% slower' headline? (256 and 1 output tokens)",
        task="qwen_txt", group="fair", kind="ab", frameworks=("sglang-omni", "vllm-omni", MSTAR), cases=("txt-1k", "txt-1k-1tok"), repeats=2,
        question="qwen-fair-txt: with every prefix cache off M* is 2.4% slower than vLLM on the 1,072-token TXT request (1,648 against 1,610 ms), all of it first-token latency. qwen-prefill-len: ~32 of the "
                 "47 ms TTFT gap at 1,072 tokens is the default prefill-graph ladder (1,024 + 48 in two forwards, or one step padded to 2,048 at budget 4,096); a 1,152 rung with budget 4,096 gives "
                 "TTFT 113 against 145 ms. Does the 256-token E2E headline follow, i.e. does M* with the rung catch up with or pass vLLM?",
        reading="analyze_decode_itl.py: E2E (txt-1k) and TTFT (txt-1k-1tok) per arm. mstar-off is the shipped config, mstar-b1152 the counterfactual (budget 4,096 + 1,152 rung), sglang-off / vllm-off the "
                "same arms as qwen-fair-txt (repeat check against its numbers). mstar-b1152 E2E within ~1% of vllm-off or below = the TXT lag is a configuration, report both rows; still ~2% behind = a kernel-level "
                "residual (qwen-prefill-len: ~11-15 ms from 1,024 tokens up). ITL must be equal to mstar-off (the ladder only changes prefill).",
        arms=(Arm("sglang-off", frameworks=("sglang-omni",), opts=(ELIGIBLE, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(command_has=(_vfile("sglang_omni/qwen_no_radix"),), log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("vllm-off", frameworks=("vllm-omni",), opts=(ELIGIBLE,), expect=_moe(log_lacks=("enable_prefix_caching=True",))),
              Arm("mstar-off", frameworks=(MSTAR,), opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("mstar-b1152", frameworks=(MSTAR,), opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_B1152}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_bucket1152"),)))),
        post=("{py} exps/tools/analyze_decode_itl.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("sglang-off", "vllm-off", "mstar-off", "mstar-b1152"))
              + " --markdown {results}/qwen-fair-txt-b1152_{hardware}.md",),
        caveats="Same protocol as qwen-fair-txt (cache off, lookahead-eligible request, 2 starts); C=1 only. The 1,152 config also sets the chunk budget to 4,096, so a concurrent decode row shares its step with a "
                "longer prefill: invisible here. It displaces the 2,048 prefill graph (qwen-prefill-len, section 9.6)."),
    Experiment(
        name="qwen-prefill-len-dense", title="Qwen3-Omni TXT, one output token: TTFT at 1,100 / 1,300 / 1,536 / 1,800 prompt tokens, M* ladders against vLLM-Omni",
        task="qwen_txt", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni"),
        cases=("txt-1100-1tok", "txt-1300-1tok", "txt-1536-1tok", "txt-1800-1tok"), repeats=1, measured=10, preconditions=3,
        question="qwen-prefill-len: a 1,152 rung fixed 1,072 (TTFT 113 against 145 / 156 ms) and displaced the 2,048 graph. Between the rungs the shipped ladder pads every step to the next captured graph, so the "
                 "TTFT line against prompt length should be a staircase. Where are the steps for each ladder (default / budget 4,096 / + 1,152 / + 1,152 + 1,536), and does a second rung at 1,536 get captured on this 48 GiB card?",
        reading="analyze_decode_itl.py, TTFT per case and arm, with vLLM as the reference line. Default: 1,100 = 1,024 + 76 (two forwards, ~+38 ms), 1,300 and 1,536 likewise, 1,800 = 1,024 + 776. Budget 4,096: all four padded to "
                "the 2,048 graph (flat ~155-165 ms). + 1,152: 1,100 on the 1,152 rung, 1,300 / 1,536 / 1,800 on a graph only if one exists (else eager at the actual size). + 1,152 + 1,536: 1,300 and 1,536 on the 1,536 rung "
                "IF the log shows it captured (`Captured accelerator graph for Thinker: prefill_text[bs=1,tokens=1536]`); if not, that arm is the + 1,152 arm and the card cannot hold another rung.",
        arms=(Arm("chunk-1024", frameworks=(MSTAR,), opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("chunk-4096", frameworks=(MSTAR,), opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096"),))),
              Arm("chunk-4096-b1152", frameworks=(MSTAR,), opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_B1152}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_bucket1152"),))),
              Arm("chunk-4096-b1152-1536", frameworks=(MSTAR,), opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_B1152_1536}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_bucket1152_1536"),))),
              Arm("vllm", frameworks=("vllm-omni",), opts=(ELIGIBLE,), expect=_moe())),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:chunk-1024} exps/runs/{campaign:chunk-4096} exps/runs/{campaign:chunk-4096-b1152} "
              "exps/runs/{campaign:chunk-4096-b1152-1536} exps/runs/{campaign:vllm} --markdown {results}/qwen-prefill-len-dense_{hardware}.md",),
        caveats="Nominal server-side lengths (body = target - 48 template tokens, not asserted by the harness). One start per arm and case. After the run, read the capture list of each M* arm "
                "(`grep 'Captured accelerator graph for Thinker: prefill_text' server.log`, and the `captured N of M buckets` ERROR line): a rung that was not captured is not a rung. C=1 only."),
    Experiment(
        name="qwen-prefill-len-sglang", title="Qwen3-Omni TXT, one output token: SGLang-Omni's TTFT against prompt length (56 / 512 / 1,024 / 1,072 / 2,048), cache off",
        task="qwen_txt", group="fair", kind="ab", frameworks=("sglang-omni",),
        cases=("txt-tiny-1tok", "txt-512-1tok", "txt-1024-1tok", "txt-1k-1tok", "txt-2048-1tok"), repeats=1, measured=10, preconditions=3,
        question="qwen-prefill-len compares M* with vLLM-Omni only. SGLang-Omni's 1,072-token TTFT was 117 ms (qwen-ttft-txt, M* 146, vLLM 99). Is SGLang's line a straight 56 -> 2,048 line like vLLM's or does it "
                 "also have a step, and where does it sit between the two?",
        reading="analyze_decode_itl.py, TTFT per case against qwen-prefill-len's vllm arm and M* arms (same cases, same protocol, same request). SGLang with the radix cache off (config no_radix) so every request is a cold prefill.",
        arms=(Arm("sglang", frameworks=("sglang-omni",), opts=(ELIGIBLE, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(command_has=(_vfile("sglang_omni/qwen_no_radix"),), log_lacks=(SGLANG_NO_PREFIX_HIT,))),),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:sglang} --markdown {results}/qwen-prefill-len-sglang_{hardware}.md",),
        caveats="SGLang-Omni returns the first text chunk differently from M* / vLLM (a chunk can carry several tokens); with one output token TTFT is the first chunk. One start per case."),
    Experiment(
        name="qwen-prefill-len-repeat", title="Qwen3-Omni TXT, one output token: start-to-start repeatability of the 512- and 1,072-token TTFT, M* default against the 1,152 rung (3 starts)",
        task="qwen_txt", group="fair", kind="ab", frameworks=(MSTAR,), cases=("txt-512-1tok", "txt-1k-1tok"), repeats=3, measured=10, preconditions=3,
        question="qwen-prefill-len's default arm, run twice, agreed to 0.2-3.9 ms everywhere except 512 tokens (71.3 against 78.7 ms), and every conclusion rests on one start per arm. How large is the start-to-start spread "
                 "at 512 and 1,072 tokens, and does the 113 against 145 ms (1,152 rung against default) survive 3 starts each?",
        reading="Per-start TTFT p50 (analyze_decode_itl.py lists the start range). A spread of 5+ ms at 512 = differences below ~8 ms in the length sweeps are not readable; 1,152 below default in all 3 starts at 1,072 = "
                "the 32 ms is not an artefact of one start.",
        arms=(Arm("chunk-1024", opts=(ELIGIBLE, MSTAR_CACHE_OFF), expect=_mstar_cold()),
              Arm("chunk-4096-b1152", opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_CHUNK4096_B1152}"),
                  expect=_mstar_cold(command_has=(_vfile("mstar/qwen_chunk4096_bucket1152"),)))),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:chunk-1024} exps/runs/{campaign:chunk-4096-b1152} --markdown {results}/qwen-prefill-len-repeat_{hardware}.md",),
        caveats="Starts are separate server launches back to back, not independent days."),
    Experiment(
        name="pi05-nsys", title="Pi0.5: GPU timeline of one action request in M*, SGLang core and vLLM-Omni (nsys)",
        task="pi05", group="profile", kind="ab", frameworks=(MSTAR, "sglang-core", "vllm-omni"), cases=("droid-46row",), repeats=1, measured=4, preconditions=2,
        question="baseline-pi05: M* 104 ms, SGLang core 164, vLLM-Omni (eager) 220 for the same 10 flow-matching steps; M* with every ablation off (mstar-ablation-pi05) is still 120 ms. Nothing says where the "
                 "40-100 ms go. Is it GPU work (different kernels per step, un-fused attention / GEMM, a larger vision encoder pass) or idle GPU between launches (launch gaps, host-side image / state handling)?",
        reading="analyze_nsys_burst_gaps.py, last burst per process: GPU busy against span and launch count per framework. Same busy and a longer span = launch / host gaps; different busy = different kernels, then "
                "read the top-kernel list (vision encoder, VLM prefill, action expert x 10 steps). Structure only: nsys inflates host time, never read E2E from it.",
        arms=(Arm("mstar", frameworks=(MSTAR,), nsys=True, serve_args=("mstar:--enable-nvtx",)),
              # H200: the SGLang worker exits before the default 1,000 ms CUDA flush and its capture holds no kernels (2026-11-14 run
              # 01 and 02; pi05-nsys-sglang-diag: flush 100 ms records them, graph-level trace and CUDA graph off do not). 5880 keeps 1,000 ms.
              Arm("sglang-core", frameworks=("sglang-core",), nsys=True, nsys_flush_ms="{hw:pi05_sglang_nsys_flush_ms}"),
              Arm("vllm", frameworks=("vllm-omni",), nsys=True)),
        post=tuple("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:" + a + "}/nsys --min-launches 300 --gap-ms 10 --last 4 --top 14 "
                   "--markdown {results}/pi05-nsys_" + a + "_{hardware}.md" for a in ("mstar", "sglang-core", "vllm")),
        caveats="No fairness audit of the Pi0.5 configs exists yet (attention backend, dtype, compile, image preprocessing, number of denoise steps is the only equalised setting); this run only says where the time is."),
    Experiment(
        name="pi05-nsys-sglang-diag", title="Pi0.5 SGLang core under nsys: why did the H200 capture hold no kernels, and which variant records them?",
        task="pi05", group="profile", kind="ab", frameworks=("sglang-core",), cases=("droid-46row",), repeats=1, measured=4, preconditions=2,
        question="pi05-nsys on the H200 captured SGLang core's server and its worker process but no CUDA kernel at all (1,128 runtime calls, 564 per process, no "
                 "CUPTI_ACTIVITY_KIND_KERNEL table; M* and vLLM-Omni captured 195k and 14k launches in the same harness; the rtx5880 capture of the same arm is fine; "
                 "torch, CUDA and CUPTI are 13.0 / 13.0.85 in all three environments, so it is not a library mismatch). Is it node-level tracing of the graphs SGLang captures lazily, "
                 "the buffers of a worker that is stopped by SIGTERM, or something the CUDA graphs have to do with at all?",
        reading="analyze_nsys_burst_gaps.py per arm: the arm whose capture has kernels names the cause. graph-trace-graph (`--cuda-graph-trace=graph`, no per-node tracing) recording "
                "kernels = node-level tracing of lazily captured graphs; flush-100ms recording them = buffers lost at shutdown (a shorter interval bounds the loss); no-cuda-graph "
                "recording them while the other two do not = the graphs are the cause, and that capture is a different workload (SGLang with graphs off, 10 ms of launch gaps "
                "per step), so read it as structure only. None recording = not the capture options; read server.log and try another nsys build "
                "(the rtx5880 box captures with ~/.local/opt/nsys-2026.6.1, this one with /usr/local/bin/nsys 2026.4.1). Three one-cell arms, ~10 minutes. The first arm that "
                "records kernels is also the missing SGLang-core column of pi05-nsys.",
        arms=(Arm("graph-trace-graph", frameworks=("sglang-core",), nsys=True, nsys_graph_trace="graph"),
              Arm("flush-100ms", frameworks=("sglang-core",), nsys=True, nsys_flush_ms=100),
              Arm("no-cuda-graph", frameworks=("sglang-core",), nsys=True, opts=(f"config.sglang={PI05_SGLANG_GRAPH_OFF}",),
                  expect=Expect(command_has=("pi05_pipeline_graph_off.json",)),
                  note="counterfactual: prefix and action CUDA graphs off, a different workload")),
        post=tuple("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:" + a + "}/nsys --min-launches 300 --gap-ms 10 --last 4 --top 14 "
                   "--markdown {results}/pi05-nsys-sglang-diag_" + a + "_{hardware}.md" for a in ("graph-trace-graph", "flush-100ms", "no-cuda-graph")),
        caveats="A diagnostic of the capture, not a measurement: nsys inflates host time. Each arm is one start of one cell."),
    Experiment(
        name="orpheus-nsys", title="Orpheus TTS: GPU timeline of one request (b1) and one 16-request wave (b16) in M* and VoxServe (nsys)",
        task="orpheus", group="profile", kind="ab", frameworks=(MSTAR, "voxserve"), cases=("b1", "b16"), repeats=1, measured=2, preconditions=1,
        question="baseline-orpheus: M* leads VoxServe 1.09x at b1 and 1.38x at b16 (E2E 2,531 against 2,751 and 5,000 against 6,925 ms) with sampling matched. The lead grows with the batch. Is VoxServe's extra time "
                 "GPU work (a slower decode step, a slower SNAC decoder) or gaps between launches, and does the gap per step grow with the batch?",
        reading="analyze_nsys_burst_gaps.py per framework and case (the last burst): span against busy and the launch count per decode step. Equal busy and a longer VoxServe span = scheduling / launch gaps; unequal busy "
                "= kernels (read the top-kernel list). Compare b1 against b16 for how the gap scales.",
        arms=(Arm("mstar", frameworks=(MSTAR,), nsys=True, serve_args=("mstar:--enable-nvtx",)),
              Arm("voxserve", frameworks=("voxserve",), nsys=True)),
        post=tuple("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:" + a + "}/nsys --min-launches 300 --gap-ms 15 --last 4 --top 14 "
                   "--markdown {results}/orpheus-nsys_" + a + "_{hardware}.md" for a in ("mstar", "voxserve")),
        caveats="No fairness audit of VoxServe's serve flags beyond the matched sampling (its CUDA graph / max batch settings are what `frameworks.voxserve_serve` passes). Two measured requests per cell: structure, not speed."),
    Experiment(
        name="pi05-audit", title="Pi0.5: what do each framework's own switches (CUDA graphs, torch.compile, attention kernel) account for? M* 104 / SGLang core 164 / vLLM-Omni 220 ms",
        task="pi05", group="fair", kind="ab", frameworks=(MSTAR, "sglang-core", "vllm-omni"), cases=("droid-46row",), repeats=2, measured=12, preconditions=5,
        question="baseline-pi05: M* 104 ms, SGLang core 164, vLLM-Omni (eager) 220 for the same 10 flow-matching steps and [50,32] action. mstar-ablation-pi05 switched M*'s own chain / Walk / compile "
                 "off and left 120 ms, still 1.4x ahead of SGLang core, and never touched the other two. Source reading found three differences that are not 'the engine': M* captures SigLIP, "
                 "prefill and the 10-step action expert in CUDA graphs and torch.compiles them (max-autotune); SGLang core has two independent graph switches (prefix, action; both on); "
                 "vLLM-Omni's `_attend` is a hand-written matmul + softmax, whatever --enforce-eager says. Which of those is the gap?",
        reading="Within a framework, each arm against its first (mstar / sglang / vllm): mstar-nocompile (TORCHDYNAMO_DISABLE=1) and mstar-nograph (hook: no accelerator graph at all) alone, "
                "and mstar-eager both: M* with the same 'no graph, no compile' as vLLM's eager recipe. sglang-nograph: both of SGLang core's CUDA graphs off. vllm-sdpa: the same attention math "
                "through F.scaled_dot_product_attention; vllm-graph: enforce_eager off (earlier: no change, the pipeline is not graph-captured); vllm-sdpa-graph both. "
                "Then across frameworks: mstar-eager against vllm (and sglang-nograph) is the like-for-like 'no graph, no compile' comparison; vllm-sdpa against sglang is vLLM with a modern attention kernel. "
                "A switch that moves a framework by <5% is not part of the gap. Hook arms need their proof file (applied=true) or the cell is rejected.",
        arms=(Arm("mstar", frameworks=(MSTAR,)),
              Arm("mstar-nocompile", frameworks=(MSTAR,), env=NO_COMPILE, expect=_cmd_has(*NO_COMPILE)),
              Arm("mstar-nograph", frameworks=(MSTAR,), env=PI05_MSTAR_NO_GRAPH, expect=Expect(proof="hook_applied"), note="counterfactual: no SigLIP / prefill / action-expert graph"),
              Arm("mstar-eager", frameworks=(MSTAR,), env=PI05_MSTAR_NO_GRAPH + NO_COMPILE, expect=Expect(proof="hook_applied"),
                  note="counterfactual: M* with neither graphs nor compile, the closest M* has to vLLM-Omni's eager recipe"),
              Arm("sglang", frameworks=("sglang-core",)),
              Arm("sglang-nograph", frameworks=("sglang-core",), opts=(f"config.sglang={PI05_SGLANG_GRAPH_OFF}",),
                  expect=Expect(command_has=("pi05_pipeline_graph_off.json",)), note="counterfactual: prefix and action CUDA graphs off"),
              Arm("vllm", frameworks=("vllm-omni",)),
              Arm("vllm-sdpa", frameworks=("vllm-omni",), env=PI05_VLLM_SDPA, expect=Expect(proof="hook_applied"), note="counterfactual: attention through SDPA"),
              Arm("vllm-graph", frameworks=("vllm-omni",), opts=("pi05.vllm_graph=1",), note="counterfactual: enforce_eager off"),
              Arm("vllm-sdpa-graph", frameworks=("vllm-omni",), env=PI05_VLLM_SDPA, opts=("pi05.vllm_graph=1",), expect=Expect(proof="hook_applied"),
                  note="counterfactual: SDPA attention and enforce_eager off")),
        caveats="Counterfactual arms are not baselines: hooks change how a framework runs (they are listed in the manifest, `external/` is untouched). The output actions are not compared; "
                "the SDPA hook changes the softmax from an explicit fp32 upcast to the kernel's own accumulation (rounding level, CPU-checked: fp32 5e-7, bf16 one ulp). "
                "Not audited: attention backends inside SGLang core and M* (both are their own kernels), image preprocessing (outside the timer on every path), the transport (HTTP against msgpack WebSocket)."),
    Experiment(
        name="pi05-steps", title="Pi0.5 at 1 / 5 / 10 / 20 flow steps: how much of each framework's latency is the denoise loop?",
        task="pi05", group="fair", kind="compare", frameworks=(MSTAR, "sglang-core", "vllm-omni"),
        cases=("droid-46row-s1", "droid-46row-s5", "droid-46row", "droid-46row-s20"), repeats=2, measured=12, preconditions=5,
        question="baseline-pi05 runs 10 flow-matching steps, so its ratios say nothing about the denoise loop. Each step is one action-expert "
                 "forward on the prefix KV, so E2E = fixed + steps x per-step. Which framework pays more per step, and which pays more fixed cost "
                 "(images, prefix, request path)? Does M*'s lead come from the step, the fixed part, or both?",
        reading="Per case (step count), the ratio best baseline / M* as in baseline-pi05, pooled over starts with the per-start range beside it. "
                "Then the slope per framework: (p50 at 20 steps - p50 at 1 step) / 19. A lead that grows with the step count is a per-step lead; a "
                "lead the same at 1 and at 20 steps is a fixed-cost lead. The intercept (p50 at 1 step minus one slope) is the fixed cost.",
        caveats="Steps 1, 5 and 20 are counterfactual M* services (`exps/configs/mstar/pi05_bfloat16_steps{N}.yaml`, identical to the shipped "
                "10-step file except `num_flow_steps`). vLLM-Omni and SGLang core take the count per request, so their configs stay at 10 "
                "(verified in source: vLLM-Omni `sampling_params.num_inference_steps`, SGLang core payload `num_inference_steps`). Latency only: "
                "the actions are shape-checked, not compared across step counts. The 1-step point is a timing point, not a usable policy."),
    Experiment(
        name="pi05-kv", title="Pi0.5: does a prefix cache change the action latency? SGLang core cache on / off, M*'s default, vLLM-Omni as control",
        task="pi05", group="fair", kind="compare", frameworks=(MSTAR, "sglang-core", "vllm-omni"), cases=("droid-row0-x20",),
        repeats=2, measured=20, preconditions=5,
        question="A Pi0.5 request carries three images, the prompt and the discretised state; a prefix cache reuses the prefix KV only when a "
                 "later request repeats it. The 46-row fixture never repeats a row, so this case sends row 0 on every request of a start. Is "
                 "SGLang core's global prefix cache on in its shipped config (no: enable_global_prefix_cache=false), and does M*'s default KV "
                 "prefix cache change Pi0.5 latency (source says it never opens for Pi0.5)?",
        reading="sglang-core against sglang-cache on the same row: a lower p50 for sglang-cache is the prefix cache, and the difference is the saving "
                "per request. The log line shows the setting took effect, not that it hit: a hit shows as a fall after the first request of each "
                "start, so read the per-request column. M*: its Pi0.5 prefix cache is not opened (no keyed KV label, no checkpoint name, no "
                "prefix_cache_salt), so its row-0 latency should match its 46-row latency; a fall on repeated rows would mean the source reading is "
                "wrong. vLLM-Omni has no Pi0.5 prefix cache and is the control for a repeated row.",
        arms=(Arm("mstar", frameworks=(MSTAR,),
                  note="M*'s default. The per-request prefix_cache=false knob (bagel.prefix_cache=off) would have nothing to switch off: the cache never opens for Pi0.5"),
              Arm("sglang-core", frameworks=("sglang-core",),
                  expect=Expect(command_has=("pi05_pipeline.json",), log_has=("Pi05 memory config: prefix_cache=False/",)),
                  note="shipped config: enable_global_prefix_cache=false"),
              Arm("sglang-cache", frameworks=("sglang-core",), opts=(f"config.sglang={PI05_SGLANG_PREFIX_CACHE}",),
                  expect=Expect(command_has=("pi05_pipeline_prefix_cache.json",), log_has=("Pi05 memory config: prefix_cache=True/",)),
                  note="counterfactual: enable_global_prefix_cache=true, otherwise the shipped config"),
              Arm("vllm-omni", frameworks=("vllm-omni",), note="control: Pi0.5 has no prefix cache in this pipeline")),
        caveats="No M* cache-off arm: M*'s pi05 prefix cache is closed by construction, so an off arm would be the same configuration. "
                "The shipped SGLang config has enable_global_prefix_cache=false; the cache also needs conditioning_cache_max_size_mb > 0 "
                "(default 512). The `Pi05 memory config` log lines are source-verified (sglang multimodal_gen pipelines/pi05.py), not yet seen in "
                "a server.log: the first run confirms them."),
    Experiment(
        name="pi05-graph-split", title="Pi0.5 on SGLang core: which of its two CUDA graphs (prefix or action expert) carries the latency?",
        task="pi05", group="fair", kind="ab", frameworks=("sglang-core",), cases=("droid-46row",), repeats=2, measured=12, preconditions=5,
        question="pi05-audit turned both SGLang core graphs off together (sglang-nograph). The prefix graph captures the images and the prompt; "
                 "the action graph captures the 10-step expert loop. Which graph is the cost of running without graphs? M* captures both, so "
                 "the answer bears on whether SGLang core's gap is a graph gap.",
        reading="Each arm against sglang (both graphs on). The arm that slows more holds the graph that matters; a change under 5% is not part of "
                "the gap (the pi05-audit rule). If the two single-graph penalties add up to pi05-audit's sglang-nograph, the graphs act "
                "independently; if they do not, the two interact.",
        arms=(Arm("sglang", frameworks=("sglang-core",),
                  expect=Expect(log_has=("cuda_graph=prefix:True/", "action:True/")), note="shipped: both graphs on"),
              Arm("sglang-prefix-off", frameworks=("sglang-core",), opts=(f"config.sglang={PI05_SGLANG_PREFIX_GRAPH_OFF}",),
                  expect=Expect(command_has=("pi05_pipeline_prefix_graph_off.json",), log_has=("cuda_graph=prefix:False/", "action:True/")),
                  note="counterfactual: prefix CUDA graph off, action graph on"),
              Arm("sglang-action-off", frameworks=("sglang-core",), opts=(f"config.sglang={PI05_SGLANG_ACTION_GRAPH_OFF}",),
                  expect=Expect(command_has=("pi05_pipeline_action_graph_off.json",), log_has=("cuda_graph=prefix:True/", "action:False/")),
                  note="counterfactual: action CUDA graph off, prefix graph on")),
        caveats="Counterfactual configs are copies of `pi05_pipeline.json` with one graph flag changed (exps/configs/sglang_core/). The M* split is "
                "not built: M*'s Pi0.5 graphs (SigLIP, prefill, action expert) are switched only together, by the `mstar_pi05_nograph` hook, and a "
                "per-graph split needs that hook to take a family list. Not built here."),
    Experiment(
        name="pi05-sglang-buckets", title="Pi0.5 on SGLang core: is the shipped config its best? Prompt-length buckets against the default, 46-row fixture",
        task="pi05", group="fair", kind="ab", frameworks=("sglang-core",), cases=("droid-46row",), repeats=2, measured=12, preconditions=5,
        question="The shipped SGLang core Pi0.5 config leaves prompt_token_buckets empty. Its denoise CUDA graph is keyed by the exact prefix length and "
                 "holds 4 signatures (action_cuda_graph_max_entries); once full, new signatures run eagerly ('VLA denoise CUDA graph capacity reached'). "
                 "The 46 rows give several prefix lengths, so most denoise steps may run eager, while a repeated row (pi05-kv) hits the graph. Padding every "
                 "prompt to one bucket gives one signature. Does the best config for SGLang core move its ratio to M*?",
        reading="sglang (shipped) against sglang-buckets (prompt_token_buckets [200], the config's max_token_len, so no prompt misses a bucket). A drop of 5% or "
                "more is a config effect and replaces the SGLang row of baseline-pi05; a smaller change means the eager fallback was not the cost. The buckets "
                "arm must log no capacity message.",
        arms=(Arm("sglang", frameworks=("sglang-core",),
                  expect=Expect(log_has=("cuda_graph=prefix:True/", "action:True/", "VLA denoise CUDA graph capacity reached")),
                  note="shipped: no prompt buckets, so denoise graphs are keyed by exact prefix length"),
              Arm("sglang-buckets", frameworks=("sglang-core",), opts=(f"config.sglang={PI05_SGLANG_BUCKETS}",),
                  expect=Expect(command_has=("pi05_pipeline_buckets200.json",), log_has=("buckets:[200]",),
                                log_lacks=("VLA denoise CUDA graph capacity reached",)),
                  note="counterfactual: prompts padded to one bucket of 200 tokens")),
        caveats="Counterfactual config = `pi05_pipeline.json` plus prompt_token_buckets [200] (exps/configs/sglang_core/). 200 is the config's max_token_len, "
                "not a tuned value: a tighter bucket could be faster and is not tested. The captured requests in the baseline log had 140 to 148 prompt "
                "tokens, so the padding is about 50 tokens. Latency only; actions are shape-checked."),
    Experiment(
        name="cosmos3-sglang", title="Cosmos3-Nano T2I 256x256: M* against vLLM-Omni and SGLang-Omni (direct-generation pipeline)",
        task="cosmos3", group="fair", kind="compare", frameworks=(MSTAR, "vllm-omni", "sglang-omni"), cases=("t2i-256",), repeats=3, measured=5, preconditions=2,
        question="baseline-cosmos3: M* 1,290 ms against vLLM-Omni 3,133 ms (2.46x), decomposed into batched CFG 1.54x x compile 1.38x x host launch gaps 1.17x. SGLang-Omni serves Cosmos3 through native SGLang "
                 "multimodal_gen (Cosmos3PipelineConfig; docs/cookbook/cosmos3_nano.md) and was never run. Where does it sit, and is the 2.46x a vLLM-Omni property or an M* one?",
        reading="Same 5 prompts, 256x256, 50 steps, guidance 6.0, interval [400, 1000], seed 42, one image per response, size verified from the PNG header. SGLang-Omni guardrails are off "
                "(SGLANG_DISABLE_COSMOS3_GUARDRAILS=1, as vLLM-Omni's `guardrails: false`). SGLang between the two = the structure story holds in a third implementation; SGLang near vLLM = the gap is "
                "what M* does (batched CFG, compile); SGLang at or below M* = vLLM-Omni's Cosmos3 pipeline is the slow one. First start of the SGLang arm may include native warmup: check start 1 against 2-3.",
        caveats="First run of the SGLang-Omni Cosmos3 arm in this harness: if the server does not come up, read server.log (this is a result). The direct-generation pipeline, not the interleaved Reasoner pipeline. "
                "The M* config is the machine's one-GPU recipe (the kv512 overlay on rtx5880, the default on h200nvl); the SGLang stage is placed on GPU 0 of the cell's visible devices."),
    Experiment(
        name="cosmos3-size", title="Cosmos3-Nano T2I at 256 / 512 / 1,024: does the M* lead shrink as the GEMMs become compute-bound?",
        task="cosmos3", group="fair", kind="compare", frameworks=(MSTAR, "vllm-omni", "sglang-omni"), cases=("t2i-256", "t2i-512", "t2i-1024"), repeats=2, measured=3, preconditions=1,
        question="At 256x256 a denoise step is weight- and launch-bound (nsys: GPU time per forward equal, vLLM-Omni has 2x the kernels per request); the host-gap and batched-CFG terms of the 2.46x should "
                 "shrink when the per-step GEMMs grow with the pixel count. The doc says so; no larger size was ever run (only one case existed). How does the ratio move?",
        reading="Per-size ratio vLLM-Omni / M* and SGLang-Omni / M*, with the per-start range. The M* batched-CFG term (1.54x at 256) is a weight-reuse effect and should fall toward 1 as the GEMMs fill the "
                "GPU; a ratio that stays ~2x at 1,024 means the lead is not launch-bound. M*'s denoise CUDA graph is capped by graph_max_latent_area (2,000): the larger sizes run eager in M*.",
        caveats="Same 5 prompts at every size; 1,024x1,024 x 50 steps is slow (tens of seconds per request) so only 3 measured requests per start. A framework that runs out of memory at 1,024 is a result. "
                "Image size is verified from the PNG header at every size."),
    Experiment(
        name="cosmos3-sglang-nsys", title="Cosmos3-Nano 256x256: GPU kernel mix and idle of M* (default), vLLM-Omni and SGLang-Omni (nsys)",
        task="cosmos3", group="profile", kind="ab", frameworks=(MSTAR, "vllm-omni", "sglang-omni"), cases=("t2i-256",), repeats=1, measured=4, preconditions=2,
        question="cosmos3-nsys: against M* with sequential CFG / eager / no compile, vLLM-Omni is still 1.17x slower and the extra time is idle GPU between launches. Where is SGLang-Omni, and does it batch the "
                 "two CFG branches (one forward per step like M*, or two like vLLM)?",
        reading="analyze_nsys_burst_gaps.py, the last burst per process: GPU busy against span and the number of DiT forwards (attention launches / 36 layers) per request. ~50 forwards = batched CFG, ~100 = two "
                "sequential branches. Compare busy first (same work?), then span minus busy (launch gaps). nsys inflates host time; structure only.",
        arms=(Arm("mstar", frameworks=(MSTAR,), nsys=True, serve_args=("mstar:--enable-nvtx",), note="M* default recipe (batched CFG, graph cap, compile)"),
              Arm("vllm", frameworks=("vllm-omni",), nsys=True),
              Arm("sglang", frameworks=("sglang-omni",), nsys=True)),
        post=tuple("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:" + a + "}/nsys --min-launches 2000 --gap-ms 20 --last 4 --top 12 "
                   "--markdown {results}/cosmos3-sglang-nsys_" + a + "_{hardware}.md" for a in ("mstar", "vllm", "sglang")),
        caveats="The SGLang-Omni generation stage is a child process of the launcher; nsys follows children by default, if a report is empty read server.log first."),
    Experiment(
        name="cosmos3-sglang-resident", title="Cosmos3-Nano T2I, SGLang-Omni on the 48 GB card: DiT kept resident against the auto offload policy",
        task="cosmos3", group="fair", kind="ab", frameworks=("sglang-omni",), cases=("t2i-256", "t2i-512", "t2i-1024"), repeats=2, measured=3, preconditions=1,
        question="cosmos3-sglang-nsys: on the 5880 the SGLang-Omni run copies about 29 GB of DiT weights host-to-device and back on every request. The server ran the auto memory policy with dit_cpu_offload on "
                 "(47 GiB free, below the 90 GiB keep-resident threshold). Is the ~2.9 s per request that puts SGLang-Omni at 3.6x M* at 256 this copy, and does pinning the DiT resident bring it near the H200 result?",
        reading="Same protocol as cosmos3-size, same session: auto arm (the default config) against resident arm (`--generation.factory.server_args_overrides.dit_cpu_offload false`; `sgl-omni serve` has no bare `--dit-cpu-offload` flag). If the copy is the cost, the resident arm drops by about the same ~2.9 s at every size "
                "(a constant term, not a compute term), and the SGLang / M* ratio moves toward the H200 range (0.9-1.0). Read dit_cpu_offload in each start's server.log.",
        caveats="5880 only. The resident arm needs the 27 GB DiT plus activations inside 48 GB; a 1,024 failure is a result. This card throttles (sw_power_cap) during these runs, as in cosmos3-size.",
        arms=(Arm("auto", frameworks=("sglang-omni",)),
              Arm("resident", frameworks=("sglang-omni",),
                  serve_args=("sglang:--generation.factory.server_args_overrides.dit_cpu_offload false",),
                  expect=Expect(log_has=('"dit_cpu_offload": false',)))),
    ),
    Experiment(
        name="bagel-concurrency-fair", title="BAGEL I2T: B = 1 / 4 / 16 concurrent requests with the Thinker prefix cache ON in both M* and vLLM-Omni two-stage",
        task="bagel", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), cases=("i2t-512", "i2t-512-b4", "i2t-512-b16"), repeats=2,
        question="bagel-concurrency ran vLLM-Omni two-stage with its Thinker prefix cache OFF (the upstream default) and M* with its cache ON; bagel-sampling-parity showed that is the whole I2T TTFT gap at B = 1. "
                 "Does the batch scaling (wave time against B) look the same once the cache state is equal?",
        reading="analyze_concurrency.py per arm: wave wall time, requests/s and throughput scaling against B = 1. m-cache-on is M* (cache on by default); vllm-cache-on is two-stage with "
                "`enable_prefix_caching: true` on the Thinker. Compare the scaling curves, not only B = 1.",
        arms=(Arm("m-cache-on", frameworks=(MSTAR,)),
              Arm("vllm-cache-on", frameworks=("vllm-omni-two-stage",), opts=(f"config.vllm-omni={BAGEL_VLLM_PREFIX_CACHE}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_prefix_cache"),), log_has=("enable_prefix_caching=True",)))),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/bagel-concurrency-fair_" + a + "_{hardware}.md"
                   for a in ("m-cache-on", "vllm-cache-on")),
        caveats="The fixture repeats one prompt + image, so every request after the first is a hot prefix (best case for any cache). vLLM-Omni single-stage has no Thinker prefix cache option and is not in this run."),
    Experiment(
        name="bagel-concurrency-seqs", title="BAGEL I2T: B = 1 / 4 / 16 with vLLM-Omni's Thinker max_num_seqs 3 (upstream recipe) / 8 / 16, against M*",
        task="bagel", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), cases=("i2t-512", "i2t-512-b4", "i2t-512-b16"), repeats=2,
        question="bagel-concurrency-fair (both caches on): M* wave time 2,293 / 2,665 / 3,647 ms at B = 1 / 4 / 16, vLLM two-stage 2,258 / 4,545 / 13,647. The vLLM numbers are almost exactly 1x / 2x / 6x "
                 "the B = 1 time, and the upstream recipe sets the Thinker's `max_num_seqs: 3` (sized for three CFG branches). Is the 3.7x at B = 16 a scheduler cap rather than an engine property?",
        reading="analyze_concurrency.py per arm: wave wall time against B. Prediction if it is the cap: seqs3 = ceil(B/3) rounds (B=4: 2x, B=16: 6x the B=1 time), seqs8 = ceil(B/8) rounds (B=16: 2 rounds), "
                "seqs16 = one round, and the vLLM curve moves to or below M*'s. If seqs16 still sits far above M* at B = 16, the cap is not (only) the cause and the Thinker stage itself is slower per batched step; "
                "then read bagel-i2t-nsys style timelines at B = 16.",
        arms=(Arm("m-cache-on", frameworks=(MSTAR,)),
              Arm("vllm-seqs3", frameworks=("vllm-omni-two-stage",), opts=(f"config.vllm-omni={BAGEL_VLLM_PREFIX_CACHE}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_prefix_cache"),), log_has=("enable_prefix_caching=True",))),
              Arm("vllm-seqs8", frameworks=("vllm-omni-two-stage",), opts=(f"config.vllm-omni={BAGEL_VLLM_SEQS8}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_prefix_cache_seqs8"),), log_has=("enable_prefix_caching=True",))),
              Arm("vllm-seqs16", frameworks=("vllm-omni-two-stage",), opts=(f"config.vllm-omni={BAGEL_VLLM_SEQS16}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_prefix_cache_seqs16"),), log_has=("enable_prefix_caching=True",)))),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/bagel-concurrency-seqs_" + a + "_{hardware}.md"
                   for a in ("m-cache-on", "vllm-seqs3", "vllm-seqs8", "vllm-seqs16")),
        caveats="The fixture repeats one prompt + image, so every request after the first is a hot prefix. max_num_seqs also changes KV memory planning in vLLM; a start failure at 16 is a result (read server.log). "
                "This is a counterfactual deploy config, not the upstream recipe: the fair-profile comparison in the report keeps the upstream value and states the cap."),
    Experiment(
        name="t2s-concurrency-fair", title="Qwen3-Omni T2S: B = 1 / 4 / 16 concurrent requests with the shapes pre-warmed (no first-seen-shape JIT in the measured waves)",
        task="qwen_t2s", group="fair", kind="ab", frameworks=QWEN_FW, cases=("verbatim-en", "verbatim-en-b4", "verbatim-en-b16"),
        metric="ms_per_audio_s", repeats=1, measured=6, preconditions=10,
        question="t2s-concurrency ran with cold rows: qwen-t2s-fit showed vLLM-Omni pays inference-time JIT on every first-seen shape, which a wave of 16 different sentences hits many times. "
                 "With the rows cycled during preconditions (qwen.rows=cycle), what is the real scaling per audio second at B = 16?",
        reading="analyze_concurrency.py: per-audio-second time and requests/s against B. Compare with t2s-concurrency (cold rows): a framework whose B = 16 number falls a lot here was measuring JIT before. "
                "vLLM-Omni runs the max_num_seqs=16 variant of its recipe (a failed start is a result).",
        arms=(Arm("warm", opts=(WARM_ROWS,)),),
        post=("{py} exps/tools/analyze_concurrency.py --campaign {campaign:warm} --markdown {results}/t2s-concurrency-fair_{hardware}.md",),
        caveats="10 preconditions per start is the number of waves, not requests: at B = 16 that is 160 requests over the 20-sentence fixture, so every sentence is seen before measurement; at B = 1 only 10 of the 20 are."),
    Experiment(
        name="t2s-conc-compile-ab", title="Qwen3-Omni T2S, M* only: torch.compile on vs off at B = 1 / 4 / 16 in one session (warm rows)",
        task="qwen_t2s", group="fair", kind="ab", frameworks=(MSTAR,), cases=("verbatim-en", "verbatim-en-b4", "verbatim-en-b16"),
        metric="ms_per_audio_s", repeats=2, measured=6, preconditions=10,
        question="Compile on is the M* T2S default on the 5880. Its B = 16 gain was never measured in one session: the compile-off curve (24.6 ms per audio second at B = 16) came from the "
                 "earlier t2s-concurrency campaign and the compile-on curve (21.4) from t2s-concurrency-fair, so the difference mixes sessions. Does compile still help at B = 16 when both arms run "
                 "back to back in one session with the same warm rows?",
        reading="analyze_concurrency.py per arm: ms per audio second at B = 1, 4 and 16. Compile counts as a gain at a given B only if compile-off is slower than both compile-on arms (A1 and A2 of the "
                "pattern) by more than their spread. If the two compile-on arms disagree at a B by more than the compile gap, the session drifted and no gain is claimed at that B.",
        arms=(Arm("compile-on", opts=(WARM_ROWS, "qwen.mstar_compile=on"), expect=_moe(log_lacks=("OutOfResources",))),
              Arm("compile-off", opts=(WARM_ROWS, "qwen.mstar_compile=off"), expect=_moe(log_lacks=("OutOfResources",))),
              Arm("compile-on-2", opts=(WARM_ROWS, "qwen.mstar_compile=on"), expect=_moe(log_lacks=("OutOfResources",)))),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/t2s-conc-compile-ab_" + a + "_{hardware}.md"
                   for a in ("compile-on", "compile-off", "compile-on-2")),
        caveats="Compile-off is selected with `qwen.mstar_compile=off`, the same switch as qwen-mstar-compile-t2s. On rtx5880 it sets TORCHDYNAMO_DISABLE=1 for the M* service, and the manifest "
                "records the setting, so the NO_COMPILE env is not needed. Rows are pre-warmed as in t2s-concurrency-fair, so a recompile does not land in the measured waves."),
    Experiment(
        name="t2s-conc-b1-cap16", title="Qwen3-Omni T2S: vLLM-Omni at B = 1 on the B > 1 deploy (max_num_seqs 16) against its C = 1 deploy (max_num_seqs 1)",
        task="qwen_t2s", group="fair", kind="ab", frameworks=("vllm-omni",), cases=("verbatim-en",),
        metric="ms_per_audio_s", repeats=2, measured=6, preconditions=10,
        question="In t2s-concurrency-fair, vLLM-Omni's B = 1 point runs the C = 1 recipe (max_num_seqs 1) while B = 4 and B = 16 run the max_num_seqs 16 "
                 "variant, so the first point of the curve comes from a different deploy. Does the B = 1 number move when the same single request runs on the 16-cap overlay?",
        reading="Compare the vllm-omni B = 1 row here with the B = 1 row of t2s-concurrency-fair (same warm rows and counts, two starts here against one there). "
                "If the two agree within the start spread, the cap does not touch the B = 1 point and the curve is fine as it is.",
        arms=(Arm("cap16", opts=(WARM_ROWS, f"config.vllm-omni={QWEN_VLLM_BATCH}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_batch_async_c2wgraph"),))),),
        post=("{py} exps/tools/analyze_concurrency.py --campaign {campaign:cap16} --markdown {results}/t2s-conc-b1-cap16_{hardware}.md",),
        caveats="Only the vLLM-Omni arm is run: M* and SGLang-Omni already take the same B = 1 cell in t2s-concurrency-fair, and their configs do not change with B. "
                "The cap does not limit a single request, so a difference here would come from graph capture sizes or KV pool size, not from scheduling."),
    Experiment(
        name="t2s-conc-outliers", title="Qwen3-Omni T2S at B = 1 / 4 / 16: how often does a response fail the read-aloud contract (record, do not void)?",
        task="qwen_t2s", group="fair", kind="ab", frameworks=QWEN_FW, cases=("verbatim-en", "verbatim-en-b4", "verbatim-en-b16"),
        metric="ms_per_audio_s", repeats=1, measured=30, preconditions=10,
        question="t2s-concurrency-fair voided two B = 16 cells in the precondition phase: M* returned the right text but 20.5 s of audio for a 64-character sentence (wave 5, request 2), vLLM-Omni returned 1.42 s of audio "
                 "for a 43-character sentence (band floor 1.43 s, so a 0.7% miss: the 30 chars/s ceiling was too tight and is now 36). Is the M* long-audio response a rare random event or tied to concurrency, and is "
                 "anything similar happening in vLLM-Omni / SGLang-Omni?",
        reading="analyze_t2s_outliers.py: every response's chars per second against the band, per framework and B, with the fixture row of each outlier. Outliers at B = 16 only = concurrency-dependent "
                "(state shared between requests in a wave); outliers at B = 1 too = sampling or the Talker's stop condition; the same fixture row failing again = prompt-specific. The contract runs in record mode "
                "(`qwen.contract=record`): responses are saved and flagged, never void the cell.",
        arms=(Arm("record", opts=(WARM_ROWS, "qwen.contract=record")),),
        post=("{py} exps/tools/analyze_t2s_outliers.py --campaign {campaign:record} --markdown {results}/t2s-conc-outliers_{hardware}.md",),
        caveats="Durations are the only judge of 'not the target sentence' besides the returned-text similarity (M* returned the right text for the 20.5 s response, so duration is what caught it); nobody listened to the audio. "
                "30 measured waves + 10 precondition waves per cell: 640 responses at B = 16."),
    Experiment(
        name="t2s-conc-shuffle", title="Qwen3-Omni T2S at B = 1 / 4 / 16 with shuffled rows: is the ~20.5 s runaway tied to a sentence, a wave composition, or chance?",
        task="qwen_t2s", group="fair", kind="ab", frameworks=QWEN_FW, cases=("verbatim-en", "verbatim-en-b4", "verbatim-en-b16"),
        metric="ms_per_audio_s", repeats=1, measured=100, preconditions=10,
        question="t2s-conc-outliers: at B = 16, M* returned ~20.5 s of audio for fixture row seedtts_en_0590 and SGLang-Omni for seedtts_en_0870, each failing in both waves with the same composition "
                 "(rows cycle every 25 waves) and passing in the other 4 times the row was served; no framework failed at B = 1 / 4, but B = 1 only reached rows 0-39. Does a failure follow the sentence, "
                 "the batch neighbours, or neither (a per-response chance that grows with B)?",
        reading="analyze_t2s_outliers.py on a campaign where every pass over the 100 sentences is a fresh seeded permutation (qwen.rows=shuffle): the 'Outlier rows across batch sizes' table gives failed / served "
                "per row and B. The same row failing in different compositions = prompt-specific; failures on many different rows at B = 16 and none at B = 1 / 4 = a per-response chance that grows with concurrency; "
                "no failure at all = the earlier two were tied to that exact wave composition. B = 1 now serves all 100 rows (110 responses).",
        arms=(Arm("shuffle", opts=("qwen.rows=shuffle", "qwen.contract=record")),),
        post=("{py} exps/tools/analyze_t2s_outliers.py --campaign {campaign:shuffle} --markdown {results}/t2s-conc-shuffle_{hardware}.md",),
        caveats="One start per cell. Shuffle changes the warm-up property of cycle mode: the precondition phase (10 waves) at B = 1 / 4 does not cover all rows, so a first-seen-shape JIT in vLLM-Omni can add latency "
                "spikes to the timing; read the outlier table, not the p50, for this campaign."),
    Experiment(
        name="bagel-i2i-concurrency", title="BAGEL I2I (512x512, 5 steps): B = 1 / 4 / 16 against vLLM-Omni two-stage (Mooncake + Thinker cache) with Thinker max_num_seqs 3 and 16",
        task="bagel", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), repeats=2, measured=5, preconditions=1,
        question="bagel-concurrency-seqs: the I2T 3.7x lead at B = 16 was the upstream Thinker max_num_seqs=3; with it at 16 vLLM two-stage is 12% ahead of M*. Image editing has a second stage: every request "
                 "also hands three 5k-token KV branches to the DiT, whose stage runs max_num_seqs=1. Does the DiT stage serialise concurrent I2I requests in vLLM-Omni, and how does M*'s single engine scale?",
        reading="analyze_concurrency.py per arm: wave wall time against B. A serialised DiT shows as wave time ~ B x the B = 1 time in both vllm arms, whatever the Thinker cap; M* below that line = it overlaps "
                "or batches requests. If seqs16 sits well under seqs3 the Thinker cap matters for I2I too (the three CFG branches of one request already fill 3 slots).",
        arms=(Arm("m", frameworks=(MSTAR,), cases=("i2i-512-srv-steps5", "i2i-512-srv-steps5-b4", "i2i-512-srv-steps5-b16")),
              Arm("vllm-seqs3", frameworks=("vllm-omni-two-stage",), cases=("i2i-512-steps5", "i2i-512-steps5-b4", "i2i-512-steps5-b16"),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"), log_lacks=("Failed to create connector",))),
              Arm("vllm-seqs16", frameworks=("vllm-omni-two-stage",), cases=("i2i-512-steps5", "i2i-512-steps5-b4", "i2i-512-steps5-b16"),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_SEQS16}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"), log_lacks=("Failed to create connector",)))),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/bagel-i2i-concurrency_" + a + "_{hardware}.md"
                   for a in ("m", "vllm-seqs3", "vllm-seqs16")),
        caveats="The fixture repeats one prompt + image, so every request after the first is a hot prefix in both engines. Counterfactual deploy configs for seqs16, not the upstream recipe. A start failure at B = 16 "
                "(KV memory for 16 x 3 branches) is a result: read server.log. Server-side 5 steps for M*, request-level 5 steps for vLLM."),
    Experiment(
        name="bagel-i2i-kvtimeout", title="BAGEL I2I B = 4 / 16 on vLLM-Omni two-stage with the DiT's KV receive timeout at 900 s instead of 30 s",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-two-stage",), repeats=2, measured=5, preconditions=1,
        question="bagel-i2i-concurrency: at B = 16 vLLM two-stage needs 97 s (Thinker cap 3) / 226 s (cap 16) per wave against 46 s for M*, and server.log has 'Timeout waiting for KV cache ... after 30.0s' "
                 "(10 and 37 lines). Is the collapse the DiT stage's 30 s KV receive timeout firing while its queue is long (each timeout drops the KV and the request is retried or degraded), or is the DiT simply slow?",
        reading="analyze_concurrency.py per arm against the 30 s runs of bagel-i2i-concurrency. With the timeout at 900 s and no timeout line in server.log: wave time near B x the B = 1 time (about 56 s at B = 16) "
                "= the collapse was timeout handling; wave time unchanged = the DiT or the transfer is the bottleneck. Also compare the outputs' sha256 with the 30 s runs: a timed-out request may return a degraded image.",
        arms=(Arm("seqs3-t900", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_KVT900}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_kvtimeout900"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector", "after 30.0s"))),
              Arm("seqs16-t900", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_SEQS16_KVT900}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16_kvtimeout900"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector", "after 30.0s")))),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/bagel-i2i-kvtimeout_" + a + "_{hardware}.md" for a in ("seqs3-t900", "seqs16-t900")),
        caveats="Counterfactual config (recv_timeout is an undocumented knob of the stage's omni_kv_config; the KV roles stay with the pipeline topology, and vllm-02 merges the yaml's recv_timeout into it). The Expect lacks 'after 30.0s' "
                "to prove the setting took effect. One prompt + image repeated."),
    Experiment(
        name="bagel-i2i-pool-size", title="BAGEL I2I B = 4 / 16 on vLLM-Omni two-stage (Thinker cap 3): how small can the Mooncake sender pool be?",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-two-stage",), repeats=2, measured=5, preconditions=1,
        question="bagel-i2i-pool showed the 4 GiB sender pool exhausts at B = 16 (about 11 GB per wave) and 16 GiB removes the collapse. "
                 "Which sizes in between serve B = 16: 8 GiB, 12 GiB? Each request puts about 680 MB into the pool.",
        reading="Count 'Pool exhausted' lines per arm and read the wave time. The pool-8 and pool-12 arms are NOT asserted to be free of "
                "exhaustion: an exhausted arm is the answer, recorded, not a failed cell. A size that serves B = 16 with no exhaustion line and a "
                "wave near the 16 GiB arm is the smallest working pool. If 12 GiB still exhausts, the working size is between 12 and 16 GiB.",
        arms=(Arm("seqs3-t30-pool8", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_POOL8}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool8g"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector",))),
              Arm("seqs3-t30-pool12", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_POOL12}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool12g"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector",))),
              Arm("seqs3-t30-pool16", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_POOL16}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool16g"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector",)),
                  note="same-session reference for the two smaller pools")),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/bagel-i2i-pool-size_" + a + "_{hardware}.md"
                   for a in ("seqs3-t30-pool8", "seqs3-t30-pool12", "seqs3-t30-pool16")),
        caveats="Same prompt and image in every request (hot prefix), as bagel-i2i-pool. The pool-size question is about the sender side only; the 900 s receive timeout is not varied here."),
    Experiment(
        name="bagel-i2i-pool", title="BAGEL I2I B = 4 / 16 on vLLM-Omni two-stage with the Mooncake sender pool at 16 GiB: is the B = 16 collapse the pool or the receive timeout?",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-two-stage",), repeats=2, measured=5, preconditions=1,
        question="At B = 16 the vLLM-Omni two-stage I2I wave took 97 s (Thinker cap 3) and 226 s (cap 16) against 46 s for M*, with 'Timeout waiting for KV cache ... after 30.0s' in server.log. "
                 "The Mooncake sender pool is 4 GiB and each request puts about 680 MB into it (main KV plus the cfg_text branch), so a 16-request wave needs about 11 GB: "
                 "'Pool exhausted' appears in every B = 16 campaign, the 30 s ones and the 900 s ones alike. bagel-i2i-kvtimeout changed only the receive timeout, and its B = 16 cell failed on the pool. "
                 "Is the collapse the pool, the 30 s receive timeout, or a slow DiT?",
        reading="Decisive pair at B = 16 with the pool at 16 GiB: seqs3-t30-pool16 against seqs3-t900-pool16. If the 30 s arm finishes near the 900 s arm, both near B x the B = 1 wave time of bagel-i2i-concurrency (this group has no B = 1 cell), "
                "and neither logs 'Pool exhausted', the earlier collapse was the pool and the receive timeout is not implicated. If the 30 s arm still collapses (wave well above the 900 s arm) with no "
                "'Pool exhausted' line, the timeout is implicated. If both collapse alike, the DiT stage is the bottleneck. The seqs16 pair repeats the test at Thinker cap 16. Count the timeout lines per arm "
                "and compare output sha256 across arms, as in bagel-i2i-kvtimeout.",
        arms=(Arm("seqs3-t30-pool16", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_POOL16}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool16g"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector", "Pool exhausted"))),
              Arm("seqs3-t900-pool16", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_POOL16_KVT900}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_pool16g_kvtimeout900"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector", "Pool exhausted", "after 30.0s"))),
              Arm("seqs16-t30-pool16", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_SEQS16_POOL16}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16_pool16g"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector", "Pool exhausted"))),
              Arm("seqs16-t900-pool16", frameworks=("vllm-omni-two-stage",), cases=('i2i-512-steps5-b4', 'i2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_SEQS16_POOL16_KVT900}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_seqs16_pool16g_kvtimeout900"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector", "Pool exhausted", "after 30.0s")))),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/bagel-i2i-pool_" + a + "_{hardware}.md"
                   for a in ("seqs3-t30-pool16", "seqs3-t900-pool16", "seqs16-t30-pool16", "seqs16-t900-pool16")),
        caveats="Counterfactual configs (pool16g): the Mooncake sender pool is 16 GiB instead of 4 GiB. Besides that, the t900 arms differ from the t30 arms only in the DiT receive timeout, and the seqs16 arms only in Thinker max_num_seqs. "
                "The cap-16 arms take about 2 h because a 16-request wave takes about 226 s there; the cap-3 pair alone answers the question. The Expect forbids 'Pool exhausted' in every arm, so a cell whose "
                "wave exhausts the pool is rejected at analysis rather than counted."),
    Experiment(
        name="bagel-t2i-concurrency", title="BAGEL T2I (512x512, 5 steps): B = 1 / 4 / 16, M* against vLLM-Omni two-stage (Mooncake + Thinker cache), KV timeout 30 s and 900 s",
        task="bagel", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), repeats=2, measured=5, preconditions=1,
        question="Concurrency has been measured for I2T and I2I only. T2I has no input image, so the Thinker prefill is small and the three CFG branches' KV is smaller than for I2I; does the DiT stage still serialise "
                 "concurrent requests, and does vLLM two-stage collapse at B = 16 like I2I?",
        reading="analyze_concurrency.py per arm: wave time against B, throughput (req/s) flat = serialised. The t900 arm separates a timeout effect from DiT speed exactly as in bagel-i2i-kvtimeout.",
        arms=(Arm("m", frameworks=(MSTAR,), cases=("t2i-512-srv-steps5", "t2i-512-srv-steps5-b4", "t2i-512-srv-steps5-b16")),
              Arm("vllm-seqs3", frameworks=("vllm-omni-two-stage",), cases=('t2i-512-steps5', 't2i-512-steps5-b4', 't2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector",))),
              Arm("vllm-seqs3-t900", frameworks=("vllm-omni-two-stage",), cases=('t2i-512-steps5', 't2i-512-steps5-b4', 't2i-512-steps5-b16'),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX_KVT900}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_kvtimeout900"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"),
                                log_lacks=("Failed to create connector", "after 30.0s")))),
        post=tuple("{py} exps/tools/analyze_concurrency.py --campaign {campaign:" + a + "} --markdown {results}/bagel-t2i-concurrency_" + a + "_{hardware}.md" for a in ("m", "vllm-seqs3", "vllm-seqs3-t900")),
        caveats="Server-side 5 steps for M*, request-level 5 steps for vLLM. The 30 s arm has no timeout Expect on purpose: timeouts there are a result. T2I here is the default two-branch CFG request."),
    Experiment(
        name="t2s-conc-probe", title="Qwen3-Omni T2S: what in a B = 16 wave makes seedtts_en_0590 (M*) / seedtts_en_0139 (SGLang-Omni) run to ~20.5 s of audio?",
        task="qwen_t2s", group="fair", kind="ab", frameworks=("mstar", "sglang-omni"), cases=("verbatim-en-b4", "verbatim-en-b8", "verbatim-en-b12", "verbatim-en-b16"),
        metric="ms_per_audio_s", repeats=1, measured=40, preconditions=5,
        question="t2s-conc-shuffle: with rows shuffled the failures stayed on one sentence per framework (M* 0590: 13 of 17 servings at B = 16; SGLang 0139: 3 of 18), clean at B = 1 / 4 (4 servings each at B = 4). "
                 "So the sentence needs a large batch to fail. Is that the batch size itself, what the other 15 slots hold (long or short sentences, the same sentence), or the sentence's position in the wave?",
        reading="analyze_t2s_outliers.py per arm: the probe sentence is in every wave (40 measured + 5 precondition waves) at position wave % B, so each cell serves it 45 times and the failed / served column gives a rate per B. "
                "Rate rising with B under every fill = batch size; rate depends on fill (short or long neighbours, same sentence) = composition; rate depends on position (outlier table, position column) = slot. Neighbours are fixed per fill, "
                "so the 'random' fill is the shuffle-run condition at a known rate.",
        arms=(Arm("m-short", cases=("verbatim-en-b16",), frameworks=("mstar",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0590", "qwen.probe_fill=short", "qwen.contract=record")), Arm("m-long", cases=("verbatim-en-b16",), frameworks=("mstar",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0590", "qwen.probe_fill=long", "qwen.contract=record")), Arm("m-same", cases=("verbatim-en-b16",), frameworks=("mstar",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0590", "qwen.probe_fill=same", "qwen.contract=record")), Arm("m-random", frameworks=("mstar",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0590", "qwen.probe_fill=random", "qwen.contract=record")),
              Arm("sg-short", cases=("verbatim-en-b16",), frameworks=("sglang-omni",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0139", "qwen.probe_fill=short", "qwen.contract=record")), Arm("sg-long", cases=("verbatim-en-b16",), frameworks=("sglang-omni",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0139", "qwen.probe_fill=long", "qwen.contract=record")), Arm("sg-same", cases=("verbatim-en-b16",), frameworks=("sglang-omni",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0139", "qwen.probe_fill=same", "qwen.contract=record")), Arm("sg-random", frameworks=("sglang-omni",), opts=("qwen.rows=probe", "qwen.probe_row=seedtts_en_0139", "qwen.probe_fill=random", "qwen.contract=record"))),
        post=tuple("{py} exps/tools/analyze_t2s_outliers.py --campaign {campaign:" + a + "} --markdown {results}/t2s-conc-probe_" + a + "_{hardware}.md"
                   for a in ("m-short", "m-long", "m-same", "m-random", "sg-short", "sg-long", "sg-same", "sg-random")),
        caveats="Record-mode contract; one start per cell. The 'random' fill runs B = 4 / 8 / 12 / 16 (batch-size dependence), short / long / same only B = 16. B = 8 and 12 are new shapes for the warm-up. The probe sentence is not from the shuffle run's neighbour distribution, so a rate here is conditional on the fill, not the in-the-wild rate. "
                "'same' fills the wave with 16 copies of one sentence (identical prompts in a batch)."),
    Experiment(
        name="bagel-prefill-chunk", title="BAGEL on M*: the LLM's 1,024-token chunked-prefill budget against 8,192 (I2T TTFT and 5-step I2I)",
        task="bagel", group="fair", kind="ab", frameworks=(MSTAR,), cases=("i2t-512", "i2i-512-srv-steps5"), repeats=2,
        question="M* cuts BAGEL's ~5k-token branch prefill into 1,024-token chunks (nsys: 168 prefill-attention launches = 6 chunks x 28 layers) while vLLM-Omni prefills it in one "
                 "pass. With both caches on, M*'s I2T TTFT is 134 ms against vLLM's 85 ms and nothing explains the 44 ms; for Qwen the same budget turns out to be one whole "
                 "extra forward per chunk. A dense 7B chunk of 1,024 tokens is compute-bound, so the per-chunk cost should be far smaller than Qwen's MoE ~40 ms. How large is it here?",
        reading="TTFT of i2t-512 and E2E of i2i-512-srv-steps5 by arm (analyze_decode_itl.py). chunk-8192 TTFT well below chunk-1024 (towards vLLM's 85 ms) = part of D2 is M*'s "
                "chunk budget; no change = the 44 ms is elsewhere. Both arms have the prefix cache at its default, so a cached prompt may skip chunks: the cache hit and the chunk "
                "count interact, read the I2I E2E as well (it did not move with the cache switch).",
        arms=(Arm("chunk-1024"),
              Arm("chunk-8192", opts=(f"config.mstar={BAGEL_MSTAR_CHUNK8192}",), expect=Expect(command_has=(_vfile("mstar/bagel_chunk8192"),)))),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:chunk-1024} exps/runs/{campaign:chunk-8192} --markdown {results}/bagel-prefill-chunk_{hardware}.md",),
        caveats="One image size (512²). The budget of 8,192 is a counterfactual, not M*'s shipped config; it also lengthens the decode steps that share a step with a prefill chunk."),
    Experiment(
        name="mstar-tp-cache-probe", title="M* Qwen3-Omni TP=2 Thinker prefix cache with the whole prompt in one chunk (is the chunk boundary the crash trigger?)",
        task="qwen_txt", group="fair", kind="ab", frameworks=(MSTAR,), cases=("txt-1k-1tok",), repeats=1, measured=10, preconditions=3,
        min_gpus=2,          # a TP=2 Thinker: there is nothing to probe on one card (one rank opens the prefix cache by default)
        question="qwen3omni_3gpu_thinker_tp2_prefix_cache.yaml (prefix_cache_tp: true, offload 0, the documented opt-in) crashes on the first cache hit: `TP ranks diverged: KV "
                 "thinker_kv: rank 0 leased 8 cached pages for 0/main, whose stream here already holds tokens or has no keys` (RuntimeError from KVManager.adopt_cached_prefix on "
                 "the follower; the config passed M*'s own load-time checks). The 8 leased pages are 1,024 tokens, exactly the first chunk of the 1,072-token prompt. M*'s "
                 "unit tests (test_kv_prefix_tp.py, 14 pass here) use stubs, not the Qwen3-Omni path, and no shipped TP config turns prefix_cache_tp on.",
        reading="Read server.log (grep 'TP ranks diverged'). No crash and the measured requests complete = the chunked-row path under TP is the trigger (the prompt is one chunk "
                "here); the same crash = the chunk boundary is not it and the follower's chain / stored_len state is. Either way it is an M* defect in an opt-in path, "
                "to be reported with the log; this probe only narrows where. If it runs, E2E here against qwen-prefill-chunk's chunk-4096 is the prefix cache's TTFT effect.",
        arms=(Arm("tp-cache-chunk4096", opts=(ELIGIBLE, f"config.mstar={QWEN_MSTAR_PREFIX_CHUNK4096}"),
                  expect=_moe(command_has=(_vfile("mstar/qwen_prefix_cache_chunk4096"),), log_lacks=("prefix cache off at world size", "TP ranks diverged"))),),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:tp-cache-chunk4096} --markdown {results}/mstar-tp-cache-probe_{hardware}.md",),
        caveats="The analyze step reports nothing when the cell crashed again; the evidence is then the server.log line and the cell's requests.jsonl (503s)."),
    Experiment(
        name="qwen-a2t-cache-warm", title="Qwen3-Omni A2T, one output token: does vLLM's slow audio TTFT with the Thinker prefix cache on survive three warm passes?",
        task="qwen_a2t", group="fair", kind="ab", frameworks=("sglang-omni", "vllm-omni"), cases=("a2t-en-1tok",), repeats=2, measured=10, preconditions=30,
        question="qwen-fair-a2t: vLLM-Omni with the Thinker prefix cache on has TTFT 481 ms against 77 ms with it off (SGLang 119 on vs 88 off), although the measured "
                 "phase repeats clips the precondition phase has already served once. 3 of the 7 slow requests were fast (95-107 ms) on the first pass and 3 were ~500 ms "
                 "on both. With the cache on, the second pass prefills only the uncached tail, which is a new token count for every clip: Triton JIT of the "
                 "fused-MoE kernel for first-seen shapes would fit, a real cost of the cached path would not.",
        reading="Preconditions = 30 = three passes over the ten clips (`qwen.rows=cycle`), the measured phase is the fourth. vllm-on at ~77-90 ms = the 481 ms was shapes "
                "first seen on the cached path (JIT), and the earlier arm just under-warmed; vllm-on still ~480 ms = a real cost of the cached audio path. "
                "Count the `Triton kernel JIT compilation during inference` lines in each server.log by phase. sglang-on vs sglang-off 30 ms: same reading.",
        arms=(Arm("vllm-off", frameworks=("vllm-omni",), opts=(ELIGIBLE, WARM_ROWS), expect=_moe(log_lacks=("enable_prefix_caching=True",))),
              Arm("vllm-on", frameworks=("vllm-omni",), opts=(ELIGIBLE, WARM_ROWS, f"config.vllm-omni={QWEN_VLLM_PREFIX}"),
                  expect=_moe(command_has=(_vfile("vllm_omni/qwen_prefix_cache"),), log_has=("enable_prefix_caching=True",))),
              Arm("sglang-off", frameworks=("sglang-omni",), opts=(ELIGIBLE, WARM_ROWS, f"config.sglang={QWEN_SGLANG_NO_RADIX}"),
                  expect=_moe(command_has=(_vfile("sglang_omni/qwen_no_radix"),), log_lacks=(SGLANG_NO_PREFIX_HIT,))),
              Arm("sglang-on", frameworks=("sglang-omni",), opts=(ELIGIBLE, WARM_ROWS), expect=_moe(log_has=(SGLANG_PREFIX_HIT,)))),
        post=("{py} exps/tools/analyze_decode_itl.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("vllm-off", "vllm-on", "sglang-off", "sglang-on"))
              + " --markdown {results}/qwen-a2t-cache-warm_{hardware}.md",),
        caveats="M* is not in this experiment (its cache is inert on audio prompts of this length: 252 vs 255 ms). One output token, so E2E is the first token."),
    Experiment(
        name="bagel-connector", title="BAGEL I2I two-stage: the Thinker -> DiT KV hand-off over SharedMemory vs Mooncake (TCP) vs NIXL (UCX)",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-two-stage", "vllm-omni-single-stage"),
        cases=("i2i-512-steps5", "i2i-512"), repeats=2,
        question="The two-stage I2I is 4.4 s slower than M* because the Thinker's KV (2 x 340 MB) is serialised to bytes (~0.47 s per branch), copied "
                 "through shared memory at ~0.9 GB/s and copied back, with ~3.7 s of idle GPU between the stages. vLLM-Omni also ships connectors that "
                 "carry the GPU tensor itself (Mooncake Transfer Engine, NIXL). Is the hand-off an architectural cost of two stages, or of the default "
                 "connector? Installed in a user directory by exps/tools/install_omni_connectors.sh, not in the baseline environment.",
        reading="analyze_kv_handoff.py: sender-side serialise + transfer per request, and E2E, per connector. `steps5` exposes the fixed cost "
                "(E2E 6.9 s on shm, 2.9 s single-stage); `i2i-512` shows it against the 24-28 s total. mooncake / nixl E2E near single-stage = the "
                "gap was the default connector, and 'M* has no stage boundary' is not an architectural advantage; E2E near shm = the gap is "
                "elsewhere (receiver side, orchestration, the Thinker's three separate prefills) and the connector is not the lever. A connector "
                "that fails to start is a result (read server.log): the box has no RDMA NIC and no NVLink, so Mooncake runs its TCP transport and "
                "UCX picks host-staged or PCIe paths between the two cards.",
        arms=(Arm("shm", frameworks=("vllm-omni-two-stage",), expect=Expect(log_lacks=("Created connector: Mooncake", "Created connector: Nixl"))),
              Arm("mooncake-tcp", frameworks=("vllm-omni-two-stage",),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector",), log_lacks=("Failed to create connector",))),
              Arm("nixl", frameworks=("vllm-omni-two-stage",),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_NIXL}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_nixl"), CONNECTOR_SITE),
                                log_has=("Created connector: NixlConnector",), log_lacks=("Failed to create connector",))),
              Arm("single-stage", frameworks=("vllm-omni-single-stage",), note="no hand-off at all: the floor for the connector arms")),
        post=("{py} exps/tools/analyze_kv_handoff.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("shm", "mooncake-tcp", "nixl", "single-stage"))
              + " --markdown {results}/bagel-connector_{hardware}.md",),
        caveats="One pair of cards on PCIe without NVLink; the harness picks the lowest free pair, which may sit on different NUMA nodes. Mooncake uses "
                "its TCP transport (no RDMA device). The hand-off cost seen here is the sender side plus whatever the client sees; a faster connector "
                "does not change the Thinker's three separate prefills."),
    Experiment(
        name="bagel-connector-timeline", title="BAGEL I2I two-stage, 5 steps: millisecond timeline of one request per connector (and two UCX transports for NIXL)",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-two-stage",), cases=("i2i-512-steps5",), repeats=2,
        question="bagel-connector showed Mooncake-TCP saves 2.25 s of the 3.9 s two-stage gap to single-stage and NIXL 1.5 s; 1.6 s (Mooncake) and 2.4 s "
                 "(NIXL) are still unexplained, and the log stamps there have one-second resolution. Where does a request's time go "
                 "(client -> engine, Thinker, send, orchestrator gap, DiT side), and does NIXL's UCX pick a slow path (host-staged TCP) where cuda_ipc exists? "
                 "nvidia-smi topo -p2p r reports P2P OK between every pair of cards.",
        reading="analyze_i2i_timeline.py: the segment that differs between connectors is the cost of the connector; the segment that is the same across all "
                "arms (stage 0, pre) is not. nixl-cuda-ipc faster than nixl means UCX's default transport choice, not NIXL, was the limit; equal to "
                "nixl means the default already uses it. `DiT side` includes the receiver's KV load, so a difference there is the receiver cost. "
                "Neither connector is expected to remove stage 0 (three separate Thinker prefills), which stays in every two-stage arm.",
        arms=(Arm("shm", env=(MS_LOG,), expect=Expect(command_has=("ms_timestamps.json",), log_lacks=("Created connector: Mooncake", "Created connector: Nixl"))),
              Arm("mooncake-tcp", env=(MS_LOG, f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}"), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP}",),
                  expect=Expect(command_has=("ms_timestamps.json", _vfile("vllm_omni/bagel_two_stage_mooncake_tcp")),
                                log_has=("Created connector: MooncakeTransferEngineConnector",), log_lacks=("Failed to create connector",))),
              Arm("nixl", env=(MS_LOG, f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}"), opts=(f"config.vllm-omni={BAGEL_NIXL}",),
                  expect=Expect(command_has=("ms_timestamps.json", _vfile("vllm_omni/bagel_two_stage_nixl")),
                                log_has=("Created connector: NixlConnector",), log_lacks=("Failed to create connector",))),
              Arm("nixl-cuda-ipc", env=(MS_LOG, f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}", "vllm-omni:UCX_TLS=cuda_copy,cuda_ipc,tcp"),
                  opts=(f"config.vllm-omni={BAGEL_NIXL}",),
                  expect=Expect(command_has=("UCX_TLS=cuda_copy,cuda_ipc,tcp", _vfile("vllm_omni/bagel_two_stage_nixl")),
                                log_has=("Created connector: NixlConnector",), log_lacks=("Failed to create connector",))),
              Arm("nixl-tcp", env=(MS_LOG, f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}", "vllm-omni:UCX_TLS=cuda_copy,tcp"),
                  opts=(f"config.vllm-omni={BAGEL_NIXL}",),
                  expect=Expect(command_has=("UCX_TLS=cuda_copy,tcp", _vfile("vllm_omni/bagel_two_stage_nixl")),
                                log_has=("Created connector: NixlConnector",), log_lacks=("Failed to create connector",))),
              Arm("mooncake-tcp-prefix", env=(MS_LOG, f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}"), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX}",),
                  note="Mooncake + Thinker prefix cache on: the arm that is 763 ms behind M* in bagel-i2i-fair; its stage 0 should drop by ~1 s",
                  expect=Expect(command_has=("ms_timestamps.json", _vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache")),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"), log_lacks=("Failed to create connector",)))),
        post=("{py} exps/tools/analyze_i2i_timeline.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("shm", "mooncake-tcp", "nixl", "nixl-cuda-ipc", "nixl-tcp", "mooncake-tcp-prefix"))
              + " --markdown {results}/bagel-connector-timeline_{hardware}.md",),
        caveats="Single-stage has no KV markers in its log and is read from bagel-connector. UCX_TLS names the transports UCX may use; an unavailable one is dropped "
                "silently, so a nixl-cuda-ipc that equals nixl is a result, not a failed switch (the arm's server.log would show UCX's own transport lines at higher log level)."),
    Experiment(
        name="bagel-connector-1024", title="BAGEL I2I two-stage at 1024^2, 5 steps: does the connector saving grow with the KV size?",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-two-stage", "vllm-omni-single-stage"), cases=("i2i-1024-steps5",), repeats=2,
        question="At 512^2 Mooncake-TCP saves 2.25 s per request and the earlier 1024^2 gap to M* was 1.36x against 1.18x at 512^2. The KV is ~4x larger at 1024^2 "
                 "(about 1.3 GB per branch, if it scales with tokens). Does the saving scale with the KV, and is the 1.36x also mostly the default connector?",
        reading="E2E per arm. saving(1024) / saving(512) near 4 = the hand-off cost scales with bytes and the connector is the lever at every size; near 1 = a fixed "
                "orchestration cost the connector does not touch. The Mooncake arm must not log a pool / put failure (4 GiB pinned pool vs ~2.7 GB of in-flight KV): "
                "read the server.log before trusting it. vLLM-Omni only: M* ignores request-level steps, compare against the 50-step M* number from bagel-i2i-steps.",
        arms=(Arm("shm", frameworks=("vllm-omni-two-stage",), expect=Expect(log_lacks=("Created connector: Mooncake", "Created connector: Nixl"))),
              Arm("mooncake-tcp", frameworks=("vllm-omni-two-stage",), env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",),
                  opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp"),), log_has=("Created connector: MooncakeTransferEngineConnector",),
                                log_lacks=("Failed to create connector",))),
              Arm("nixl", frameworks=("vllm-omni-two-stage",), env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_NIXL}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_nixl"),), log_has=("Created connector: NixlConnector",), log_lacks=("Failed to create connector",))),
              Arm("single-stage", frameworks=("vllm-omni-single-stage",), note="no hand-off: the floor")),
        post=("{py} exps/tools/analyze_kv_handoff.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("shm", "mooncake-tcp", "nixl", "single-stage"))
              + " --markdown {results}/bagel-connector-1024_{hardware}.md",),
        caveats="Single request at a time on one pair of cards; one fixture. 1024^2 is explicit-only elsewhere because a long request crosses the thermal knee: with 5 steps a request is ~10-20 s."),
    Experiment(
        name="bagel-connector-nsys", title="BAGEL I2I, 5 steps: Thinker / hand-off / DiT on the GPU timeline (nsys): single-stage, two-stage on SharedMemory / Mooncake / Mooncake + prefix cache, and M*",
        task="bagel", group="profile", kind="ab", frameworks=(MSTAR, *VLLM_BAGEL), repeats=1, measured=4, preconditions=1, cases=("i2i-512-steps5",),
        question="With Mooncake the hand-off itself is ~13 ms on the sender, yet two-stage is still ~1.6 s slower than single-stage at 5 steps, and "
                 "server.log has no receiver-side line (the DiT process logs nothing). The first run (2 measured requests) showed per request: DiT 1.83 s of GPU "
                 "work identical in every arm, the Thinker process 2.05-2.1 s busy (15.3k launches: two 5k-token branch prefills) against ~0.85 s (6.4k launches) "
                 "for everything before the DiT in single-stage, and a Thinker -> DiT hand-off of ~0.28 s on Mooncake. bagel-i2i-fair then showed the Thinker prefix "
                 "cache takes 1.04 s off the vLLM two-stage request and M*'s cache switch changes nothing. Does the Thinker burst of the prefix-cache arm drop to "
                 "single-stage's, how many Thinker launches does M* make for the same request, and where is the remaining ~0.76 s?",
        reading="analyze_nsys_burst_gaps.py --handoff, one table per two-stage arm (Thinker first..last, GPU busy, launches, Thinker last -> DiT first split into the "
                "KV D2H -> first H2D and first H2D -> first DiT kernel, DiT span) plus the per-process burst table; single-stage and M* run in one process, so their "
                "last burst is Thinker prefill + DiT together (single-stage: 16,605 launches, 2,627 ms busy, of which the DiT is 10,239 / ~1,775). "
                "mooncake-prefix Thinker busy near single-stage's ~0.85 s = the second branch's prefill is gone and the rest is hand-off + front end; still ~2 s = "
                "the prefix cache does not cover what the two branches share. M* launches near 16.6k = one prefill like single-stage; near 25k = two prefills "
                "(then its cache switch is inert because the work is real, not cached). nsys inflates host time: structure only. The Thinker's trace has been "
                "seen to end 3-12 s before the DiT's (the tool prints a warning): read the rows followed by a complete one, which is why four requests are measured.",
        arms=(Arm("single-stage", frameworks=("vllm-omni-single-stage",), nsys=True),
              Arm("shm", frameworks=("vllm-omni-two-stage",), nsys=True),
              Arm("mooncake-tcp", frameworks=("vllm-omni-two-stage",), nsys=True,
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector",), log_lacks=("Failed to create connector",))),
              Arm("mooncake-prefix", frameworks=("vllm-omni-two-stage",), nsys=True,
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"), log_lacks=("Failed to create connector",))),
              Arm("mstar", frameworks=(MSTAR,), cases=("i2i-512-srv-steps5",), nsys=True, serve_args=("mstar:--enable-nvtx",),
                  note="M* takes the 5 steps from the server (request-level steps are inert), cache at its default; NVTX on so the Thinker / DiT split is readable")),
        post=tuple("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:" + a + "}/nsys --min-launches 1000 --gap-ms 40 --last 5 --top 10"
                   + (" --handoff" if a in ("shm", "mooncake-tcp", "mooncake-prefix") else "") + " --markdown {results}/bagel-connector-nsys_" + a + "_{hardware}.md"
                   for a in ("single-stage", "shm", "mooncake-tcp", "mooncake-prefix", "mstar"))),
    Experiment(
        name="bagel-i2i-fair", title="BAGEL I2I with every known difference removed: Mooncake connector + Thinker prefix cache on, against M* with its cache on and off",
        task="bagel", group="fair", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), cases=("i2i-512-srv-steps5", "i2i-512-steps5", "i2i-512"), repeats=2,
        question="vLLM-Omni two-stage I2I still has a ~1.6 s fixed gap to M* after the connector is swapped, and its stage 0 alone takes ~2.2 s before the first KV "
                 "branch is out. The Thinker's prefill of the two 5k-token branches is ~0.74 s each (I2T nsys) and every vLLM-Omni deploy config has prefix caching "
                 "off while M* opens its cache by default. Whether M* actually hits its cache on an I2I request is not logged, and the 50-step cache on/off pair "
                 "(23,528 / 23,583 ms) cannot tell. With the Thinker cache on in vLLM and M*'s cache switched on and off at 5 steps, how much of the fixed gap is prefill?",
        reading="Per case, E2E by arm (analyze_decode_itl.py). M* cases: `i2i-512-srv-steps5` (M* takes the step count from the server, request-level steps are inert) and "
                "`i2i-512` (50 steps); vLLM cases `i2i-512-steps5` and `i2i-512`. m-cache-off vs m-cache-on at 5 steps = the prefill M* skips (or not). "
                "vllm-mooncake vs vllm-mooncake-prefix = the prefill vLLM skips when its Thinker caches the repeated input. Equal-state pairs: m-cache-on vs vllm-mooncake-prefix "
                "and m-cache-off vs vllm-mooncake. If the 5-step M* pair differs by ~1.5 s, the two-stage gap in every earlier I2I number included M*'s cache hit, and the "
                "I2I row of the fairness audit changes from `no difference` to `M*`.",
        arms=(Arm("m-cache-on", frameworks=(MSTAR,), cases=("i2i-512-srv-steps5", "i2i-512")),
              Arm("m-cache-off", frameworks=(MSTAR,), cases=("i2i-512-srv-steps5", "i2i-512"), opts=("bagel.prefix_cache=off",),
                  note="every M* request carries prefix_cache=false"),
              Arm("vllm-mooncake", frameworks=("vllm-omni-two-stage",), cases=("i2i-512-steps5", "i2i-512"),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",), opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector",), log_lacks=("Failed to create connector",))),
              Arm("vllm-mooncake-prefix", frameworks=("vllm-omni-two-stage",), cases=("i2i-512-steps5", "i2i-512"),
                  env=(f"vllm-omni:PYTHONPATH={CONNECTOR_SITE}",),
                  opts=(f"config.vllm-omni={BAGEL_MOONCAKE_TCP_PREFIX}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache"), CONNECTOR_SITE),
                                log_has=("Created connector: MooncakeTransferEngineConnector", "enable_prefix_caching=True"), log_lacks=("Failed to create connector",)))),
        post=("{py} exps/tools/analyze_decode_itl.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("m-cache-on", "m-cache-off", "vllm-mooncake", "vllm-mooncake-prefix"))
              + " --markdown {results}/bagel-i2i-fair_{hardware}.md",),
        caveats="One fixture repeated (a hot prefix, best case for every cache). The 50-step cells cross the thermal knee on this box; the 5-step pair is the clean one."),
    *(Experiment(
        name=f"qwen-mstar-compile-{tag}", title=f"M* on Qwen3-Omni {tag.upper()}: torch.compile off vs on (rtx5880: off for TXT / A2T, on for T2S since 2026-11-13)",
        task=task, group="fair", kind="ab", frameworks=(MSTAR,), cases=cases, repeats=2, **extra,
        question="M* runs Qwen with TORCHDYNAMO_DISABLE=1 on the 5880 because the first smoke died with a Talker shared-memory OutOfResources. That kernel is "
                 "M*'s own _cp_linear_kernel and SMOKE_5880_03 (with the flag already set) failed identically, so the flag was not the fix; the pinned "
                 "M* now falls back to fewer stages / a shorter K step. Does M* compile on sm_89 now, what does it cost to start, and what is it worth?",
        reading="A1 / compile-on / A2: ratio beyond the A1-A2 spread. compile-on must have no OutOfResources / Traceback in server.log (the arm refuses "
                "to report otherwise). If it runs and is faster, M*'s Qwen numbers here understate it and the fair-profile arms should use "
                "`qwen.mstar_compile=on` (EXPS_QWEN_MSTAR_COMPILE=on); if it is slower or fails, the stale-workaround story is wrong and the "
                "failure is the thing to report. The arms pin the setting explicitly, because the per-task default is now on for T2S.",
        arms=(Arm("A1", opts=(("qwen.mstar_compile=off",) + ((WARM_ROWS,) if tag != "txt" else ())), expect=_moe(log_lacks=("OutOfResources",))),
              Arm("compile-on", opts=(("qwen.mstar_compile=on",) + ((WARM_ROWS,) if tag != "txt" else ())),
                  expect=_moe(log_lacks=("OutOfResources",))),
              Arm("A2", opts=(("qwen.mstar_compile=off",) + ((WARM_ROWS,) if tag != "txt" else ())), expect=_moe(log_lacks=("OutOfResources",)))),
        post=("{py} exps/tools/" + ("analyze_t2s_fit.py" if tag == "t2s" else "analyze_decode_itl.py") + " "
              + " ".join(f"exps/runs/{{campaign:{a}}}" for a in ("A1", "compile-on", "A2"))
              + f" --markdown {{results}}/qwen-mstar-compile-{tag}_{{hardware}}.md",),
        caveats="Compile changes the first-seen-shape cost: the A2T / T2S arms use pre-warmed rows so a recompile does not land in the measured phase.")
      for tag, task, cases, extra in (("txt", "qwen_txt", ("txt-1k",), {"measured": 5}),
                                      ("a2t", "qwen_a2t", ("a2t-en",), {"measured": 10, "preconditions": 10}),
                                      ("t2s", "qwen_t2s", ("verbatim-en",), {"measured": 10, "preconditions": 10}))),
    Experiment(
        name="bagel-sampling-parity", title="BAGEL I2T, fair profile: neutral sampler + prefix cache on in both, vLLM's yaml repetition penalty removed",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-two-stage", MSTAR), cases=("i2t-512",), repeats=2,
        question="vLLM-Omni's BAGEL deploy yaml gives the Thinker repetition_penalty 1.05 / top_p 0.9 as server defaults and the request only "
                 "sets temperature, so vLLM decodes with a repetition penalty M* does not apply (the two engines' captions differ for the same "
                 "image). Does the penalty cost vLLM per-token time, and what is the fully fair I2T (neutral sampler, cache on in both)?",
        reading="ITL per arm. vllm-neutral below vllm-default: the penalty was a cost vLLM paid alone. vllm-fair (neutral + Thinker cache on) against "
                "m-fair (neutral + M* default cache) is the I2T comparison with nothing left unequal that we know of.",
        arms=(Arm("vllm-default", frameworks=("vllm-omni-two-stage",)),
              Arm("vllm-neutral", frameworks=("vllm-omni-two-stage",), opts=("bagel.sampling=neutral",)),
              Arm("vllm-fair", frameworks=("vllm-omni-two-stage",), opts=("bagel.sampling=neutral", f"config.vllm-omni={BAGEL_VLLM_PREFIX_CACHE}"),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_prefix_cache"),))),
              Arm("m-default", frameworks=(MSTAR,)),
              Arm("m-fair", frameworks=(MSTAR,), opts=("bagel.sampling=neutral",))),
        post=("{py} exps/tools/analyze_decode_itl.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in (
              "vllm-default", "vllm-neutral", "vllm-fair", "m-default", "m-fair")) + " --markdown {results}/bagel-sampling-parity_{hardware}.md",),
        caveats="Captions are not byte-identical across engines even with the same sampler (different kernels, near-tied tokens); lengths are pinned."),
    Experiment(
        name="bagel-i2t-single-fair", title="BAGEL I2T, fair profile: vLLM-Omni single-stage against M* with its prefix cache on and off",
        task="bagel", group="fair", kind="ab", frameworks=("vllm-omni-single-stage", MSTAR), cases=("i2t-512",), repeats=2,
        question="The fair I2T comparison so far is two-stage only (vllm-fair against m-fair). Single-stage runs understanding inside the DiT engine: "
                 "its KV is a NaiveCache, there is no Thinker prefix cache to switch on, and text is decoded greedily. What are its E2E, TTFT and ITL "
                 "under the same protocol, and how does it compare with M* once M*'s prefix cache is off too, the only cache state single-stage can have?",
        reading="E2E first, TTFT and ITL as extra (analyze_decode_itl.py). single-stage against m-cache-off is the cache-equal pair: neither side has a prefix "
                "cache. single-stage against m-fair (cache on) is reference only, because one side has a cache the other cannot have. The sampler needs no "
                "switch on the single-stage side: its text decode is greedy unless do_sample is set (default false), which matches M*'s neutral arm (top_k 1).",
        arms=(Arm("single-stage", frameworks=("vllm-omni-single-stage",),
                  expect=Expect(command_has=("bagel_single_stage.yaml",))),
              Arm("m-fair", frameworks=(MSTAR,), opts=("bagel.sampling=neutral",)),
              Arm("m-cache-off", frameworks=(MSTAR,), opts=("bagel.sampling=neutral", "bagel.prefix_cache=off"),
                  note="M* with prefix_cache=false per request: the cache-equal pair for single-stage")),
        post=("{py} exps/tools/analyze_decode_itl.py " + " ".join(f"exps/runs/{{campaign:{a}}}" for a in (
              "single-stage", "m-fair", "m-cache-off")) + " --markdown {results}/bagel-i2t-single-fair_{hardware}.md",),
        caveats="One repeated prompt + image, so every request after the first is a hot prefix for M*'s cache-on arm. Lengths are pinned (128 tokens, "
                "ignore_eos). The single-stage deploy has no prefix cache option (enable_prefix_caching false in bagel_single_stage.yaml); its gap to "
                "two-stage is the whole decode path, not only the cache. Read TTFT and ITL separately from E2E."),
    Experiment(
        name="bagel-async-sched", title="BAGEL I2T: vLLM-Omni two-stage Thinker with async scheduling off",
        task="bagel", group="advantage", kind="ab", frameworks=("vllm-omni-two-stage", MSTAR), cases=("i2t-512",), repeats=2,
        question="BAGEL I2T decodes at 16.9 ms/token on M* and 17.1 on vLLM-Omni two-stage; M* with its launch chain off takes "
                 "19.5. Does vLLM lose the same ~2.5 ms/token when its async scheduling is off?",
        reading="ITL per arm: vllm-async-off near 19.5-20 means vLLM already has the same overlap and the per-token parity is "
                "explained; unchanged means vLLM's per-token cost has another reason.",
        arms=(Arm("vllm-async-on", frameworks=("vllm-omni-two-stage",)),
              Arm("vllm-async-off", frameworks=("vllm-omni-two-stage",), opts=(f"config.vllm-omni={BAGEL_SYNC_SCHED}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_thinker_sync_sched"),)),
                  note="Thinker stage on the non-async scheduler: a counterfactual, not a baseline"),
              Arm("mstar-spec-on", frameworks=(MSTAR,)),
              Arm("mstar-spec-off", frameworks=(MSTAR,), env=SPEC_OFF, expect=_cmd_has(*SPEC_OFF))),
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:vllm-async-on} exps/runs/{campaign:vllm-async-off} "
              "exps/runs/{campaign:mstar-spec-on} exps/runs/{campaign:mstar-spec-off} --markdown {results}/bagel-async-sched_{hardware}.md",)),
    Experiment(
        name="bagel-prefix-cache", title="BAGEL I2T / I2I: M*'s KV prefix cache on vs off, against vLLM-Omni (prefix caching off)",
        task="bagel", group="advantage", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), cases=("i2t-512", "i2i-512"),
        repeats=2,
        question="Every request of a cell sends the same prompt and the same image. M* opens a KV prefix cache by default; every "
                 "vLLM-Omni deploy config sets enable_prefix_caching: false. nsys shows M*'s I2T LLM prefill at ~28 ms of GPU work "
                 "against ~740 ms for vLLM-Omni's 4,935-token prefill, and the I2T TTFT gap (130 vs 808 ms) is exactly that. Is the gap "
                 "a prefix-cache hit on a repeated input, and how much of the I2I gap is too?",
        reading="TTFT and E2E per arm (analyze_decode_itl.py). m-cache-off TTFT near vllm's (~800 ms for I2T) means the TTFT lead was "
                "the cache, not the engine: the equal-work comparison is m-cache-off against vllm. Unchanged TTFT means the cache is "
                "not the reason and the profile above is read wrong.",
        arms=(Arm("m-cache-on", frameworks=(MSTAR,)),
              Arm("m-cache-off", frameworks=(MSTAR,), opts=("bagel.prefix_cache=off",),
                  note="every M* request carries prefix_cache=false: a counterfactual that makes M* recompute like the baseline"),
              Arm("vllm", frameworks=("vllm-omni-two-stage",))),
        caveats="Not a claim that caching is unfair in general (a served system keeps it), but the baseline configs forbid it and the "
                "fixture repeats one input, so the cell measures the cache. A bagel.prefix_cache=off arm is recorded in the manifest's "
                "overrides; the proof it took effect is the TTFT change itself.",
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:m-cache-on} exps/runs/{campaign:m-cache-off} "
              "exps/runs/{campaign:vllm} --markdown {results}/bagel-prefix-cache_{hardware}.md",)),
    Experiment(
        name="bagel-vllm-prefix-cache", title="BAGEL: does vLLM-Omni get the same TTFT when its Thinker prefix cache is on?",
        task="bagel", group="advantage", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), cases=("i2t-512", "i2i-512"),
        repeats=2,
        question="bagel-prefix-cache showed M*'s I2T lead (TTFT 138 vs 805 ms) is its KV prefix cache hitting a repeated image: "
                 "with the cache off M* takes 777 ms. Is that an M* feature vLLM-Omni cannot match, or a switch the baseline left off?",
        reading="TTFT per arm. vllm-cache-on near m-cache-on (~140 ms): the lead is a deploy flag, not the engine, and the fair "
                "comparison is cache-on against cache-on or off against off. TTFT unchanged (or the server fails; read server.log): "
                "vLLM-Omni's BAGEL path cannot reuse the multimodal prefix, so the cache is a real M* capability, only one the "
                "fixture (one repeated image) overstates. I2I: any change here is the Thinker context being reused.",
        arms=(Arm("vllm-cache-off", frameworks=("vllm-omni-two-stage",)),
              Arm("vllm-cache-on", frameworks=("vllm-omni-two-stage",), opts=(f"config.vllm-omni={BAGEL_VLLM_PREFIX_CACHE}",),
                  expect=Expect(command_has=(_vfile("vllm_omni/bagel_two_stage_prefix_cache"),)),
                  note="Thinker stage with enable_prefix_caching: true: a counterfactual, not a baseline"),
              Arm("m-cache-on", frameworks=(MSTAR,))),
        caveats="If vLLM-Omni's BAGEL Thinker does not support prefix caching with image inputs the cell may fail or serve stale "
                "context: check the output text against vllm-cache-off before trusting a faster TTFT.",
        post=("{py} exps/tools/analyze_decode_itl.py exps/runs/{campaign:vllm-cache-off} exps/runs/{campaign:vllm-cache-on} "
              "exps/runs/{campaign:m-cache-on} --markdown {results}/bagel-vllm-prefix-cache_{hardware}.md",)),
    Experiment(
        name="bagel-i2i-steps", title="BAGEL I2I: fixed and per-step cost, three deployments",
        task="bagel", group="advantage", frameworks=BAGEL_FW, kind="ab",
        cases=("i2i-512", "i2i-512-steps5", "i2i-512-steps25", "i2i-512-srv-steps5", "i2i-512-srv-steps25"), repeats=2,
        question="I2I takes 23.8 s on M*, 24.2 s on vLLM-Omni single-stage and 28.2 s on two-stage (T2I is within 3% on all three). "
                 "Is the two-stage 4.4 s a per-request fixed cost (KV hand-off from the Thinker over the shared-memory connector, "
                 "the source-image prefill) or a per-denoise-step cost?",
        reading="Fit E2E = a + b*steps per framework (the I2I table of the fit). Gap in `a`: hand-off / prefill, once per request. "
                "Gap in `b`: every step is slower (attention over the longer image context, more KV per step). A zero gap to "
                "single-stage in both is the 2% M* lead and nothing else.",
        arms=(Arm("main"),), post=("{py} exps/tools/analyze_bagel_ablation.py --campaign {campaign:main} "
                                    "--markdown {results}/bagel-i2i-steps_fit_{hardware}.md",)),
    Experiment(
        name="cosmos3-decompose", title="Cosmos3: M*'s 2.4x over vLLM-Omni, switched off one implementation choice at a time",
        task="cosmos3", group="advantage", kind="ab", frameworks=(MSTAR, "vllm-omni"), cases=("t2i-256",), repeats=3,
        measured=10, preconditions=3,
        question="M* takes 1.29 s and vLLM-Omni 3.13 s for the same 50-step 256x256 image. M* packs cond + uncond into one forward "
                 "(batched CFG), may replay a CUDA graph (its captured tiers are 192x320 and 480x832: is 256x256 one?) and compiles. "
                 "If M* is given vLLM's structure (two sequential forwards, no graph, no compile), does the 2.4x disappear?",
        reading="E2E per arm against `vllm`. m-nograph ~ m-control: the graph is not used at 256x256. m-seqcfg: what batching the two "
                "CFG branches is worth. m-all-off ~ vllm: the gap is these three choices and nothing else (then it is implementation, "
                "not architecture). m-capture256: what a 256x256 graph would add. A leftover gap after m-all-off is unexplained.",
        arms=(Arm("m-control", frameworks=(MSTAR,)),
              Arm("m-nograph", frameworks=(MSTAR,), env=COSMOS3_NO_GRAPH, expect=_cmd_has(*COSMOS3_NO_GRAPH)),
              Arm("m-seqcfg", frameworks=(MSTAR,), env=COSMOS3_SEQ_CFG, expect=Expect(proof="hook_applied")),
              Arm("m-seqcfg-nograph", frameworks=(MSTAR,), env=COSMOS3_SEQ_CFG + COSMOS3_NO_GRAPH, expect=Expect(proof="hook_applied")),
              Arm("m-all-off", frameworks=(MSTAR,), env=COSMOS3_SEQ_CFG + COSMOS3_NO_GRAPH + NO_COMPILE,
                  expect=Expect(proof="hook_applied"), note="M* with vLLM-Omni's structure: sequential CFG, eager, no compile"),
              Arm("m-capture256", frameworks=(MSTAR,), env=("mstar:COSMOS3_GEN_CAPTURE_RES=192x320,256x256",),
                  expect=_cmd_has("mstar:COSMOS3_GEN_CAPTURE_RES=192x320,256x256"), note="adds a 256x256 denoise graph"),
              Arm("vllm", frameworks=("vllm-omni",))),
        caveats="Counterfactual M* arms are not baselines. Sequential CFG changes M*'s graph/eager path selection; outputs are not "
                "compared (M*'s own docs say the two paths do the same math)."),
    Experiment(
        name="t2s-concurrency", title="T2S: B = 1 / 4 / 8 / 16 concurrent requests, three frameworks", task="qwen_t2s",
        group="advantage", kind="ab", frameworks=QWEN_FW,
        cases=("verbatim-en", "verbatim-en-b4", "verbatim-en-b16"),
        metric="ms_per_audio_s", repeats=2, measured=8, preconditions=4,
        question="The paper's headline T2S claim is B=16 throughput; the C=1 cells cannot test it. How does each framework scale?",
        reading="Per-audio-second time against B per framework; the ratio between frameworks at B=16 is the throughput claim. "
                "vLLM-Omni runs the max_num_seqs=16 variant of its best recipe (untested at that setting: a failed start is a result).",
        arms=(Arm("main"),),
        post=("{py} exps/tools/analyze_concurrency.py --campaign {campaign:main} --markdown {results}/t2s-concurrency_{hardware}.md",)),
)

# --- fairness: what is each baseline's own configuration / tuning worth? ---------------------------------------------------
FAIRNESS = (
    Experiment(
        name="bagel-mot-tuned-vs-default", title="BAGEL: what is tuning vLLM-Omni's MoT GEMM worth end to end?",
        task="bagel", group="fairness", kind="ab", frameworks=("vllm-omni-two-stage",),
        cases=("t2i-512-nocfg", "t2i-512-cfg3"), repeats=2,
        question="Per-step GPU busy dropped 32-58% with tuned tiles on the H200; does E2E follow, and on this GPU?",
        reading="untuned / tuned. ~1.0 (what was seen at 512^2 on the H200) means the E2E gap is not mostly this kernel.",
        # The two-stage DiT subprocess does not forward its INFO log to the server log, so "MoT config loaded from" /
        # "No tuned MoT config" are invisible there (the single-stage DiffusionWorker does log them); the serve command's
        # VLLM_TUNED_CONFIG_FOLDER is checked instead.
        arms=(Arm("tuned", expect=Expect(command_has=("VLLM_TUNED_CONFIG_FOLDER=",), log_lacks=("No tuned MoT config",))),
              Arm("untuned", opts=("bagel.vllm_mot=default",), expect=Expect(command_lacks=("VLLM_TUNED_CONFIG_FOLDER",)),
                  note="vLLM-Omni's default tiles: NOT its best config"))),
    Experiment(
        name="t2s-deploy-config", title="T2S: what vLLM-Omni's async-chunk + Code2Wav graph and SGLang's partial start are worth",
        task="qwen_t2s", group="fairness", kind="ab", frameworks=("sglang-omni", "vllm-omni"), cases=("verbatim-en",),
        metric="ms_per_audio_s",
        question="The baselines run their best recipe. How far behind is each in its plain single-GPU recipe?",
        reading="plain / best per framework: the part of the baselines' T2S numbers that is configuration, not engine.",
        arms=(Arm("best"),
              Arm("plain", opts=("qwen.vllm_deploy_config=exps/configs/vllm_omni/qwen3_omni_full_1gpu.yaml",
                                 "qwen.sglang_config=exps/configs/sglang_omni/qwen3_omni_colocated_h200.yaml")))),
    Experiment(
        name="a2t-moe-tuned-vs-default", title="A2T: what is M*'s own MoE tile tuning worth end to end?",
        task="qwen_a2t", group="fairness", kind="ab", frameworks=(MSTAR,), cases=("a2t-en",),
        repeats=2, measured=8, preconditions=6,
        question="vLLM and SGLang run tuned MoE tiles; M* only does if a table for this exact GPU exists (the package ships H100 / "
                 "H200 / GB200 / B300; an H200 NVL or RTX 5880 matches none). How much does M*'s own sweep (`mstar-moe-tune`) buy on "
                 "the Thinker's MoE layers?",
        reading="builtin / tuned. The old finding (M* ~1.6x the GPU time per token of the baselines on A2T, concentrated in MoE) was "
                "measured with built-in tiles; if tuning closes most of it, the A2T / TXT 'M* is slower' results were a tuning "
                "artefact, and every Qwen3-Omni ratio must be re-read with tuned M*.",
        arms=(Arm("tuned", expect=Expect(log_has=("mstar@kernel table: moe_gemm",), log_lacks=("mstar@kernel table: no moe_gemm",))),
              Arm("builtin", env=("mstar:MSTAR_KERNEL_TABLES={campaign_dir}/no_tables",),
                  expect=Expect(log_has=("mstar@kernel table: no moe_gemm",)),
                  note="M*'s built-in tiles: NOT its best config (an empty table dir hides the tables of this machine)"))),
)

# --- profiles and tools --------------------------------------------------------------------------------------------------------
PROFILES = (
    Experiment(
        name="bagel-nsys-equal-work", title="BAGEL: GPU busy / kernel mix at equal work (nsys), M* vs vLLM two-stage",
        task="bagel", group="profile", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), repeats=1, measured=2,
        cases=("t2i-512-nocfg", "t2i-512-cfg3"),
        question="Per-step GPU busy and launch count at equal work with vLLM-Omni tuned: is M*'s lead less GPU work per step?",
        reading="Busy per step and kernels by name; nsys E2E is not a benchmark number. The attention kernels' share says whether "
                "attention is the lever.",
        arms=(Arm("nsys", nsys=True, serve_args=("mstar:--enable-nvtx",)),),
        post=("{py} -m exps.tools.analyze_nsys_step_kernels exps/runs/{campaign}/nsys --markdown {results}/bagel-nsys-step-kernels_{hardware}.md",)),
    Experiment(
        name="bagel-t2i-nsys-single", title="BAGEL T2I 512: GPU busy per denoise step on one card, M* against vLLM-Omni single-stage (nsys, one session)",
        task="bagel", group="profile", kind="ab", frameworks=(MSTAR, "vllm-omni-single-stage"), repeats=1, measured=2,
        cases=("t2i-512-nocfg", "t2i-512-cfg3"),
        question="The two-stage vLLM-Omni per-step numbers were read from its DiT card. On one card, in one session, with the same nsys capture: "
                 "does vLLM-Omni's single-stage denoise loop cost the same GPU time per step as M*?",
        reading="GPU busy per step and span per step, by framework, at -nocfg and -cfg3. Busy within about 2% of M*: the denoise loop is equal on the "
                "5880 whatever the stage split, so the per-step gap is not where M*'s lead comes from. Busy clearly above M*: M* does less GPU work "
                "per step on one card, which the two-stage comparison could not show. Busy clearly below M*: check the kernel names before "
                "reading anything into it.",
        arms=(Arm("nsys", nsys=True, serve_args=("mstar:--enable-nvtx",)),),
        post=("{py} -m exps.tools.analyze_nsys_step_kernels exps/runs/{campaign}/nsys --markdown {results}/bagel-t2i-nsys-single_{hardware}.md",),
        caveats="nsys inflates host time: the GPU busy column is the evidence, not the span. One card (the one the run is pinned to), one start per framework."),
    Experiment(
        name="bagel-kv-pool", title="BAGEL on M*, one 48 GiB card: KV pool 1,024 pages against the shipped 512, at one request at a time",
        task="bagel", group="profile", kind="ab", frameworks=(MSTAR,), cases=("t2i-1024-cfg3", "i2i-512", "i2t-512"),
        repeats=2, measured=3,
        question="The 5880 runs M* BAGEL with a 512-page KV pool (~65k tokens) where the H200 default is 2,048 pages. Does a pool twice as big change the "
                 "latency of one request at a time (1024x1024 three-branch T2I, 512 I2I, 512 I2T)?",
        reading="E2E p50 per case, kv1024 against the two shipped-pool arms around it (A1 / A2 is the drift check). kv1024 inside the A1-A2 spread: the pool "
                "is not limiting one request, so the smaller 5880 advantage is not a pool effect. kv1024 faster than both by more than the spread: the 512-page "
                "pool is limiting something at B = 1 (eviction or allocation waits; read server.log). kv1024 failing on memory: the 1,024-page pool does not fit "
                "beside the weights on this card, which is a memory limit, not a latency one.",
        arms=(Arm("A1"),
              Arm("kv1024", opts=(f"config.mstar={BAGEL_MSTAR_KV1024}",), expect=Expect(command_has=(_vfile("mstar/bagel_kv1024"),))),
              Arm("A2")),
        caveats="B = 1 only: the pool matters more at B = 4 / 16, which this does not run. The 2,048-page default is not run (its config comment says it "
                "does not fit beside the weights). 1,024 pages add ~3.75 GB to the ~12 GB the log reports free after graph capture, so a 1,024-pixel "
                "request could still fail on activation memory."),
    Experiment(
        name="bagel-i2i-nsys", title="BAGEL I2I: GPU busy / idle per card (nsys), M* vs vLLM-Omni single- and two-stage",
        task="bagel", group="profile", kind="ab", frameworks=(MSTAR, *VLLM_BAGEL), repeats=1, measured=2, preconditions=1,
        cases=("i2i-512",),
        question="Where do the extra ~4 s of vLLM-Omni two-stage I2I go: GPU work (Thinker prefill, more attention per step), idle "
                 "time between the Thinker card and the DiT card (KV hand-off), or the denoise loop itself?",
        reading="analyze_nsys_burst_gaps.py, one table per GPU. Two-stage: the Thinker card's burst, the gap, then the DiT card's "
                "burst (the hand-off is the idle time between them); busy/span of the DiT card against single-stage's single burst "
                "says whether the loop itself differs. nsys inflates host time (M* most): structure only.",
        arms=(Arm("nsys", nsys=True, serve_args=("mstar:--enable-nvtx",)),),
        post=("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign}/nsys --min-launches 5000 --gap-ms 40 --last 4 --top 10 "
              "--markdown {results}/bagel-i2i-nsys_{hardware}.md",)),
    Experiment(
        name="bagel-i2t-nsys", title="BAGEL I2T: where do vLLM-Omni's extra ~650 ms before the first token go? (nsys)",
        task="bagel", group="profile", kind="ab", frameworks=(MSTAR, "vllm-omni-two-stage"), repeats=1, measured=4,
        preconditions=2, cases=("i2t-512",),
        question="Per-token cost is equal (16.9 vs 17.1 ms) but time to first token is 130 ms on M* and 808 ms on vLLM-Omni "
                 "two-stage. Is the difference GPU work or idle time before the first kernel?",
        reading="Answered on the RTX 5880 (kept for the H200): the vLLM burst is 98% GPU-busy from its first kernel, and its first "
                "step is one ~736 ms prefill of 4,935 tokens (GEMMs of 5-10 ms each) where M*'s whole LLM prefill is ~28 ms of "
                "GPU work. That is GPU work not done, not idle time: see bagel-prefix-cache for why. Read with "
                "analyze_nsys_burst_gaps.py (top kernels of the last burst).",
        arms=(Arm("nsys", nsys=True, serve_args=("mstar:--enable-nvtx",)),),
        post=("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign}/nsys --min-launches 20000 --last 4 "
              "--markdown {results}/bagel-i2t-nsys_{hardware}.md",)),
    Experiment(
        name="cosmos3-nsys", title="Cosmos3: GPU kernel mix of M* (sequential CFG, eager, no compile) vs vLLM-Omni at 256x256 (nsys)",
        task="cosmos3", group="profile", kind="ab", frameworks=(MSTAR, "vllm-omni"), cases=("t2i-256",), repeats=1, measured=4,
        preconditions=2,
        question="With CFG, graph and compile all matched (m-all-off) vLLM-Omni is still 1.17x (~440 ms) slower than M*. Is that "
                 "GPU work (a different attention / GEMM / VAE kernel) or time the GPU is idle?",
        reading="Last burst per cell (analyze_nsys_burst_gaps.py): GPU busy against span says idle vs work; the top-kernel "
                "lists say which kernel family carries the 440 ms. Structure only, nsys inflates host time.",
        arms=(Arm("m-all-off", frameworks=(MSTAR,), nsys=True, serve_args=("mstar:--enable-nvtx",),
                  env=COSMOS3_SEQ_CFG + COSMOS3_NO_GRAPH + NO_COMPILE, expect=Expect(proof="hook_applied")),
              Arm("vllm", frameworks=("vllm-omni",), nsys=True)),
        post=("{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:m-all-off}/nsys --min-launches 2000 --gap-ms 20 "
              "--last 4 --top 12 --markdown {results}/cosmos3-nsys_m-all-off_{hardware}.md",
              "{py} exps/tools/analyze_nsys_burst_gaps.py exps/runs/{campaign:vllm}/nsys --min-launches 2000 --gap-ms 20 "
              "--last 4 --top 12 --markdown {results}/cosmos3-nsys_vllm_{hardware}.md")),
    Experiment(
        name="t2s-stage-nsys", title="T2S: per-stage GPU timeline (nsys), three frameworks", task="qwen_t2s", group="profile",
        kind="ab", frameworks=QWEN_FW, cases=("verbatim-en",), repeats=1, measured=3, preconditions=3,
        question="Where does each framework spend GPU time per stage (Thinker / Talker / Code2Wav)?",
        reading="Busy per audio second by stage. nsys inflates the timeline 1.6-2.2x, M* most: structure only, never E2E.",
        arms=(Arm("nsys", nsys=True, serve_args=("mstar:--enable-nvtx",)),),
        post=("{py} exps/tools/analyze_nsys_t2s_stages.py --cells {campaign}/mstar {campaign}/vllm-omni {campaign}/sglang-omni "
              "--markdown {results}/t2s-stage-nsys_{hardware}.md",)),
    Experiment(
        name="qwen-t2s-nsys-warm", title="T2S: per-stage GPU timeline (nsys), three frameworks, rows pre-warmed",
        task="qwen_t2s", group="profile", kind="ab", frameworks=QWEN_FW, cases=("verbatim-en",), repeats=1, measured=5, preconditions=5,
        question="qwen-t2s-fit: after the first audio chunk the frame loop costs M* 54, SGLang 65, vLLM 100 ms per audio second. M*'s own launch chain explains 1.24x; "
                 "vLLM's extra 1.27x has only a structural nsys hint (3x launches in the Talker, measured on cold rows) and its mechanism is unproven. Where is the "
                 "per-frame time of vLLM-Omni's Talker + Code2Wav loop on the GPU when no request carries first-seen-shape JIT?",
        reading="analyze_nsys_t2s_stages.py: busy per audio second by stage (Thinker / Talker / Code2Wav) and the idle remainder. vLLM Talker busy per audio second near "
                "M*'s but a long idle remainder = host launch / scheduling gaps in the frame loop (the chain M* has); Talker busy itself ~1.3x = more or slower kernels per "
                "frame (no CUDA graph, unfused sampling). nsys inflates the timeline 1.6-2.2x, M* most: structure only, never E2E.",
        arms=(Arm("nsys", nsys=True, opts=(WARM_ROWS,), serve_args=("mstar:--enable-nvtx",)),),
        post=("{py} exps/tools/analyze_nsys_t2s_stages.py --cells {campaign}/mstar {campaign}/vllm-omni {campaign}/sglang-omni "
              "--markdown {results}/qwen-t2s-nsys-warm_{hardware}.md",)),
    Experiment(
        name="bagel-mot-tune", title="BAGEL: tune vLLM-Omni's MoT GEMM tiles for this GPU", task=None, group="tool",
        question="vLLM-Omni's BAGEL MoT linears run a Triton GEMM with per-GPU tuned tiles; without a tuned file every "
                 "cell used conservative defaults. Generate the file on every new GPU and after a vLLM-Omni upgrade.",
        reading="Writes exps/configs/vllm_omni/tuned/device_name=<GPU>,dtype=w16a16.json (needs `ray` in envs/vllm-omni). "
                "Batches 1 3 4 12 16 48 cover 256..1024 at one and three branches. Every vLLM-Omni BAGEL cell then loads it "
                "and the server log must say `MoT config loaded from`.",
        tool=("env", "VAE_CHUNK_SIZE=256", "envs/vllm-omni/bin/python", "external/vllm-omni/benchmarks/kernels/mot_linear_benchmarks.py",
              "--model", "{bagel_snapshot}", "--tp-size", "1", "--dtype", "w16a16", "--tune", "--batch-size", "1", "3", "4",
              "12", "16", "48", "--save-dir", "{repo}/exps/configs/vllm_omni/tuned")),
    Experiment(
        name="mstar-moe-tune", title="M*: sweep the MoE tile tables for Qwen3-Omni on this GPU", task=None, group="tool",
        question="M* picks fused-MoE tiles from per-GPU tables (exact torch device name); without one it uses built-in tiles while "
                 "vLLM / SGLang run tuned ones. Sweep both Qwen3-Omni shapes on every new GPU.",
        reading="Writes exps/configs/mstar/kernel_tables/moe_gemm/<GPU>.json (Thinker E=128,N=1536/TP,K=2048,top_k=8 -- N=768 for the TP=2 Thinker of the RTX 5880 overlays; Talker "
                "E=128,N=768,K=1024,top_k=6). Every M* cell then loads it (MSTAR_KERNEL_TABLES is set by the harness); the server "
                "log must say `kernel table: moe_gemm ... on <GPU>`. Needs an idle GPU; M*'s sweep checks every candidate against a "
                "reference output before it is timed, so a config that changes the maths is never written.",
        tool=("bash", "-c",
              "set -e; cd {repo}/external/mstar; for spec in 'E=128,N={mstar_thinker_n},K=2048,top_k=8:thinker:{thinker_tp}' "
              "'E=128,N=768,K=1024,top_k=6:talker:1'; "
              "do shape=${spec%%:*}; rest=${spec#*:}; ../../envs/mstar/bin/python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 "
              "--shape \"$shape\" --note \"Qwen3-Omni ${rest%%:*}, TP${rest##*:}\" --out {repo}/exps/configs/mstar/kernel_tables; done")),
    Experiment(
        name="vllm-moe-tune", title="vLLM: tune the fused-MoE Triton tiles for Qwen3-Omni on this GPU", task=None, group="tool",
        question="vLLM-Omni's Thinker / Talker MoE layers look up per-GPU tuned tiles (vLLM maps every H200 variant to NVIDIA_H200, "
                 "whose table ships in the wheel; an RTX 5880 has none and runs default tiles with a warning).",
        reading="Writes exps/configs/vllm_omni/tuned/E=128,N=768/TP,device_name=<GPU>.json (Thinker; N=384 at TP=2, the same file as the TP=1 Talker, so they share one table and the Thinker, tuned last, wins) and E=128,N=384,... (Talker); "
                "every vLLM-Omni cell then sets VLLM_TUNED_CONFIG_FOLDER to that folder. Tuner: external/vllm "
                "benchmarks/kernels/benchmark_moe.py (v0.32.0, matching envs/vllm-omni); needs `ray` in envs/vllm-omni and an idle "
                "GPU. Inputs are Qwen3Moe-shaped configs under exps/configs/tuner_inputs/. Not needed on the H200 NVL (wheel table).",
        tool=("bash", "-c",
              "set -e; for spec in {moe_roles}; do role=${spec%%:*}; {repo}/envs/vllm-omni/bin/python "
              "{repo}/external/vllm/benchmarks/kernels/benchmark_moe.py --model {repo}/exps/configs/tuner_inputs/$role "
              "--tp-size ${spec##*:} --dtype auto --tune --save-dir {repo}/exps/configs/vllm_omni/tuned; done")),
    Experiment(
        name="sglang-moe-tune", title="SGLang: tune the fused-MoE Triton tiles for Qwen3-Omni on this GPU and Triton version",
        task=None, group="tool",
        question="SGLang-Omni looks tables up as <dir>/configs/triton_<version>/E=..,N=..,device_name=<GPU>.json and falls back to "
                 "the newest *other* Triton version's table with a warning when its own is missing (the H200 tables are from "
                 "triton 3.7.1; the env has 3.8.0). Re-tune on the installed version.",
        reading="Writes exps/configs/sglang_omni/moe_<hardware profile>/configs/triton_<installed version>/E=128,N=768,... "
                "(Thinker) and E=128,N=384,... (Talker); the harness points SGLANG_MOE_CONFIG_DIR there. Tuner: external/sglang "
                "benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py (v0.5.22, matching envs/sglang-omni); it writes "
                "into the current directory, which this command sets. Needs an idle GPU; long (it searches a large Triton config "
                "space per batch size).",
        tool=("bash", "-c",
              "set -e; export PATH=\"{repo}/envs/sglang-omni/bin:$PATH\"; "
              "v=$({repo}/envs/sglang-omni/bin/python -c 'import triton; print(\"triton_\" + triton.__version__.replace(\".\", \"_\"))'); "
              "d={repo}/exps/configs/sglang_omni/moe_{hardware}/configs/$v; mkdir -p \"$d\"; cd \"$d\"; "
              "for spec in {moe_roles}; do role=${spec%%:*}; {repo}/envs/sglang-omni/bin/python "
              "{repo}/external/sglang/benchmark/kernels/fused_moe_triton/tuning_fused_moe_triton.py "
              "--model {repo}/exps/configs/tuner_inputs/$role --tp-size ${spec##*:} --dtype auto --tune; done")),
    Experiment(
        name="thermal-probe", title="GPU: sustained-load thermal probe (~10 min)", task=None, group="tool",
        question="Does the card throttle or drift under sustained load? (Confirms a machine really has stable cooling before "
                 "its default protocol is back-to-back.)",
        reading="Median clock under burn, time to the throttle knee, cooling curve; compare with an earlier probe.",
        tool=("{py}", "exps/tools/gpu_thermal_probe.py", "run", "--out", "exps/runs/THERMAL_PROBE_{run_id}")),
)

ALL: dict[str, Experiment] = {e.name: e for e in (*BASELINES, *PROTOCOL_AB, *ABLATIONS_MSTAR, *ADVANTAGE, *FAIRNESS, *PROFILES)}


def get(name: str) -> Experiment:
    try:
        return ALL[name]
    except KeyError:
        raise KeyError(f"unknown experiment {name!r}; try `python -m exps.suite list`") from None
