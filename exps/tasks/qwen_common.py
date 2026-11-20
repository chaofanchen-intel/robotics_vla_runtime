#!/usr/bin/env python3
"""Shared Qwen3-Omni service recipes for the T2S, TXT and A2T tasks.

All three tasks serve the same checkpoint on the same three runtimes and differ
only in what they send. Defining the serve commands once means a change to how
SGLang-Omni is launched cannot apply to text-to-speech but not to
audio-to-text, which would quietly make those two tasks incomparable.
"""
from __future__ import annotations

import os
from pathlib import Path

import exps.frameworks as frameworks
from exps import hardware
from exps.core import overrides, paths
from exps.core.lifecycle import ServiceSpec

SNAPSHOT = (
    paths.MODELS_CACHE
    / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct"
    / "snapshots"
    / "26291f793822fb6be9555850f06dfe95f2d7e695"
)
MODEL_REVISION = "26291f793822fb6be9555850f06dfe95f2d7e695"
MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SERVED_MODEL_NAME = "qwen3_omni"

#: Each framework runs its best single-GPU recipe, not its first-draft one (the earlier rounds ran the plain ones and a
#: later "final config" round found them slower, so every old T2S ratio had to be redone):
#:   vLLM-Omni  `async_chunk: true` (upstream default) and Code2Wav `enforce_eager: false` (upstream: "stage 2 keeps
#:              enforce_eager=false so Code2Wav can use its inner cudagraph"); the local single-GPU yaml had both off.
#:   SGLang-Omni  Talker partial start on (the default of its 2-GPU speech variant; measured better on one GPU too).
#:   M*  its one config.
#: The plain recipes stay available for the ablations that ask what each setting is worth:
#: `--opt qwen.vllm_deploy_config=exps/configs/vllm_omni/qwen3_omni_full_1gpu.yaml`, `--opt qwen.sglang_config=...`.
MSTAR_CONFIG = paths.CONFIGS / "mstar" / "qwen3omni_1gpu.yaml"
VLLM_CONFIG = paths.CONFIGS / "vllm_omni" / "qwen3_omni_full_1gpu_default_async_c2wgraph.yaml"
SGLANG_CONFIG = paths.CONFIGS / "sglang_omni" / "qwen3_omni_colocated_h200_partial_start.yaml"
VLLM_PLAIN_CONFIG = paths.CONFIGS / "vllm_omni" / "qwen3_omni_full_1gpu.yaml"
SGLANG_PLAIN_CONFIG = paths.CONFIGS / "sglang_omni" / "qwen3_omni_colocated_h200.yaml"
VLLM_STAGE_TIMING_CONFIG = (
    paths.CONFIGS / "vllm_omni" / "qwen3_omni_full_1gpu_stage_timing.yaml"
)
SGLANG_STAGE_TIMING_CONFIG = (
    paths.CONFIGS / "sglang_omni" / "qwen3_omni_colocated_h200_stage_timing.yaml"
)
#: Concurrency variant: identical to VLLM_CONFIG except max_num_seqs=16 per stage. The C=1 recipe
#: pins max_num_seqs=1, which would serialise a B>1 sweep. SGLang-Omni and M* need no variant.
#: (Untested at max_num_seqs=16: a start that fails is a result, not a retry.)
VLLM_BATCH_CONFIG = paths.CONFIGS / "vllm_omni" / "qwen3_omni_full_1gpu_batch_async_c2wgraph.yaml"
def sglang_moe_dir():
    """SGLang's tuned Triton MoE tables for this machine: `configs/sglang_omni/moe_<hardware profile>/`.

    SGLang finds them as `<dir>/configs/triton_<triton version>/E=..,N=..,device_name=<GPU>.json`, so a table tuned under one
    Triton version is silently not found under another (the upgraded env has triton 3.8.0; the H200 tables were tuned under
    3.7.1). `python -m exps.tuning status` shows what is present. None when the machine has no tuned directory: SGLang then
    runs its default MoE config and the cell says so in its manifest context.
    """
    from exps import hardware

    folder = paths.CONFIGS / "sglang_omni" / f"moe_{hardware.current().name}"
    return folder if folder.is_dir() else None

#: Loading 30B of weights takes minutes, and SGLang-Omni's colocated launch
#: brings up five stages before it binds its port.
READY_TIMEOUT_S = 2400.0

#: vLLM-Omni gives each stage 300 s (and the whole engine 600 s) to become ready.
VLLM_SLOW_LOAD_ARGS = "--stage-init-timeout 1800 --init-timeout 2100"

TOPOLOGY_NOTE = (
    "Single H200 with every stage colocated. The paper runs Qwen3-Omni on 2xH200 with Thinker "
    "and Talker on separate devices; that topology is not reproduced here and no result should "
    "be read as if it were."
)

CONFIG_BY_FRAMEWORK = {
    "mstar": MSTAR_CONFIG,
    "vllm-omni": VLLM_CONFIG,
    "sglang-omni": SGLANG_CONFIG,
}

STAGE_TIMING_CONFIG_BY_FRAMEWORK = {
    "mstar": MSTAR_CONFIG,
    "vllm-omni": VLLM_STAGE_TIMING_CONFIG,
    "sglang-omni": SGLANG_STAGE_TIMING_CONFIG,
}


RTX5880_TOPOLOGY_NOTE = (
    "Three RTX 5880 (48 GB), the fewest that hold the 61 GB Thinker: Thinker TP=2 on two cards, Talker and Code2Wav on "
    "the third. The paper runs Qwen3-Omni on 2xH200 with Thinker and Talker on separate devices; this keeps that split "
    "but is not the same hardware, so no result should be read as if it were."
)


def client_model_instance_kwargs(framework: str) -> dict[str, str]:
    if framework == "mstar":
        return {}
    if framework in {"vllm-omni", "sglang-omni"}:
        return {"served_model_name": SERVED_MODEL_NAME}
    raise KeyError(f"Qwen3-Omni has no framework {framework!r}")


def service(
    framework: str,
    task: str,
    tag: str,
    *,
    stage_timing: bool = False,
    batch_config: bool = False,
) -> ServiceSpec:
    """Serve command for one runtime on one Qwen3-Omni task."""
    port = frameworks.port(framework, task)
    health = f"http://127.0.0.1:{port}/health"
    config = config_path(framework, stage_timing=stage_timing, batch_config=batch_config)

    if framework == "mstar":
        # compile off is the same switch on every machine: `_compile_default` is on everywhere but rtx5880, so only an explicit
        # `qwen.mstar_compile=off` reaches this on the others. (Until 2026-11-15 the flag was set on rtx5880 only, so an h200nvl
        # "compile off" arm ran with compile ON: the three arms of qwen-mstar-compile-{txt,a2t,t2s} on h200nvl were one config.)
        extra_env = {"TORCHDYNAMO_DISABLE": "1"} if not mstar_compile(task) else None
        command = frameworks.mstar_serve(
            model="qwen3_omni",
            config=config,
            port_number=port,
            tag=f"qwen-{tag}",
            extra_env=extra_env,
        )
    elif framework == "vllm-omni":
        command = frameworks.vllm_omni_serve(
            model_path=SNAPSHOT,
            deploy_config=config,
            port_number=port,
            served_model_name=SERVED_MODEL_NAME,
            # a TP=2 Thinker reads the 66 GB checkpoint on both ranks: ~4.5 min on this box, over the 300 s default
            extra=VLLM_SLOW_LOAD_ARGS if hardware.current().name == "rtx5880" else "",
        )
    elif framework == "sglang-omni":
        command = frameworks.sglang_omni_serve(
            model_path=SNAPSHOT,
            config=config,
            port_number=port,
            model_name=SERVED_MODEL_NAME,
            moe_config_dir=sglang_moe_dir(),
            colocate=hardware.current().name != "rtx5880",
        )
    else:
        raise KeyError(f"Qwen3-Omni has no framework {framework!r}")

    return ServiceSpec(command=command, health_url=health, ready_timeout_s=READY_TIMEOUT_S)


def config_path(framework: str, *, stage_timing: bool = False, batch_config: bool = False):
    if batch_config and framework == "vllm-omni":
        return _config_override("vllm_batch_config", VLLM_BATCH_CONFIG)
    if stage_timing:
        return STAGE_TIMING_CONFIG_BY_FRAMEWORK[framework]
    if framework == "vllm-omni":
        return _config_override("vllm_deploy_config", VLLM_CONFIG)
    if framework == "sglang-omni":
        return _config_override("sglang_config", SGLANG_CONFIG)
    return CONFIG_BY_FRAMEWORK[framework]


def _config_override(key: str, default: Path) -> Path:
    """`--opt qwen.<key>=PATH` swaps a deploy config (the T2S async-chunk / partial-start ablations)."""
    env_name = f"--opt qwen.{key}"
    raw_path = overrides.current().option("qwen", key)
    if not raw_path:
        return default

    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = paths.REPO_ROOT / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{env_name} points to a missing config file: {path}")
    hardware.check_config_path(path)
    return path


def base_url(framework: str, task: str) -> str:
    return f"http://127.0.0.1:{frameworks.port(framework, task)}"


def shared_context(framework: str, *, stage_timing: bool = False, batch_config: bool = False, task: str | None = None) -> dict:
    config = config_path(framework, stage_timing=stage_timing, batch_config=batch_config)
    return {
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION},
        "config": paths.relative_to_repo(config),
        "topology": RTX5880_TOPOLOGY_NOTE if hardware.current().name == "rtx5880" else TOPOLOGY_NOTE,
        "served_model_name": SERVED_MODEL_NAME,
        "request_sampling_mode": sampling_mode(),
        "request_rows_mode": rows_mode(),
        **({"mstar_torch_compile": _compile_label(task)} if framework == "mstar" else {}),
    }


SAMPLING_MODES = ("pinned", "lookahead-eligible")


def sampling_mode() -> str:
    """`--opt qwen.sampling=lookahead-eligible` (TXT and A2T clients): drop `seed` (and TXT's `min_tokens`) from the request.

    SGLang-Omni's one-step lookahead decode (`enable_async_decode`, default on for the Thinker) routes a batch to the
    synchronous loop when any request carries a sampling seed or `min_new_tokens > 0`
    (`ThinkerModelRunner.lookahead_eligible`). The default requests carry `seed: 42` (and TXT `min_tokens`) so every runtime is
    pinned the same way; that silently turns SGLang's own overlap off. Greedy decoding needs no seed, and TXT stays pinned to
    256 tokens by `ignore_eos` + `max_tokens`, so this mode keeps the work equal. Counterfactual, not a baseline.
    """
    choice = overrides.current().option("qwen", "sampling", "pinned")
    if choice not in SAMPLING_MODES:
        raise ValueError(f"--opt qwen.sampling={choice!r}: expected one of {SAMPLING_MODES}")
    return choice


def _explicit_compile_choice() -> str | None:
    """`on` / `off` when the option or EXPS_QWEN_MSTAR_COMPILE says so, else None (the per-task default applies)."""
    import os

    choice = overrides.current().option("qwen", "mstar_compile", os.environ.get("EXPS_QWEN_MSTAR_COMPILE"))
    if choice is not None and choice not in ("on", "off"):
        raise ValueError(f"--opt qwen.mstar_compile={choice!r}: expected 'on' or 'off'")
    return choice


def _compile_default(task: str | None) -> bool:
    """Per-task default for M* Qwen. Off-profile hardware compiles (as before); on rtx5880 only T2S does.

    2026-11-13, the rtx5880 T2S A/B (qwen-mstar-compile-t2s, two batches): torch.compile is about 2% faster per audio second
    (A1 95.7 / 94.9 vs compile-on 92.8 / 93.4 ms, A1-A2 drift 0.6%), while TXT is 4% slower and A2T 1% slower. So TXT and A2T keep
    the compile-off default, and T2S takes compile-on.
    """
    if hardware.current().name != "rtx5880":
        return True
    return task == "qwen_t2s"


def mstar_compile(task: str | None = None) -> bool:
    """Whether M* Qwen cells run with torch.compile.

    Precedence: `--opt qwen.mstar_compile=on|off`, then EXPS_QWEN_MSTAR_COMPILE, then the per-task default (`_compile_default`).
    Off on rtx5880 meant TORCHDYNAMO_DISABLE=1, which dates from the first smoke's Talker shared-memory OutOfResources. That kernel is
    M*'s own `_cp_linear_kernel` (code predictor), tuned on an H200; the pinned M* retries it with fewer stages / a shorter K step
    (`_CP_SMEM_FIT`), and SMOKE_5880_05 completed, so the old workaround was stale; the qwen-mstar-compile-* experiments measured it.
    """
    choice = _explicit_compile_choice()
    if choice is None:
        return _compile_default(task)
    return choice == "on"


def _compile_label(task: str | None) -> str:
    """The value written to the manifest: what the cell actually ran, and whether it came from the default."""
    on = mstar_compile(task)
    on_5880 = hardware.current().name == "rtx5880"
    explicit = _explicit_compile_choice() is not None
    if on:
        return "on" if explicit or not on_5880 or task != "qwen_t2s" else "on (rtx5880 T2S default)"
    return "off" if explicit or not on_5880 else "off (rtx5880 default)"


CONTRACT_MODES = ("enforce", "record")


def contract_mode() -> str:
    """`--opt qwen.contract=record` (T2S client): a response that fails the read-aloud contract is flagged in the row
    (`contract_honoured: false`, `violation`) instead of voiding the cell, so a diagnostic run can count how often it happens.
    Never used for a measurement that is compared across frameworks."""
    choice = overrides.current().option("qwen", "contract", "enforce")
    if choice not in CONTRACT_MODES:
        raise ValueError(f"--opt qwen.contract={choice!r}: expected one of {CONTRACT_MODES}")
    return choice


ROWS_MODES = ("sequential", "cycle", "shuffle", "probe")
PROBE_FILLS = ("short", "long", "same", "random")


def rows_mode() -> str:
    """`--opt qwen.rows=cycle` (T2S and A2T clients): request i uses fixture row `i % measured`, so a precondition phase of
    `measured` requests has already served every row (and every input shape) the measured phase will send.

    vLLM-Omni JIT-compiles Triton fused-MoE kernels at inference time for token counts it has not seen (its `jit_monitor`
    warns "This causes a latency spike; consider extending warmup to cover this shape/config"): ~400-500 ms on the first request
    of a new shape, 2 of 10 measured A2T-zh requests and ~1 in 4 T2S requests per start. With `sequential` rows a short
    precondition phase leaves those shapes for the measured phase. Counterfactual / warm profile, not a baseline.

    `shuffle` (T2S client only): every pass over the fixture uses a fresh seeded permutation, so the same sentence meets different
    neighbours in different waves (sequential rows repeat the same wave composition every lcm(B, rows)/B waves).

    `probe` (T2S client only): every wave carries one fixed sentence (`--opt qwen.probe_row=<fixture_id>`) at a position that rotates
    from wave to wave, and the other B-1 slots are filled per `--opt qwen.probe_fill=short|long|same|random` (the shortest / longest
    other sentences, the probe sentence again, or a seeded shuffle of the others). Used to find what in a wave makes a sentence run away.
    """
    choice = overrides.current().option("qwen", "rows", "sequential")
    if choice not in ROWS_MODES:
        raise ValueError(f"--opt qwen.rows={choice!r}: expected one of {ROWS_MODES}")
    return choice


def probe_options() -> tuple[str, str]:
    row = overrides.current().option("qwen", "probe_row", "")
    fill = overrides.current().option("qwen", "probe_fill", "random")
    if fill not in PROBE_FILLS:
        raise ValueError(f"--opt qwen.probe_fill={fill!r}: expected one of {PROBE_FILLS}")
    if rows_mode() == "probe" and not row:
        raise ValueError("--opt qwen.rows=probe needs --opt qwen.probe_row=<fixture_id>")
    return row, fill


def client_env() -> dict[str, str]:
    env = {"HF_HOME": str(paths.MODELS_CACHE), "HF_HUB_OFFLINE": "1"}
    if sampling_mode() != "pinned":
        env["EXPS_QWEN_SAMPLING"] = sampling_mode()
    if rows_mode() != "sequential":
        env["EXPS_QWEN_ROWS"] = rows_mode()
    if contract_mode() != "enforce":
        env["EXPS_QWEN_CONTRACT"] = contract_mode()
    if rows_mode() == "probe":
        env["EXPS_QWEN_PROBE_ROW"], env["EXPS_QWEN_PROBE_FILL"] = probe_options()
    return env
