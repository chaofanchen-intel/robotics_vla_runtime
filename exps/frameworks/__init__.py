#!/usr/bin/env python3
"""Per-runtime serve recipes: how to start each engine and how to talk to it.

Each runtime gets a `Framework` record (identity, virtualenv, wire protocol)
and a module-level helper that builds its serve command. Task packages compose
those helpers rather than embedding shell strings, so a change to how M* is
launched happens once rather than in eight places.

Why each runtime needs its own virtualenv
-----------------------------------------
These engines pin mutually incompatible Torch builds -- M* on 2.10.0+cu130,
vLLM-Omni on 2.13.0+cu130, SGLang-Omni on 2.14.1+cu130, and VoxServe on
2.8.0+cu128. They cannot share an interpreter, which is why every measured
client runs as a subprocess under `envs/<name>/bin/python` instead of being
imported by the harness.

Port allocation
---------------
Ports are assigned per (framework, task) from a single table below. Cells run
one at a time behind a GPU lock, so collisions should be impossible -- but a
leaked process from a previous cell is not impossible, and a fixed, documented
port makes that diagnosable instead of mysterious.
"""
from __future__ import annotations

import hashlib
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path

from exps import hardware
from exps.core import overrides, paths

#: Shared by every serve command: weights come from the local cache, offline.
def host_nvcc() -> str:
    """The host CUDA compiler FlashInfer's JIT uses for VoxServe (it compiles at server start with `PYTORCH_NVCC`).

    This used to be the literal `/usr/bin/nvcc`, which only exists where apt's `nvidia-cuda-toolkit` is installed (the
    RTX 5880 host, nvcc 12.0). The H200 host has CUDA 13.3 under `/usr/local/cuda` and no `/usr/bin/nvcc`, so every
    VoxServe cell there died building FlashInfer ("/usr/bin/nvcc: not found") while the health endpoint stayed up and the
    harness waited out the full 1800 s. Order: `$VOXSERVE_NVCC`, `nvcc` on PATH, `/usr/local/cuda/bin/nvcc`, `/usr/bin/nvcc`.
    """
    explicit = os.environ.get("VOXSERVE_NVCC")
    if explicit:
        return explicit
    for candidate in (shutil.which("nvcc"), "/usr/local/cuda/bin/nvcc", "/usr/bin/nvcc"):
        if candidate and os.access(candidate, os.X_OK):
            return candidate
    return "/usr/bin/nvcc"


OFFLINE_ENV = 'HF_HOME="$PWD/models/cache" HF_HUB_CACHE="$PWD/models/cache" HF_HUB_OFFLINE=1'


def _gpus() -> str:
    """`CUDA_VISIBLE_DEVICES=<hardware profile / --gpus>`; one GPU (`0`) unless the run asks for more."""
    return hardware.visible_devices_env()


def _mstar_tuned_env() -> dict[str, str]:
    """M* looks tuned kernel tables up in `MSTAR_KERNEL_TABLES/<op>/<GPU name>.json`, then in its package.

    The package ships tables for a few GPUs (H100, H200, GB200, B300) by exact torch device name, so an H200 *NVL*
    (`NVIDIA H200 NVL`) or an RTX 5880 matches none of them. Sweep with `python -m exps.suite run mstar-moe-tune --execute`.
    """
    folder = paths.CONFIGS / "mstar" / "kernel_tables"
    return {"MSTAR_KERNEL_TABLES": str(folder)} if any(folder.rglob("*.json")) else {}


def _tuned_env() -> dict[str, str]:
    """vLLM-Omni reads tuned kernel configs (BAGEL MoT GEMM, fused-MoE) from this folder, matched by device name."""
    folder = paths.CONFIGS / "vllm_omni" / "tuned"
    return {"VLLM_TUNED_CONFIG_FOLDER": str(folder)} if any(folder.glob("*.json")) else {}


@dataclass(frozen=True)
class Framework:
    """Identity and wire protocol of one runtime under test."""

    name: str
    title: str
    #: Directory under envs/ holding this runtime's interpreter.
    env: str
    #: Directory under external/ whose commit is recorded in every manifest.
    external_repo: str
    #: How responses are shaped: decides which decoder in core.codecs applies.
    api_style: str
    notes: str = ""
    #: Name of the adapter in M*'s benchmark.request. Defaults to `name`, and
    #: differs only where one runtime is entered under two deployment recipes
    #: (BAGEL single-stage vs two-stage), which are different services but the
    #: same wire protocol.
    adapter: str | None = None

    @property
    def adapter_name(self) -> str:
        return self.adapter or self.name

    @property
    def python(self) -> Path:
        return paths.env_python(self.env)


MSTAR = Framework(
    name="mstar",
    title="M*",
    env="mstar",
    external_repo="mstar",
    api_style="mstar-ndjson",
    notes=(
        "POST /generate with a multipart form; the response is NDJSON with one base64 chunk per "
        "line. Audio arrives as raw PCM16 with no container, so the client wraps it before "
        "measuring duration."
    ),
)

VLLM_OMNI = Framework(
    name="vllm-omni",
    title="vLLM-Omni",
    env="vllm-omni",
    external_repo="vllm-omni",
    api_style="openai-chat",
    notes="OpenAI-compatible /v1/chat/completions, with modality extensions for image and audio.",
)

SGLANG_OMNI = Framework(
    name="sglang-omni",
    title="SGLang-Omni",
    env="sglang-omni",
    external_repo="sglang-omni",
    api_style="openai-chat",
    notes=(
        "Colocated multi-stage pipeline (image/audio encoders, Thinker, Talker, Code2Wav) on one "
        "GPU, driven through an OpenAI-compatible endpoint."
    ),
)

SGLANG_CORE = Framework(
    name="sglang-core",
    title="SGLang core (multimodal_gen)",
    env="sglang-omni",
    external_repo="sglang-omni",
    api_style="websocket-json",
    notes=(
        "sglang.multimodal_gen.runtime.launch_server with Pi05Pipeline. This is SGLang core, NOT "
        "SGLang-Omni Stage; the two must never be relabelled as each other in a result."
    ),
)

VLLM_OMNI_SINGLE_STAGE = Framework(
    name="vllm-omni-single-stage",
    title="vLLM-Omni (BAGEL single-stage)",
    env="vllm-omni",
    external_repo="vllm-omni",
    api_style="openai-chat",
    adapter="vllm-omni",
    notes=(
        "BAGEL served through the single-stage deploy config: understanding and generation in "
        "one engine. Kept distinct from the two-stage recipe because they are different "
        "deployments of the same runtime, not the same thing measured twice."
    ),
)

VLLM_OMNI_TWO_STAGE = Framework(
    name="vllm-omni-two-stage",
    title="vLLM-Omni (BAGEL two-stage)",
    env="vllm-omni",
    external_repo="vllm-omni",
    api_style="openai-chat",
    adapter="vllm-omni",
    notes=(
        "BAGEL served through the two-stage deploy config: a Thinker stage feeding a diffusion "
        "stage. Its YAML defaults allow 2048 tokens at temperature 0.4, which the client "
        "overrides per request and then asserts."
    ),
)

VOXSERVE = Framework(
    name="voxserve",
    title="VoxServe",
    env="voxserve",
    external_repo="vox-serve",
    api_style="voxserve-wav",
    notes=(
        "POST /generate with a multipart form; the response body is a complete WAV when "
        "streaming=false. Identified by the M* paper as its Orpheus baseline."
    ),
)

META_NATIVE = Framework(
    name="meta-native",
    title="Meta FAIR upstream V-JEPA 2-AC",
    env="vjepa2_repo",
    external_repo="vjepa2",
    api_style="inprocess",
    notes=(
        "Not a serving runtime. Facebook Research's own vjepa2_ac_vit_giant rollout, run in "
        "process and timed end to end -- the same pairing the paper uses for its baseline."
    ),
)

PYTORCH_EAGER = Framework(
    name="pytorch-eager",
    title="PyTorch eager (M* components)",
    env="mstar",
    external_repo="mstar",
    api_style="inprocess",
    notes=(
        "M*'s own component modules run without its engine, compile or CUDA-graph runner. Kept "
        "available for component-level probes; it is NOT the V-JEPA baseline this round uses."
    ),
)

ALL: dict[str, Framework] = {
    framework.name: framework
    for framework in (
        MSTAR, VLLM_OMNI, VLLM_OMNI_SINGLE_STAGE, VLLM_OMNI_TWO_STAGE,
        SGLANG_OMNI, SGLANG_CORE, VOXSERVE, META_NATIVE, PYTORCH_EAGER,
    )
}


def get(name: str) -> Framework:
    try:
        return ALL[name]
    except KeyError:
        raise KeyError(f"unknown framework {name!r}; available: {', '.join(sorted(ALL))}") from None


#: (framework, task) -> port. One table so a leak is traceable to a cell.
PORTS: dict[tuple[str, str], int] = {
    ("mstar", "bagel"): 19301,
    ("vllm-omni-single-stage", "bagel"): 19302,
    ("vllm-omni-two-stage", "bagel"): 19303,
    ("mstar", "qwen_t2s"): 19311,
    ("sglang-omni", "qwen_t2s"): 19312,
    ("vllm-omni", "qwen_t2s"): 19313,
    ("mstar", "qwen_txt"): 19321,
    ("sglang-omni", "qwen_txt"): 19322,
    ("vllm-omni", "qwen_txt"): 19323,
    ("mstar", "qwen_a2t"): 19331,
    ("sglang-omni", "qwen_a2t"): 19332,
    ("vllm-omni", "qwen_a2t"): 19333,
    ("mstar", "orpheus"): 19341,
    ("voxserve", "orpheus"): 19342,
    ("mstar", "vjepa_ac"): 19351,
    ("meta-native", "vjepa_ac"): 19352,
    ("pytorch-eager", "vjepa_ac"): 19353,
    ("mstar", "cosmos3"): 19361,
    ("vllm-omni", "cosmos3"): 19362,
    ("sglang-omni", "cosmos3"): 19363,
    ("mstar", "pi05"): 19371,
    ("vllm-omni", "pi05"): 19372,
    ("sglang-core", "pi05"): 19373,
}


def port(framework: str, task: str) -> int:
    try:
        return PORTS[(framework, task)]
    except KeyError:
        raise KeyError(f"no port allocated for framework {framework!r} on task {task!r}") from None


def mstar_serve(
    *,
    model: str,
    config: Path,
    port_number: int,
    tag: str,
    extra: str = "",
    extra_env: dict[str, str] | None = None,
    cache_dir: str = "models/cache",
) -> str:
    """Serve command for `mstar serve <model>`.

    `cache_dir` (repo-relative) replaces the default HF cache for both the HF env vars and
    `--cache-dir`; used by the BAGEL server-side step ablation, whose cache dir holds a
    config.json with a different `num_timesteps`.

    `mstar_cuda_env.sh` must be sourced first: M*'s JIT CUDA extensions have to
    build against the same CUDA 13.0 toolkit minor version as its cu130 Torch
    build, and the ambient system toolkit is a different one.

    `--socket-path-prefix` and `--upload-dir` are tagged per cell so a leaked
    socket or upload file from a previous start cannot be picked up by the next.
    Overlong tags use a stable digest to stay within Linux's IPC socket limit.
    """
    socket_dir = f"/tmp/exps-{tag}"
    socket_path = f"{socket_dir}/api_server_preprocess_worker.ipc"
    if len(socket_path.encode("utf-8")) > 107:
        digest = hashlib.sha256(tag.encode("utf-8")).hexdigest()[:16]
        socket_dir = f"/tmp/exps-{digest}"

    offline_env = OFFLINE_ENV.replace("models/cache", cache_dir)
    # Counterfactual / instrumentation switches come from `core.overrides` (`--env mstar:TORCHDYNAMO_DISABLE=1`,
    # `--serve-arg mstar:--enable-nvtx`) and land in the manifest's serve command. `mstar serve` does not accept
    # --enable-nvtx (only the low-level server does), so that flag swaps in tools/mstar_serve_nvtx.py.
    active = overrides.current()
    config = active.config_for("mstar", hardware.resolve_config(config))
    merged_env = {**_mstar_tuned_env(), **(extra_env or {}), **active.env_for("mstar")}
    extra_env = " ".join(shlex.quote(f"{key}={value}") for key, value in sorted(merged_env.items()))
    extra = " ".join(part for part in (extra, *active.args_for("mstar")) if part)
    launcher = "envs/mstar/bin/mstar"
    if "--enable-nvtx" in extra.split():
        extra = " ".join(part for part in extra.split() if part != "--enable-nvtx")
        launcher = "envs/mstar/bin/python exps/tools/mstar_serve_nvtx.py"
    return (
        f"export {_gpus()} && source exps/frameworks/mstar_cuda_env.sh && {_gpus()} {offline_env}"
        f"{(' ' + extra_env) if extra_env else ''} "
        f"exec {launcher} serve {model} "
        f"--config {shlex.quote(str(config))} "
        f'--host 127.0.0.1 --port {port_number} --cache-dir "$PWD/{cache_dir}" '
        f"--socket-path-prefix {socket_dir} --upload-dir {socket_dir}"
        f"{(' ' + extra) if extra else ''}"
    )


def vllm_omni_serve(
    *,
    model_path: Path,
    deploy_config: Path,
    port_number: int,
    served_model_name: str | None = None,
    extra: str = "",
    env: dict[str, str | None] | None = None,
) -> str:
    """Serve command for `vllm serve --omni`.

    `env` is the task's own recipe environment (BAGEL's tuned MoT config folder); `--env vllm-omni:K=V` overrides
    layer on top of it, so a counterfactual can still replace a recipe value.

    `--disable-log-stats` is always on: the stats logger wakes on a timer and
    adds periodic GPU work that would land inside measured requests.
    """
    name_flag = f"--served-model-name {served_model_name} " if served_model_name else ""
    # Overrides for every vLLM-Omni cell (`--env vllm-omni:VLLM_TUNED_CONFIG_FOLDER=...`), recorded in the manifest's
    # serve command; the stage workers inherit them.
    active = overrides.current()
    deploy_config = active.config_for("vllm-omni", hardware.resolve_config(deploy_config))
    # A recipe value of None *removes* a default (the tuned-config folder every vLLM-Omni cell otherwise gets), so a
    # counterfactual such as BAGEL's untuned MoT arm really runs without it.
    merged_env = {**_tuned_env(), **(env or {}), **active.env_for("vllm-omni")}
    merged_env = {key: value for key, value in merged_env.items() if value is not None}
    extra_env = " ".join(shlex.quote(f"{key}={value}") for key, value in sorted(merged_env.items()))
    extra = " ".join(part for part in (extra, *active.args_for("vllm-omni")) if part)
    return (
        f'{_gpus()} {OFFLINE_ENV}{(" " + extra_env) if extra_env else ""} '
        'CUDA_HOME="$PWD/envs/vllm-omni/lib/python3.12/site-packages/nvidia/cu13" '
        'PATH="$PWD/envs/vllm-omni/lib/python3.12/site-packages/nvidia/cu13/bin:$PWD/envs/vllm-omni/bin:$PATH" '
        'LD_LIBRARY_PATH="$PWD/envs/vllm-omni/lib/python3.12/site-packages/nvidia/cu13/lib:$PWD/envs/vllm-omni/lib'
        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" '
        f"exec envs/vllm-omni/bin/vllm serve {shlex.quote(str(model_path))} --omni "
        f"{name_flag}--host 127.0.0.1 --port {port_number} "
        f"--deploy-config {shlex.quote(str(deploy_config))} --disable-log-stats"
        f"{(' ' + extra) if extra else ''}"
    )


def _sglang_ucx_env() -> str:
    """Prefer the SGLang v0.2.0 UCX build without replacing the host library."""
    ucx_root = "$PWD/envs/sglang-omni/ucx-1.20.1/usr"
    cuda_home = "$PWD/envs/sglang-omni/lib/python3.12/site-packages/nvidia/cu13"
    return (
        f'CUDA_HOME="{cuda_home}"; export CUDA_HOME; export CUDACXX="$CUDA_HOME/bin/nvcc"; '
        f'UCX_ROOT="{ucx_root}"; '
        'export UCX_MODULE_DIR="$UCX_ROOT/lib/ucx"; '
        'export LD_LIBRARY_PATH="$CUDA_HOME/lib:$UCX_ROOT/lib:$UCX_ROOT/lib/ucx${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"; '
        'export PATH="$CUDA_HOME/bin:$UCX_ROOT/bin:$PWD/envs/sglang-omni/bin:$PATH"; '
    )


def sglang_omni_serve(
    *,
    model_path: Path,
    config: Path,
    port_number: int,
    model_name: str,
    moe_config_dir: Path | None = None,
    colocate: bool = True,
    extra: str = "",
) -> str:
    """Serve command for Qwen3-Omni on SGLang-Omni.

    `SGLANG_MOE_CONFIG_DIR` points at the H200-tuned Triton MoE kernel configs;
    without it the first Talker forward pays an autotune inside the measured
    window. The config file is the local copy under `exps/configs/`, so
    `external/sglang-omni` stays clean at its release tag. The colocated CLI
    flag is only used with the single-GPU Qwen pipeline; TP deployments use the
    regular speech pipeline config without `--colocate`.
    """
    config = overrides.current().config_for("sglang", hardware.resolve_config(config))
    moe = f'SGLANG_MOE_CONFIG_DIR={shlex.quote(str(moe_config_dir))} ' if moe_config_dir else ""
    active = overrides.current()
    user_env = active.env_words("sglang")
    moe += f"{user_env} " if user_env else ""
    extra = " ".join(part for part in (extra, *active.args_for("sglang")) if part)
    colocate_arg = "--colocate " if colocate else ""
    return (
        f'{_sglang_ucx_env()}PYTHONPATH="$PWD/external/sglang-omni:$PYTHONPATH" '
        f"{OFFLINE_ENV} {_gpus()} {moe}"
        f"exec envs/sglang-omni/bin/sgl-omni serve --model-path {shlex.quote(str(model_path))} "
        f"--config {shlex.quote(str(config))} {colocate_arg}"
        f"--host 127.0.0.1 --port {port_number} --model-name {model_name}"
        f"{(' ' + extra) if extra else ''}"
    )


def sglang_core_serve(
    *,
    model_path: Path,
    pipeline: str,
    pipeline_config: Path,
    port_number: int,
    extra: str = "",
) -> str:
    """Serve command for SGLang core's multimodal_gen launcher."""
    active = overrides.current()
    pipeline_config = active.config_for("sglang", hardware.resolve_config(pipeline_config))
    user_env = active.env_words("sglang")
    extra = " ".join(part for part in (extra, *active.args_for("sglang")) if part)
    return (
        f'{_sglang_ucx_env()}{OFFLINE_ENV} {_gpus()} {user_env + " " if user_env else ""}'
        "exec envs/sglang-omni/bin/python -m sglang.multimodal_gen.runtime.launch_server "
        f"--model-path {shlex.quote(str(model_path))} --pipeline {pipeline} "
        f"--pipeline-config-path {shlex.quote(str(pipeline_config))} "
        f"--host 127.0.0.1 --port {port_number}"
        f"{(' ' + extra) if extra else ''}"
    )


def voxserve_serve(
    *,
    model_path: Path,
    port_number: int,
    max_batch_size: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int,
    extra: str = "--enable-cuda-graph",
) -> str:
    """Serve command for `vox-serve`.

    Sampling parameters are passed explicitly rather than left at VoxServe's
    defaults, because M*'s `OrpheusModelConfig` fixes temperature 0.6, top_p
    0.8, repetition_penalty 1.3 and max_new_tokens 4096 in code. VoxServe's own
    defaults differ (temperature is unset, which its sampler reads as 1.0), so
    without these flags the two engines would be decoding different
    distributions and the token counts -- hence the audio lengths and the
    latencies -- would not be comparable.
    """
    active = overrides.current()
    user_env = active.env_words("voxserve")
    nvcc = host_nvcc()
    extra = " ".join(part for part in (extra, *active.args_for("voxserve")) if part)
    return (
        f'{_gpus()} {OFFLINE_ENV} {user_env + " " if user_env else ""}'
        'CUDA_HOME="$PWD/envs/voxserve/lib/python3.12/site-packages/nvidia/cuda_runtime" '
        f'PYTORCH_NVCC={shlex.quote(nvcc)} CUDACXX={shlex.quote(nvcc)} '
        'FLASHINFER_EXTRA_LDFLAGS="-L$PWD/envs/voxserve/lib/python3.12/site-packages/nvidia/cuda_runtime/lib" '
        'LD_LIBRARY_PATH="$PWD/envs/voxserve/lib/python3.12/site-packages/nvidia/cuda_runtime/lib'
        '${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PATH="$PWD/envs/voxserve/bin:$PATH" '
        f"exec envs/voxserve/bin/vox-serve --model {shlex.quote(str(model_path))} "
        f"--host 127.0.0.1 --port {port_number} --max-batch-size {max_batch_size} "
        f"--temperature {temperature} --top-p {top_p} "
        f"--repetition-penalty {repetition_penalty} --max-tokens {max_tokens}"
        f"{(' ' + extra) if extra else ''}"
    )
