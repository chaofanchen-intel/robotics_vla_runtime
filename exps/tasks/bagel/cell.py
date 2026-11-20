#!/usr/bin/env python3
"""Cell plans for BAGEL: M* and the two vLLM-Omni deployment recipes."""
from __future__ import annotations

import exps.frameworks as frameworks
from exps.core import overrides, paths
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.lifecycle import ServiceSpec
from exps.core.measure import MeasurementConfig
from exps.core.registry import Case, CellRef
from exps.tasks.bagel.cases import I2T_MAX_TOKENS, SERVED_MODEL_NAME, TASK, TIMESTEP_POINTS

TASK_NAME = TASK.name
SNAPSHOT = (
    paths.MODELS_CACHE
    / "models--ByteDance-Seed--BAGEL-7B-MoT"
    / "snapshots"
    / "5019f57d168e5816e8f3f701b17cc816bb7cf24b"
)
REVISION = "5019f57d168e5816e8f3f701b17cc816bb7cf24b"
MODEL_ID = "ByteDance-Seed/BAGEL-7B-MoT"

MSTAR_CONFIG = paths.CONFIGS / "mstar" / "bagel_single_gpu.yaml"
VLLM_SINGLE_STAGE_CONFIG = paths.CONFIGS / "vllm_omni" / "bagel_single_stage.yaml"
VLLM_TWO_STAGE_CONFIG = paths.CONFIGS / "vllm_omni" / "bagel.yaml"

CONFIG_BY_FRAMEWORK = {
    "mstar": MSTAR_CONFIG,
    "vllm-omni-single-stage": VLLM_SINGLE_STAGE_CONFIG,
    "vllm-omni-two-stage": VLLM_TWO_STAGE_CONFIG,
}


#: vLLM-Omni's BAGEL MoT linears run through a Triton GEMM with per-device tuned tile configs. Without a tuned
#: file every cell logs "No tuned MoT config found" and uses conservative tiles (so the baseline would not be
#: at its best). The folder below is produced by `python -m exps.suite run bagel-mot-tune`; re-run it after a
#: vLLM-Omni upgrade or for a new resolution (the tuned M values are 258 x {1, 3, 4, 12}).
MOT_TUNED_DIR = paths.CONFIGS / "vllm_omni" / "tuned"


def mot_config_filename(dtype: str = "w16a16") -> str | None:
    """File name vLLM-Omni looks for on this GPU (`mot_gemm.get_device_name`: strip non-alphanumerics and the vendor prefix)."""
    import re

    from exps import hardware

    raw = hardware.gpu_name(hardware.current().primary_gpu)
    if not raw:
        return None  # no GPU visible (planning on a CPU box): fall back to "any tuned file"
    name = re.sub(r"[^a-zA-Z0-9]", "", raw.replace(" ", ""))
    for prefix in ("NVIDIA", "AMD"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for key, target in {"A800": "A100", "H800": "H100"}.items():
        if key in name:
            name = name.replace(key, target)
            break
    return f"device_name={name},dtype={dtype}.json"


def vllm_mot_config() -> tuple[str, dict[str, str | None]]:
    """The vLLM-Omni MoT GEMM config for this run: (label, serve environment).

    ``--opt bagel.vllm_mot=tuned`` (default, the best config), ``default`` (untuned tiles, only to measure what tuning
    is worth), or a directory path.
    """
    choice = overrides.current().option("bagel", "vllm_mot", "tuned")
    if choice == "default":
        # None removes the folder every vLLM-Omni cell gets by default (frameworks._tuned_env); returning {} here ran the
        # "untuned" arm of bagel-mot-tuned-vs-default with the tuned config.
        return "untuned (vLLM-Omni default tiles)", {"VLLM_TUNED_CONFIG_FOLDER": None}
    folder = MOT_TUNED_DIR if choice == "tuned" else paths.REPO_ROOT / choice
    wanted = mot_config_filename()
    if wanted is not None and not (folder / wanted).is_file():
        raise FileNotFoundError(
            f"no tuned MoT config for this GPU: expected {folder / wanted}. Run `python -m exps.suite run bagel-mot-tune "
            "--execute` on this machine first (tuned files are per device), or pass `--opt bagel.vllm_mot=default` to "
            "measure vLLM-Omni untuned on purpose"
        )
    if wanted is None and not list(folder.glob("device_name=*.json")):
        raise FileNotFoundError(f"no tuned MoT config in {folder}: run `python -m exps.suite run bagel-mot-tune --execute` first")
    return f"tuned ({paths.relative_to_repo(folder)})", {"VLLM_TUNED_CONFIG_FOLDER": str(folder)}


VLLM_SINGLE_STAGE_GRAPH_CONFIG = paths.CONFIGS / "vllm_omni" / "bagel_single_stage_graph.yaml"


def steps_cache_dir(server_timesteps: int) -> str:
    """Repo-relative HF cache dir holding a config.json with num_timesteps=N (M* ablation)."""
    return f"models/cache_bagel_steps{server_timesteps}"


#: `--opt bagel.topology=cfg-parallel`: the paper's headline topology, one rank per CFG branch (3 GPUs; vLLM-Omni two-stage
#: puts its Thinker beside DiT rank 0 on an 80 GB card, so it needs a 4th card where that does not fit). Hardware overlays
#: (`exps/configs/hardware/<profile>/`) replace these for machines with smaller cards. Not defined for I2T (no CFG).
CFG_PARALLEL_CONFIG_BY_FRAMEWORK = {
    "mstar": paths.CONFIGS / "mstar" / "bagel_cfg_parallel.yaml",
    "vllm-omni-single-stage": paths.CONFIGS / "vllm_omni" / "bagel_single_stage_cfg_parallel.yaml",
    "vllm-omni-two-stage": paths.CONFIGS / "vllm_omni" / "bagel_cfg_parallel.yaml",
}


def topology() -> str | None:
    """`--opt bagel.topology=cfg-parallel` (default: the single-GPU topology of `CONFIG_BY_FRAMEWORK`)."""
    choice = overrides.current().option("bagel", "topology")
    if choice not in (None, "cfg-parallel"):
        raise ValueError(f"--opt bagel.topology={choice!r}: expected 'cfg-parallel' (or leave it unset)")
    return choice


def mstar_prefix_cache_off() -> bool:
    """`--opt bagel.prefix_cache=off`: M* requests carry prefix_cache=false (counterfactual; default on = M*'s own default)."""
    choice = overrides.current().option("bagel", "prefix_cache", "on")
    if choice not in ("on", "off"):
        raise ValueError(f"--opt bagel.prefix_cache={choice!r}: expected 'on' or 'off'")
    return choice == "off"


def neutral_sampling() -> bool:
    """`--opt bagel.sampling=neutral`: every I2T request carries repetition_penalty=1.0, top_k=1, top_p=1.0.

    vLLM-Omni's BAGEL deploy yaml gives the Thinker `repetition_penalty: 1.05` (top_p 0.9, top_k 1) as server defaults, and the
    request only sets temperature, so vLLM decodes with a repetition penalty M* does not apply. Neutral makes the sampler the same.
    """
    choice = overrides.current().option("bagel", "sampling", "default")
    if choice not in ("default", "neutral"):
        raise ValueError(f"--opt bagel.sampling={choice!r}: expected 'default' or 'neutral'")
    return choice == "neutral"


VLLM_BATCH_CONFIG_BY_FRAMEWORK = {
    "vllm-omni-single-stage": paths.CONFIGS / "vllm_omni" / "bagel_single_stage_batch.yaml",
    "vllm-omni-two-stage": paths.CONFIGS / "vllm_omni" / "bagel_batch.yaml",
}


def _deploy_config(framework: str, batch: bool, variant: str | None = None):
    """Concurrency cases use the max_num_seqs=16 variants; C=1 cases never do."""
    if topology() == "cfg-parallel":
        if batch or variant:
            raise ValueError("bagel.topology=cfg-parallel is for the B=1 CFG cases only (no batch / deploy-variant cases)")
        return CFG_PARALLEL_CONFIG_BY_FRAMEWORK[framework]
    if variant == "graph" and framework == "vllm-omni-single-stage":
        return VLLM_SINGLE_STAGE_GRAPH_CONFIG
    return (VLLM_BATCH_CONFIG_BY_FRAMEWORK.get(framework) if batch else None) or CONFIG_BY_FRAMEWORK[framework]


def _service(
    framework: str,
    tag: str,
    batch: bool = False,
    server_timesteps: int | None = None,
    variant: str | None = None,
) -> ServiceSpec:
    port = frameworks.port(framework, TASK_NAME)
    if framework == "mstar":
        command = frameworks.mstar_serve(
            model="bagel", config=_deploy_config(framework, batch, variant), port_number=port, tag=f"bagel-{tag}",
            cache_dir=steps_cache_dir(server_timesteps) if server_timesteps else "models/cache",
        )
    else:
        command = frameworks.vllm_omni_serve(
            model_path=SNAPSHOT,
            deploy_config=_deploy_config(framework, batch, variant),
            port_number=port,
            served_model_name=SERVED_MODEL_NAME,
            env=vllm_mot_config()[1],
        )
    return ServiceSpec(
        command=command,
        health_url=f"http://127.0.0.1:{port}/health",
        ready_timeout_s=1800.0,
    )


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    framework = frameworks.get(ref.framework)
    bagel_task = case.params["task"]
    resolution = case.params["resolution"]
    timesteps = case.params.get("timesteps", TIMESTEP_POINTS)
    cfg_off = bool(case.params.get("cfg_off", False))
    cfg_img_scale = case.params.get("cfg_img_scale")
    max_tokens = case.params.get("max_tokens", I2T_MAX_TOKENS)
    batch_size = int(case.params.get("batch_size", 1))
    server_timesteps = case.params.get("server_timesteps")
    variant = case.params.get("deploy_variant")
    port = frameworks.port(ref.framework, TASK_NAME)
    if topology() == "cfg-parallel":
        from exps import hardware

        hw = hardware.current()
        need = hw.gpus_needed(TASK_NAME, ref.framework)
        if need > max(hw.gpu_count, 1):
            raise SystemExit(f"bagel.topology=cfg-parallel needs {need} GPUs for {ref.framework}; hardware profile {hw.name!r} "
                             f"has {hw.gpu_count}. Plan it on a multi-GPU machine.")
        if bagel_task == "i2t":
            raise SystemExit("bagel.topology=cfg-parallel has no I2T case: CFG applies to image generation only")

    context = {
        "case_label": case.label,
        "bagel_task": bagel_task,
        "resolution": f"{resolution}x{resolution}",
        "ablation": {
            "timesteps": timesteps, "cfg_off": cfg_off, "cfg_img_scale": cfg_img_scale, "max_tokens": max_tokens,
            "batch_size": batch_size,
            "server_timesteps": server_timesteps,
            "deploy_variant": variant,
            "is_ablation": any(
                k in case.params
                for k in ("timesteps", "cfg_off", "cfg_img_scale", "max_tokens", "batch_size", "server_timesteps",
                          "deploy_variant")
            ),
        },
        "comparison_label": TASK.comparison_label,
        "timing_boundary": TASK.timing_boundary,
        "model": {"id": MODEL_ID, "revision": REVISION},
        "config": paths.relative_to_repo(_deploy_config(ref.framework, batch_size > 1, variant)),
        "deployment_recipe": framework.title,
        "vllm_mot_config": vllm_mot_config()[0] if ref.framework != "mstar" else None,
        "request_sampling": "neutral (repetition_penalty 1.0, top_k 1, top_p 1.0)" if neutral_sampling() else "default",
        "mstar_prefix_cache": ("off" if mstar_prefix_cache_off() else "on (M* default)") if ref.framework == "mstar" else None,
        "served_model_name": SERVED_MODEL_NAME if ref.framework != "mstar" else None,
    }
    if topology():
        context["topology"] = topology()
    if ref.framework == "vllm-omni-two-stage":
        context["recipe_caveats"] = {
            "rng": (
                "stage 0 (Thinker) takes the request seed while stage 1 (diffusion) keeps the "
                "YAML seed. Carried forward unchanged from the earlier rounds; affects pixels, "
                "not step count."
            ),
            "sampling_defaults": (
                "the YAML sets max_tokens 2048 and temperature 0.4. The client overrides both "
                "per request and asserts the realised completion length, so these defaults "
                "cannot silently decide how much work the engine does."
            ),
        }

    return CellPlan(
        ref=ref,
        service=_service(
            ref.framework, f"{case.name}-{ref.repeat}", batch_size > 1, server_timesteps, variant
        ),
        client_argv=[
            str(framework.python),
            "-m", "exps.tasks.bagel.client",
            "--framework", ref.framework,
            "--case", case.name,
            "--base-url", f"http://127.0.0.1:{port}",
            "--output-dir", str(config.output_dir),
            "--task", bagel_task,
            "--resolution", str(resolution),
            "--max-tokens", str(max_tokens),
            "--timesteps", str(timesteps),
            "--batch-size", str(batch_size),
            *(["--cfg-off"] if cfg_off else []),
            *(["--no-prefix-cache"] if ref.framework == "mstar" and mstar_prefix_cache_off() else []),
            *(["--cfg-img-scale", str(cfg_img_scale)] if cfg_img_scale is not None else []),
            *protocol_argv(config),
        ],
        context=context,
        provenance_repos=[framework.external_repo, "mstar"],
        client_env={"HF_HOME": str(paths.MODELS_CACHE), "HF_HUB_OFFLINE": "1",
                    **({"EXPS_BAGEL_SAMPLING": "neutral"} if neutral_sampling() else {})},
    )
