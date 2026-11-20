#!/usr/bin/env python3
"""Profile the M* Qwen3-Omni AudioEncoder on one frozen FLEURS clip.

This is a component-only diagnostic, not an E2E benchmark. It reuses M*'s
AudioEncoder implementation, HF feature extraction and checkpoint prefix,
then compares FlashAttention 2 with Transformers SDPA on identical tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
MSTAR_SOURCE = REPO_ROOT / "external" / "mstar"
if str(MSTAR_SOURCE) not in sys.path:
    sys.path.insert(0, str(MSTAR_SOURCE))

from exps.core import gpu, paths, thermal
from exps.core.cell import gpu_lock
from exps.core.manifest import write_json
from exps.tasks.qwen_a2t.audio_io import read_audio_wav

FIXTURE_MANIFEST = paths.INPUTS / "QW-A2T" / "fleurs_test_100_manifest.json"
AUDIO_ROOT = paths.INPUTS / "QW-A2T" / "audio"
MODEL_SNAPSHOT = (
    paths.MODELS_CACHE
    / "models--Qwen--Qwen3-Omni-30B-A3B-Instruct"
    / "snapshots"
    / "26291f793822fb6be9555850f06dfe95f2d7e695"
)
PRECONDITION_DTYPE = "torch.bfloat16"
MSTAR_FA3_ATTENTION_IMPLEMENTATION = "mstar_flash_attention_3"


def _fa3_varlen_attention_forward(
    module: Any,
    query: Any,
    key: Any,
    value: Any,
    attention_mask: Any,
    dropout: float = 0.0,
    scaling: float | None = None,
    sliding_window: int | None = None,
    softcap: float | None = None,
    is_causal: bool | None = None,
    **kwargs: Any,
) -> tuple[Any, None]:
    """Transformers attention-interface adapter for the FA3 varlen kernel."""
    import torch
    from fa3_fwd_interface import flash_attn_varlen_func

    if query.ndim != 4 or query.shape[0] != 1:
        raise ValueError(
            "Qwen3-Omni's FA3 AudioEncoder probe expects one packed batch with "
            f"[1, heads, tokens, head_dim] query layout; got {tuple(query.shape)}"
        )
    if attention_mask is not None:
        raise ValueError("the packed FA3 AudioEncoder path does not accept an attention mask")
    if dropout != 0.0:
        raise ValueError("the inference-only FA3 AudioEncoder probe requires zero dropout")
    if sliding_window is not None:
        raise ValueError("the FA3 AudioEncoder probe does not support sliding-window attention")
    if kwargs.get("output_attentions", False):
        raise ValueError("the FA3 AudioEncoder probe does not return attention weights")

    cu_seqlens_q = kwargs.get("cu_seq_lens_q")
    cu_seqlens_k = kwargs.get("cu_seq_lens_k")
    max_length_q = kwargs.get("max_length_q")
    max_length_k = kwargs.get("max_length_k")
    if any(value is None for value in (cu_seqlens_q, cu_seqlens_k, max_length_q, max_length_k)):
        raise ValueError("the FA3 AudioEncoder path requires packed sequence lengths")

    def _as_cu_seqlens(lengths: Any) -> Any:
        if not isinstance(lengths, torch.Tensor) or lengths.ndim != 1:
            raise TypeError("cu_seq_lens_q/k must be one-dimensional tensors")
        return lengths.to(device=query.device, dtype=torch.int32).contiguous()

    def _as_max_length(length: Any) -> int:
        if isinstance(length, torch.Tensor):
            if length.numel() != 1:
                raise ValueError("max_length_q/k must be a scalar")
            return int(length.item())
        return int(length)

    if is_causal is None:
        is_causal = bool(getattr(module, "is_causal", False))

    batch_size, num_heads, sequence_length, head_dim = query.shape
    q = query.transpose(1, 2).reshape(-1, num_heads, head_dim).contiguous()
    k = key.transpose(1, 2).reshape(-1, key.shape[1], key.shape[-1]).contiguous()
    v = value.transpose(1, 2).reshape(-1, value.shape[1], value.shape[-1]).contiguous()
    output = flash_attn_varlen_func(
        q,
        k,
        v,
        _as_cu_seqlens(cu_seqlens_q),
        _as_cu_seqlens(cu_seqlens_k),
        _as_max_length(max_length_q),
        _as_max_length(max_length_k),
        softmax_scale=scaling,
        causal=is_causal,
        softcap=softcap or 0.0,
    )
    if isinstance(output, tuple):
        output = output[0]
    return output.reshape(batch_size, sequence_length, num_heads, output.shape[-1]), None


def _register_fa3_attention_interface() -> str:
    import torch
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ALL_ATTENTION_FUNCTIONS.register(
        MSTAR_FA3_ATTENTION_IMPLEMENTATION,
        torch.compiler.disable(_fa3_varlen_attention_forward),
    )
    return MSTAR_FA3_ATTENTION_IMPLEMENTATION


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-relpath",
        required=True,
        help="Exact audio_relpath from the frozen FLEURS manifest.",
    )
    parser.add_argument(
        "--backend",
        choices=("flash_attention_2", "flash_attention_3", "sdpa"),
        required=True,
        help=(
            "AudioEncoder attention implementation. FA3 uses M*'s fa3-fwd varlen adapter."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("compiled", "eager"),
        default="compiled",
        help="compiled matches M*'s default torch.compile wrapper; eager is a control.",
    )
    parser.add_argument("--preconditions", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--measured", type=int, default=3)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Unique output directory. Defaults to exps/runs/.",
    )
    parser.add_argument("--cooldown-target-c", type=float, default=70.0)
    parser.add_argument("--cooldown-continuous-s", type=float, default=30.0)
    parser.add_argument("--cooldown-timeout-s", type=float, default=900.0)
    parser.add_argument("--loaded-idle-continuous-s", type=float, default=30.0)
    parser.add_argument("--loaded-idle-timeout-s", type=float, default=1800.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def load_fixture_index() -> dict[str, dict[str, Any]]:
    if not FIXTURE_MANIFEST.is_file():
        raise FileNotFoundError(f"FLEURS manifest missing: {FIXTURE_MANIFEST}")
    manifest = json.loads(FIXTURE_MANIFEST.read_text(encoding="utf-8"))
    configs = manifest["dataset"]["configs"]
    index: dict[str, dict[str, Any]] = {}
    for config in configs.values():
        for sample in config["samples"]:
            relpath = sample["audio_relpath"]
            if relpath in index:
                raise ValueError(f"duplicate audio_relpath in FLEURS manifest: {relpath}")
            index[relpath] = sample
    return index


def resolve_audio(
    audio_relpath: str,
    fixture_index: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], Path]:
    relative_path = Path(audio_relpath)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"audio_relpath must stay under the fixture audio root: {audio_relpath}")
    try:
        sample = fixture_index[audio_relpath]
    except KeyError:
        raise ValueError(
            f"audio_relpath is not in the frozen FLEURS manifest: {audio_relpath}"
        ) from None

    path = AUDIO_ROOT / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"FLEURS audio missing: {path}")
    expected_sha256 = sample.get("audio_sha256")
    if expected_sha256:
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"{audio_relpath} hash mismatch: expected {expected_sha256}, "
                f"found {actual_sha256}"
            )
    return sample, path


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _timed_forward(
    *,
    forward: Any,
    audio_features: Any,
    audio_seqlens: Any,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    started = time.perf_counter()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        result = forward(
            "prefill_audio",
            None,
            audio_features=audio_features,
            audio_seqlens=audio_seqlens,
        )
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - started) * 1000.0
    cuda_ms = start_event.elapsed_time(end_event)
    embeddings = result["audio_embeds"][0]
    finite = bool(torch.isfinite(embeddings).all().item())
    if not finite:
        raise RuntimeError("AudioEncoder returned non-finite embeddings")
    return {
        "audio_encoder_wall_ms": wall_ms,
        "audio_encoder_cuda_ms": cuda_ms,
        "embedding_shape": list(embeddings.shape),
        "embedding_mean": float(embeddings.float().mean().item()),
        "embedding_std": float(embeddings.float().std(unbiased=False).item()),
        "embedding_finite": finite,
        "_embedding": embeddings.detach(),
    }


def _run(args: argparse.Namespace, sample: dict[str, Any], audio_path: Path) -> Path:
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeAudioEncoder,
    )

    from mstar.model.qwen3_omni.components.attention import patch_hf_fa2_int_maxlen
    from mstar.model.qwen3_omni.config import Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.submodules import AudioEncoderSubmodule
    from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the AudioEncoder backend probe")
    if args.gpu_index < 0 or args.gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"gpu-index {args.gpu_index} is outside the visible CUDA device count "
            f"{torch.cuda.device_count()}"
        )
    device = torch.device(f"cuda:{args.gpu_index}")
    model_path = MODEL_SNAPSHOT
    if not model_path.is_dir():
        raise FileNotFoundError(f"local Qwen3-Omni snapshot missing: {model_path}")

    run_dir = args.output_dir
    if run_dir is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = paths.RUNS / "qwen_a2t_audio_encoder" / f"{stamp}-pid{os.getpid()}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "cooldowns").mkdir()
    (run_dir / "embeddings").mkdir()

    audio, sample_rate = read_audio_wav(audio_path)
    processor = AutoProcessor.from_pretrained(
        str(model_path),
        cache_dir=str(paths.MODELS_CACHE),
        trust_remote_code=True,
    )
    feature_extractor = processor.feature_extractor
    expected_rate = getattr(feature_extractor, "sampling_rate", 16_000)
    if expected_rate != sample_rate:
        raise ValueError(
            f"fixture rate {sample_rate} does not match feature extractor rate {expected_rate}"
        )
    feature_started = time.perf_counter()
    encoded = feature_extractor(
        audio,
        sampling_rate=sample_rate,
        padding=True,
        truncation=False,
        return_attention_mask=True,
        return_tensors="pt",
    )
    audio_features = (
        encoded["input_features"]
        .permute(0, 2, 1)[encoded["attention_mask"].bool()]
        .permute(1, 0)
    )
    audio_seqlens = encoded["attention_mask"].sum(-1).to(torch.long)
    feature_extraction_ms = (time.perf_counter() - feature_started) * 1000.0
    audio_features = audio_features.to(device)
    audio_seqlens = audio_seqlens.to(device)

    if args.backend == "flash_attention_2":
        patch_hf_fa2_int_maxlen()
        hf_attention_implementation = "flash_attention_2"
    elif args.backend == "flash_attention_3":
        hf_attention_implementation = "sdpa"
    else:
        hf_attention_implementation = "sdpa"
    hf_config = AutoConfig.from_pretrained(
        str(model_path),
        cache_dir=str(paths.MODELS_CACHE),
        trust_remote_code=True,
    )
    audio_config = hf_config.thinker_config.audio_config
    encoder = Qwen3OmniMoeAudioEncoder._from_config(
        audio_config,
        attn_implementation=hf_attention_implementation,
    )
    if args.backend == "flash_attention_3":
        hf_attention_implementation = _register_fa3_attention_interface()
        encoder.config._attn_implementation = hf_attention_implementation
    load_weights_from_hf_shards(
        repo_dir=model_path,
        modules=[ModuleAndPrefix(encoder, prefix="thinker.audio_tower")],
        device=str(device),
    )
    encoder.eval()
    submodule = AudioEncoderSubmodule(
        audio_encoder=encoder,
        config=Qwen3OmniModelConfig.from_pretrained(str(model_path)),
    ).to(device)

    if args.mode == "compiled":
        forward = torch.compile(submodule.forward, fullgraph=False, dynamic=None)
    else:
        forward = submodule.forward

    metadata = {
        "schema_version": 1,
        "probe": "qwen3_omni_audio_encoder_component",
        "measurement_scope": (
            "AudioEncoderSubmodule.forward only; excludes audio file I/O, HF feature extraction "
            "from encoder wall/CUDA spans, Thinker, API and response handling."
        ),
        "backend": args.backend,
        "transformers_attention_implementation": hf_attention_implementation,
        "attention_kernel_adapter": (
            "fa3-fwd varlen (probe override of Transformers flash_attention_2 interface)"
            if args.backend == "flash_attention_3"
            else args.backend
        ),
        "mode": args.mode,
        "autocast_dtype": PRECONDITION_DTYPE,
        "audio_relpath": sample["audio_relpath"],
        "audio_sha256": sample.get("audio_sha256"),
        "audio_duration_seconds": sample.get("duration_seconds"),
        "audio_features_shape": list(audio_features.shape),
        "audio_feature_lengths": audio_seqlens.cpu().tolist(),
        "feature_extraction_ms_cpu": feature_extraction_ms,
        "preconditions": args.preconditions,
        "warmups": args.warmups,
        "measured": args.measured,
        "model_revision": "26291f793822fb6be9555850f06dfe95f2d7e695",
        "torch_version": torch.__version__,
        "transformers_version": importlib.metadata.version("transformers"),
        "gpu": gpu.probe(args.gpu_index),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(run_dir / "metadata.json", metadata)
    results_path = run_dir / "encoder_requests.jsonl"
    results_path.write_text("", encoding="utf-8")

    def run_one(phase: str, sequence: int, cooldown: dict[str, Any] | None) -> dict[str, Any]:
        before = gpu.sample(args.gpu_index)
        monitor = thermal.RequestThermalMonitor(args.gpu_index, period_seconds=0.05)
        with monitor:
            result = _timed_forward(
                forward=forward,
                audio_features=audio_features,
                audio_seqlens=audio_seqlens,
                device=device,
                torch=torch,
            )
        verdict = monitor.verdict()
        embedding = result.pop("_embedding")
        embedding_path = run_dir / "embeddings" / f"{phase}_{sequence:03d}.pt"
        torch.save(embedding.to(device="cpu"), embedding_path)
        row = {
            "phase": phase,
            "sequence": sequence,
            "success": True,
            "backend": args.backend,
            "mode": args.mode,
            "audio_relpath": sample["audio_relpath"],
            **result,
            "embedding_output_file": str(embedding_path.relative_to(run_dir)),
            "embedding_dtype": str(embedding.dtype),
            "cooldown_waited_s": (
                cooldown.get("waited_seconds") if cooldown is not None else None
            ),
            "cooldown_timed_out": (
                bool(cooldown.get("timed_out")) if cooldown is not None else None
            ),
            "temperature_c_before": before.get("temperature_c"),
            "sm_clock_mhz_before": before.get("clocks_sm_mhz"),
            "sm_clock_mhz_nominal": before.get("clocks_max_sm_mhz"),
            "thermally_clean": verdict.get("thermally_clean"),
            "thermal_monitor": verdict,
        }
        _append_jsonl(results_path, row)
        return row

    loaded_idle = thermal.loaded_idle_gate(
        run_dir / "loaded_idle_after_load.csv",
        gpu_index=args.gpu_index,
        max_temperature_celsius=args.cooldown_target_c,
        continuous_seconds=args.loaded_idle_continuous_s,
        timeout_seconds=args.loaded_idle_timeout_s,
    )
    if loaded_idle.get("status") != "passed":
        raise RuntimeError(f"loaded-idle gate failed after AudioEncoder load: {loaded_idle}")

    precondition_rows = [
        run_one("precondition", index, None)
        for index in range(1, args.preconditions + 1)
    ]
    warmup_rows = [
        run_one("warmup", index, None)
        for index in range(1, args.warmups + 1)
    ]
    if precondition_rows or warmup_rows:
        after_warmup = thermal.loaded_idle_gate(
            run_dir / "loaded_idle_after_warmup.csv",
            gpu_index=args.gpu_index,
            max_temperature_celsius=args.cooldown_target_c,
            continuous_seconds=args.loaded_idle_continuous_s,
            timeout_seconds=args.loaded_idle_timeout_s,
        )
        if after_warmup.get("status") != "passed":
            raise RuntimeError(f"loaded-idle gate failed after warmup: {after_warmup}")

    measured_rows = []
    for index in range(1, args.measured + 1):
        cooldown = thermal.cooldown_gate(
            gpu_index=args.gpu_index,
            target_celsius=args.cooldown_target_c,
            continuous_seconds=args.cooldown_continuous_s,
            timeout_seconds=args.cooldown_timeout_s,
            output=run_dir / "cooldowns" / f"measured_{index:03d}.csv",
        )
        measured_rows.append(run_one("measured", index, cooldown))

    eligible = [row for row in measured_rows if not row["cooldown_timed_out"]]
    summary = {
        "schema_version": 1,
        "metadata": "metadata.json",
        "requests": "encoder_requests.jsonl",
        "measured_count": len(measured_rows),
        "cooldown_eligible_count": len(eligible),
        "audio_encoder_wall_ms_median": (
            statistics.median(row["audio_encoder_wall_ms"] for row in eligible)
            if eligible
            else None
        ),
        "audio_encoder_cuda_ms_median": (
            statistics.median(row["audio_encoder_cuda_ms"] for row in eligible)
            if eligible
            else None
        ),
        "cold_precondition_wall_ms": (
            precondition_rows[0]["audio_encoder_wall_ms"] if precondition_rows else None
        ),
        "cold_precondition_cuda_ms": (
            precondition_rows[0]["audio_encoder_cuda_ms"] if precondition_rows else None
        ),
    }
    write_json(run_dir / "summary.json", summary)
    print(json.dumps({"output_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))
    if not eligible:
        raise RuntimeError(
            f"no cooldown-eligible encoder measurements; inspect {run_dir / 'cooldowns'}"
        )
    return run_dir


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.preconditions < 0 or args.warmups < 0 or args.measured < 1:
        parser.error("preconditions/warmups must be >= 0 and measured must be >= 1")

    fixture_index = load_fixture_index()
    sample, audio_path = resolve_audio(args.audio_relpath, fixture_index)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "backend": args.backend,
                    "mode": args.mode,
                    "audio_relpath": sample["audio_relpath"],
                    "audio_duration_seconds": sample.get("duration_seconds"),
                    "audio_sha256": sample.get("audio_sha256"),
                    "audio_file_exists": audio_path.is_file(),
                    "preconditions": args.preconditions,
                    "warmups": args.warmups,
                    "measured": args.measured,
                    "model_snapshot_exists": MODEL_SNAPSHOT.is_dir(),
                    "dry_run": True,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    with gpu_lock(timeout_s=0.0):
        run_dir = _run(args, sample, audio_path)
    return 0 if (run_dir / "summary.json").is_file() else 1


if __name__ == "__main__":
    raise SystemExit(main())
