#!/usr/bin/env python3
"""Optional, profile-only stage timing hooks for the three Qwen3-Omni servers.

The hook is activated only when QWEN_A2T_STAGE_TRACE_PATH is set in a service
environment. Stage-profile runs use eager model execution so Python wrappers
observe each request rather than only CUDA-graph capture.
"""
from __future__ import annotations

import contextvars
import fcntl
import functools
import importlib.abc
import importlib.machinery
import inspect
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

_GRAPH_WALK: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "qwen_a2t_graph_walk", default=None
)
_EVENT_SEQUENCE = 0
_EVENT_SEQUENCE_LOCK = threading.Lock()
_PATCHED_MODULES: set[str] = set()


def _next_sequence() -> int:
    global _EVENT_SEQUENCE
    with _EVENT_SEQUENCE_LOCK:
        _EVENT_SEQUENCE += 1
        return _EVENT_SEQUENCE


def _tensor_summaries(arguments: dict[str, Any]) -> dict[str, dict[str, Any]]:
    torch = sys.modules.get("torch")
    if torch is None:
        return {}
    result: dict[str, dict[str, Any]] = {}
    for name, value in arguments.items():
        if isinstance(value, torch.Tensor):
            result[name] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "device": str(value.device),
            }
        elif isinstance(value, dict):
            nested = {
                str(key): {
                    "shape": list(item.shape),
                    "dtype": str(item.dtype),
                    "device": str(item.device),
                }
                for key, item in list(value.items())[:16]
                if isinstance(item, torch.Tensor)
            }
            if nested:
                result[name] = {"tensors": nested}
    return result


def _find_cuda_device(
    torch: Any,
    arguments: dict[str, Any],
    instance: Any | None = None,
) -> Any | None:
    instance_device = getattr(instance, "device", None)
    if instance_device is not None:
        device = torch.device(instance_device)
        if device.type == "cuda":
            return device
    parameters = getattr(instance, "parameters", None)
    if callable(parameters):
        first_parameter = next(iter(parameters()), None)
        if isinstance(first_parameter, torch.Tensor) and first_parameter.device.type == "cuda":
            return first_parameter.device
    for value in arguments.values():
        candidates = value.values() if isinstance(value, dict) else (value,)
        for candidate in candidates:
            if isinstance(candidate, torch.Tensor) and candidate.device.type == "cuda":
                return candidate.device
            if isinstance(candidate, (list, tuple)):
                for item in candidate:
                    if isinstance(item, torch.Tensor) and item.device.type == "cuda":
                        return item.device
    return None


def _request_ids(arguments: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    explicit = arguments.get("request_id")
    if explicit is not None:
        ids.append(str(explicit))
    for name in ("payload", "engine_inputs", "forward_batch"):
        value = arguments.get(name)
        if value is None:
            continue
        request_id = getattr(value, "request_id", None)
        if request_id is not None:
            ids.append(str(request_id))
        request_ids = getattr(value, "request_ids", None)
        if isinstance(request_ids, (list, tuple)):
            ids.extend(str(item) for item in request_ids[:8])
        reqs = getattr(value, "reqs", None)
        if isinstance(reqs, (list, tuple)):
            for request in reqs[:8]:
                rid = (
                    getattr(request, "request_id", None)
                    or getattr(request, "rid", None)
                )
                if rid is not None:
                    ids.append(str(rid))
    return list(dict.fromkeys(ids))


def _bind_arguments(function: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        bound = inspect.signature(function).bind_partial(*args, **kwargs)
    except (TypeError, ValueError):
        return dict(kwargs)
    bound.apply_defaults()
    arguments = dict(bound.arguments)
    arguments.pop("self", None)
    arguments.pop("cls", None)
    return arguments


def _append_event(event: dict[str, Any]) -> None:
    trace_path = os.environ.get("QWEN_A2T_STAGE_TRACE_PATH")
    if not trace_path:
        return
    path = Path(trace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        handle.flush()
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _run_timed(
    function: Callable[..., Any],
    stage: str,
    phase_resolver: Callable[[dict[str, Any]], str | None] | None,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
    torch = sys.modules.get("torch")
    arguments = _bind_arguments(function, args, kwargs)
    phase = phase_resolver(arguments) if phase_resolver else None
    if phase is None:
        phase = _GRAPH_WALK.get()

    instance = args[0] if args else None
    device = (
        _find_cuda_device(torch, arguments, instance)
        if torch is not None
        else None
    )
    start_event = end_event = None
    capturing = False
    if device is not None and torch.cuda.is_available():
        capturing = torch.cuda.is_current_stream_capturing()
        if not capturing:
            stream = torch.cuda.current_stream(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)

    started_unix_ns = time.time_ns()
    started_perf_ns = time.perf_counter_ns()
    result = None
    error: BaseException | None = None
    try:
        result = function(*args, **kwargs)
        return result
    except BaseException as exc:
        error = exc
        raise
    finally:
        cuda_ms = None
        if end_event is not None and start_event is not None:
            end_event.record(stream)
            end_event.synchronize()
            cuda_ms = start_event.elapsed_time(end_event)
        ended_perf_ns = time.perf_counter_ns()
        ended_unix_ns = time.time_ns()
        _append_event(
            {
                "schema_version": 1,
                "sequence": _next_sequence(),
                "framework": os.environ.get("QWEN_A2T_STAGE_TRACE_FRAMEWORK"),
                "stage": stage,
                "phase": phase,
                "method": f"{function.__module__}.{function.__qualname__}",
                "request_ids": _request_ids(arguments),
                "pid": os.getpid(),
                "thread_id": threading.get_ident(),
                "started_unix_ns": started_unix_ns,
                "finished_unix_ns": ended_unix_ns,
                "wall_ms": (ended_perf_ns - started_perf_ns) / 1_000_000,
                "cuda_event_ms": cuda_ms,
                "cuda_graph_capture_call": capturing,
                "input_tensors": _tensor_summaries(arguments),
                "error": (
                    f"{type(error).__name__}: {error}" if error is not None else None
                ),
            }
        )


def _wrap(
    function: Callable[..., Any],
    stage: str,
    phase_resolver: Callable[[dict[str, Any]], str | None] | None = None,
) -> Callable[..., Any]:
    if getattr(function, "__qwen_a2t_stage_trace__", False):
        return function
    if inspect.iscoroutinefunction(function):
        @functools.wraps(function)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            torch = sys.modules.get("torch")
            arguments = _bind_arguments(function, args, kwargs)
            phase = phase_resolver(arguments) if phase_resolver else None
            if phase is None:
                phase = _GRAPH_WALK.get()
            started_unix_ns = time.time_ns()
            started_perf_ns = time.perf_counter_ns()
            result = None
            error: BaseException | None = None
            try:
                result = await function(*args, **kwargs)
                return result
            except BaseException as exc:
                error = exc
                raise
            finally:
                ended_perf_ns = time.perf_counter_ns()
                ended_unix_ns = time.time_ns()
                _append_event(
                    {
                        "schema_version": 1,
                        "sequence": _next_sequence(),
                        "framework": os.environ.get("QWEN_A2T_STAGE_TRACE_FRAMEWORK"),
                        "stage": stage,
                        "phase": phase,
                        "method": f"{function.__module__}.{function.__qualname__}",
                        "request_ids": _request_ids(arguments),
                        "pid": os.getpid(),
                        "thread_id": threading.get_ident(),
                        "started_unix_ns": started_unix_ns,
                        "finished_unix_ns": ended_unix_ns,
                        "wall_ms": (ended_perf_ns - started_perf_ns) / 1_000_000,
                        "cuda_event_ms": None,
                        "cuda_graph_capture_call": False,
                        "input_tensors": _tensor_summaries(arguments),
                        "error": (
                            f"{type(error).__name__}: {error}"
                            if error is not None
                            else None
                        ),
                    }
                )

        setattr(async_wrapper, "__qwen_a2t_stage_trace__", True)
        return async_wrapper

    @functools.wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        token = None
        if stage == "thinker_submodule_step":
            graph_walk = _mstar_walk(_bind_arguments(function, args, kwargs))
            if graph_walk is not None:
                token = _GRAPH_WALK.set(graph_walk)
        try:
            return _run_timed(function, stage, phase_resolver, args, kwargs)
        finally:
            if token is not None:
                _GRAPH_WALK.reset(token)

    setattr(wrapper, "__qwen_a2t_stage_trace__", True)
    return wrapper


def _patch_method(
    module: Any,
    class_name: str,
    method_name: str,
    stage: str,
    phase_resolver: Callable[[dict[str, Any]], str | None] | None = None,
) -> None:
    cls = getattr(module, class_name, None)
    if cls is None:
        return
    method = getattr(cls, method_name, None)
    if method is None:
        return
    setattr(cls, method_name, _wrap(method, stage, phase_resolver))


def _mstar_walk(arguments: dict[str, Any]) -> str | None:
    walk = arguments.get("graph_walk")
    return str(walk) if walk is not None else None


def _model_forward_phase(arguments: dict[str, Any]) -> str | None:
    forward_batch = arguments.get("forward_batch")
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is not None:
        value = getattr(mode, "is_extend", None)
        if callable(value):
            return "prefill" if value() else "decode"
        mode_name = str(mode).lower()
        if "extend" in mode_name or "prefill" in mode_name:
            return "prefill"
        if "decode" in mode_name:
            return "decode"
        return mode_name

    input_ids = arguments.get("input_ids")
    input_id_shape = getattr(input_ids, "shape", None)
    if input_id_shape:
        token_count = int(input_ids.numel())
        return "prefill" if token_count > 1 else "decode"
    for name in ("input_embeds", "inputs_embeds", "positions"):
        value = arguments.get(name)
        shape = getattr(value, "shape", None)
        if shape:
            token_count = int(shape[-2] if name in ("input_embeds", "inputs_embeds") and len(shape) > 1 else shape[-1])
            return "prefill" if token_count > 1 else "decode"
    return None


def _patch_module(module: Any) -> None:
    name = module.__name__
    if name in _PATCHED_MODULES:
        return
    _PATCHED_MODULES.add(name)

    if name == "transformers.models.whisper.feature_extraction_whisper":
        _patch_method(
            module,
            "WhisperFeatureExtractor",
            "__call__",
            "audio_feature_extractor",
        )
    elif name == "mstar.model.qwen3_omni.qwen3_omni_model":
        _patch_method(
            module,
            "Qwen3OmniModel",
            "process_prompt",
            "request_preprocessor_total",
        )
    elif name == "mstar.model.qwen3_omni.submodules":
        audio_cls = getattr(module, "AudioEncoderSubmodule", None)
        thinker_cls = getattr(module, "ThinkerSubmodule", None)
        if audio_cls is not None:
            audio_cls.disable_torch_compile = True
            _patch_method(module, "AudioEncoderSubmodule", "forward", "audio_encoder")
        if thinker_cls is not None:
            thinker_cls.disable_torch_compile = True
            thinker_cls.get_cuda_graph_configs = lambda self, device, tp_world_size=1: []
            thinker_cls.get_piecewise_cuda_graph_configs = (
                lambda self, device, autocast_dtype, tp_world_size=1: {}
            )
            _patch_method(
                module,
                "ThinkerSubmodule",
                "forward_batched",
                "thinker_submodule_step",
                _mstar_walk,
            )
    elif name == "mstar.model.qwen3_omni.components.thinker":
        _patch_method(
            module,
            "Qwen3OmniThinkerModel",
            "forward",
            "thinker_transformer_core",
            _model_forward_phase,
        )
    elif name == "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker":
        _patch_method(module, "Qwen3OmniMoeAudioEncoder", "forward", "audio_encoder")
        _patch_method(
            module,
            "Qwen3MoeLLMModel",
            "forward",
            "thinker_transformer_core",
            _model_forward_phase,
        )
        _patch_method(
            module,
            "Qwen3OmniMoeThinkerForConditionalGeneration",
            "forward",
            "thinker_runner_step",
            _model_forward_phase,
        )
        _patch_method(
            module,
            "Qwen3OmniMoeThinkerMultiModalProcessor",
            "apply",
            "request_preprocessor_total",
        )
    elif name == "sglang_omni.models.qwen3_omni.components.preprocessor":
        _patch_method(
            module,
            "Qwen3OmniPreprocessor",
            "call_impl",
            "request_preprocessor_total",
        )
    elif name == "sglang_omni.models.qwen3_omni.components.audio_encoder":
        _patch_method(module, "Qwen3OmniAudioEncoder", "forward", "audio_encoder")
    elif name == "sglang_omni.models.qwen3_omni.components.sglang_thinker":
        _patch_method(
            module,
            "Qwen3OmniThinkerForCausalLM",
            "forward",
            "thinker_runner_step",
            _model_forward_phase,
        )
    elif name == "sglang.srt.models.qwen3_vl_moe":
        _patch_method(
            module,
            "Qwen3MoeLLMModel",
            "forward",
            "thinker_transformer_core",
            _model_forward_phase,
        )

    _append_event(
        {
            "schema_version": 1,
            "sequence": _next_sequence(),
            "framework": os.environ.get("QWEN_A2T_STAGE_TRACE_FRAMEWORK"),
            "stage": "trace_module_patched",
            "module": name,
            "pid": os.getpid(),
            "thread_id": threading.get_ident(),
            "started_unix_ns": time.time_ns(),
            "finished_unix_ns": time.time_ns(),
            "wall_ms": 0.0,
            "cuda_event_ms": None,
            "cuda_graph_capture_call": False,
            "input_tensors": {},
            "error": None,
        }
    )


class _TraceLoader(importlib.abc.Loader):
    def __init__(self, loader: Any) -> None:
        self._loader = loader

    def create_module(self, spec: Any) -> Any:
        create = getattr(self._loader, "create_module", None)
        return create(spec) if create is not None else None

    def exec_module(self, module: Any) -> None:
        self._loader.exec_module(module)
        _patch_module(module)


class _TraceFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        if fullname not in _TARGET_MODULES:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _TraceLoader(spec.loader)
        return spec


_TARGET_MODULES = {
    "transformers.models.whisper.feature_extraction_whisper",
    "mstar.model.qwen3_omni.qwen3_omni_model",
    "mstar.model.qwen3_omni.submodules",
    "mstar.model.qwen3_omni.components.thinker",
    "vllm_omni.model_executor.models.qwen3_omni.qwen3_omni_moe_thinker",
    "sglang_omni.models.qwen3_omni.components.preprocessor",
    "sglang_omni.models.qwen3_omni.components.audio_encoder",
    "sglang_omni.models.qwen3_omni.components.sglang_thinker",
    "sglang.srt.models.qwen3_vl_moe",
}


def install() -> bool:
    """Install import-time hooks when stage-trace output is explicitly set."""
    if not os.environ.get("QWEN_A2T_STAGE_TRACE_PATH"):
        return False
    if not any(isinstance(finder, _TraceFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _TraceFinder())
    return True
