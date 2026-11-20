#!/usr/bin/env python3
"""Drive every runtime through one request path, and map its metrics to ours.

The fairness problem this solves
--------------------------------
In the previous round each engine was driven by whatever client was convenient:
M* through its own in-tree `benchmark.runner`, VoxServe through a bespoke
script, Qwen3-Omni through a third. Those clients did not agree on when the
timer started, what counted as the first byte, which text they sent, or which
metrics they emitted -- M* Orpheus reported no real-time factor at all. Any
ratio computed across them was partly a comparison of clients.

Here all four served runtimes go through **M*'s own `benchmark.request`
adapters**, imported read-only from `external/mstar`. That package already
contains an adapter per runtime (`OurSystem`, `VLLMOmni`, `SGLangOmni`,
`VoxServe`), each with the same instrumentation contract: the timer is anchored
immediately before `session.post(...)`, per-modality first-chunk times and
every chunk arrival are recorded, and byte and token counts are tallied the
same way. Using the paper harness's own adapters means no engine is measured by
a client written with it in mind. The pinned M* adapter receives a minimal,
recorded patch for BAGEL vLLM-Omni per-request kwargs and served-model name;
the runtime imports that verified checkout without editing it during a run.

What this module adds on top
----------------------------
1. **One request at a time, under our control**, so the per-request cooldown
   gate can run between requests. `benchmark.runner` owns its own loop and
   offers no hook, which is why the loop lives here instead.
2. **Translation into `core.metrics`' schema**, so BAGEL, Orpheus, Qwen3-Omni
   and Pi0.5 rows are directly comparable and every task reports RTF, TTFT,
   TTFA and ITL wherever they are meaningful.

Timing boundary
---------------
`e2e_ms` is from immediately before the HTTP send to the end of the response
body, as anchored by the adapter. Decoding, validation, hashing and disk writes
all happen afterwards and are never inside the measured interval.
"""
from __future__ import annotations

import asyncio
import importlib
import os
import sys
from pathlib import Path
from typing import Any

from . import metrics as metrics_mod
from . import paths

#: M* PCM output is mono 16-bit at this rate (benchmark/request.py:374).
MSTAR_PCM_SAMPLE_RATE_HZ = 24_000
MSTAR_PCM_BYTES_PER_SAMPLE = 2

#: Framework name -> the adapter class in M*'s benchmark.request.
ADAPTER_BY_FRAMEWORK = {
    "mstar": "OurSystem",
    "vllm-omni": "VLLMOmni",
    "sglang-omni": "SGLangOmni",
    "voxserve": "VoxServe",
}


def ensure_mstar_benchmark_importable() -> None:
    """Put `external/mstar` on `sys.path` so `benchmark.*` resolves.

    Import only -- nothing in `external/mstar` is written, and the harness
    never runs from inside that checkout. The package imports cleanly under all
    four runtime virtualenvs, which is what makes a single request path
    possible across incompatible Torch builds.
    """
    mstar = str(paths.EXTERNAL / "mstar")
    if mstar not in sys.path:
        sys.path.insert(0, mstar)


def load_benchmark_module(name: str):
    ensure_mstar_benchmark_importable()
    return importlib.import_module(name)


def inference_system(framework: str):
    """Instantiate M*'s adapter for one runtime.

    The lookup is by adapter name, not framework name: BAGEL's single-stage and
    two-stage recipes are different services but speak the same protocol, so
    both resolve to `VLLMOmni`.
    """
    try:
        import exps.frameworks as frameworks_mod

        class_name = ADAPTER_BY_FRAMEWORK[frameworks_mod.get(framework).adapter_name]
    except KeyError:
        raise KeyError(
            f"no benchmark.request adapter for framework {framework!r}; "
            f"available: {', '.join(sorted(ADAPTER_BY_FRAMEWORK))}"
        ) from None
    request_module = load_benchmark_module("benchmark.request")
    return getattr(request_module, class_name)()


def model_instance(model_type: str, **kwargs):
    """Instantiate M*'s `Model` for a workload, e.g. `vjepa2ac`, `bagel`."""
    base = load_benchmark_module("benchmark.base")
    return base.ModelType(model_type).inst(**kwargs)


def request_type(name: str):
    base = load_benchmark_module("benchmark.base")
    return base.RequestType(name)


def new_session(*, total_timeout_s: float | None = None, connection_limit: int = 64):
    """An aiohttp session sized for the largest batch any task issues."""
    import aiohttp

    return aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=total_timeout_s, sock_read=600),
        connector=aiohttp.TCPConnector(limit=connection_limit),
    )


def response_text(result) -> str:
    """The text channel of a response as one string.

    M*'s benchmark adapter keeps `_text_chunks` as raw bytes, the OpenAI-style
    adapters keep str; a UTF-8 sequence may also be split between chunks, so
    bytes are joined before they are decoded.
    """
    chunks = getattr(result, "_text_chunks", None) or []
    raw = b"".join(c if isinstance(c, bytes) else c.encode("utf-8") for c in chunks)
    return raw.decode("utf-8", errors="replace")


def _pcm_duration_s(pcm_bytes: int) -> float | None:
    if not pcm_bytes:
        return None
    return pcm_bytes / (MSTAR_PCM_SAMPLE_RATE_HZ * MSTAR_PCM_BYTES_PER_SAMPLE)


def _primary_modality(result) -> str | None:
    """Whichever streamed modality best represents the work of this request."""
    for modality in ("audio", "video", "action", "image", "text"):
        if modality in result.chunk_arrivals:
            return modality
    return None


def to_metric_row(result, *, started_at_monotonic: float | None = None) -> dict[str, Any]:
    """Map one `benchmark.request.RequestMetrics` into our per-request schema.

    Every field in `metrics.REQUEST_METRIC_FIELDS` is present; the ones this
    request could not produce are ``None``, never missing.
    """
    row = metrics_mod.empty_request_metrics()
    status = getattr(result.status, "value", str(result.status))
    succeeded = status.lower() == "success"

    ttft = result.ttft or {}
    audio_bytes = result._audio_pcm.tell() if hasattr(result, "_audio_pcm") else 0  # noqa: SLF001
    audio_duration = _pcm_duration_s(audio_bytes)

    primary = _primary_modality(result)
    arrivals = (result.chunk_arrivals or {}).get(primary or "", [])
    itl = [
        (later - earlier) * 1000.0
        for earlier, later in zip(arrivals, arrivals[1:], strict=False)
    ]

    row.update(
        {
            "e2e_ms": (result.e2e_latency * 1000.0) if result.e2e_latency else None,
            "ttfb_ms": (min(ttft.values()) * 1000.0) if ttft else None,
            "ttft_ms": (ttft["text"] * 1000.0) if "text" in ttft else None,
            "ttfa_ms": (ttft["audio"] * 1000.0) if "audio" in ttft else None,
            "itl_ms": itl,
            "output_tokens": result.output_text_tokens or None,
            "output_bytes": sum((result.output_bytes or {}).values()) or None,
            "audio_duration_s": audio_duration,
        }
    )
    row["rtf"] = metrics_mod.compute_rtf(row["e2e_ms"], audio_duration)
    row.update(
        {
            "success": succeeded,
            "error": result.error,
            "status": status,
            "primary_modality": primary,
            "per_modality_ttfb_ms": {name: value * 1000.0 for name, value in ttft.items()},
            "per_modality_bytes": dict(result.output_bytes or {}),
            "per_modality_chunks": dict(result.response_chunks or {}),
            "input_tokens": result.input_tokens or None,
            "started_at_monotonic": started_at_monotonic,
        }
    )
    return row


async def _send_one(system, session, req_input, base_url, request_id, model, model_kwargs):
    return await system.send_request(
        session=session,
        req_input=req_input,
        base_url=base_url,
        request_id=request_id,
        model=model,
        **({"additional_model_kwargs": model_kwargs} if model_kwargs else {}),
    )


def send_sequential(
    system,
    req_input,
    *,
    base_url: str,
    request_id: int,
    model,
    model_kwargs: dict[str, Any] | None = None,
    timeout_s: float | None = None,
):
    """Issue exactly one request and return its raw `RequestMetrics`.

    A fresh event loop and session per request keeps connection reuse from
    carrying state across a cooldown gate, which would otherwise make the first
    request after a gate structurally different from the rest.
    """

    async def runner():
        async with new_session(total_timeout_s=timeout_s) as session:
            return await _send_one(
                system, session, req_input, base_url, request_id, model, model_kwargs
            )

    return asyncio.run(runner())


def send_batch(
    system,
    req_inputs: list,
    *,
    base_url: str,
    first_request_id: int,
    model,
    model_kwargs: dict[str, Any] | None = None,
    timeout_s: float | None = None,
):
    """Issue a wave of requests concurrently; used only by the throughput phase."""

    async def runner():
        async with new_session(
            total_timeout_s=timeout_s, connection_limit=max(64, len(req_inputs) + 8)
        ) as session:
            return await asyncio.gather(
                *(
                    _send_one(
                        system, session, req_input, base_url, first_request_id + offset,
                        model, model_kwargs,
                    )
                    for offset, req_input in enumerate(req_inputs)
                )
            )

    return asyncio.run(runner())


def write_request_outputs(result, output_dir: Path, label: str) -> int:
    """Persist a request's generated artefacts, after the timer has stopped.

    Outputs are kept because a latency number nobody can audit is not evidence:
    an image of the wrong size or an audio clip of the wrong utterance is only
    discoverable if the artefact survives.
    """
    target = Path(output_dir) / label
    target.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    try:
        return result.write_files(str(target))
    except Exception:  # noqa: BLE001 - never fail a measured request on a disk write
        return 0
    finally:
        if os.getcwd() != previous:
            os.chdir(previous)
