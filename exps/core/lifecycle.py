#!/usr/bin/env python3
"""Fresh-service lifecycle: gate, start, hand off to the client, stop, gate.

Every cell gets its own service process. That is expensive -- a Qwen3-Omni
start costs about 150 s -- but it is the only way to separate engine behaviour
from start-to-start variance, which on this host has been measured at 1.39x on
an identical workload. Reusing one service across cells would fold that
variance into whichever engine happened to run second.

Division of labour
------------------
This module runs under the **harness** interpreter and owns everything that
does not need a runtime's own dependencies: the service-level gates, process
supervision, GPU telemetry, and teardown.

The measured loop runs in a **separate client process** under the runtime's own
virtualenv (`envs/<name>/bin/python`), because M*, vLLM-Omni, SGLang-Omni and
VoxServe require incompatible Torch builds: M* main requires
`2.10 <= torch < 2.13`, SGLang-Omni v0.2.0 pins `2.14.1`, vLLM-Omni v0.32.0
is tested with `2.13.0`/CUDA 13.0, and VoxServe pins `2.8.0`. A single venv
cannot satisfy these upstream constraints. The client owns the per-request
cooldown gates, since it is the only process that knows when one request has
ended and the next is about to begin.

Teardown is not optional
------------------------
The shutdown path kills the whole process group, then polls `nvidia-smi` until
no compute app remains, because a leaked worker holding 40 GiB makes the next
cell's prestart gate wait forever on a memory condition it cannot satisfy. The
post-shutdown gate is in a `finally` so an aborted run still leaves the card in
a known state for whatever runs next.
"""
from __future__ import annotations

import contextlib
import dataclasses
import os
import shlex
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from exps import hardware

from . import gpu, overrides, thermal
from .manifest import write_json


class ServiceStartupError(RuntimeError):
    """The service died or never became ready; the cell cannot produce a number."""


@dataclass
class ServiceSpec:
    """Everything needed to start one runtime and know when it is usable."""

    command: str
    health_url: str
    health_mode: str = "http"  # "http" or "tcp"
    ready_timeout_s: float = 1800.0
    post_ready_grace_s: float = 0.0
    #: when set, the service is only ready once this text also appears in server.log. For servers whose HTTP port (and /health)
    #: comes up before the model does: VoxServe answers /health 4 s after launch and loads the model and captures CUDA graphs
    #: for another minute, so the first request used to hit a scheduler that was not there yet and time out.
    ready_log_marker: str | None = None
    shutdown_grace_s: float = 30.0
    env: dict[str, str] = field(default_factory=dict)


def nsys_wrap(command: str, run_root: Path) -> tuple[str, float]:
    """Wrap a serve command in `nsys profile` when `--nsys-dir` is set (profiling runs only).

    The report is named `<framework>-<case>-<rNN>` after the cell directory. Returns (command,
    shutdown_grace_s): nsys needs time to write the report after the group receives SIGTERM.
    Unset by default, so ordinary cells are untouched.
    """
    nsys = overrides.current().nsys
    if nsys is None:
        return command, 0.0
    out_dir = nsys.out_dir
    # `node` shows per-kernel records inside CUDA graphs (needed for graph-decoded paths) but its trace
    # volume truncated the vLLM-Omni two-stage reports; `--nsys-graph-trace graph` is the fallback.
    graph_trace = nsys.graph_trace
    # Multi-process engines (vLLM-Omni stages) lose their last unflushed CUDA buffers when the parent kills
    # them: two-stage I2T reports stopped after ~30 s of kernels while runtime calls continued. A short
    # flush interval bounds the loss at the price of some profiler overhead (busy/launch counts unaffected).
    flush = f"--cuda-flush-interval={int(nsys.flush_ms)} " if nsys.flush_ms else ""
    os.makedirs(out_dir, exist_ok=True)
    # Cells run under `<campaign>/<TASK>/<framework>/<case>/<rNN>`; some runners instead use
    # `_service_lifecycles/<task>_<framework>_<case>_<rNN>` (names carry hyphens, never underscores).
    if run_root.parent.name == "_service_lifecycles":
        name = "-".join(run_root.name.split("_")[-3:])
    else:
        name = "-".join(run_root.parts[-3:])
    return (
        f"nsys profile --trace=cuda,nvtx --cuda-graph-trace={graph_trace} {flush}--sample=none --cpuctxsw=none --force-overwrite=true "
        f"--stop-on-exit=true -o {shlex.quote(os.path.join(out_dir, name))} bash -c {shlex.quote(command)}",
        300.0,
    )


def _tail(path: Path, limit: int = 4000) -> str:
    try:
        return path.read_bytes()[-limit:].decode(errors="replace")
    except OSError:
        return "<no server log>"


def _http_ready(url: str, timeout: float = 5.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _tcp_ready(url: str, timeout: float = 5.0) -> bool:
    from urllib.parse import urlsplit

    parsed = urlsplit(url if "://" in url else f"tcp://{url}")
    if parsed.hostname is None or parsed.port is None:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=timeout):
            return True
    except OSError:
        return False


def wait_until_ready(spec: ServiceSpec, process: subprocess.Popen, server_log: Path) -> float:
    """Block until the service answers, failing fast if it exits first.

    Polling a health URL while ignoring the process is how a crashed service
    turns into a 30-minute timeout. Checking `poll()` each round surfaces the
    traceback in seconds instead.
    """
    probe = _tcp_ready if spec.health_mode == "tcp" else _http_ready
    started = time.monotonic()
    deadline = started + spec.ready_timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise ServiceStartupError(
                f"service exited with code {process.returncode} before becoming ready.\n"
                f"Last {4000} bytes of server.log:\n{_tail(server_log)}"
            )
        if probe(spec.health_url) and (spec.ready_log_marker is None or spec.ready_log_marker in _tail(server_log, 1_000_000)):
            return time.monotonic() - started
        time.sleep(2.0)
    raise ServiceStartupError(
        f"service did not become ready within {spec.ready_timeout_s}s at {spec.health_url}.\n"
        f"Last {4000} bytes of server.log:\n{_tail(server_log)}"
    )


def _terminate_group(process: subprocess.Popen, grace_s: float) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=10)


def _reap_leftover_compute_apps(group_pid: int, gpu_index: int, attempts: int = 15) -> bool:
    """Kill stragglers from our own process group that still hold a CUDA context.

    Only processes in this cell's group are touched: killing every compute app
    would take down an unrelated session that happens to share the machine.
    """
    for _ in range(attempts):
        leftover = _compute_apps_on_used_gpus(gpu_index)
        if not leftover:
            return True
        for pid in leftover:
            try:
                if os.getpgid(pid) == group_pid:
                    os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                continue
        time.sleep(2.0)
    return not _compute_apps_on_used_gpus(gpu_index)


def used_gpu_indices(primary: int) -> list[int]:
    """Every physical GPU the run uses (`--gpus 0,1,2,3`): the primary one the gates read, plus the others."""
    return sorted({primary, *hardware.current().gpu_indices})


def _compute_apps_on_used_gpus(primary: int) -> list[int]:
    return [pid for index in used_gpu_indices(primary) for pid in gpu.compute_apps(index)]


@contextlib.contextmanager
def service_lifecycle(
    spec: ServiceSpec | None,
    *,
    run_root: Path,
    gpu_index: int = 0,
    gate_temperature_c: float = thermal.DEFAULT_COOLDOWN_TARGET_C,
    gate_continuous_s: float = thermal.DEFAULT_SERVICE_GATE_CONTINUOUS_S,
    gate_timeout_s: float = 1800.0,
    skip_prestart_gate: bool = False,
) -> Iterator[dict[str, Any]]:
    """Run one fresh service for the duration of the `with` block.

    Yields a mutable record that accumulates gate verdicts, readiness timing
    and teardown state; it is written to `lifecycle.json` on exit whether the
    block succeeded or raised, because a failed cell's thermal history is
    exactly what explains the failure.

    `spec` may be ``None`` for an in-process workload such as the Meta FAIR
    V-JEPA baseline, which has no server to start. The gates, telemetry and
    teardown discipline still apply: the baseline must face the same thermal
    protocol as the served side, or the comparison measures the protocol rather
    than the engines.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    server_log_path = run_root / "server.log"
    record: dict[str, Any] = {
        "schema_version": 1,
        "mode": "served" if spec is not None else "in_process_no_service",
        "command": spec.command if spec is not None else None,
        "health_url": spec.health_url if spec is not None else None,
        "gate_policy": {
            "max_temperature_celsius": gate_temperature_c,
            "continuous_seconds": gate_continuous_s,
            "timeout_seconds": gate_timeout_s,
            "prestart_enabled": not skip_prestart_gate,
            "post_shutdown_enabled": True,
        },
        "gpus_in_use": used_gpu_indices(gpu_index),
        "gate_scope": (
            f"temperature / clock / utilisation gates read GPU {gpu_index} only; all {len(used_gpu_indices(gpu_index))} "
            "GPUs in use must be free of compute processes before start and after shutdown"
        ),
        "hardware_profile": hardware.current().name,
        "clock_control_mode": "unlocked",
        "clock_control_note": (
            "No SM clock lock. find_sustainable_clock.py found no frequency this chassis can "
            "hold: every value from 1785 MHz down to 600 MHz settled at the same 345 MHz / 99 C, "
            "because -lgc is only a ceiling. Gates and per-request telemetry are used instead."
        ),
    }
    process: subprocess.Popen | None = None
    log_handle = None

    try:
        if not skip_prestart_gate:
            for other in used_gpu_indices(gpu_index):
                if other != gpu_index and not thermal.wait_for_empty_gpu(gpu_index=other, timeout_seconds=120.0):
                    raise ServiceStartupError(f"GPU {other} (in use by this run) still has a compute process before start")
            record["prestart_gate"] = thermal.idle_gate(
                run_root / "prestart_gate.csv",
                gpu_index=gpu_index,
                max_temperature_celsius=gate_temperature_c,
                continuous_seconds=gate_continuous_s,
                timeout_seconds=gate_timeout_s,
            )
            if record["prestart_gate"]["status"] != "passed":
                raise ServiceStartupError(
                    "prestart idle gate failed: "
                    f"{record['prestart_gate'].get('blocking_reason')}"
                )

        if spec is not None:
            log_handle = server_log_path.open("w", encoding="utf-8")
            command, nsys_grace = nsys_wrap(spec.command, run_root)
            spec = dataclasses.replace(spec, shutdown_grace_s=max(spec.shutdown_grace_s, nsys_grace))
            process = subprocess.Popen(
                ["bash", "-lc", command],
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group, so teardown can reap children
                cwd=str(Path(__file__).resolve().parents[2]),
                env={**os.environ, **spec.env},
            )
            record["service_pid"] = process.pid
            record["ready_after_seconds"] = wait_until_ready(spec, process, server_log_path)
            if spec.post_ready_grace_s:
                time.sleep(spec.post_ready_grace_s)
        record["state_at_ready"] = gpu.probe(gpu_index)
        record["state_at_ready_all"] = {str(i): gpu.probe(i) for i in used_gpu_indices(gpu_index)}

        yield record

    finally:
        if process is not None and spec is not None:
            _terminate_group(process, spec.shutdown_grace_s)
            record["gpu_cleared_after_shutdown"] = _reap_leftover_compute_apps(process.pid, gpu_index)
        else:
            record["gpu_cleared_after_shutdown"] = all(
                thermal.wait_for_empty_gpu(gpu_index=i) for i in used_gpu_indices(gpu_index))
        if log_handle is not None:
            log_handle.close()
        try:
            record["post_shutdown_gate"] = thermal.idle_gate(
                run_root / "post_shutdown_gate.csv",
                gpu_index=gpu_index,
                max_temperature_celsius=gate_temperature_c,
                continuous_seconds=gate_continuous_s,
                timeout_seconds=gate_timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - teardown telemetry must not mask the real error
            record["post_shutdown_gate"] = {"status": "error", "error": str(exc)}
        write_json(run_root / "lifecycle.json", record)
