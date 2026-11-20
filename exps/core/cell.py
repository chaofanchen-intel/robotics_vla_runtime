#!/usr/bin/env python3
"""Run one cell: gate, start the service, run the client, record, tear down.

A *cell* is one (task, framework, case, repeat). It is the unit of scheduling,
of resumption and of evidence: everything needed to interpret its numbers lives
inside its own directory.

    runs/<campaign>/<TASK>/<framework>/<case>/<repeat>/
      manifest.json        provenance, serve command, protocol, verdict
      summary.json         the metrics, written by the client
      requests.jsonl       one row per request, including excluded ones
      cooldowns/*.csv      the per-request gate traces
      prestart_gate.csv    service-level gates
      post_shutdown_gate.csv
      lifecycle.json       readiness timing, teardown state, gate verdicts
      server.log           the engine's own output
      client.log           the client's stdout and stderr
      .cell_done           resumption sentinel, written only on success

Why the client is a subprocess
------------------------------
The four runtimes pin incompatible Torch builds, so the measured client has to
run under the runtime's own virtualenv. This module stays dependency-free and
supervises; the client measures.

Why a GPU lock
--------------
Two cells sharing the card would measure contention. The lock is advisory and
file-based: it stops the common accident (two terminals, two sweeps) without
pretending to be a cluster scheduler.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from . import manifest as manifest_mod
from exps import hardware
from . import overrides
from . import paths
from .lifecycle import ServiceSpec, ServiceStartupError, service_lifecycle
from .measure import MeasurementConfig
from .registry import CellRef


@dataclass
class CellPlan:
    """Everything needed to execute one cell, assembled by the task package."""

    ref: CellRef
    #: ``None`` for an in-process workload with no server to start.
    service: ServiceSpec | None
    #: argv for the measured client, run under the runtime's own interpreter.
    client_argv: list[str]
    #: Merged verbatim into the manifest: endpoint, fixture, contract, boundary.
    context: dict[str, Any] = field(default_factory=dict)
    #: External checkouts whose commit and cleanliness belong in this manifest.
    provenance_repos: list[str] = field(default_factory=list)
    #: Extra environment for the client process.
    client_env: dict[str, str] = field(default_factory=dict)
    #: Client wall-clock ceiling. Generous: the client does its own cooldowns.
    client_timeout_s: float = 14_400.0


@contextlib.contextmanager
def gpu_lock(timeout_s: float = 0.0) -> Iterator[None]:
    """Advisory exclusive lock on the GPU for the duration of the block."""
    paths.RUNS.mkdir(parents=True, exist_ok=True)
    lock_path = paths.RUNS / ".gpu.lock"
    handle = lock_path.open("w", encoding="utf-8")
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"another run holds {lock_path}. Two sweeps sharing the GPU would "
                        "measure contention, not engines. Wait for it, or stop it first."
                    ) from None
                time.sleep(5.0)
        handle.write(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        handle.flush()
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def describe(plan: CellPlan, config: MeasurementConfig) -> dict[str, Any]:
    """What `--dry-run` prints: the full plan, without touching the GPU."""
    return {
        "cell": str(plan.ref),
        "output_dir": paths.relative_to_repo(Path(config.output_dir)),
        "service_command": plan.service.command if plan.service else "(in-process, no service)",
        "health_url": plan.service.health_url if plan.service else None,
        "client_argv": plan.client_argv,
        "measured_requests": config.measured_requests,
        "precondition_requests": config.precondition_requests,
        "protocol": (
            "fixed-gap" if config.back_to_back and config.inter_request_gap_s > 0
            else "back-to-back" if config.back_to_back else "gated"
        ),
        "inter_request_gap_s": config.inter_request_gap_s,
        "cooldown_target_c": config.cooldown_target_c,
        "cooldown_gates_frequency": config.cooldown_gates_frequency,
        "calibration_source": config.calibration_source,
        "overrides": overrides.current().as_manifest(),
        "hardware": {"profile": hardware.current().name, "visible_devices": hardware.current().visible_devices},
        "estimated_minutes": estimate_minutes(config),
    }


def estimate_minutes(config: MeasurementConfig) -> float:
    """Rough wall-clock estimate, for planning a night rather than for billing.

    Built from measured behaviour on this host: service-level gates take about
    60-90 s each, a service start 45-150 s, and a per-request cooldown to 70 C
    about 100-160 s. Request time itself is small next to the waiting, so it is
    folded into the per-request term rather than modelled per task.
    """
    gate_s = config.loaded_idle_continuous_s + 30  # a gate holds its condition this long, plus polling slack
    service_overhead_s = gate_s + 120 + gate_s  # prestart gate, service start, post-shutdown gate
    if config.back_to_back:
        # one loaded-idle gate, no per-request gate: a request costs its own time plus the optional fixed gap
        loaded_gates_s = gate_s
        per_request_s = 20 + config.inter_request_gap_s
    else:
        loaded_gates_s = 2 * gate_s
        # a chassis that heats needs minutes of cooling before each request; one with stable cooling only waits out the gate's hold
        cooldown_s = config.cooldown_continuous_s + 10 if hardware.current().thermal_stable else 160
        per_request_s = cooldown_s + 20
    total_s = (
        service_overhead_s
        + loaded_gates_s
        + config.precondition_requests * per_request_s
        + config.measured_requests * per_request_s
    )
    return round(total_s / 60.0, 1)


def run_cell(plan: CellPlan, config: MeasurementConfig, *, campaign: str | None = None) -> dict[str, Any]:
    """Execute one cell end to end and write its manifest.

    Returns the manifest. Never raises for an engine-side failure: a cell that
    fails is recorded as failed, with its gate traces and server log intact, so
    the sweep can continue and the failure can be diagnosed afterwards.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in (".cell_done", "VOID.json", "summary.json", "manifest.json"):
        (output_dir / name).unlink(missing_ok=True)
    lifecycle_root = paths.lifecycle_dir(
        plan.ref.task, plan.ref.framework, plan.ref.case, plan.ref.repeat, campaign=campaign
    )
    client_log = output_dir / "client.log"
    started = time.monotonic()
    record: dict[str, Any] = {
        "schema_version": 1,
        "campaign": campaign or paths.campaign_id(),
        "cell": str(plan.ref),
        "task": plan.ref.task,
        "framework": plan.ref.framework,
        "case": plan.ref.case,
        "repeat": plan.ref.repeat,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        **plan.context,
        "protocol": config.as_record(),
        "overrides": overrides.current().as_manifest(),
        "hardware": {"profile": hardware.current().name, "visible_devices": hardware.current().visible_devices,
                     "gpu_name": hardware.gpu_name(hardware.current().primary_gpu)},
        "service_command": plan.service.command if plan.service else None,
        "client_argv": plan.client_argv,
        "provenance": manifest_mod.provenance(plan.provenance_repos or None),
    }

    client_returncode: int | None = None
    failure: str | None = None
    try:
        with service_lifecycle(
            plan.service,
            run_root=lifecycle_root,
            gpu_index=config.gpu_index,
            gate_temperature_c=config.loaded_idle_target_c,
            gate_continuous_s=config.loaded_idle_continuous_s,
            gate_timeout_s=config.loaded_idle_timeout_s,
        ):
            with client_log.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    plan.client_argv,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    cwd=str(paths.REPO_ROOT),
                    env={**os.environ, "PYTHONPATH": str(paths.REPO_ROOT), **plan.client_env},
                    timeout=plan.client_timeout_s,
                )
            client_returncode = completed.returncode
    except ServiceStartupError as exc:
        failure = f"ServiceStartupError: {exc}"
    except subprocess.TimeoutExpired:
        failure = f"client exceeded its {plan.client_timeout_s}s ceiling"
    except Exception as exc:  # noqa: BLE001 - one bad cell must not end the sweep
        failure = f"{type(exc).__name__}: {exc}"

    summary = _read_json(output_dir / "summary.json")
    voided = (output_dir / "VOID.json").is_file()
    status = _status(failure, client_returncode, summary, voided)

    record.update(
        {
            "status": status,
            "failure": failure,
            "client_returncode": client_returncode,
            "wall_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "lifecycle": _read_json(lifecycle_root / "lifecycle.json"),
            "result_headline": _headline(summary),
            "outputs": {
                "summary_json": "summary.json",
                "requests_jsonl": "requests.jsonl",
                "client_log": "client.log",
                "server_log": paths.relative_to_repo(lifecycle_root / "server.log"),
                "cooldown_csvs": "cooldowns/",
                "void_marker": "VOID.json" if voided else None,
            },
        }
    )
    manifest_mod.write_json(output_dir / "manifest.json", record)
    _copy_lifecycle_artifacts(lifecycle_root, output_dir)
    manifest_mod.mark_cell_done(output_dir, status)
    return record


def _status(failure: str | None, returncode: int | None, summary: dict, voided: bool) -> str:
    if voided:
        return "void_contract_violation"
    if failure:
        return "failed"
    if returncode not in (0, None):
        return "failed"
    if not summary:
        return "failed_no_summary"
    if summary.get("status") != "completed":
        return "failed_measurement"
    return "completed"


def _headline(summary: dict) -> dict[str, Any]:
    """The few numbers worth seeing in a sweep log, pulled out of the summary."""
    results = summary.get("results") or {}
    e2e = (results.get("metrics") or {}).get("e2e_ms") or {}
    return {
        "headline_requests": results.get("headline_requests"),
        "measured_requests": results.get("measured_requests"),
        "e2e_ms_p50": e2e.get("p50"),
        "intra_start_drift": (results.get("intra_start_drift") or {}).get("status"),
        "all_requests_started_at_full_clock": (
            (results.get("thermal_start_state") or {}).get("all_requests_started_at_full_clock")
        ),
    }


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _copy_lifecycle_artifacts(lifecycle_root: Path, output_dir: Path) -> None:
    """Mirror the gate traces and server log next to the results.

    The cell directory has to be self-contained: reading a number should never
    require knowing that `_service_lifecycles/` exists.
    """
    import shutil

    for name in ("prestart_gate.csv", "post_shutdown_gate.csv", "lifecycle.json", "server.log"):
        source = lifecycle_root / name
        if source.is_file():
            with contextlib.suppress(OSError):
                shutil.copy2(source, output_dir / name)
