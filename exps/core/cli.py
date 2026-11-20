#!/usr/bin/env python3
"""Protocol arguments shared by `run.py` and every task client.

Defining these once means a flag like `--cooldown-target-c` has the same name,
the same default and the same meaning whether it is typed on the command line,
read from a calibration profile, or passed down to a client subprocess. It also
means the protocol a cell actually ran under can be reconstructed from its
manifest without consulting the code that produced it.

Precedence, highest first:

1. an explicit command-line value
2. `profiles/calibrated.json` for this task and case, if calibration has run
3. `profiles/defaults.json` -- the conservative pre-calibration values

Which of the three supplied each parameter is recorded as `calibration_source`
in every summary, because a number measured under calibrated parameters and one
measured under untested defaults are not the same evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from exps import hardware

from . import paths, thermal
from .measure import MeasurementConfig

PROTOCOLS = ("gated", "back-to-back")
DEFAULTS_PROFILE = paths.PROFILES / "defaults.json"
CALIBRATED_PROFILE = paths.PROFILES / "calibrated.json"

#: Used when neither profile file exists. Kept in code as well as in
#: profiles/defaults.json so a deleted profile degrades to the documented
#: values rather than to whatever argparse happens to default to.
BUILTIN_DEFAULTS: dict[str, Any] = {
    "repeats": 5,
    "measured": 5,
    "preconditions": 1,
    "cooldown_target_c": thermal.DEFAULT_COOLDOWN_TARGET_C,
    "cooldown_continuous_s": thermal.DEFAULT_COOLDOWN_CONTINUOUS_S,
    "cooldown_timeout_s": 900.0,
    "cooldown_gates_frequency": True,
    "loaded_idle_target_c": thermal.DEFAULT_COOLDOWN_TARGET_C,
    "loaded_idle_continuous_s": thermal.DEFAULT_SERVICE_GATE_CONTINUOUS_S,
    "loaded_idle_timeout_s": 1800.0,
    "memory_cooldown_target_c": None,
}


def _load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def resolve_parameters(task: str, case: str) -> tuple[dict[str, Any], str]:
    """Merge built-in, default-profile and calibrated values for one case.

    Returns the parameters and a `calibration_source` string naming where the
    case-specific values came from.
    """
    values = dict(BUILTIN_DEFAULTS)
    source = "builtin_defaults"

    defaults = _load(DEFAULTS_PROFILE)
    if defaults:
        values.update(defaults.get("global", {}))
        values.update(defaults.get("tasks", {}).get(task, {}).get("global", {}))
        values.update(defaults.get("tasks", {}).get(task, {}).get("cases", {}).get(case, {}))
        source = "default_not_measured"

    gate_s = hardware.current().service_gate_continuous_s
    if gate_s is not None:  # a machine with stable cooling needs no 60 s settling hold; paid twice per cell
        values["loaded_idle_continuous_s"] = gate_s
        source = f"{source}+hardware({hardware.current().name}).service_gate_continuous_s"

    calibrated = _load(CALIBRATED_PROFILE)
    case_values = calibrated.get("tasks", {}).get(task, {}).get("cases", {}).get(case)
    if case_values:
        values.update(case_values)
        source = f"calibrated:{calibrated.get('measured_at_utc', 'unknown')}"
    elif calibrated:
        source = "calibrated_profile_exists_but_case_not_measured"

    return values, source


def add_protocol_arguments(parser: argparse.ArgumentParser) -> argparse._ArgumentGroup:
    """Add the thermal-protocol flags. Defaults stay None so profiles can win.

    An explicit `None` default is what lets `resolve_parameters` distinguish
    "the operator asked for 65 C" from "nobody said, use the profile".
    """
    group = parser.add_argument_group(
        "thermal protocol",
        "Unset flags fall back to profiles/calibrated.json, then profiles/defaults.json.",
    )
    group.add_argument("--measured", type=int, default=None,
                       help="Measured requests per start, each preceded by a cooldown gate.")
    group.add_argument("--preconditions", type=int, default=None,
                       help="Unscored warmup requests that absorb JIT and CUDA-graph capture.")
    group.add_argument("--cooldown-target-c", type=float, default=None,
                       help="Per-request gate temperature. Default 70 C: 10 C below the ~80 C "
                            "knee where this card starts dropping clocks.")
    group.add_argument("--memory-cooldown-target-c", type=float, default=None,
                       help="Optional memory-junction ceiling for the per-request gate.")
    group.add_argument("--cooldown-continuous-s", type=float, default=None,
                       help="How long all three gate conditions must hold before measuring.")
    group.add_argument("--cooldown-timeout-s", type=float, default=None,
                       help="Give up waiting after this long; the request is kept but flagged.")
    group.add_argument("--cooldown-gates-frequency", choices=("0", "1"), default=None,
                       help="1: the cooldown also waits for full SM clock and no event reason. 0: it waits for temperature "
                            "only, and the clock and event reasons are recorded. Default: the hardware profile.")
    group.add_argument("--loaded-idle-target-c", type=float, default=None)
    group.add_argument("--loaded-idle-continuous-s", type=float, default=None)
    group.add_argument("--loaded-idle-timeout-s", type=float, default=None)
    group.add_argument("--gpu-index", type=int, default=None,
                       help="Physical GPU the gates and telemetry read (default: the first of --gpus / the hardware profile).")
    group.add_argument("--protocol", choices=PROTOCOLS, default=None,
                       help="gated (default): per-request cooldown gate, the card is cool and at full clock before every "
                            "measured request. back-to-back: no post-precondition gate and no per-request gate, requests go "
                            "out consecutively. Available for every task; record which one a number was measured under.")
    group.add_argument("--back-to-back", action="store_true",
                       help="Alias for --protocol back-to-back.")
    group.add_argument("--inter-request-gap-s", type=float, default=0.0,
                       help="With back-to-back: sleep this many seconds before each measured request, with no thermal "
                            "condition (separates an idle gap from the thermal gate).")
    group.add_argument("--with-throughput", action="store_true",
                       help="Also run a separate back-to-back batch after a full cooldown. "
                            "Reported as thermally_drifting_throughput, never merged into latency.")
    group.add_argument("--throughput-requests", type=int, default=20)
    group.add_argument("--throughput-concurrency", type=int, default=1)
    return group


def _gpu_index(args: argparse.Namespace) -> int:
    value = getattr(args, "gpu_index", None)
    return int(value) if value is not None else hardware.current().primary_gpu


def wants_back_to_back(args: argparse.Namespace, task: str | None = None) -> bool:
    """`--protocol back-to-back` or its `--back-to-back` alias; a gap implies it (a gap is not a gate)."""
    protocol = getattr(args, "protocol", None)
    if protocol == "back-to-back" or bool(getattr(args, "back_to_back", False)):
        return True
    if protocol is None and float(getattr(args, "inter_request_gap_s", 0.0) or 0.0) == 0.0 \
            and getattr(args, "calibration_source", None) is None:
        # no explicit choice on a harness command line: the hardware profile decides (stable cooling: back-to-back)
        from exps import protocols

        return protocols.default_for(task) == "back-to-back"
    if protocol == "gated":
        if float(getattr(args, "inter_request_gap_s", 0.0) or 0.0) > 0:
            raise ValueError("--inter-request-gap-s is a back-to-back variant; it cannot be combined with --protocol gated")
        return False
    return float(getattr(args, "inter_request_gap_s", 0.0) or 0.0) > 0


def measurement_config(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    task: str,
    case: str,
) -> MeasurementConfig:
    """Build a `MeasurementConfig` from CLI values layered over the profiles."""
    resolved, source = resolve_parameters(task, case)

    def pick(name: str, profile_key: str | None = None) -> Any:
        explicit = getattr(args, name, None)
        if explicit is not None:
            return explicit
        return resolved.get(profile_key or name)

    explicit_names = [
        name
        for name in (
            "measured", "preconditions", "cooldown_target_c", "cooldown_continuous_s",
            "cooldown_timeout_s", "loaded_idle_target_c", "loaded_idle_continuous_s",
            "loaded_idle_timeout_s", "memory_cooldown_target_c",
        )
        if getattr(args, name, None) is not None
    ]
    if explicit_names:
        source = f"{source}+cli_override({','.join(sorted(explicit_names))})"

    return MeasurementConfig(
        output_dir=output_dir,
        measured_requests=int(pick("measured")),
        precondition_requests=int(pick("preconditions")),
        gpu_index=_gpu_index(args),
        cooldown_target_c=float(pick("cooldown_target_c")),
        cooldown_continuous_s=float(pick("cooldown_continuous_s")),
        cooldown_timeout_s=float(pick("cooldown_timeout_s")),
        cooldown_gates_frequency=bool(hardware.current().cooldown_gates_frequency),
        loaded_idle_target_c=float(pick("loaded_idle_target_c")),
        loaded_idle_continuous_s=float(pick("loaded_idle_continuous_s")),
        loaded_idle_timeout_s=float(pick("loaded_idle_timeout_s")),
        throughput_requests=(
            getattr(args, "throughput_requests", 0) if getattr(args, "with_throughput", False) else 0
        ),
        throughput_concurrency=getattr(args, "throughput_concurrency", 1),
        back_to_back=wants_back_to_back(args, task),
        inter_request_gap_s=float(getattr(args, "inter_request_gap_s", 0.0) or 0.0),
        calibration_source=source,
    )


def protocol_argv(config: MeasurementConfig) -> list[str]:
    """Render a resolved config back into flags for a client subprocess.

    The harness resolves the profile once, then passes concrete numbers down.
    If each client re-resolved them, a profile edited mid-run could give two
    cells of the same sweep different protocols without anything recording it.
    """
    argv = [
        "--measured", str(config.measured_requests),
        "--preconditions", str(config.precondition_requests),
        "--cooldown-target-c", str(config.cooldown_target_c),
        "--cooldown-continuous-s", str(config.cooldown_continuous_s),
        "--cooldown-timeout-s", str(config.cooldown_timeout_s),
        "--loaded-idle-target-c", str(config.loaded_idle_target_c),
        "--loaded-idle-continuous-s", str(config.loaded_idle_continuous_s),
        "--loaded-idle-timeout-s", str(config.loaded_idle_timeout_s),
        "--gpu-index", str(config.gpu_index),
        "--calibration-source", config.calibration_source,
        "--cooldown-gates-frequency", "1" if config.cooldown_gates_frequency else "0",
    ]
    if config.back_to_back:
        argv += ["--protocol", "back-to-back"]
        if config.inter_request_gap_s > 0:
            argv += ["--inter-request-gap-s", str(config.inter_request_gap_s)]
    if config.throughput_requests:
        argv += [
            "--with-throughput",
            "--throughput-requests", str(config.throughput_requests),
            "--throughput-concurrency", str(config.throughput_concurrency),
        ]
    return argv


def add_client_protocol_arguments(parser: argparse.ArgumentParser) -> None:
    """Client-side counterpart: the same flags, but with concrete defaults.

    Clients are always invoked by the harness with every value supplied, so
    these defaults only matter when a client is run by hand for debugging.
    """
    add_protocol_arguments(parser)
    parser.add_argument("--calibration-source", default="direct_invocation",
                        help="Recorded verbatim in the summary; set by the harness.")


def client_measurement_config(args: argparse.Namespace, output_dir: Path) -> MeasurementConfig:
    """Build a config inside a client from fully-specified flags."""
    def value(name: str) -> Any:
        explicit = getattr(args, name, None)
        return explicit if explicit is not None else BUILTIN_DEFAULTS[name]

    return MeasurementConfig(
        output_dir=output_dir,
        measured_requests=int(value("measured")),
        precondition_requests=int(value("preconditions")),
        gpu_index=_gpu_index(args),
        cooldown_target_c=float(value("cooldown_target_c")),
        cooldown_continuous_s=float(value("cooldown_continuous_s")),
        cooldown_timeout_s=float(value("cooldown_timeout_s")),
        cooldown_gates_frequency=bool(int(value("cooldown_gates_frequency"))),
        loaded_idle_target_c=float(value("loaded_idle_target_c")),
        loaded_idle_continuous_s=float(value("loaded_idle_continuous_s")),
        loaded_idle_timeout_s=float(value("loaded_idle_timeout_s")),
        throughput_requests=(
            getattr(args, "throughput_requests", 0) if getattr(args, "with_throughput", False) else 0
        ),
        throughput_concurrency=getattr(args, "throughput_concurrency", 1),
        back_to_back=wants_back_to_back(args),
        inter_request_gap_s=float(getattr(args, "inter_request_gap_s", 0.0) or 0.0),
        calibration_source=getattr(args, "calibration_source", "direct_invocation"),
    )
