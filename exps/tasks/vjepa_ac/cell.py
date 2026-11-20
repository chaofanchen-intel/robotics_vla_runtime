#!/usr/bin/env python3
"""Cell plans for V-JEPA 2-AC: the served M* side and the in-process baseline."""
from __future__ import annotations

import os
import shlex

import exps.frameworks as frameworks
from exps import hardware
from exps.core import paths
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.lifecycle import ServiceSpec
from exps.core.measure import MeasurementConfig
from exps.core.registry import Case, CellRef
from exps.tasks.vjepa_ac.cases import TASK

MSTAR_CONFIG = paths.CONFIGS / "mstar" / "vjepa2_ac.yaml"
UPSTREAM_REPO = paths.EXTERNAL / "vjepa2"


def _meta_native_pythonpath() -> str:
    """Upstream repo, M*'s benchmark package, M*'s Torch, and this harness.

    `envs/vjepa2_repo` deliberately holds only the dependencies upstream needs
    that M* does not ship (OpenCV, iopath, tqdm); Torch and torchcodec are
    imported from `envs/mstar` so the baseline runs against the same Torch
    build as the served side rather than a second, differently-compiled one.
    """
    return ":".join(
        (
            str(UPSTREAM_REPO),
            str(paths.EXTERNAL / "mstar"),
            str(paths.ENVS / "mstar" / "lib" / "python3.12" / "site-packages"),
            str(paths.REPO_ROOT),
        )
    )


def _mstar_client_library_env() -> dict[str, str]:
    """The client decodes the fixture videos with TorchCodec, whose libtorchcodec links `libnppicc.so.13`.

    That library lives in M*'s CUDA 13 toolkit directory, which `mstar_cuda_env.sh` puts on LD_LIBRARY_PATH for the *server*;
    the client process is started by the harness without it, so every request failed with "the fixture provided 0".
    """
    lib = paths.ENVS / "mstar" / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "lib"
    existing = os.environ.get("LD_LIBRARY_PATH")
    return {"LD_LIBRARY_PATH": f"{lib}:{existing}" if existing else str(lib)}


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    horizon = case.params["horizon"]
    frames = case.params["num_video_frames"]
    output_dir = config.output_dir

    if ref.framework == "mstar":
        port = frameworks.port("mstar", TASK.name)
        service = ServiceSpec(
            command=frameworks.mstar_serve(
                model="vjepa2_ac",
                config=MSTAR_CONFIG,
                port_number=port,
                tag=f"vjepa-{case.name}-{ref.repeat}",
            ),
            health_url=f"http://127.0.0.1:{port}/health",
            ready_timeout_s=1800.0,
        )
        client_argv = [
            str(frameworks.MSTAR.python),
            "-m", "exps.tasks.vjepa_ac.client",
            "--framework", "mstar",
            "--case", case.name,
            "--base-url", f"http://127.0.0.1:{port}",
            "--output-dir", str(output_dir),
            "--horizon", str(horizon),
            "--num-video-frames", str(frames),
            *protocol_argv(config),
        ]
        return CellPlan(
            ref=ref,
            service=service,
            client_argv=client_argv,
            context={
                "case_label": case.label,
                "horizon": horizon,
                "comparison_label": TASK.comparison_label,
                "timing_boundary": TASK.timing_boundary,
                "config": paths.relative_to_repo(MSTAR_CONFIG),
                "required_patch": "mstar-01-vjepa-max-rollout-horizon-32 (needed for H=30)",
            },
            provenance_repos=["mstar"],
            client_env={"HF_HOME": str(paths.MODELS_CACHE), "HF_HUB_OFFLINE": "1", **_mstar_client_library_env()},
        )

    if ref.framework == "meta-native":
        # No service: the baseline runs in process. The lifecycle still applies
        # its prestart and post-shutdown gates, so both sides of the comparison
        # face the same thermal protocol.
        inner = " ".join(
            shlex.quote(part)
            for part in (
                str(frameworks.META_NATIVE.python),
                "-m", "exps.tasks.vjepa_ac.meta_native_client",
                "--case", case.name,
                "--output-dir", str(output_dir),
                "--horizon", str(horizon),
                "--num-video-frames", str(frames),
                *protocol_argv(config),
            )
        )
        command = (
            "source exps/frameworks/mstar_cuda_env.sh >/dev/null && "
            f"export PYTHONPATH={shlex.quote(_meta_native_pythonpath())} && "
            f"export HF_HOME={shlex.quote(str(paths.MODELS_CACHE))} "
            f"HF_HUB_CACHE={shlex.quote(str(paths.MODELS_CACHE))} HF_HUB_OFFLINE=1 "
            f"VJEPA2_REPO={shlex.quote(str(UPSTREAM_REPO))} {hardware.visible_devices_env()} && "
            f"exec {inner}"
        )
        return CellPlan(
            ref=ref,
            service=None,
            client_argv=["bash", "-lc", command],
            context={
                "case_label": case.label,
                "horizon": horizon,
                "comparison_label": TASK.comparison_label,
                "timing_boundary": TASK.timing_boundary,
                "baseline_note": (
                    "Meta FAIR upstream, not an eager run of M*'s own components. Round 1 used "
                    "the latter, which answers a different question than the paper's."
                ),
                "baseline_dtype": "bfloat16 (autocast, same as M*'s engine default; FP32 is not measured)",
            },
            provenance_repos=["vjepa2", "mstar"],
        )

    raise KeyError(f"vjepa_ac has no framework {ref.framework!r}")
