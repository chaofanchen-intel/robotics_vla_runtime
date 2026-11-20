#!/usr/bin/env python3
"""Cell plans for Orpheus: M* and VoxServe, launched with matched sampling."""
from __future__ import annotations

import exps.frameworks as frameworks
from exps.core import paths
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.lifecycle import ServiceSpec
from exps.core.measure import MeasurementConfig
from exps.core.registry import Case, CellRef
from exps.tasks.orpheus.cases import SAMPLING, TASK

MSTAR_CONFIG = paths.CONFIGS / "mstar" / "orpheus_colocated.yaml"
ORPHEUS_SNAPSHOT = (
    paths.MODELS_CACHE
    / "models--canopylabs--orpheus-3b-0.1-ft"
    / "snapshots"
    / "4206a56e5a68cf6cf96900a8a78acd3370c02eb6"
)
ORPHEUS_REVISION = "4206a56e5a68cf6cf96900a8a78acd3370c02eb6"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"

#: VoxServe's server-side batch ceiling. Kept at least 8, matching the recipe
#: the previous round used, so a change in this value is not silently mixed in
#: with the change of client. Offered concurrency is still exactly the case's B.
MIN_SERVER_BATCH = 8


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    batch = case.params["batch_size"]
    port = frameworks.port(ref.framework, TASK.name)
    output_dir = config.output_dir

    shared_context = {
        "case_label": case.label,
        "batch_size": batch,
        "comparison_label": TASK.comparison_label,
        "timing_boundary": TASK.timing_boundary,
        "model": {"id": "canopylabs/orpheus-3b-0.1-ft", "revision": ORPHEUS_REVISION},
        "snac_decoder_revision": SNAC_REVISION,
        "sampling": SAMPLING,
        "sampling_parity_note": (
            "These are M*'s OrpheusModelConfig values. They are passed explicitly on the VoxServe "
            "command line because VoxServe's own defaults differ -- its temperature is unset, "
            "which its sampler reads as 1.0 -- and two engines decoding different distributions "
            "would produce different audio lengths and incomparable latencies."
        ),
    }

    if ref.framework == "mstar":
        service = ServiceSpec(
            command=frameworks.mstar_serve(
                model="orpheus",
                config=MSTAR_CONFIG,
                port_number=port,
                tag=f"orpheus-{case.name}-{ref.repeat}",
            ),
            health_url=f"http://127.0.0.1:{port}/health",
            ready_timeout_s=1800.0,
        )
        return CellPlan(
            ref=ref,
            service=service,
            client_argv=[
                str(frameworks.MSTAR.python),
                "-m", "exps.tasks.orpheus.client",
                "--framework", "mstar",
                "--case", case.name,
                "--base-url", f"http://127.0.0.1:{port}",
                "--output-dir", str(output_dir),
                "--batch-size", str(batch),
                *protocol_argv(config),
            ],
            context={
                **shared_context,
                "config": paths.relative_to_repo(MSTAR_CONFIG),
                "response_protocol": "NDJSON, base64-framed raw PCM16 chunks",
                "required_patch": (
                    "mstar-02-orpheus-offline-tokenizer -- without it the service cannot start "
                    "under HF_HUB_OFFLINE=1"
                ),
                "sampling_source": "M*'s OrpheusModelConfig defaults, used as served",
            },
            provenance_repos=["mstar"],
            client_env={"HF_HOME": str(paths.MODELS_CACHE), "HF_HUB_OFFLINE": "1"},
        )

    if ref.framework == "voxserve":
        service = ServiceSpec(
            command=frameworks.voxserve_serve(
                model_path=ORPHEUS_SNAPSHOT,
                port_number=port,
                max_batch_size=max(batch, MIN_SERVER_BATCH),
                temperature=SAMPLING["temperature"],
                top_p=SAMPLING["top_p"],
                repetition_penalty=SAMPLING["repetition_penalty"],
                max_tokens=SAMPLING["max_tokens"],
            ),
            health_url=f"http://127.0.0.1:{port}/health",
            ready_timeout_s=1800.0,
            ready_log_marker="started successfully with model",   # vox_serve.launch logs it once the scheduler is up
        )
        return CellPlan(
            ref=ref,
            service=service,
            client_argv=[
                str(frameworks.VOXSERVE.python),
                "-m", "exps.tasks.orpheus.client",
                "--framework", "voxserve",
                "--case", case.name,
                "--base-url", f"http://127.0.0.1:{port}",
                "--output-dir", str(output_dir),
                "--batch-size", str(batch),
                *protocol_argv(config),
            ],
            context={
                **shared_context,
                "server_max_batch_size": max(batch, MIN_SERVER_BATCH),
                "offered_concurrency": batch,
                "response_protocol": "complete WAV response body (streaming=false)",
                "sampling_source": "forced on the command line to match M*'s defaults",
                "environment_patch_required": (
                    "envs/voxserve's pinned flashinfer hardcodes -std=c++17, which fails to "
                    "compile against Torch 2.8's ATen headers. Both occurrences in "
                    "flashinfer/jit/core.py must read -std=c++20; see doctor.py."
                ),
            },
            provenance_repos=["vox-serve"],
            client_env={"HF_HOME": str(paths.MODELS_CACHE), "HF_HUB_OFFLINE": "1"},
        )

    raise KeyError(f"orpheus has no framework {ref.framework!r}")
