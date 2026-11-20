"""Fresh-service launch plans for Cosmos3-Nano."""
from __future__ import annotations

import exps.frameworks as frameworks
from exps.core import paths
from exps.core.cell import CellPlan
from exps.core.cli import protocol_argv
from exps.core.lifecycle import ServiceSpec
from exps.core.measure import MeasurementConfig
from exps.core.registry import Case, CellRef
from exps.tasks.cosmos3.cases import TASK

REVISION = "e59a53c25979a090fa8706c9acc0c254a6e89b92"
SNAPSHOT = paths.MODELS_CACHE / "models--nvidia--Cosmos3-Nano" / "snapshots" / REVISION
CONFIGS = {
    "mstar": paths.CONFIGS / "mstar" / "cosmos3_nano.yaml",
    "vllm-omni": paths.CONFIGS / "vllm_omni" / "cosmos3_omni.yaml",
    "sglang-omni": paths.CONFIGS / "sglang_omni" / "cosmos3_direct.yaml",
}


def plan(ref: CellRef, case: Case, config: MeasurementConfig) -> CellPlan:
    framework = frameworks.get(ref.framework)
    port = frameworks.port(ref.framework, TASK.name)
    if ref.framework == "mstar":
        command = frameworks.mstar_serve(
            model="cosmos3", config=CONFIGS[ref.framework], port_number=port,
            tag=f"cosmos3-{case.name}-{ref.repeat}",
        )
        model = "cosmos3"
    elif ref.framework == "vllm-omni":
        command = frameworks.vllm_omni_serve(
            model_path=SNAPSHOT, deploy_config=CONFIGS[ref.framework],
            port_number=port, extra="--init-timeout 1800",
        )
        model = str(SNAPSHOT)
    elif ref.framework == "sglang-omni":
        # Direct generation, one visible GPU. The native image route takes the OpenAI fields plus SGLang's own (steps, guidance, seed).
        # `export` first: the launcher's own env prelude is a `;`-separated shell list, so a bare `VAR=1 ` prefix would not reach the server.
        command = "export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1; " + frameworks.sglang_omni_serve(
            model_path=SNAPSHOT, config=CONFIGS[ref.framework], port_number=port, model_name="cosmos3", colocate=False,
        )
        model = "cosmos3"
    else:
        raise KeyError(f"cosmos3 has no framework {ref.framework!r}")

    return CellPlan(
        ref=ref,
        service=ServiceSpec(
            command=command, health_url=f"http://127.0.0.1:{port}/health",
            ready_timeout_s=1800.0,
        ),
        client_argv=[
            str(framework.python), "-m", "exps.tasks.cosmos3.client",
            "--framework", ref.framework, "--case", case.name,
            "--base-url", f"http://127.0.0.1:{port}",
            "--output-dir", str(config.output_dir), "--model", model,
            "--width", str(case.params["width"]), "--height", str(case.params["height"]),
            "--steps", str(case.params["steps"]), *protocol_argv(config),
        ],
        context={
            "case_label": case.label, "comparison_label": TASK.comparison_label,
            "timing_boundary": TASK.timing_boundary,
            "model": {"id": "nvidia/Cosmos3-Nano", "revision": REVISION},
            "config": paths.relative_to_repo(CONFIGS[ref.framework]),
            "width": case.params["width"], "height": case.params["height"],
            "num_inference_steps": case.params["steps"],
            **({"guardrails": "disabled (SGLANG_DISABLE_COSMOS3_GUARDRAILS=1)"} if ref.framework == "sglang-omni" else {}),
        },
        provenance_repos=[framework.external_repo],
        client_env={"HF_HOME": str(paths.MODELS_CACHE), "HF_HUB_OFFLINE": "1"},
    )
