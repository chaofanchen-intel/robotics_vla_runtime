"""Round 7: Cosmos3 on SGLang-Omni and at larger sizes, and the Pi0.5 audit hooks."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from exps.core.measure import MeasurementConfig
from exps.core.registry import CellRef
from exps.suite import catalogue
from exps.tasks.cosmos3 import cell as cosmos_cell
from exps.tasks.cosmos3 import client as cosmos_client
from exps.tasks.cosmos3.cases import SIZES, TASK as COSMOS

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("case", sorted(SIZES))
def test_cosmos_sglang_plan_serves_the_direct_pipeline_without_guardrails(case, tmp_path):
    ref = CellRef("cosmos3", "sglang-omni", case, "r01")
    plan = cosmos_cell.plan(ref, COSMOS.case(case), MeasurementConfig(tmp_path))
    command = plan.service.command
    assert command.startswith("export SGLANG_DISABLE_COSMOS3_GUARDRAILS=1; ")
    assert "cosmos3_direct.yaml" in command and "--model-name cosmos3" in command and "--colocate" not in command
    size = str(SIZES[case])
    assert plan.client_argv[plan.client_argv.index("--width") + 1] == size == plan.client_argv[plan.client_argv.index("--height") + 1]
    assert plan.context["guardrails"].startswith("disabled")


def test_cosmos_sizes_are_outside_the_standard_profiles_except_256():
    assert {name for name in SIZES if "full" in COSMOS.case(name).profiles} == {"t2i-256"}


def test_cosmos_client_rejects_a_size_that_does_not_match_the_case(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", ["cosmos3.client", "--framework", "sglang-omni", "--case", "t2i-512", "--base-url", "http://127.0.0.1:1",
                                     "--output-dir", str(tmp_path), "--model", "m", "--width", "256", "--height", "256"])
    with pytest.raises(SystemExit) as error:
        cosmos_client.main()
    assert error.value.code == 2


def test_cosmos_sglang_payload_carries_native_fields():
    payload = cosmos_client.build_payload("sglang-omni", "cosmos3", "a cat", 512, 512, 50)
    assert payload["size"] == "512x512" and payload["guidance_interval"] == [400.0, 1000.0] and payload["seed"] == 42
    assert payload["response_format"] == "b64_json" and "extra_params" not in payload


def test_pi05_audit_hook_arms_demand_their_proof_file():
    experiment = catalogue.get("pi05-audit")
    arms = {arm.name: arm for arm in experiment.arms}
    for name in ("mstar-nograph", "mstar-eager", "vllm-sdpa", "vllm-sdpa-graph"):
        assert arms[name].expect.proof == "hook_applied", name
        assert any("hook_proof.jsonl" in item for item in arms[name].env), name
    assert "pi05_pipeline_graph_off.json" in arms["sglang-nograph"].expect.command_has[0]
    assert arms["mstar-eager"].env[-1] == "mstar:TORCHDYNAMO_DISABLE=1"
    for arm in experiment.arms:                       # every arm names exactly one framework
        assert arm.frameworks is not None and len(arm.frameworks) == 1


def test_mstar_pi05_nograph_hook_empties_every_graph_config(tmp_path):
    proof = tmp_path / "proof.jsonl"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MSTAR_PI05_NO_GRAPH": "1", "MSTAR_PI05_NO_GRAPH_LOG": str(proof),
           "PYTHONPATH": str(REPO / "exps/tools/mstar_pi05_nograph_hook") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    code = ("import mstar.model.pi05.submodules as s;"
            "print([c.get_accelerator_graph_configs(object.__new__(c), device=None) for n, c in vars(s).items() "
            "if isinstance(c, type) and c.__module__ == s.__name__ and 'get_accelerator_graph_configs' in vars(c)])")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-400:]
    assert result.stdout.strip().endswith("[[], []]")
    lines = [json.loads(line) for line in proof.read_text().splitlines()]
    assert lines and all(line["applied"] for line in lines)


def test_mstar_pi05_nograph_hook_is_inert_without_its_flag(tmp_path):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "MSTAR_PI05_NO_GRAPH": "0", "MSTAR_PI05_NO_GRAPH_LOG": str(tmp_path / "p.jsonl"),
           "PYTHONPATH": str(REPO / "exps/tools/mstar_pi05_nograph_hook")}
    code = "import mstar.model.pi05.submodules as s; print(s.Pi05LLMSubmodule.get_accelerator_graph_configs.__qualname__)"
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0 and "Pi05LLMSubmodule.get_accelerator_graph_configs" in result.stdout
    assert not (tmp_path / "p.jsonl").exists()
