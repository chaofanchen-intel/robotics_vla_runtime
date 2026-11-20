"""CPU-only checks for the two image/action workload cell plans."""
from __future__ import annotations

import base64
import contextlib
import json
import os
import shlex
import struct
import sys
from pathlib import Path

import pytest
import yaml

from exps.core.codecs import ContractViolation
from exps.core.cell import CellPlan, _status, run_cell
from exps.core.measure import MeasurementConfig
from exps.core.measure import _Collector, _build_summary
from exps.core.registry import TASK_ORDER, Case, CellRef, expand, load_tasks
from exps.core import paths
from exps.frameworks import mstar_serve
from exps.tasks.cosmos3 import cell as cosmos_cell
from exps.tasks.cosmos3 import client as cosmos_client
from exps.tasks.cosmos3.cases import TASK as COSMOS
from exps.tasks.bagel.cases import TASK as BAGEL
from exps.tasks.orpheus.cases import BATCH_SIZES as ORPHEUS_BATCH_SIZES, TASK as ORPHEUS
from exps.tasks.pi05 import cell as pi05_cell
from exps.tasks.pi05 import client as pi05_client
from exps.tasks.pi05.cases import TASK as PI05
from exps.tasks import qwen_common
from exps.tasks.qwen_t2s.client import sampling_kwargs as qwen_t2s_sampling_kwargs
from exps.tasks.vjepa_ac.meta_native_client import _verified_rollout_steps
from exps.run import _config_for, _select_campaign, build_parser, main as run_main


def test_retry_clears_previous_void_before_reading_new_summary(monkeypatch, tmp_path):
    output_dir = tmp_path / "cell"
    output_dir.mkdir()
    (output_dir / "VOID.json").write_text('{"status":"void_contract_violation"}')
    (output_dir / "summary.json").write_text('{"status":"void"}')
    monkeypatch.setattr("exps.core.cell.service_lifecycle", lambda *args, **kwargs: contextlib.nullcontext())
    monkeypatch.setattr(paths, "lifecycle_dir", lambda *args, **kwargs: tmp_path / "lifecycle")
    plan = CellPlan(
        ref=CellRef("vjepa_ac", "meta-native", "h4", "r01"),
        service=None,
        client_argv=[
            sys.executable,
            "-c",
            "import json,sys; open(sys.argv[1], 'w').write(json.dumps({'status':'completed'}))",
            str(output_dir / "summary.json"),
        ],
    )
    record = run_cell(plan, MeasurementConfig(output_dir), campaign="retry-test")
    assert record["status"] == "completed"
    assert not (output_dir / "VOID.json").exists()


def test_mstar_serve_shortens_only_socket_tags_that_exceed_unix_path_limit():
    config = paths.CONFIGS / "mstar" / "qwen3omni_1gpu.yaml"

    def socket_path(tag):
        command = shlex.split(
            mstar_serve(
                model="qwen3_omni",
                config=config,
                port_number=19331,
                tag=tag,
            )
        )
        return command[command.index("--socket-path-prefix") + 1]

    short_tag = "qwen-a2t-en-r01"
    assert socket_path(short_tag) == f"/tmp/exps-{short_tag}"

    long_tag = "qwen-a2t-a2t-en-long-shape-matched-other-audio-precondition-with-extra-padding-r01"
    long_path = socket_path(long_tag)
    ipc_path = f"{long_path}/api_server_preprocess_worker.ipc"
    assert len(ipc_path.encode("utf-8")) <= 107
    assert long_path != f"/tmp/exps-{long_tag}"
    assert long_path != socket_path(f"{long_tag[:-3]}r02")


@pytest.mark.parametrize("framework", PI05.frameworks)
def test_pi05_cell_plan(framework, tmp_path):
    ref = CellRef("pi05", framework, "droid-46row", "r01")
    plan = pi05_cell.plan(ref, PI05.case(ref.case), MeasurementConfig(tmp_path))
    assert "--num-inference-steps" in plan.client_argv
    assert plan.client_argv[plan.client_argv.index("--num-inference-steps") + 1] == "10"
    assert str(pi05_cell.CONFIGS[framework]) in plan.service.command
    assert plan.service.health_url.endswith(f":{plan.service.command.split('--port ')[1].split()[0]}/health")
    if framework == "vllm-omni":
        assert "--custom-pipeline-args" in plan.service.command
        command = shlex.split(plan.service.command)
        args = json.loads(command[command.index("--custom-pipeline-args") + 1])
        assert args == {"tokenizer": str(pi05_cell.TOKENIZER)}
        assert "--enforce-eager" in plan.service.command
        assert command[command.index("--robot-openpi-idle-timeout") + 1] == "1800"
    if framework == "sglang-core":
        assert "--pipeline Pi05Pipeline" in plan.service.command
        assert plan.client_argv[plan.client_argv.index("--base-url") + 1].startswith("ws://127.0.0.1:")


@pytest.mark.parametrize("framework", COSMOS.frameworks)
def test_cosmos_cell_plan(framework, tmp_path):
    ref = CellRef("cosmos3", framework, "t2i-256", "r01")
    plan = cosmos_cell.plan(ref, COSMOS.case(ref.case), MeasurementConfig(tmp_path))
    assert "--steps" in plan.client_argv
    assert plan.client_argv[plan.client_argv.index("--steps") + 1] == "50"
    assert str(cosmos_cell.CONFIGS[framework]) in plan.service.command
    payload = cosmos_client.build_payload(framework, "model", "hello", 256, 256, 50)
    assert payload["size"] == "256x256"
    assert payload["num_inference_steps"] == 50
    assert payload.get("guidance_interval", payload.get("extra_params", {}).get("guidance_interval")) == [400.0, 1000.0]


def test_cosmos_image_contract():
    png = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", 256, 256) + b"\x08\x02\x00\x00\x00"
    body = json.dumps({"data": [{"b64_json": base64.b64encode(png).decode()}]}).encode()
    assert cosmos_client.decode_image(body, 256, 256) == png
    with pytest.raises(ContractViolation, match="returned image"):
        cosmos_client.decode_image(body, 512, 512)
    with pytest.raises(ContractViolation, match="invalid base64"):
        cosmos_client.decode_image(b'{"data":[{"b64_json":"%%%"}]}', 256, 256)


def test_pi05_rejects_unequal_steps(monkeypatch, tmp_path):
    monkeypatch.setattr("sys.argv", [
        "pi05.client", "--framework", "mstar", "--case", "droid-46row",
        "--base-url", "http://127.0.0.1:19371", "--output-dir", str(tmp_path),
        "--num-inference-steps", "8",
    ])
    with pytest.raises(SystemExit) as error:
        pi05_client.main()
    assert error.value.code == 2


@pytest.mark.parametrize("case,steps", [("droid-46row-s1", 1), ("droid-46row-s5", 5), ("droid-46row", 10), ("droid-46row-s20", 20)])
def test_pi05_steps_reach_the_mstar_config_and_every_request(case, steps, tmp_path):
    expected = paths.CONFIGS / "mstar" / ("pi05_bfloat16.yaml" if steps == 10 else f"pi05_bfloat16_steps{steps}.yaml")
    for framework in PI05.frameworks:
        ref = CellRef("pi05", framework, case, "r01")
        plan = pi05_cell.plan(ref, PI05.case(case), MeasurementConfig(tmp_path))
        argv = plan.client_argv
        assert argv[argv.index("--num-inference-steps") + 1] == str(steps), framework
        if framework == "mstar":   # M* reads its count from the service config only
            assert str(expected) in plan.service.command and plan.context["config"] == paths.relative_to_repo(expected)
        else:
            assert str(pi05_cell.CONFIGS[framework]) in plan.service.command


def test_pi05_rejects_flow_step_counts_outside_the_listed_ones(tmp_path):
    with pytest.raises(ValueError, match="flow steps"):
        pi05_cell.plan(CellRef("pi05", "mstar", "droid-46row-s7", "r01"),
                       Case("droid-46row-s7", "7 steps", {"num_inference_steps": 7}), MeasurementConfig(tmp_path))


def test_pi05_step_configs_differ_from_the_shipped_one_only_in_flow_steps():
    shipped = yaml.safe_load((paths.CONFIGS / "mstar" / "pi05_bfloat16.yaml").read_text())
    assert shipped["model_kwargs"] == {"num_flow_steps": 10}
    for steps in (1, 5, 20):
        other = yaml.safe_load((paths.CONFIGS / "mstar" / f"pi05_bfloat16_steps{steps}.yaml").read_text())
        assert other["model_kwargs"] == {"num_flow_steps": steps}
        assert {k: v for k, v in other.items() if k != "model_kwargs"} == {k: v for k, v in shipped.items() if k != "model_kwargs"}


def test_bagel_pool16g_variants_change_only_the_mooncake_sender_pool():
    overlays = paths.REPO_ROOT / "exps" / "configs" / "hardware" / "rtx5880" / "vllm_omni"
    pairs = (
        ("bagel_two_stage_2gpu_mooncake_tcp_prefix_cache.yaml", "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_pool16g.yaml"),
        ("bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_kvtimeout900.yaml",
         "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_pool16g_kvtimeout900.yaml"),
        ("bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16.yaml",
         "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16_pool16g.yaml"),
        ("bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16_kvtimeout900.yaml",
         "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16_pool16g_kvtimeout900.yaml"),
    )

    def body(text):
        return [line for line in text.splitlines() if not line.startswith("#")]

    for source, variant in pairs:
        src_text = (overlays / source).read_text(encoding="utf-8")
        dst_text = (overlays / variant).read_text(encoding="utf-8")
        assert dst_text.startswith("# COUNTERFACTUAL (pool-16G)"), variant
        assert yaml.safe_load(src_text)["connectors"]["kv_connector"]["extra"]["memory_pool_size"] == 4294967296, source
        assert yaml.safe_load(dst_text)["connectors"]["kv_connector"]["extra"]["memory_pool_size"] == 17179869184, variant
        assert body(dst_text) == [line.replace("memory_pool_size: 4294967296", "memory_pool_size: 17179869184")
                                  for line in body(src_text)], variant


def test_pi05_sglang_variants_change_exactly_one_switch():
    def pipeline(name):
        return json.loads((paths.CONFIGS / "sglang_core" / name).read_text())

    shipped = pipeline("pi05_pipeline.json")
    assert shipped["enable_global_prefix_cache"] is False          # the shipped SGLang core config: prefix cache off
    assert shipped["enable_prefix_cuda_graph"] is True and shipped["enable_action_cuda_graph"] is True
    variants = {
        "pi05_pipeline_prefix_cache.json": {"enable_global_prefix_cache": True},
        "pi05_pipeline_prefix_graph_off.json": {"enable_prefix_cuda_graph": False},
        "pi05_pipeline_action_graph_off.json": {"enable_action_cuda_graph": False},
        "pi05_pipeline_graph_off.json": {"enable_prefix_cuda_graph": False, "enable_action_cuda_graph": False},
        "pi05_pipeline_buckets200.json": {"prompt_token_buckets": [200]},
    }
    for name, changed in variants.items():
        assert pipeline(name) == {**shipped, **changed}, name


def test_pi05_fixture_row_option_repeats_one_row_and_the_default_cycles():
    from exps.tasks.pi05.client import select_examples

    rows = [{"episode_index": index} for index in range(3)]
    assert [row["episode_index"] for row in select_examples(rows, 5)] == [0, 1, 2, 0, 1]
    assert [row["episode_index"] for row in select_examples(rows, 4, fixture_row=2)] == [2, 2, 2, 2]
    with pytest.raises(ValueError, match="outside"):
        select_examples(rows, 4, fixture_row=3)


def test_pi05_row_repeat_case_sends_one_row_at_the_default_steps(tmp_path):
    for framework in PI05.frameworks:
        plan = pi05_cell.plan(CellRef("pi05", framework, "droid-row0-x20", "r01"), PI05.case("droid-row0-x20"),
                              MeasurementConfig(tmp_path))
        argv = plan.client_argv
        assert argv[argv.index("--fixture-row") + 1] == "0" and argv[argv.index("--num-inference-steps") + 1] == "10"
        assert plan.context["fixture_row"] == 0


def test_pi05_step_and_row_cases_stay_out_of_the_sweeps():
    assert {ref.case for ref in expand(PI05, profile="full", repeats=1)} == {"droid-46row"}
    assert {ref.case for ref in expand(PI05, profile="nightly", repeats=1)} == {"droid-46row"}


@pytest.mark.parametrize("horizon", [4, 15, 30])
def test_meta_native_horizon_counts_prediction_steps_not_spatial_tokens(horizon):
    predictions = [type("Prediction", (), {"shape": (1, 256, 1408)})() for _ in range(horizon)]
    assert _verified_rollout_steps(predictions, horizon, 256) == horizon


def test_meta_native_horizon_rejects_a_missing_prediction_step():
    predictions = [type("Prediction", (), {"shape": (1, 256, 1408)})() for _ in range(3)]
    with pytest.raises(ContractViolation, match="produced 3 steps"):
        _verified_rollout_steps(predictions, 4, 256)


def test_meta_native_horizon_rejects_wrong_spatial_token_count():
    predictions = [type("Prediction", (), {"shape": (1, 255, 1408)})() for _ in range(4)]
    with pytest.raises(ContractViolation, match="256 spatial tokens"):
        _verified_rollout_steps(predictions, 4, 256)


def test_unknown_task_not_skipped():
    with pytest.raises(KeyError, match="unknown task"):
        load_tasks(("pi05", "misspelled"))


def test_smoke_profile_is_one_measured_request():
    args = build_parser().parse_args(["pi05", "--profile", "smoke", "--dry-run"])
    ref = CellRef("pi05", "mstar", "droid-46row", "r01")
    assert _config_for(args, ref).measured_requests == 1


def test_bagel_nightly_defaults_to_256_and_full_adds_512():
    """Profiles contain only equal-work headline cases: T2I at one and at three CFG branches, I2I, I2T."""
    nightly = list(expand(BAGEL, profile="nightly", repeats=1))
    full = list(expand(BAGEL, profile="full", repeats=1))
    headline = ("t2i-{r}-nocfg", "t2i-{r}-cfg3", "i2i-{r}", "i2t-{r}")
    assert {ref.case for ref in nightly} == {c.format(r=256) for c in headline}
    assert {ref.case for ref in full} == {c.format(r=r) for c in headline for r in (256, 512)}
    assert len(nightly) == 12 and len(full) == 24
    assert "t2i-256" not in {c.name for c in BAGEL.cases}  # the unequal default request is only `t2i-256-default`


def test_orpheus_nightly_includes_all_batch_sizes():
    nightly = list(expand(ORPHEUS, profile="nightly", repeats=1))

    assert {ref.case for ref in nightly} == {
        f"b{batch}" for batch in ORPHEUS_BATCH_SIZES
    }
    assert len(nightly) == len(ORPHEUS_BATCH_SIZES) * len(ORPHEUS.frameworks)


def test_smoke_campaign_is_separate_from_nightly():
    import argparse

    try:
        paths.set_campaign(None)
        _select_campaign(argparse.Namespace(profile="smoke", campaign=None))
        assert paths.campaign_id() == f"{paths.DEFAULT_CAMPAIGN_ID}-smoke"
        _select_campaign(argparse.Namespace(profile="smoke", campaign="custom-smoke"))
        assert paths.campaign_id() == "custom-smoke"
        _select_campaign(argparse.Namespace(profile="nightly", campaign=None))
        assert paths.campaign_id() == "custom-smoke"  # a fresh process picks the default; within one process the last --campaign sticks
    finally:
        paths.set_campaign(None)


@pytest.mark.parametrize("profile", ["smoke", "nightly", "full"])
def test_default_sweep_executes_priority_tasks_first(profile, capsys, monkeypatch):
    monkeypatch.delenv("EXPS_CAMPAIGN_ID", raising=False)
    assert run_main(["sweep", "--profile", profile, "--repeats", "1", "--dry-run"]) == 0
    plans = json.loads(capsys.readouterr().out)["plans"]
    assert tuple(dict.fromkeys(plan["cell"].split("/")[0] for plan in plans)) == TASK_ORDER


def test_explicit_task_order_is_respected(capsys):
    assert run_main([
        "sweep", "--tasks", "qwen_t2s bagel", "--profile", "smoke", "--dry-run",
    ]) == 0
    plans = json.loads(capsys.readouterr().out)["plans"]
    assert tuple(dict.fromkeys(plan["cell"].split("/")[0] for plan in plans)) == (
        "qwen_t2s", "bagel",
    )


def test_qwen_config_overrides_select_t2s_candidates():
    from exps.core import overrides

    overrides.reset()
    # defaults are each framework's best single-GPU recipe
    assert qwen_common.config_path("vllm-omni").name == "qwen3_omni_full_1gpu_default_async_c2wgraph.yaml"
    assert qwen_common.config_path("sglang-omni").name == "qwen3_omni_colocated_h200_partial_start.yaml"
    plain_vllm = paths.CONFIGS / "vllm_omni" / "qwen3_omni_full_1gpu.yaml"
    plain_sglang = paths.CONFIGS / "sglang_omni" / "qwen3_omni_colocated_h200.yaml"
    overrides.set_current(overrides.build(options=[
        f"qwen.vllm_deploy_config={plain_vllm.relative_to(paths.REPO_ROOT)}",
        f"qwen.sglang_config={plain_sglang.relative_to(paths.REPO_ROOT)}"]))
    try:
        assert qwen_common.config_path("vllm-omni") == plain_vllm
        assert qwen_common.config_path("sglang-omni") == plain_sglang
        assert qwen_common.config_path("vllm-omni", stage_timing=True) == qwen_common.VLLM_STAGE_TIMING_CONFIG
        assert qwen_common.config_path("vllm-omni", batch_config=True) == qwen_common.VLLM_BATCH_CONFIG
    finally:
        overrides.reset()


def test_qwen_config_override_rejects_a_missing_file():
    from exps.core import overrides

    overrides.set_current(overrides.build(options=["qwen.vllm_deploy_config=exps/configs/missing.yaml"]))
    try:
        with pytest.raises(FileNotFoundError, match="qwen.vllm_deploy_config"):
            qwen_common.config_path("vllm-omni")
    finally:
        overrides.reset()


def test_vllm_qwen_t2s_candidates_match_talker_sampling_and_only_toggle_async_chunk():
    config_dir = paths.CONFIGS / "vllm_omni"
    sync_config = yaml.safe_load(
        (config_dir / "qwen3_omni_full_1gpu_greedy.yaml").read_text(encoding="utf-8")
    )
    async_config = yaml.safe_load(
        (config_dir / "qwen3_omni_full_1gpu_async_chunk.yaml").read_text(encoding="utf-8")
    )

    assert sync_config.pop("async_chunk") is False
    assert async_config.pop("async_chunk") is True
    assert sync_config == async_config

    client_params = qwen_t2s_sampling_kwargs()
    expected_talker_params = {
        "seed": client_params["seed"],
        "temperature": client_params["talker_temperature"],
        "top_p": client_params["talker_top_p"],
        "top_k": client_params["talker_top_k"],
        "max_tokens": client_params["talker_max_new_tokens"],
        "repetition_penalty": client_params["talker_repetition_penalty"],
    }
    for candidate in (sync_config, async_config):
        talker = next(stage for stage in candidate["stages"] if stage["stage_id"] == 1)
        assert talker["default_sampling_params"] == expected_talker_params


def test_failed_start_gate_does_not_complete_cell(tmp_path):
    collector = _Collector(rows=[{
        "phase": "measured", "success": True, "e2e_ms": 100,
        "thermally_clean": True, "temperature_c_before": 65,
        "sm_clock_mhz_before": 1785, "sm_clock_mhz_nominal": 1785,
    }])
    summary = _build_summary(
        MeasurementConfig(tmp_path), {}, collector,
        {"loaded_idle_after_ready": {"status": "failed_timeout"}}, None,
    )
    assert summary["status"] == "failed_loaded_idle_gate"
    assert _status(None, 0, summary, False) == "failed_measurement"


def test_no_clean_request_does_not_complete_cell(tmp_path):
    collector = _Collector(rows=[{
        "phase": "measured", "success": True, "e2e_ms": 100,
        "cooldown_timed_out": True, "temperature_c_before": 75,
    }])
    summary = _build_summary(MeasurementConfig(tmp_path), {}, collector, {}, None)
    assert summary["status"] == "no_valid_requests"
    assert _status(None, 0, summary, False) == "failed_measurement"


def test_qwen_t2s_top_k_values_pass_mstar_validation():
    """A rejected knob is an HTTP 400 on every request of the cell (it was `code_predictor_top_k=-1`)."""
    from mstar.engine.resources.sampler.config import INT_KNOB_RULES, check_int_knob

    knobs = {k: v for k, v in qwen_t2s_sampling_kwargs().items() if k.endswith("top_k")}
    assert "code_predictor_top_k" in knobs
    for key, value in knobs.items():
        check_int_knob(key, value, *INT_KNOB_RULES["top_k"])


def test_voxserve_is_ready_only_when_its_scheduler_has_started_not_when_health_answers(tmp_path, monkeypatch):
    """VoxServe answers /health 4 s after launch and loads the model for another minute; the first request used to time out."""
    import subprocess
    import threading
    import time

    from exps.core import lifecycle
    from exps.core.lifecycle import ServiceSpec
    from exps.tasks.orpheus import cell as orpheus_cell
    from exps.core.registry import CellRef
    from exps.core.measure import MeasurementConfig
    from exps.core import registry

    plan = orpheus_cell.plan(CellRef("orpheus", "voxserve", "b1", "r01"), registry.load_task("orpheus").case("b1"),
                             MeasurementConfig(tmp_path))
    assert plan.service.ready_log_marker == "started successfully with model"

    monkeypatch.setattr(lifecycle, "_http_ready", lambda url, timeout=5.0: True)       # /health answers at once
    real_sleep = time.sleep
    monkeypatch.setattr(lifecycle.time, "sleep", lambda s: real_sleep(0.01))
    log = tmp_path / "server.log"
    log.write_text("INFO: Uvicorn running\n")
    process = subprocess.Popen(["sleep", "30"])
    try:
        spec = ServiceSpec(command="x", health_url="http://x/health", ready_timeout_s=0.3, ready_log_marker="started successfully")
        try:
            lifecycle.wait_until_ready(spec, process, log)
            raise AssertionError("health alone must not make the service ready")
        except lifecycle.ServiceStartupError:
            pass
        threading.Timer(0.1, lambda: log.write_text("INFO: Uvicorn running\nScheduler started successfully with model\n")).start()
        spec = ServiceSpec(command="x", health_url="http://x/health", ready_timeout_s=3.0, ready_log_marker="started successfully")
        assert lifecycle.wait_until_ready(spec, process, log) >= 0.0
    finally:
        process.kill()


def test_vjepa_mstar_client_can_load_torchcodec_libraries(tmp_path):
    """The client process is not started through mstar_cuda_env.sh, so it needs the CUDA 13 lib dir (libnppicc) itself."""
    from exps.core.registry import CellRef
    from exps.core.measure import MeasurementConfig
    from exps.core import registry
    from exps.tasks.vjepa_ac import cell as vjepa_cell

    plan = vjepa_cell.plan(CellRef("vjepa_ac", "mstar", "h4", "r01"), registry.load_task("vjepa_ac").case("h4"),
                           MeasurementConfig(tmp_path))
    assert "nvidia/cu13/lib" in plan.client_env["LD_LIBRARY_PATH"]


def test_qwen_vllm_gets_a_long_stage_timeout_on_the_5880_only():
    from exps import hardware
    from exps.tasks import qwen_common

    hardware.set_current("rtx5880", "2,3,4")
    assert "--stage-init-timeout 1800" in qwen_common.service("vllm-omni", "qwen_t2s", "t").command
    hardware.set_current("h200nvl")
    assert "--stage-init-timeout" not in qwen_common.service("vllm-omni", "qwen_t2s", "t").command


def _bagel_plan(framework, case, tmp_path, options=()):
    from exps.core import overrides
    from exps.tasks.bagel import cell as bagel_cell

    overrides.set_current(overrides.build(options=list(options)))
    try:
        ref = CellRef("bagel", framework, case, "r01")
        return bagel_cell.plan(ref, BAGEL.case(case), MeasurementConfig(tmp_path))
    finally:
        overrides.reset()


@pytest.mark.parametrize("framework, name", [
    ("mstar", "bagel_cfg_parallel_3gpu_kv512.yaml"),                 # rtx5880 overlay of mstar/bagel_cfg_parallel.yaml
    ("vllm-omni-single-stage", "bagel_single_stage_cfg_parallel.yaml"),
    ("vllm-omni-two-stage", "bagel_cfg_parallel_4gpu.yaml"),          # rtx5880 overlay: Thinker gets its own card
])
def test_bagel_cfg_parallel_topology_selects_the_three_gpu_configs(framework, name, tmp_path, monkeypatch):
    from exps import hardware

    monkeypatch.setattr(hardware, "detect", lambda *a, **k: hardware.load("rtx5880"))
    hardware.set_current("rtx5880")
    try:
        plan = _bagel_plan(framework, "t2i-1024-cfg3", tmp_path, ["bagel.topology=cfg-parallel"])
    finally:
        hardware.reset()
    assert name in plan.service.command
    assert "cfg_parallel" in plan.service.command
    assert plan.client_argv[plan.client_argv.index("--case") + 1] == "t2i-1024-cfg3"


def test_bagel_cfg_parallel_is_refused_where_the_machine_has_one_gpu(tmp_path):
    with pytest.raises(SystemExit, match="needs 3 GPUs"):
        _bagel_plan("mstar", "t2i-1024-cfg3", tmp_path, ["bagel.topology=cfg-parallel"])


def test_bagel_topology_option_is_validated_and_default_stays_single_gpu(tmp_path):
    with pytest.raises(ValueError, match="cfg-parallel"):
        _bagel_plan("mstar", "t2i-512-nocfg", tmp_path, ["bagel.topology=tp2"])
    plan = _bagel_plan("mstar", "t2i-512-nocfg", tmp_path)
    assert "bagel_single_gpu.yaml" in plan.service.command and "cfg_parallel" not in plan.service.command
    assert "topology" not in plan.context


def test_vjepa_frozen_inputs_pin_the_first_encoding(tmp_path):
    """The dataset re-encodes ep<N>.mp4 on every client start and the encoder is not byte-stable; the first build wins."""
    from types import SimpleNamespace

    from exps.tasks.vjepa_ac import frozen_inputs

    source = tmp_path / "ep1.mp4"
    source.write_bytes(b"first-encoding")
    frozen = tmp_path / "frozen"
    first = frozen_inputs.freeze([SimpleNamespace(video_path=str(source))], horizon=15, num_video_frames=8, directory=frozen)
    source.write_bytes(b"second-encoding-with-different-bytes")  # the next client start re-encodes the same episode
    second = frozen_inputs.freeze([SimpleNamespace(video_path=str(source))], horizon=15, num_video_frames=8, directory=frozen)
    assert first[0].video_path == second[0].video_path
    assert Path(second[0].video_path).read_bytes() == b"first-encoding"
    other_horizon = frozen_inputs.freeze([SimpleNamespace(video_path=str(source))], horizon=30, num_video_frames=8, directory=frozen)
    assert other_horizon[0].video_path != first[0].video_path


def test_bagel_batch_configs_are_servable_on_the_5880():
    """vLLM-Omni's BagelPipeline cannot batch requests (max_num_seqs>1 refuses to start), and the two-stage batch config
    co-locates both stages (OOM on 48 GiB) unless the 5880 overlay replaces it."""
    import yaml

    from exps import hardware
    from exps.core import paths

    single = yaml.safe_load((paths.CONFIGS / "vllm_omni" / "bagel_single_stage_batch.yaml").read_text())
    assert all(stage["max_num_seqs"] == 1 for stage in single["stages"])
    profile = hardware.profiles()["rtx5880"]
    overlay = profile.config_overlay_files["vllm_omni/bagel_batch.yaml"]
    two_stage = yaml.safe_load((paths.CONFIGS / "hardware" / "rtx5880" / overlay).read_text())
    assert [stage["devices"] for stage in two_stage["stages"]] == ["0", "1"]
    assert two_stage["stages"][0]["max_num_seqs"] == 16 and two_stage["stages"][1]["max_num_seqs"] == 1


def test_qwen_first_token_cases_and_fair_profile_reach_the_client(tmp_path):
    from exps.core import overrides
    from exps.tasks.qwen_a2t import cell as a2t_cell
    from exps.tasks.qwen_a2t.cases import TASK as A2T
    from exps.tasks.qwen_txt import cell as txt_cell
    from exps.tasks.qwen_txt.cases import TASK as TXT

    try:
        overrides.set_current(overrides.build(options=["qwen.sampling=lookahead-eligible", "qwen.rows=cycle"]))
        tiny = txt_cell.plan(CellRef("qwen_txt", "sglang-omni", "txt-tiny-1tok", "r01"), TXT.case("txt-tiny-1tok"), MeasurementConfig(tmp_path))
        argv = tiny.client_argv
        assert argv[argv.index("--prompt-kind") + 1] == "tiny" and argv[argv.index("--max-output-tokens") + 1] == "1"
        assert tiny.client_env["EXPS_QWEN_SAMPLING"] == "lookahead-eligible" and tiny.client_env["EXPS_QWEN_ROWS"] == "cycle"
        assert tiny.context["request_sampling_mode"] == "lookahead-eligible" and tiny.context["request_rows_mode"] == "cycle"
        one = a2t_cell.plan(CellRef("qwen_a2t", "mstar", "a2t-en-1tok", "r01"), A2T.case("a2t-en-1tok"), MeasurementConfig(tmp_path))
        assert one.client_argv[one.client_argv.index("--max-output-tokens") + 1] == "1"
    finally:
        overrides.reset()
    base = txt_cell.plan(CellRef("qwen_txt", "mstar", "txt-1k", "r01"), TXT.case("txt-1k"), MeasurementConfig(tmp_path))
    assert "EXPS_QWEN_SAMPLING" not in base.client_env and "EXPS_QWEN_ROWS" not in base.client_env
    assert base.client_argv[base.client_argv.index("--max-output-tokens") + 1] == "256"


def test_mstar_qwen_compile_switch_flips_only_the_dynamo_flag_on_rtx5880():
    from exps import hardware
    from exps.core import overrides

    hardware.set_current("rtx5880", "2,3,4")
    try:
        overrides.reset()
        assert "TORCHDYNAMO_DISABLE=1" in qwen_common.service("mstar", "qwen_txt", "t").command
        assert qwen_common.shared_context("mstar")["mstar_torch_compile"] == "off (rtx5880 default)"
        overrides.set_current(overrides.build(options=["qwen.mstar_compile=on"]))
        assert "TORCHDYNAMO_DISABLE" not in qwen_common.service("mstar", "qwen_txt", "t").command
        assert qwen_common.shared_context("mstar")["mstar_torch_compile"] == "on"
        overrides.reset()
        # T2S defaults to compile on on rtx5880 (2026-11-13 A/B); TXT / A2T keep it off
        assert "TORCHDYNAMO_DISABLE" not in qwen_common.service("mstar", "qwen_t2s", "t", batch_config=False).command
        assert qwen_common.shared_context("mstar", task="qwen_t2s")["mstar_torch_compile"] == "on (rtx5880 T2S default)"
        assert "TORCHDYNAMO_DISABLE=1" in qwen_common.service("mstar", "qwen_a2t", "t").command
        overrides.set_current(overrides.build(options=["qwen.mstar_compile=off"]))
        assert "TORCHDYNAMO_DISABLE=1" in qwen_common.service("mstar", "qwen_t2s", "t").command
        assert qwen_common.shared_context("mstar", task="qwen_t2s")["mstar_torch_compile"] == "off"
        overrides.reset()
        overrides.set_current(overrides.build(options=["qwen.mstar_compile=maybe"]))
        with pytest.raises(ValueError):
            qwen_common.service("mstar", "qwen_txt", "t")
    finally:
        overrides.reset()
        hardware.set_current("h200nvl")


def test_mstar_qwen_compile_off_sets_the_dynamo_flag_on_every_machine():
    """An h200nvl `qwen.mstar_compile=off` arm used to run with compile ON (the flag was rtx5880-only): three identical arms."""
    from exps import hardware
    from exps.core import overrides

    hardware.set_current("h200nvl")
    try:
        overrides.reset()
        for task in ("qwen_txt", "qwen_a2t", "qwen_t2s"):
            assert "TORCHDYNAMO_DISABLE" not in qwen_common.service("mstar", task, "t").command  # default on, as before
        assert qwen_common.shared_context("mstar", task="qwen_txt")["mstar_torch_compile"] == "on"
        overrides.set_current(overrides.build(options=["qwen.mstar_compile=off"]))
        for task in ("qwen_txt", "qwen_a2t", "qwen_t2s"):
            assert "TORCHDYNAMO_DISABLE=1" in qwen_common.service("mstar", task, "t").command
        assert qwen_common.shared_context("mstar", task="qwen_txt")["mstar_torch_compile"] == "off"
        overrides.set_current(overrides.build(options=["qwen.mstar_compile=on"]))
        assert "TORCHDYNAMO_DISABLE" not in qwen_common.service("mstar", "qwen_txt", "t").command
    finally:
        overrides.reset()
        hardware.set_current("h200nvl")


def test_qwen_txt_length_cases_exist_and_build_exact_prompts():
    """The prompt-length sweep cases map to server-side lengths 512 / 1,024 / 2,048; the body is the target minus the 48 template tokens."""
    from exps.tasks.qwen_txt import client
    from exps.tasks.qwen_txt.cases import TASK as TXT

    names = {c.name: c.params for c in TXT.cases}
    for n in (512, 1024, 2048):
        assert names[f"txt-{n}-1tok"] == {"max_output_tokens": 1, "prompt_kind": str(n)}
    assert client.SERVER_TEMPLATE_TOKENS == 48
