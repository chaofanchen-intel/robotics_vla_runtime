"""CPU-only checks for the BAGEL ablation, GPU-memory report and per-request memory fields."""
from __future__ import annotations

import csv
import json

import pytest

from exps.core import metrics, thermal
from exps.core.cell import CellPlan  # noqa: F401  (import check)
from exps.core.measure import MeasurementConfig
from exps.core.registry import CellRef, expand
from exps.tasks.bagel import cell as bagel_cell
from exps.tasks.bagel import client as bagel_client
from exps.tasks.bagel.cases import TASK as BAGEL
from exps.tools import analyze_bagel_ablation as ablation
from exps.tools import report_gpu_memory as memory_report


def test_ablation_cases_are_never_in_nightly_or_full():
    """Only equal-work headline cases are in a profile; sweeps, step / token / batch ablations and the unequal default request are not."""
    ablation_names = {c.name for c in BAGEL.cases
                      if any(k in c.params for k in ("timesteps", "max_tokens", "batch_size", "server_timesteps", "deploy_variant"))
                      or c.name.endswith("-default") or c.name.startswith("i2i-512-nocfg")}
    ablation_names -= {"i2t-256", "i2t-512"}  # i2t carries max_tokens only as the headline default, no override
    assert ablation_names, "ablation cases must be registered"
    for profile in ("smoke", "nightly", "full"):
        selected = {ref.case for ref in expand(BAGEL, profile=profile, repeats=1)}
        assert not (selected & ablation_names)


def test_ablation_cell_plan_threads_knobs_to_client(tmp_path):
    ref = CellRef("bagel", "mstar", "t2i-512-steps10", "r01")
    plan = bagel_cell.plan(ref, BAGEL.case(ref.case), MeasurementConfig(tmp_path))
    argv = plan.client_argv
    assert argv[argv.index("--timesteps") + 1] == "10"
    assert "--cfg-off" not in argv

    ref = CellRef("bagel", "vllm-omni-single-stage", "t2i-256-nocfg", "r01")
    plan = bagel_cell.plan(ref, BAGEL.case(ref.case), MeasurementConfig(tmp_path))
    assert "--cfg-off" in plan.client_argv
    assert plan.context["ablation"]["is_ablation"] is True

    ref = CellRef("bagel", "mstar", "i2t-512-tokens64", "r01")
    plan = bagel_cell.plan(ref, BAGEL.case(ref.case), MeasurementConfig(tmp_path))
    assert plan.client_argv[plan.client_argv.index("--max-tokens") + 1] == "64"

    ref = CellRef("bagel", "mstar", "t2i-256-default", "r01")
    plan = bagel_cell.plan(ref, BAGEL.case(ref.case), MeasurementConfig(tmp_path))
    assert plan.client_argv[plan.client_argv.index("--timesteps") + 1] == "50"
    assert plan.context["ablation"]["is_ablation"] is False


def test_cfg_ladder_cases_change_only_the_cfg_img_scale(tmp_path):
    """`-cfg3` makes both engines run three CFG branches; the default request runs two on vLLM-Omni and (always) three on M*."""
    for resolution in (256, 512, 768):
        ref = CellRef("bagel", "vllm-omni-single-stage", f"t2i-{resolution}-cfg3", "r01")
        plan = bagel_cell.plan(ref, BAGEL.case(ref.case), MeasurementConfig(tmp_path))
        argv = plan.client_argv
        assert argv[argv.index("--cfg-img-scale") + 1] == "2.0" and "--cfg-off" not in argv
        assert plan.context["ablation"]["is_ablation"] is True
        assert BAGEL.case(f"t2i-{resolution}-nocfg").params["cfg_off"] is True
    assert "--cfg-img-scale" not in bagel_cell.plan(CellRef("bagel", "mstar", "t2i-768", "r01"), BAGEL.case("t2i-768"),
                                                    MeasurementConfig(tmp_path)).client_argv
    base = bagel_client.model_kwargs_for("t2i", 768, 128)
    three = bagel_client.model_kwargs_for("t2i", 768, 128, cfg_img_scale=2.0)
    assert base["cfg_img_scale"] == 1.0 and three["cfg_img_scale"] == 2.0
    assert {k: v for k, v in three.items() if k != "cfg_img_scale"} == {k: v for k, v in base.items() if k != "cfg_img_scale"}
    # cfg_off wins: the two knobs are never combined
    assert bagel_client.model_kwargs_for("t2i", 256, 128, cfg_off=True, cfg_img_scale=2.0)["cfg_img_scale"] == 1.0
    # equal-work headline cases are in the profiles (256 nightly, 512 full); 768 and the unequal default request are explicit-only
    nightly = {r.case for r in expand(BAGEL, profile="nightly", repeats=1)}
    full = {r.case for r in expand(BAGEL, profile="full", repeats=1)}
    assert {"t2i-256-nocfg", "t2i-256-cfg3"} <= nightly and not any("512" in c for c in nightly)
    assert {"t2i-512-nocfg", "t2i-512-cfg3"} <= full
    for profile in ("smoke", "nightly", "full"):
        selected = {r.case for r in expand(BAGEL, profile=profile, repeats=1)}
        assert not any(c.endswith("-default") or "768" in c for c in selected)


def test_model_kwargs_ablation_changes_only_the_requested_knob():
    base = bagel_client.model_kwargs_for("t2i", 256, 128)
    steps = bagel_client.model_kwargs_for("t2i", 256, 128, timesteps=10)
    nocfg = bagel_client.model_kwargs_for("t2i", 256, 128, cfg_off=True)
    assert steps["num_timesteps"] == 10 and steps["num_inference_steps"] == 10
    assert {k: v for k, v in steps.items() if k not in ("num_timesteps", "num_inference_steps")} == {
        k: v for k, v in base.items() if k not in ("num_timesteps", "num_inference_steps")
    }
    assert nocfg["cfg_text_scale"] == 1.0 and nocfg["cfg_img_scale"] == 1.0
    assert base["cfg_text_scale"] == 4.0
    assert nocfg["num_timesteps"] == base["num_timesteps"] == 50


def test_linear_fit_recovers_intercept_and_slope():
    fit = ablation.linear_fit([(5, 600.0), (10, 700.0), (25, 1000.0), (50, 1500.0)])
    assert fit["a_ms"] == pytest.approx(500.0)
    assert fit["b_ms_per_unit"] == pytest.approx(20.0)
    assert fit["r2"] == pytest.approx(1.0)
    assert ablation.linear_fit([(5, 1.0), (10, 2.0)]) is None


def _fit(a, b):
    return {"fit": {"a_ms": a, "b_ms_per_unit": b, "r2": 0.999, "n_points": 4}, "fit_usable": True}


def test_gap_decomposition_classifies_fixed_vs_marginal():
    fixed = ablation.gap_decomposition({"mstar": _fit(300, 40), "vllm": _fit(1200, 40)}, 50)["vllm"]
    assert fixed["reading"] == "mostly_fixed_per_request_cost"
    assert fixed["fixed_share"] == pytest.approx(1.0)

    marginal = ablation.gap_decomposition({"mstar": _fit(300, 40), "vllm": _fit(300, 58)}, 50)["vllm"]
    assert marginal["reading"] == "mostly_per_step_or_per_token_cost"

    mixed = ablation.gap_decomposition({"mstar": _fit(300, 40), "vllm": _fit(600, 46)}, 50)["vllm"]
    assert mixed["reading"] == "mixed"

    slower = ablation.gap_decomposition({"mstar": _fit(900, 40), "vllm": _fit(300, 40)}, 50)["vllm"]
    assert slower["status"] == "reference_not_faster"


def test_unusable_fit_is_not_reported_as_a_number():
    bad = {"fit": {"a_ms": 1, "b_ms_per_unit": 1, "r2": 0.5, "n_points": 4}, "fit_usable": False}
    result = ablation.gap_decomposition({"mstar": _fit(300, 40), "vllm": bad}, 50)["vllm"]
    assert result["status"] == "fit_unusable"


def _write_cell(root, framework, case, repeat, p50, status="completed"):
    cell = root / framework / case / repeat
    cell.mkdir(parents=True)
    (cell / "manifest.json").write_text(json.dumps({"status": status, "framework": framework, "case": case}))
    (cell / "summary.json").write_text(json.dumps({"results": {"metrics": {"e2e_ms": {"p50": p50}}}}))


def test_collect_p50s_keeps_starts_separate_and_skips_incomplete(tmp_path):
    _write_cell(tmp_path, "mstar", "t2i-256", "r01", 2600.0)
    _write_cell(tmp_path, "mstar", "t2i-256", "r02", 2700.0)
    _write_cell(tmp_path, "mstar", "t2i-256", "r03", 99999.0, status="failed")
    found = ablation.collect_p50s(tmp_path)
    assert found[("mstar", "t2i-256")] == [2600.0, 2700.0]


def _gate_csv(path, values):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["wall_clock", "memory_used_mib"])
        writer.writeheader()
        for value in values:
            writer.writerow({"wall_clock": "t", "memory_used_mib": value})


def test_cell_footprint_subtracts_idle_baseline_and_handles_missing_peak(tmp_path):
    _gate_csv(tmp_path / "prestart_gate.csv", [14, 14, 14])
    _gate_csv(tmp_path / "loaded_idle_after_ready.csv", [40000, 40000])
    _gate_csv(tmp_path / "loaded_idle_after_precondition.csv", [48000, 48000, 48000])
    (tmp_path / "requests.jsonl").write_text(json.dumps({"phase": "measured", "success": True}) + "\n")
    footprint = memory_report.cell_footprint(tmp_path)
    assert footprint["service_resident_mib"] == 47986
    assert footprint["in_request_peak_mib"] is None and footprint["service_peak_mib"] is None

    (tmp_path / "requests.jsonl").write_text(
        json.dumps({"phase": "precondition", "memory_used_mib_max": 99999}) + "\n"
        + json.dumps({"phase": "measured", "memory_used_mib_max": 50000}) + "\n"
        + json.dumps({"phase": "measured", "memory_used_mib_max": 51000}) + "\n"
    )
    footprint = memory_report.cell_footprint(tmp_path)
    assert footprint["in_request_peak_mib"] == 51000  # precondition peak is not counted
    assert footprint["service_peak_mib"] == 51000 - 14


def test_cell_footprint_returns_none_without_loaded_gates(tmp_path):
    assert memory_report.cell_footprint(tmp_path) is None


def test_monitor_verdict_reports_memory_peak():
    monitor = thermal.RequestThermalMonitor()
    base = {"clocks_sm_mhz": 1785.0, "clocks_max_sm_mhz": 1785.0, "temperature_c": 60.0,
            "memory_temperature_c": 55.0, "power_draw_w": 300.0, "any_event_reason_active": False,
            "event_reasons": {}}
    monitor._samples = [{**base, "memory_used_mib": m} for m in (48000.0, 51000.0, 49000.0)]
    verdict = monitor.verdict()
    assert verdict["memory_used_mib_start"] == 48000.0
    assert verdict["memory_used_mib_max"] == 51000.0


def test_summary_carries_gpu_memory_block():
    rows = [
        {"phase": "measured", "success": True, "e2e_ms": 100.0, "memory_used_mib_before": 48000.0,
         "memory_used_mib_max": 50000.0, "temperature_c_before": 60.0},
        {"phase": "measured", "success": True, "e2e_ms": 110.0, "memory_used_mib_before": 48000.0,
         "memory_used_mib_max": 51000.0, "temperature_c_before": 60.0},
    ]
    summary = metrics.summarize_start(rows)
    assert summary["gpu_memory_mib"]["peak_during_request"]["max"] == 51000.0
    legacy = metrics.summarize_start([{"phase": "measured", "success": True, "e2e_ms": 1.0}])
    assert legacy["gpu_memory_mib"]["peak_during_request"]["count"] == 0


# --- concurrency and A2T parity tools -------------------------------------------------------

from exps.tools import analyze_concurrency as concurrency  # noqa: E402
from exps.tools import analyze_qwen_a2t_parity as parity  # noqa: E402


def test_wave_validator_uses_each_requests_own_fixture_index():
    """Regression: a wave must validate response k against fixture row offset+k, not offset."""
    import argparse
    from exps.core import benchmark_client

    seen: list[int] = []

    class _Result:  # minimal stand-in for an adapter result
        pass

    def fake_send_batch(system, wave, **kwargs):
        return [_Result() for _ in wave]

    def fake_row(result):
        return {"success": True, "e2e_ms": 10.0, "audio_duration_s": 1.0, "rtf": 0.1, "ttfb_ms": 1.0,
                "ttft_ms": None, "ttfa_ms": None, "itl_ms": [], "output_tokens": 1, "output_bytes": 1}

    import exps.core.adapters as adapters
    orig = (adapters.send_batch, adapters.to_metric_row, adapters.inference_system, adapters.model_instance)
    adapters.send_batch, adapters.to_metric_row = fake_send_batch, fake_row
    adapters.inference_system, adapters.model_instance = (lambda *_: None), (lambda *_a, **_k: None)
    try:
        captured = {}

        def fake_run_measurement(config, *, issue, context, throughput_batch=None):
            captured["row"] = issue(1, "measured")
            return {"status": "completed", "results": {}}

        orig_rm = benchmark_client.run_measurement
        benchmark_client.run_measurement = fake_run_measurement
        args = argparse.Namespace(
            framework="mstar", case="x", base_url="http://x", output_dir=__import__("pathlib").Path("/tmp/_wave"),
            request_timeout_s=1.0, save_outputs=False, measured=1, preconditions=0, gpu_index=0,
            cooldown_target_c=None, cooldown_continuous_s=None, cooldown_timeout_s=None,
            loaded_idle_target_c=None, loaded_idle_continuous_s=None, loaded_idle_timeout_s=None,
            with_throughput=False, throughput_requests=0, throughput_concurrency=1,
            calibration_source="t", memory_cooldown_target_c=None,
        )
        benchmark_client.run(
            args=args, task="t", model_type="m", request_inputs=list(range(8)), context={},
            validate=lambda result, row, index: seen.append(index) or {}, batch_size=4,
        )
    finally:
        benchmark_client.run_measurement = orig_rm
        adapters.send_batch, adapters.to_metric_row, adapters.inference_system, adapters.model_instance = orig
    assert seen == [0, 1, 2, 3]


def _cell(root, task, framework, case, repeat, summary=None, requests=None, status="completed"):
    cell = root / task / framework / case / repeat
    cell.mkdir(parents=True)
    (cell / "manifest.json").write_text(json.dumps({"status": status, "framework": framework, "case": case}))
    if summary is not None:
        (cell / "summary.json").write_text(json.dumps(summary))
    if requests is not None:
        (cell / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in requests) + "\n")


def test_concurrency_scaling_and_ratio(tmp_path):
    def summ(wall, audio=None):
        return {"results": {"metrics": {"e2e_ms": {"p50": wall}, "audio_duration_s": {"p50": audio},
                                        "rtf": {"p50": 0.5}}}}

    for fw, w1, w16 in (("mstar", 1000.0, 2000.0), ("vllm-omni", 1000.0, 8000.0)):
        _cell(tmp_path, "QWEN_T2S", fw, "verbatim-en", "r01", summ(w1, 4.0))
        _cell(tmp_path, "QWEN_T2S", fw, "verbatim-en-b16", "r01", summ(w16, 64.0))
    tasks = concurrency.collect(tmp_path)
    mstar = tasks["QWEN_T2S"]["mstar"]
    assert mstar[16]["requests_per_s"] == pytest.approx(8.0)
    assert mstar[16]["throughput_scaling_vs_b1"] == pytest.approx(8.0)
    assert tasks["QWEN_T2S"]["vllm-omni"][16]["throughput_scaling_vs_b1"] == pytest.approx(2.0)
    ratios = concurrency.ratios_vs_reference(tasks["QWEN_T2S"])
    assert ratios[16]["vllm-omni"] == pytest.approx(4.0)


def test_split_spikes_flags_only_the_cold_request():
    warm, dropped = parity.split_spikes([4900.0, 520.0, 410.0, 450.0])
    assert dropped == 1 and 4900.0 not in warm
    warm, dropped = parity.split_spikes([520.0, 410.0, 450.0])
    assert dropped == 0


def _starts(first, rest):
    return {"starts": 3, "p50_all": statistics_median([first] + rest), "p50_warm": statistics_median(rest),
            "first_ms": first, "spikes_per_start": 1.0, "requests_per_start": 1 + len(rest)}


def statistics_median(values):
    import statistics
    return statistics.median(values)


def test_parity_judgement_rule():
    near = {"mstar": _starts(5000, [350, 360, 355]), "vllm-omni": _starts(340, [338, 342, 340])}
    assert parity.judge(near)["verdict"] == "parity"
    far = {"mstar": {**_starts(5000, [500, 510, 505]), "p50_all": 650.0}, "vllm-omni": _starts(340, [338, 342, 340])}
    verdict = parity.judge(far)
    assert verdict["verdict"] == "residual_gap"
    assert verdict["comparisons"]["vllm-omni"]["mstar_cold_dominated"] is True
    few = {"mstar": {**_starts(1, [1, 1]), "starts": 1}, "vllm-omni": _starts(1, [1, 1])}
    assert parity.judge(few)["verdict"] == "insufficient_reference_starts"


def test_start_stats_from_requests_jsonl(tmp_path):
    rows = [{"phase": "precondition", "success": True, "e2e_ms": 9999.0}] + [
        {"phase": "measured", "success": True, "e2e_ms": v} for v in (4800.0, 520.0, 410.0)
    ]
    path = tmp_path / "requests.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows))
    stats = parity.start_stats(path)
    assert stats["first_ms"] == 4800.0 and stats["spikes"] == 1 and stats["n"] == 3


# --- M* server-side step ablation, enforce_eager variant, series mapping -------------------------


def test_mstar_server_steps_case_uses_patched_cache_dir_and_skips_vllm(tmp_path):
    ref = CellRef("bagel", "mstar", "t2i-512-srv-steps10", "r01")
    plan = bagel_cell.plan(ref, BAGEL.case(ref.case), MeasurementConfig(tmp_path))
    assert "models/cache_bagel_steps10" in plan.service.command
    assert "--cache-dir \"$PWD/models/cache_bagel_steps10\"" in plan.service.command
    assert plan.context["ablation"]["server_timesteps"] == 10
    # ordinary M* cells keep the default cache
    base = bagel_cell.plan(CellRef("bagel", "mstar", "t2i-256-nocfg", "r01"), BAGEL.case("t2i-256-nocfg"), MeasurementConfig(tmp_path))
    assert "cache_bagel_steps" not in base.service.command
    cells = {r.framework for r in expand(BAGEL, profile="full", cases=("t2i-512-srv-steps5",), repeats=1)}
    assert cells == {"mstar"}


def test_graph_variant_only_for_single_stage_and_uses_graph_config(tmp_path):
    case = BAGEL.case("i2t-512-graph-tokens64")
    cells = {r.framework for r in expand(BAGEL, profile="full", cases=(case.name,), repeats=1)}
    assert cells == {"vllm-omni-single-stage"}
    plan = bagel_cell.plan(CellRef("bagel", "vllm-omni-single-stage", case.name, "r01"), case, MeasurementConfig(tmp_path))
    assert "bagel_single_stage_graph.yaml" in plan.service.command
    assert plan.client_argv[plan.client_argv.index("--max-tokens") + 1] == "64"


def test_series_cases_mapping():
    fw, cases = ablation.series_cases("t2i", "mstar")
    assert fw == "mstar" and ("t2i-512-srv-steps5" in [c for _, c in cases]) and ("t2i-512-default" in [c for _, c in cases])
    assert "t2i-512-steps5" not in [c for _, c in cases]  # inert request-level cases are excluded for M*
    fw, cases = ablation.series_cases("t2i", "vllm-omni-two-stage")
    assert "t2i-512-steps5" in [c for _, c in cases]
    fw, cases = ablation.series_cases("i2t", ablation.GRAPH_FW)
    assert fw == "vllm-omni-single-stage" and cases[0][1].startswith("i2t-512-graph-tokens")
    assert ablation.series_cases("t2i", ablation.GRAPH_FW) is None


def test_make_bagel_steps_cache_builds_symlinks_and_patches_config(tmp_path):
    from exps.tools import make_bagel_steps_cache as mk
    root = tmp_path / "cache"
    snap = root / mk.REPO_DIR / "snapshots" / mk.REVISION
    snap.mkdir(parents=True)
    (snap / "config.json").write_text(json.dumps({"llm_config": {}}))
    (snap / "weights.bin").write_bytes(b"x")
    out = mk.build(7, cache_root=root)
    new_snap = out / mk.REPO_DIR / "snapshots" / mk.REVISION
    assert json.loads((new_snap / "config.json").read_text())["num_timesteps"] == 7
    assert (new_snap / "weights.bin").is_symlink()
    assert mk.verify(7, cache_root=root) == []
    assert json.loads((snap / "config.json").read_text()).get("num_timesteps") is None  # source untouched


def test_nsys_wrap_is_noop_without_env_and_wraps_with_it(tmp_path):
    from exps.core import overrides
    from exps.core.lifecycle import nsys_wrap

    run_root = tmp_path / "BAGEL" / "mstar" / "t2i-256" / "r01"
    overrides.reset()
    assert nsys_wrap("serve x", run_root) == ("serve x", 0.0)
    overrides.set_current(overrides.build(nsys_dir=str(tmp_path / "nsys")))
    try:
        command, grace = nsys_wrap("serve 'x'", run_root)
    finally:
        overrides.reset()
    assert command.startswith("nsys profile ")
    assert "mstar-t2i-256-r01" in command and grace >= 120


def test_nsys_wrap_names_report_from_service_lifecycles_dir(tmp_path):
    from exps.core import overrides
    from exps.core.lifecycle import nsys_wrap

    overrides.set_current(overrides.build(nsys_dir=str(tmp_path)))
    try:
        run_root = tmp_path / "C1" / "_service_lifecycles" / "bagel_vllm-omni-single-stage_t2i-256-nocfg_r01"
        command, _ = nsys_wrap("serve", run_root)
    finally:
        overrides.reset()
    assert "vllm-omni-single-stage-t2i-256-nocfg-r01" in command


def test_mstar_serve_extra_env_and_args_come_from_overrides():
    from pathlib import Path

    from exps import frameworks
    from exps.core import overrides

    kwargs = {"model": "bagel", "config": Path("c.yaml"), "port_number": 1234, "tag": "t"}
    overrides.reset()
    plain = frameworks.mstar_serve(**kwargs)
    assert "TORCHDYNAMO_DISABLE" not in plain and "--enable-nvtx" not in plain
    overrides.set_current(overrides.build(env=["mstar:TORCHDYNAMO_DISABLE=1"], serve_args=["mstar:--enable-nvtx"]))
    try:
        command = frameworks.mstar_serve(**kwargs)
        # another framework family's overrides never leak into M*'s command
        overrides.set_current(overrides.build(env=["vllm-omni:FOO=1"]))
        assert "FOO=1" not in frameworks.mstar_serve(**kwargs)
    finally:
        overrides.reset()
    # `mstar serve` itself rejects --enable-nvtx, so the flag must be carried by the launcher, not the CLI args
    assert "HF_HUB_OFFLINE=1 TORCHDYNAMO_DISABLE=1 exec envs/mstar/bin/python exps/tools/mstar_serve_nvtx.py serve" in command
    assert "--enable-nvtx" not in command.split("mstar_serve_nvtx.py", 1)[1]
    assert "exec envs/mstar/bin/mstar serve" in plain


def test_mstar_serve_nvtx_launcher_passes_the_flag_to_the_server(tmp_path):
    """Run the real `mstar serve` argument parser and check the server would receive --enable-nvtx."""
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    probe = (
        "import runpy, sys, json\n"
        "from mstar.api_server import entrypoint\n"
        "seen = []\n"
        "entrypoint.main = lambda argv=None: seen.append(list(argv))\n"
        "sys.argv = ['mstar_serve_nvtx.py', 'serve', 'qwen3_omni', '--config', sys.argv[1], '--port', '1']\n"
        "runpy.run_path(sys.argv[0] if False else %r, run_name='__main__')\n"
        "print(json.dumps(seen))\n" % str(repo / "exps/tools/mstar_serve_nvtx.py")
    )
    cfg = tmp_path / "c.yaml"
    cfg.write_text("{}")
    result = subprocess.run([sys.executable, "-c", probe, str(cfg)], capture_output=True, text=True, cwd=repo)
    assert result.returncode == 0, result.stderr[-2000:]
    import json
    (argv,) = json.loads(result.stdout.strip().splitlines()[-1])
    assert argv[-1] == "--enable-nvtx" and "--config" in argv


def test_nsys_wrap_flush_interval_is_opt_in(tmp_path):
    from exps.core import lifecycle, overrides

    run_root = tmp_path / "C" / "T" / "fw" / "case" / "r01"
    try:
        overrides.set_current(overrides.build(nsys_dir=str(tmp_path)))
        assert "--cuda-flush-interval" not in lifecycle.nsys_wrap("serve", run_root)[0]
        overrides.set_current(overrides.build(nsys_dir=str(tmp_path), nsys_flush_ms=1000))
        assert "--cuda-flush-interval=1000 " in lifecycle.nsys_wrap("serve", run_root)[0]
    finally:
        overrides.reset()


def test_resolution_sweep_cases_are_t2i_only_and_never_in_default_profiles():
    names = {c.name for c in BAGEL.cases}
    assert {"t2i-384", "t2i-768"} <= names and "i2i-384" not in names
    for profile in ("smoke", "nightly", "full"):
        selected = {ref.case for ref in expand(BAGEL, profile=profile, repeats=1)}
        assert not ({"t2i-384", "t2i-768"} & selected)
