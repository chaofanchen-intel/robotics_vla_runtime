"""Catalogue, overrides, protocol options and result validation. CPU only, no GPU, no services."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pytest
import yaml

from exps import hardware
from exps.core import cli, overrides, paths, registry
from exps.core.measure import MeasurementConfig
from exps.suite import analysis, catalogue, runner
from exps.suite.spec import Arm, Experiment, Expect, campaign_name


# --- catalogue consistency ---------------------------------------------------------------------------------------------------
def _experiments_with_cells():
    # multi-GPU experiments (min_gpus > 1) cannot be planned under the single-GPU profile the tests pin; see the skip test
    return [e for e in catalogue.ALL.values() if e.task and e.min_gpus == 1]


def test_every_experiment_refers_to_real_tasks_cases_and_frameworks():
    for experiment in _experiments_with_cells():
        spec = registry.load_task(experiment.task)
        for arm in map(experiment.resolved, experiment.arms):
            assert arm.frameworks, f"{experiment.name}/{arm.name}: no frameworks"
            assert set(arm.frameworks) <= set(spec.frameworks), (experiment.name, arm.name, arm.frameworks)
            for case in arm.cases:
                spec.case(case)  # raises KeyError for a typo
            assert expected_nonempty(experiment.task, arm)


def expected_nonempty(task, arm):
    return bool(analysis.expected_cells(task, arm))


def test_every_experiment_plans_without_a_gpu():
    for experiment in _experiments_with_cells():
        if experiment.min_gpus > hardware.current().gpu_count:
            continue   # skipped on this machine, the same rule `exps.suite run` applies (runner.run_experiments)
        for run in runner.arm_runs(experiment, dry_run=True, protocol="gated"):
            plan = runner.dry_plan(run)
            assert plan["cells"] > 0 and plan["plans"], (experiment.name, run.arm.name)
            # the overrides of the arm reach the manifest-to-be (the plan prints what the cell will record)
            want = overrides.build(env=[runner.expand_text(v, campaign=run.campaign) for v in run.arm.env]).as_manifest()
            assert plan["plans"][0]["overrides"]["env"] == want["env"]


def test_headline_experiments_use_only_equal_work_cases():
    """No baseline claim may rest on the BAGEL default request, where M* computes a CFG branch vLLM-Omni does not."""
    for experiment in catalogue.BASELINES:
        for case in experiment.cases:
            assert not case.endswith("-default"), (experiment.name, case)
    bagel = catalogue.get("baseline-bagel")
    assert set(bagel.cases) == {"t2i-512-nocfg", "t2i-512-cfg3", "i2i-512", "i2t-512"}   # 512 is the main operating point
    assert set(catalogue.get("baseline-bagel-1024").cases) == {"t2i-1024-nocfg", "t2i-1024-cfg3", "i2i-1024", "i2t-1024"}


def test_every_task_has_a_baseline_and_an_mstar_ablation():
    for task in registry.TASK_ORDER:
        slug = task.replace("_", "-")
        assert f"baseline-{slug}" in catalogue.ALL, task
        assert f"mstar-ablation-{slug}" in catalogue.ALL, task
    # protocol A/B is catalogued where it can matter (the option itself exists for every task, see the forwarding test)
    for task in registry.TASK_ORDER:  # every task can be checked: the decision needs all of them before the baselines run
        assert f"protocol-ab-{task.replace('_', '-')}" in catalogue.ALL, task


def test_mstar_ablation_matrix_switches_only_documented_env_vars_and_brackets_drift():
    for task in registry.TASK_ORDER:
        experiment = catalogue.get(f"mstar-ablation-{task.replace('_', '-')}")
        names = [a.name for a in experiment.arms]
        assert names == ["A1", "C", "S", "SC", "W", "WC", "A2"], names
        assert experiment.frameworks == ("mstar",) and experiment.kind == "ab"
        env = {a.name: set(a.env) for a in experiment.arms}
        assert env["A1"] == env["A2"] == set()
        assert env["C"] == {"mstar:TORCHDYNAMO_DISABLE=1"}
        assert env["S"] == {"mstar:MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=0"}
        assert env["SC"] == env["S"] | env["C"]
        assert env["W"] == {"mstar:MSTAR_MAX_CONSECUTIVE_SPEC_STEPS=0", "mstar:MSTAR_PRE_PLAN_SPEC=0", "mstar:MSTAR_ENGINE_STEP_SYNC=1"}
        assert env["WC"] == env["W"] | env["C"]


def test_protocol_ab_arms_fix_their_protocol_and_name_their_campaigns_distinctly():
    experiment = catalogue.get("protocol-ab-pi05")
    hardware.reset()
    names = [campaign_name(experiment, a, run_id="01", protocol="gated", gap_s=0.0) for a in experiment.arms]
    assert len(set(names)) == 3 and not any(n.endswith("_B2B") for n in names)
    assert [a.name for a in catalogue.get("protocol-ab-bagel").arms] == ["gated", "b2b"]   # gap30 is a Pi0.5-only diagnostic
    runs = {r.arm.name: r for r in runner.arm_runs(experiment)}
    assert runs["gated"].protocol == "gated" and runs["b2b"].protocol == "back-to-back"
    assert runs["gap30"].gap_s == 30.0 and "--inter-request-gap-s" in runs["gap30"].argv
    assert runs["gated"].argv[runs["gated"].argv.index("--protocol") + 1] == "gated" and "back-to-back" in runs["b2b"].argv


def test_a_suite_level_protocol_gets_its_own_campaign():
    experiment = catalogue.get("mstar-ablation-qwen-t2s")
    gated = {r.campaign for r in runner.arm_runs(experiment, protocol="gated")}
    b2b = {r.campaign for r in runner.arm_runs(experiment, protocol="back-to-back")}
    gap = {r.campaign for r in runner.arm_runs(experiment, protocol="back-to-back", gap_s=30.0)}
    assert not (gated & b2b) and not (b2b & gap) and all(c.endswith("_B2B_01_B2B") or c.endswith("_B2B") for c in b2b)


def _script_groups(script_name: str) -> dict[str, list[str]]:
    """group -> the experiment names of its `exps.suite run` line in `<script> --list` (none of this needs a GPU)."""
    import re
    import subprocess

    script = paths.REPO_ROOT / script_name
    if not script.is_file():
        pytest.skip(f"{script_name} is not in this checkout")
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    listing = subprocess.run(["bash", str(script), "--list"], capture_output=True, text=True, check=True).stdout
    groups: dict[str, list[str]] = {}
    for group, run_line in re.findall(r"^# (\S+)\n(\S[^\n]*exps\.suite run [^\n]*)$", listing, flags=re.M):
        names = re.search(r"exps\.suite run (.+?) --hardware", run_line)
        assert names and run_line.endswith("--execute"), (script_name, group, run_line)
        groups[group] = names.group(1).split()
    return groups


@pytest.mark.parametrize("script", ["run_attrib.sh", "run_h200_gap.sh", "run_h200_rest.sh"])
def test_launcher_groups_name_catalogue_experiments(script):
    """A group that names an experiment the catalogue no longer has fails at the first `exps.suite run`, hours into a queue."""
    groups = _script_groups(script)
    assert groups, script
    for group, names in groups.items():
        for name in names:
            assert name in catalogue.ALL, (script, group, name)


def test_run_attrib_pi05_groups_map_to_their_catalogue_entries():
    groups = _script_groups("run_attrib.sh")
    for group, name in (("pi05-audit", "pi05-audit"), ("pi05-nsys", "pi05-nsys")):
        assert groups[group] == [name], (group, groups[group])


def test_pi05_steps_is_three_frameworks_by_four_counts_and_24_cells():
    experiment = catalogue.get("pi05-steps")
    assert experiment.frameworks == ("mstar", "sglang-core", "vllm-omni")
    assert experiment.cases == ("droid-46row-s1", "droid-46row-s5", "droid-46row", "droid-46row-s20")
    (run,) = runner.arm_runs(experiment, protocol="back-to-back", run_id="02")
    assert runner.dry_plan(run)["cells"] == 24


def test_pi05_kv_cache_arm_turns_the_sglang_prefix_cache_on_and_the_others_keep_the_shipped_config():
    experiment = catalogue.get("pi05-kv")
    assert [arm.name for arm in experiment.arms] == ["mstar", "sglang-core", "sglang-cache", "vllm-omni"]
    assert all(len(arm.frameworks) == 1 for arm in experiment.arms)
    shipped = json.loads((paths.CONFIGS / "sglang_core" / "pi05_pipeline.json").read_text())
    assert shipped["enable_global_prefix_cache"] is False
    (opt,) = experiment.arm("sglang-cache").opts
    cache_on = json.loads((paths.REPO_ROOT / opt.split("=", 1)[1]).read_text())
    assert cache_on["enable_global_prefix_cache"] is True
    assert {k: v for k, v in cache_on.items() if k != "enable_global_prefix_cache"} == \
        {k: v for k, v in shipped.items() if k != "enable_global_prefix_cache"}
    assert experiment.arm("sglang-cache").expect.command_has == ("pi05_pipeline_prefix_cache.json",)
    assert experiment.arm("sglang-core").opts == () and experiment.arm("mstar").opts == () and experiment.arm("mstar").env == ()


def test_pi05_graph_split_arms_switch_one_graph_each():
    experiment = catalogue.get("pi05-graph-split")
    assert experiment.kind == "ab" and experiment.frameworks == ("sglang-core",)
    shipped = json.loads((paths.CONFIGS / "sglang_core" / "pi05_pipeline.json").read_text())
    expected = {"sglang": {}, "sglang-prefix-off": {"enable_prefix_cuda_graph": False},
                "sglang-action-off": {"enable_action_cuda_graph": False}}
    assert [arm.name for arm in experiment.arms] == list(expected)
    for arm in experiment.arms:
        assert len(arm.frameworks) == 1, arm.name
        config = shipped
        if arm.opts:
            (opt,) = arm.opts
            config = json.loads((paths.REPO_ROOT / opt.split("=", 1)[1]).read_text())
        assert config == {**shipped, **expected[arm.name]}, arm.name


def test_pi05_sglang_buckets_arm_changes_only_the_prompt_buckets():
    experiment = catalogue.get("pi05-sglang-buckets")
    assert experiment.kind == "ab" and experiment.frameworks == ("sglang-core",)
    shipped = json.loads((paths.CONFIGS / "sglang_core" / "pi05_pipeline.json").read_text())
    assert "prompt_token_buckets" not in shipped and shipped["max_token_len"] == 200
    assert [arm.name for arm in experiment.arms] == ["sglang", "sglang-buckets"]
    assert experiment.arm("sglang").opts == ()
    (opt,) = experiment.arm("sglang-buckets").opts
    buckets = json.loads((paths.REPO_ROOT / opt.split("=", 1)[1]).read_text())
    assert buckets == {**shipped, "prompt_token_buckets": [200]}
    assert experiment.arm("sglang-buckets").expect.log_lacks == ("VLA denoise CUDA graph capacity reached",)


def test_pool_and_compile_groups_map_to_their_catalogue_entries():
    import subprocess

    listing = subprocess.run(["bash", str(paths.REPO_ROOT / "run_attrib.sh"), "--list"], capture_output=True, text=True, check=True).stdout
    for group, name in (("bagel-i2i-pool", "bagel-i2i-pool"), ("t2s-conc-compile-ab", "t2s-conc-compile-ab")):
        run_line = listing.split(f"# {group}\n", 1)[1].split("\n", 1)[0]
        assert f"exps.suite run {name} " in run_line and run_line.endswith("--execute"), (group, run_line)


def test_bagel_i2i_pool_arms_change_only_the_pool_timeout_or_cap():
    experiment = catalogue.get("bagel-i2i-pool")
    assert experiment.kind == "ab" and experiment.frameworks == ("vllm-omni-two-stage",)
    expected = {"seqs3-t30-pool16": "pool16g", "seqs3-t900-pool16": "pool16g_kvtimeout900",
                "seqs16-t30-pool16": "seqs16_pool16g", "seqs16-t900-pool16": "seqs16_pool16g_kvtimeout900"}
    assert [arm.name for arm in experiment.arms] == list(expected)
    for arm in experiment.arms:
        (opt,) = arm.opts
        key = "vllm_omni/bagel_two_stage_mooncake_tcp_prefix_cache_" + expected[arm.name]
        assert opt == "config.vllm-omni={variant:" + key + "}", arm.name
        assert "{variant_file:" + key + "}" in arm.expect.command_has, arm.name
        assert "Pool exhausted" in arm.expect.log_lacks, arm.name
        assert ("after 30.0s" in arm.expect.log_lacks) == arm.name.endswith("t900-pool16"), arm.name
        assert arm.cases == ("i2i-512-steps5-b4", "i2i-512-steps5-b16"), arm.name


def test_t2s_compile_ab_pins_the_switch_in_a_b_a_order():
    experiment = catalogue.get("t2s-conc-compile-ab")
    assert experiment.frameworks == ("mstar",)
    assert experiment.cases == ("verbatim-en", "verbatim-en-b4", "verbatim-en-b16")
    assert experiment.metric == "ms_per_audio_s" and experiment.repeats == 2 and experiment.measured == 6
    assert [arm.name for arm in experiment.arms] == ["compile-on", "compile-off", "compile-on-2"]
    switch = {"compile-on": "qwen.mstar_compile=on", "compile-off": "qwen.mstar_compile=off", "compile-on-2": "qwen.mstar_compile=on"}
    for arm in experiment.arms:
        assert arm.opts == ("qwen.rows=cycle", switch[arm.name]), arm.name
        assert arm.env == (), arm.name


def test_hook_arms_demand_a_proof():
    for name, arm in (("bagel-attn-backend", "mstar-probe"), ("bagel-attn-backend", "mstar-no-flash-attn")):
        assert catalogue.get(name).arm(arm).expect.proof, (name, arm)


# --- overrides ----------------------------------------------------------------------------------------------------------------
def test_override_parsing_and_scopes():
    built = overrides.build(env=["mstar:A=1", "*:B=2", "vllm-omni:A=3"], serve_args=["mstar:--x"], options=["pi05.vllm_graph=1"])
    assert built.env_for("mstar") == {"A": "1", "B": "2"} and built.env_for("vllm-omni") == {"A": "3", "B": "2"}
    assert built.env_for("sglang") == {"B": "2"} and built.args_for("mstar") == ["--x"] and built.args_for("sglang") == []
    assert built.option("pi05", "vllm_graph") == "1" and built.option("pi05", "other", "d") == "d"
    for bad in (["mstar"], ["nope:A=1"], ["mstar:A"]):
        with pytest.raises(ValueError):
            overrides.build(env=bad)
    with pytest.raises(ValueError):
        overrides.build(options=["novalue"])


def test_overrides_are_recorded_in_the_dry_run_plan():
    from exps import run as run_module
    import contextlib, io

    argv = ["bagel", "--framework", "mstar", "--case", "t2i-256-nocfg", "--profile", "smoke", "--dry-run", "--campaign", "TMP",
            "--env", "mstar:TORCHDYNAMO_DISABLE=1", "--opt", "bagel.vllm_mot=default"]
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert run_module.main(argv) == 0
    plan = json.loads(buffer.getvalue())["plans"][0]
    assert plan["overrides"]["env"] == {"mstar": {"TORCHDYNAMO_DISABLE": "1"}}
    assert plan["overrides"]["options"] == {"bagel.vllm_mot": "default"}
    assert "TORCHDYNAMO_DISABLE=1" in plan["service_command"]
    overrides.reset()
    paths.set_campaign(None)


def test_bagel_vllm_loads_the_tuned_mot_config_by_default_and_untuned_only_on_request():
    from exps.tasks.bagel import cell as bagel_cell

    overrides.reset()
    label, env = bagel_cell.vllm_mot_config()
    assert label.startswith("tuned") and env["VLLM_TUNED_CONFIG_FOLDER"].endswith("tuned")
    assert list(Path(env["VLLM_TUNED_CONFIG_FOLDER"]).glob("device_name=*.json"))
    overrides.set_current(overrides.build(options=["bagel.vllm_mot=default"]))
    try:
        assert bagel_cell.vllm_mot_config()[1] == {"VLLM_TUNED_CONFIG_FOLDER": None}
        from exps.core import paths as _paths
        from exps import frameworks as _fw

        command = _fw.vllm_omni_serve(model_path=_paths.REPO_ROOT, deploy_config=_paths.CONFIGS / "vllm_omni" / "bagel.yaml",
                                      port_number=1, env=bagel_cell.vllm_mot_config()[1])
        assert "VLLM_TUNED_CONFIG_FOLDER" not in command  # the default folder must not survive an untuned request
    finally:
        overrides.reset()


def test_stale_environment_variables_are_refused(monkeypatch):
    monkeypatch.setenv("EXPERIMENTS_MSTAR_EXTRA_ENV", "TORCHDYNAMO_DISABLE=1")
    with pytest.raises(SystemExit, match="EXPERIMENTS_MSTAR_EXTRA_ENV"):
        runner.check_clean_environment()


def test_vjepa_baseline_is_bf16_only():
    from exps.tasks.vjepa_ac import meta_native_client

    parser = argparse.ArgumentParser()
    meta_native_client.add_arguments(parser) if hasattr(meta_native_client, "add_arguments") else None
    source = Path(meta_native_client.__file__).read_text()
    assert 'choices=("bfloat16",)' in source and '"float32"' not in source.split("def parse_args", 1)[-1].split("def ", 1)[0]


# --- protocol: back-to-back for every task -----------------------------------------------------------------------------------
@pytest.mark.parametrize("task", registry.TASK_ORDER)
def test_every_task_forwards_the_protocol_to_its_client(task, tmp_path):
    """`--protocol back-to-back` and the fixed gap must reach the client of every task, not just Pi0.5."""
    import importlib

    spec = registry.load_task(task)
    case = spec.cases[0]
    framework = next(f for f in spec.frameworks if f not in case.not_applicable)
    ref = registry.CellRef(task, framework, case.name, "r01")
    module = importlib.import_module(spec.runner_module)
    config = MeasurementConfig(output_dir=tmp_path, back_to_back=True, inter_request_gap_s=12.0)
    plan = module.plan(ref, case, config)
    flat = [str(a) for a in plan.client_argv]
    joined = " ".join(flat)
    assert "--protocol back-to-back" in joined and "--inter-request-gap-s 12.0" in joined, (task, framework)
    gated = module.plan(ref, case, MeasurementConfig(output_dir=tmp_path))
    assert "--protocol" not in " ".join(str(a) for a in gated.client_argv)


def test_protocol_flags_round_trip_through_the_client_parser(tmp_path):
    config = MeasurementConfig(output_dir=tmp_path, back_to_back=True, inter_request_gap_s=30.0)
    argv = cli.protocol_argv(config)
    parser = argparse.ArgumentParser()
    cli.add_client_protocol_arguments(parser)
    parsed = cli.client_measurement_config(parser.parse_args(argv), tmp_path)
    assert parsed.back_to_back is True and parsed.inter_request_gap_s == 30.0
    # the legacy alias still works, and a gap alone implies back-to-back (a gap is not a gate)
    alias = cli.client_measurement_config(parser.parse_args(["--back-to-back"]), tmp_path)
    assert alias.back_to_back is True
    gap_only = cli.client_measurement_config(parser.parse_args(["--inter-request-gap-s", "5"]), tmp_path)
    assert gap_only.back_to_back is True and gap_only.inter_request_gap_s == 5.0
    assert cli.client_measurement_config(parser.parse_args([]), tmp_path).back_to_back is False


def _stub_measurement(monkeypatch, calls):
    from exps.core import measure

    def gate(name):
        def run(*args, **kwargs):
            calls.append(name)
            return {"status": "passed", "waited_seconds": 1.0}
        return run

    monkeypatch.setattr(measure.thermal, "loaded_idle_gate", gate("loaded_idle"))
    monkeypatch.setattr(measure.thermal, "cooldown_gate", gate("cooldown"))

    def fake_issue(issue, index, phase, config, cooldown):
        calls.append(f"{phase}:{'gated' if cooldown else 'ungated'}")
        return {"phase": phase, "sequence": index, "success": True, "e2e_ms": 100.0 + index, "thermally_clean": True,
                "cooldown_timed_out": None if cooldown is None else False}

    monkeypatch.setattr(measure, "_issue_monitored", fake_issue)
    return measure


@pytest.mark.parametrize("mode", ["gated", "back-to-back", "fixed-gap"])
def test_measurement_protocols_differ_only_in_gates_and_gaps(monkeypatch, tmp_path, mode):
    calls: list[str] = []
    measure = _stub_measurement(monkeypatch, calls)
    sleeps: list[float] = []
    monkeypatch.setattr(measure.time, "sleep", lambda s: sleeps.append(s))
    config = measure.MeasurementConfig(output_dir=tmp_path, measured_requests=3, precondition_requests=2,
                                       back_to_back=mode != "gated", inter_request_gap_s=7.0 if mode == "fixed-gap" else 0.0)
    summary = measure.run_measurement(config, issue=lambda index, phase: {}, context={})
    measured = [c for c in calls if c.startswith("measured")]
    assert len(measured) == 3
    if mode == "gated":
        assert calls.count("loaded_idle") == 2 and calls.count("cooldown") == 3 and set(measured) == {"measured:gated"}
        assert summary["protocol"]["name"] == "per-request-cooldown-gated" and not sleeps
    else:
        assert calls.count("loaded_idle") == 1 and calls.count("cooldown") == 0 and set(measured) == {"measured:ungated"}
        assert summary["gates"]["per_request_cooldowns"]["count"] == 0
        if mode == "back-to-back":
            assert summary["protocol"]["name"] == "back-to-back" and not sleeps
        else:
            assert summary["protocol"]["name"] == "fixed-gap" and sleeps == [7.0, 7.0, 7.0]


def test_back_to_back_time_estimate_is_far_shorter_than_gated(tmp_path):
    from exps.core.cell import estimate_minutes

    gated = estimate_minutes(MeasurementConfig(tmp_path, measured_requests=12, precondition_requests=5))
    b2b = estimate_minutes(MeasurementConfig(tmp_path, measured_requests=12, precondition_requests=5, back_to_back=True))
    assert b2b < gated / 2


# --- analysis on fabricated cells --------------------------------------------------------------------------------------------
def _fabricate(root: Path, campaign: str, task: str, framework: str, case: str, repeat: str, e2e: list[float], *,
               env: dict | None = None, back_to_back: bool = False, gap: float = 0.0, status="completed", started=None, log="") -> Path:
    cell = root / campaign / task.upper() / framework / case / repeat
    cell.mkdir(parents=True, exist_ok=True)
    rows = [{"phase": "precondition", "success": True, "e2e_ms": 999.0}]
    rows += [{"phase": "measured", "success": True, "e2e_ms": v, "thermally_clean": True, "cooldown_timed_out": False,
              "temperature_c_before": 60.0, "thermal_monitor": {"sm_clock_mhz_mean": 1785.0}} for v in e2e]
    (cell / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    record = {"status": status, "overrides": {"env": env or {}, "serve_args": {}, "options": {}, "nsys": None},
              "protocol": {"back_to_back": back_to_back, "inter_request_gap_s": gap},
              "service_command": "serve " + " ".join(f"{k}={v}" for fam in (env or {}).values() for k, v in fam.items()),
              "started_at_utc": started or "2026-11-04T00:00:00+00:00", "finished_at_utc": "2026-11-04T01:00:00+00:00"}
    (cell / "manifest.json").write_text(json.dumps(record))
    (cell / ".cell_done").write_text(json.dumps({"status": status}))
    lifecycle = root / campaign / "_service_lifecycles" / f"{task}_{framework}_{case}_{repeat}"
    lifecycle.mkdir(parents=True, exist_ok=True)
    (lifecycle / "server.log").write_text(log)
    return cell


def _two_arm_experiment() -> Experiment:
    return Experiment(
        name="unit-ab", title="unit", task="bagel", question="q", reading="r", kind="ab", frameworks=("mstar",),
        cases=("t2i-256-nocfg",), repeats=2, measured=3,
        arms=(Arm("A1"), Arm("B", env=("mstar:TORCHDYNAMO_DISABLE=1",),
                             expect=Expect(command_has=("TORCHDYNAMO_DISABLE=1",)))))


def test_analysis_reports_pooled_p50_ratio_and_start_range(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    experiment = _two_arm_experiment()
    runs = {r.arm.name: r.campaign for r in runner.arm_runs(experiment, protocol="gated")}
    for repeat, base in (("r01", 100.0), ("r02", 110.0)):
        _fabricate(tmp_path, runs["A1"], "bagel", "mstar", "t2i-256-nocfg", repeat, [base, base + 1, base + 2])
        _fabricate(tmp_path, runs["B"], "bagel", "mstar", "t2i-256-nocfg", repeat, [base * 2, base * 2 + 1, base * 2 + 2],
                   env={"mstar": {"TORCHDYNAMO_DISABLE": "1"}})
    out = analysis.analyze(experiment, run_id="01", protocol="gated", gap_s=0.0, arms=None, output=tmp_path / "out.md")[0]
    text = out.read_text()
    # pooled p50 of {100,101,102,110,111,112} = 106; start medians 101 and 111
    assert "106 [101–111]" in text and "1.99x" in text and "q" in text


def test_analysis_marks_ablation_arms_that_sit_inside_the_a1_a2_drift(monkeypatch, tmp_path):
    """V-JEPA h15 had A1 416 ms vs A2 358 ms (14%) while every switch moved <= 7%: those ratios are drift, not effects."""
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    experiment = Experiment(
        name="unit-drift", title="unit", task="bagel", question="q", reading="r", kind="ab", frameworks=("mstar",),
        cases=("t2i-256-nocfg",), repeats=1, measured=3,
        arms=(Arm("A1"), Arm("B", env=("mstar:TORCHDYNAMO_DISABLE=1",), expect=Expect(command_has=("TORCHDYNAMO_DISABLE=1",))),
              Arm("C", env=("mstar:MSTAR_ENGINE_STEP_SYNC=1",), expect=Expect(command_has=("MSTAR_ENGINE_STEP_SYNC=1",))),
              Arm("A2")))
    runs = {r.arm.name: r.campaign for r in runner.arm_runs(experiment, protocol="gated")}
    for arm, base, env in (("A1", 100.0, {}), ("B", 107.0, {"TORCHDYNAMO_DISABLE": "1"}),
                           ("C", 200.0, {"MSTAR_ENGINE_STEP_SYNC": "1"}), ("A2", 114.0, {})):
        _fabricate(tmp_path, runs[arm], "bagel", "mstar", "t2i-256-nocfg", "r01", [base, base, base],
                   env={"mstar": env} if env else None)
    text = analysis.analyze(experiment, run_id="01", protocol="gated", gap_s=0.0, arms=None, output=tmp_path / "out.md")[0].read_text()
    assert "mstar t2i-256-nocfg 14%" in text
    assert "1.07x †" in text and "2.00x |" in text and "2.00x †" not in text


def test_analysis_refuses_to_report_around_missing_or_mislabelled_cells(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    experiment = _two_arm_experiment()
    runs = {r.arm.name: r.campaign for r in runner.arm_runs(experiment, protocol="gated")}
    _fabricate(tmp_path, runs["A1"], "bagel", "mstar", "t2i-256-nocfg", "r01", [1, 2, 3])
    # A1/r02 missing; B ran WITHOUT the switch it is supposed to test
    _fabricate(tmp_path, runs["B"], "bagel", "mstar", "t2i-256-nocfg", "r01", [1, 2, 3])
    _fabricate(tmp_path, runs["B"], "bagel", "mstar", "t2i-256-nocfg", "r02", [1, 2, 3])
    with pytest.raises(SystemExit) as caught:
        analysis.analyze(experiment, run_id="01", protocol="gated", gap_s=0.0, arms=None, output=tmp_path / "out.md")
    message = str(caught.value)
    assert "A1/mstar/t2i-256-nocfg/r02" not in message and "no .cell_done" in message
    assert "recorded overrides.env" in message and "serve command lacks 'TORCHDYNAMO_DISABLE=1'" in message
    assert not (tmp_path / "out.md").exists()


def test_analysis_checks_the_protocol_the_cells_actually_ran_under(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    experiment = Experiment(name="unit-p", title="u", task="bagel", question="q", reading="r", kind="ab",
                            frameworks=("mstar",), cases=("t2i-256-nocfg",), repeats=1, measured=3,
                            arms=(Arm("only"),))
    campaign = runner.arm_runs(experiment, protocol="back-to-back")[0].campaign
    _fabricate(tmp_path, campaign, "bagel", "mstar", "t2i-256-nocfg", "r01", [1, 2, 3], back_to_back=False)
    with pytest.raises(SystemExit, match="ran with back_to_back=False"):
        analysis.analyze(experiment, run_id="01", protocol="back-to-back", gap_s=0.0, arms=None, output=tmp_path / "o.md")


def test_attn_probe_proofs(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    experiment = catalogue.get("bagel-attn-backend")
    probe_run = next(r for r in runner.arm_runs(experiment) if r.arm.name == "mstar-probe")
    assert experiment.cases == ("t2i-512-nocfg", "i2i-512", "i2t-512")
    blocked_run = next(r for r in runner.arm_runs(experiment) if r.arm.name == "mstar-no-flash-attn")

    def cell(run, case):
        c = _fabricate(tmp_path, run.campaign, "bagel", "mstar", case, "r01", [1, 2, 3])
        return analysis.CellData("mstar", case, "r01", c, json.loads((c / "manifest.json").read_text()), [])

    def write_probe(run, **fields):
        base = {"pid": 1, "process_start_utc": "2026-11-04T00:10:00+00:00", "vit_module_loaded": True,
                "vit_flash_attn_available": True, "vit_flash_attn_calls": 0, "vit_sdpa_fallback_calls": 0,
                "attention_managers": [{"manager": "FlashInferManager"}]}
        base.update(fields)
        root = paths.campaign_root(run.campaign)
        root.mkdir(parents=True, exist_ok=True)
        (root / f"attn_probe.{base['pid']}.json").write_text(json.dumps(base))

    # text-to-image: ViT never runs, FlashInfer attention, pip flash_attn importable -> proof holds
    write_probe(probe_run)
    assert analysis.check_proof("attn_probe", probe_run, cell(probe_run, "t2i-256-nocfg")) == []
    # an image-input case that never called the ViT's flash-attn means the probe saw nothing: fail
    assert analysis.check_proof("attn_probe", probe_run, cell(probe_run, "i2t-256"))
    write_probe(probe_run, vit_flash_attn_calls=54)
    assert analysis.check_proof("attn_probe", probe_run, cell(probe_run, "i2t-256")) == []
    # a text-only case that ran the ViT contradicts the claim under test
    assert analysis.check_proof("attn_probe", probe_run, cell(probe_run, "t2i-512-nocfg"))
    # the blocked arm: not importable, zero flash calls, SDPA fallback used for image input
    write_probe(blocked_run, vit_flash_attn_available=False, vit_sdpa_fallback_calls=54)
    assert analysis.check_proof("attn_probe_blocked", blocked_run, cell(blocked_run, "i2t-256")) == []
    write_probe(blocked_run, vit_flash_attn_available=True, vit_flash_attn_calls=3)
    assert analysis.check_proof("attn_probe_blocked", blocked_run, cell(blocked_run, "i2t-256"))
    # no probe process at all in the cell's window: the probe was not loaded
    other = _fabricate(tmp_path, probe_run.campaign, "bagel", "mstar", "i2i-256", "r02", [1, 2, 3], started="2027-01-01T00:00:00+00:00")
    stranger = analysis.CellData("mstar", "i2i-256", "r02", other, {"started_at_utc": "2027-01-01T00:00:00+00:00",
                                                                     "finished_at_utc": "2027-01-01T01:00:00+00:00"}, [])
    assert "not loaded" in analysis.check_proof("attn_probe", probe_run, stranger)[0]


def test_sampler_hook_proof_must_show_applied(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    experiment = Experiment(name="unit-hook", title="u", task="bagel", question="q", reading="r", kind="ab", frameworks=("mstar",),
                            cases=("t2i-512-nocfg",), arms=(Arm("fixed", expect=Expect(proof="sampler_hook")),))
    run = runner.arm_runs(experiment)[0]
    dummy = analysis.CellData("mstar", "t2i-256-nocfg", "r01", tmp_path, {}, [])
    assert "missing or empty" in analysis.check_proof("sampler_hook", run, dummy)[0]
    root = paths.campaign_root(run.campaign)
    root.mkdir(parents=True)
    (root / "hook_proof.jsonl").write_text(json.dumps({"applied": False, "refused": "source differs"}) + "\n")
    assert "NOT applied" in analysis.check_proof("sampler_hook", run, dummy)[0]
    (root / "hook_proof.jsonl").write_text(json.dumps({"applied": True}) + "\n")
    assert analysis.check_proof("sampler_hook", run, dummy) == []


def test_best_config_table_picks_each_frameworks_best_arm(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    experiment = Experiment(
        name="unit-best", title="u", task="bagel", question="q", reading="r", kind="compare",
        frameworks=("mstar", "vllm-omni-two-stage"), cases=("t2i-256-nocfg",), repeats=1, measured=3,
        arms=(Arm("default"), Arm("mstar-nocompile", frameworks=("mstar",), env=("mstar:TORCHDYNAMO_DISABLE=1",))))
    runs = {r.arm.name: r.campaign for r in runner.arm_runs(experiment, protocol="gated")}
    _fabricate(tmp_path, runs["default"], "bagel", "mstar", "t2i-256-nocfg", "r01", [200, 200, 200])
    _fabricate(tmp_path, runs["default"], "bagel", "vllm-omni-two-stage", "t2i-256-nocfg", "r01", [250, 250, 250])
    _fabricate(tmp_path, runs["mstar-nocompile"], "bagel", "mstar", "t2i-256-nocfg", "r01", [160, 160, 160],
               env={"mstar": {"TORCHDYNAMO_DISABLE": "1"}})
    out = analysis.analyze(experiment, run_id="01", protocol="gated", gap_s=0.0, arms=None, output=tmp_path / "o.md")[0].read_text()
    # M*'s best is its no-compile arm (160), so vLLM-Omni's 250 is 1.56x, not 1.25x
    assert "| t2i-256-nocfg | mstar | mstar-nocompile |" in out and "1.56x" in out and "1.25x" not in out


def test_tool_entries_and_names_resolve():
    assert runner.select(["protocol-ab-*"]) and runner.select(["baseline-bagel"])[0].name == "baseline-bagel"
    with pytest.raises(SystemExit):
        runner.select(["no-such-experiment"])
    tune = catalogue.get("bagel-mot-tune")
    assert tune.tool and tune.task is None and runner.arm_runs(tune) == []
    sglang_tune = catalogue.get("sglang-moe-tune")
    assert 'export PATH="{repo}/envs/sglang-omni/bin:$PATH"' in sglang_tune.tool[2]


# --- hardware profiles -------------------------------------------------------------------------------------------------------
def test_hardware_profiles_load_and_describe_the_two_machines():
    profiles = hardware.profiles()
    assert {"h200nvl", "rtx5880", "generic"} <= set(profiles)
    h200, ada = profiles["h200nvl"], profiles["rtx5880"]
    assert not h200.thermal_stable and h200.default_protocol == "gated"
    assert ada.thermal_stable and ada.default_protocol == "back-to-back" and ada.gpu_count == 8 and ada.memory_gib == 48
    # a shared machine: every cell picks its own free GPUs, as many as the deployment needs and no more
    # no allowlist: any GPU that is idle is usable (an earlier version kept GPUs 0 and 1 out because another user's
    # jobs there flickered free between epochs, which made a cell sit in the idle gate; the owner has since asked
    # for every idle card to be used)
    assert ada.auto_select_gpus and ada.allowed_devices == ()
    assert ada.gpus_needed("qwen_t2s", "mstar") == ada.gpus_needed("qwen_txt") == ada.gpus_needed("qwen_a2t") == 3
    assert ada.gpus_needed("vjepa_ac", "mstar") == 1
    # Qwen3-Omni 30B BF16 does not fit one 48 GB card: those tasks must say they need a multi-GPU overlay
    assert {"qwen_t2s", "qwen_txt", "qwen_a2t"} == set(ada.needs_overlay_for_tasks)


def test_rtx5880_qwen_overlays_use_three_gpus_thinker_tp2_and_a_separate_talker_card():
    overlay = paths.CONFIGS / "hardware" / "rtx5880"
    mstar = yaml.safe_load((overlay / "mstar" / "qwen3omni_3gpu_thinker_tp2.yaml").read_text())
    vllm = yaml.safe_load(
        (overlay / "vllm_omni" / "qwen3_omni_3gpu_thinker_tp2_async_c2wgraph.yaml").read_text()
    )
    sglang = yaml.safe_load(
        (overlay / "sglang_omni" / "qwen3_omni_3gpu_thinker_tp2_partial_start.yaml").read_text()
    )
    bagel = yaml.safe_load((overlay / "vllm_omni" / "bagel_two_stage_2gpu.yaml").read_text())
    cosmos3 = yaml.safe_load((overlay / "mstar" / "cosmos3_nano_1gpu_kv512.yaml").read_text())

    assert mstar["node_groups"][0]["ranks"] == [0, 1] and mstar["node_groups"][0]["tp_size"] == 2
    assert mstar["node_groups"][2]["node_names"] == ["Talker"] and mstar["node_groups"][2]["ranks"] == [2]
    assert mstar["node_groups"][3]["node_names"] == ["Code2Wav"] and mstar["node_groups"][3]["ranks"] == [2]
    assert vllm["stages"][0]["devices"] == "0,1" and vllm["stages"][0]["tensor_parallel_size"] == 2
    assert vllm["stages"][1]["devices"] == "2" and vllm["stages"][2]["devices"] == "2"
    assert vllm["stages"][1]["gpu_memory_utilization"] + vllm["stages"][2]["gpu_memory_utilization"] < 0.9
    assert sglang["config_cls"] == "Qwen3OmniSpeechPipelineConfig"
    assert sglang["stages"]["thinker"]["gpu"] == [0, 1] and sglang["stages"]["thinker"]["tp_size"] == 2
    assert sglang["stages"]["talker_ar"]["gpu"] == 2 and sglang["stages"]["code2wav"]["gpu"] == 2
    assert sglang["stages"]["thinker"]["engine"]["disable_custom_all_reduce"] is True
    assert bagel["stages"][0]["gpu_memory_utilization"] == 0.9
    assert bagel["stages"][1]["devices"] == "1"
    assert cosmos3["node_groups"][0]["ranks"] == [0] and "tp_size" not in cosmos3["node_groups"][0]      # one card is enough


def test_rtx5880_has_an_overlay_for_every_qwen_config_the_experiments_swap_in():
    """t2s-concurrency (batch) and t2s-deploy-config (plain) must run on the 3-GPU topology, not the 1-GPU H200 yaml."""
    import json

    profile = json.loads((paths.REPO_ROOT / "exps" / "profiles" / "hardware" / "rtx5880.json").read_text())
    mapping = profile["config_overlay_files"]
    overlay = paths.CONFIGS / "hardware" / "rtx5880"
    for default in ("vllm_omni/qwen3_omni_full_1gpu_batch_async_c2wgraph.yaml", "vllm_omni/qwen3_omni_full_1gpu.yaml",
                    "sglang_omni/qwen3_omni_colocated_h200.yaml"):
        assert (paths.CONFIGS / default).is_file()
        assert (overlay / mapping[default]).is_file(), default
    batch = yaml.safe_load((overlay / mapping["vllm_omni/qwen3_omni_full_1gpu_batch_async_c2wgraph.yaml"]).read_text())
    plain = yaml.safe_load((overlay / mapping["vllm_omni/qwen3_omni_full_1gpu.yaml"]).read_text())
    sglang = yaml.safe_load((overlay / mapping["sglang_omni/qwen3_omni_colocated_h200.yaml"]).read_text())
    assert [s["max_num_seqs"] for s in batch["stages"]] == [16, 16, 16]
    assert batch["stages"][0]["tensor_parallel_size"] == 2 and batch["async_chunk"] is True
    assert plain["async_chunk"] is False and plain["stages"][2]["enforce_eager"] is True
    assert sglang["stages"]["talker_ar"]["factory"]["enable_partial_start"] is False


def test_qwen_sglang_5880_plan_uses_non_colocated_overlay():
    from exps.tasks import qwen_common

    try:
        hardware.set_current("rtx5880", "4,5,6,7")
        command = qwen_common.service("sglang-omni", "qwen_t2s", "test").command
        assert "--colocate" not in command
        assert "configs/hardware/rtx5880/sglang_omni/qwen3_omni_3gpu_thinker_tp2_partial_start.yaml" in command
        assert "CUDA_VISIBLE_DEVICES=4,5,6,7" in command
        # M* TXT keeps compile off on rtx5880; T2S takes compile on (per-task default, 2026-11-13)
        mstar_txt_command = qwen_common.service("mstar", "qwen_txt", "test").command
        assert "TORCHDYNAMO_DISABLE=1" in mstar_txt_command
        mstar_t2s_command = qwen_common.service("mstar", "qwen_t2s", "test").command
        assert "TORCHDYNAMO_DISABLE" not in mstar_t2s_command

        hardware.set_current("h200nvl")
        h200_command = qwen_common.service("sglang-omni", "qwen_t2s", "test").command
        assert "--colocate" in h200_command
        h200_mstar_command = qwen_common.service("mstar", "qwen_t2s", "test").command
        assert "TORCHDYNAMO_DISABLE" not in h200_mstar_command
    finally:
        hardware.reset()


def test_gpus_flag_reaches_serve_commands_and_gates_and_validates():
    from exps import frameworks

    try:
        hw = hardware.set_current("rtx5880", "4,5")
        assert hw.gpu_indices == [4, 5] and hw.primary_gpu == 4
        command = frameworks.mstar_serve(model="bagel", config=Path("c.yaml"), port_number=1, tag="t")
        assert "CUDA_VISIBLE_DEVICES=4,5" in command
        vllm_command = frameworks.vllm_omni_serve(model_path=Path("/m"), deploy_config=Path("/c"), port_number=1)
        assert "CUDA_VISIBLE_DEVICES=4,5" in vllm_command
        assert 'CUDA_HOME="$PWD/envs/vllm-omni/lib/python3.12/site-packages/nvidia/cu13"' in vllm_command
        voxserve_command = frameworks.voxserve_serve(
            model_path=Path("/m"), port_number=1, max_batch_size=1, temperature=0.6,
            top_p=0.8, repetition_penalty=1.3, max_tokens=8,
        )
        assert "CUDA_VISIBLE_DEVICES=4,5" in voxserve_command
        assert 'CUDA_HOME="$PWD/envs/voxserve/lib/python3.12/site-packages/nvidia/cuda_runtime"' in voxserve_command
        assert f"PYTORCH_NVCC={frameworks.host_nvcc()}" in voxserve_command   # resolved per host, not a literal /usr/bin/nvcc
        assert "CUDA_DEVICE_ORDER=PCI_BUS_ID" in voxserve_command           # CUDA's numbering == nvidia-smi's
        sglang_command = frameworks.sglang_omni_serve(
            model_path=Path("/m"), config=Path("/c"), port_number=1, model_name="test"
        )
        assert "CUDA_VISIBLE_DEVICES=4,5" in sglang_command
        assert 'CUDA_HOME="$PWD/envs/sglang-omni/lib/python3.12/site-packages/nvidia/cu13"' in sglang_command
        config = cli.measurement_config(argparse.Namespace(), output_dir=Path("/tmp/x"), task="bagel", case="t2i-512-nocfg")
        assert config.gpu_index == 4
        with pytest.raises(ValueError):
            hardware.set_current("rtx5880", "0;1")
        assert hardware.set_current("rtx5880", "2").gpu_indices == [2]
        assert hardware.set_current("rtx5880", "0").gpu_indices == [0]       # no allowlist: an explicit card is accepted
        with pytest.raises(ValueError, match="duplicate"):
            hardware.set_current("rtx5880", "4,4")
        with pytest.raises(ValueError):
            hardware.set_current("no-such-profile")
    finally:
        hardware.reset()


def test_default_protocol_follows_the_hardware_and_explicit_choices_win():
    ns = lambda **kw: argparse.Namespace(**kw)  # noqa: E731
    try:
        hardware.set_current("h200nvl")
        assert cli.wants_back_to_back(ns()) is False
        hardware.set_current("rtx5880")
        assert cli.wants_back_to_back(ns()) is True
        assert cli.wants_back_to_back(ns(protocol="gated")) is False
        with pytest.raises(ValueError):
            cli.wants_back_to_back(ns(protocol="gated", inter_request_gap_s=5.0))
        # a client subprocess is always told explicitly (it carries --calibration-source), never guessing from the machine
        assert cli.wants_back_to_back(ns(calibration_source="x")) is False
        # suite-level: the campaigns of an arm that does not fix its protocol follow the hardware default
        experiment = catalogue.get("mstar-ablation-qwen-t2s")
        assert all(r.protocol == "back-to-back" and r.campaign.endswith("_B2B") for r in runner.arm_runs(experiment))
        hardware.set_current("h200nvl")
        assert all(r.protocol == "gated" for r in runner.arm_runs(experiment))
    finally:
        hardware.reset()


def test_tasks_that_do_not_fit_one_card_are_refused_without_an_overlay_config(tmp_path):
    from exps.tasks.qwen_t2s import cell as t2s_cell

    ref = registry.CellRef("qwen_t2s", "mstar", "verbatim-en", "r01")
    spec = registry.load_task("qwen_t2s")
    import dataclasses
    import shutil

    try:
        # a throwaway overlay name: the real rtx5880 overlay holds the configs the smoke runs use and must never be touched
        hardware.set_current("rtx5880")
        hardware._CURRENT = dataclasses.replace(hardware.current(), config_overlay="rtx5880-unittest")  # noqa: SLF001
        plan = t2s_cell.plan(ref, spec.case("verbatim-en"), MeasurementConfig(tmp_path))
        with pytest.raises(SystemExit, match="does not fit one rtx5880"):
            hardware.check_task_fits("qwen_t2s", plan.service.command)
        # with a config overlay in place the same plan is accepted
        overlay_root = paths.CONFIGS / "hardware" / "rtx5880-unittest"
        overlay = overlay_root / "mstar"
        overlay.mkdir(parents=True, exist_ok=True)
        try:
            (overlay / "qwen3omni_1gpu.yaml").write_text("model: qwen3_omni\n")
            resolved = t2s_cell.plan(ref, spec.case("verbatim-en"), MeasurementConfig(tmp_path))
            assert "configs/hardware/rtx5880-unittest/mstar/qwen3omni_1gpu.yaml" in resolved.service.command
            hardware.check_task_fits("qwen_t2s", resolved.service.command)
        finally:
            shutil.rmtree(overlay_root, ignore_errors=True)
        # `--opt config.mstar=PATH` swaps the config for the whole framework family
        alt = tmp_path / "alt.yaml"
        alt.write_text("model: x\n")
        overrides.set_current(overrides.build(options=[f"config.mstar={alt}"]))
        try:
            assert str(alt) in t2s_cell.plan(ref, spec.case("verbatim-en"), MeasurementConfig(tmp_path)).service.command
        finally:
            overrides.reset()
        overrides.set_current(overrides.build(options=["config.mstar=/no/such/file.yaml"]))
        with pytest.raises(FileNotFoundError):
            t2s_cell.plan(ref, spec.case("verbatim-en"), MeasurementConfig(tmp_path))
    finally:
        overrides.reset()
        hardware.reset()


def test_suite_cases_and_frameworks_flags_replace_and_restrict_every_arm():
    experiment = catalogue.get("baseline-bagel")
    runs = runner.arm_runs(experiment, protocol="gated", cases=["t2i-1024-nocfg"], frameworks=["mstar"])
    argv = runs[0].argv
    assert argv.count("--case") == 1 and "t2i-1024-nocfg" in argv and argv.count("--framework") == 1 and "mstar" in argv
    # an arm restricted to M* disappears when the selection excludes M*
    ablation = catalogue.get("mstar-ablation-bagel")
    assert runner.arm_runs(ablation, frameworks=["vllm-omni-two-stage"]) == []
    assert "--hardware" in runner.arm_runs(experiment, protocol="gated", hardware_name="rtx5880", gpus="4,5")[0].argv


def test_bagel_1024_is_defined_and_explicit_only():
    from exps.tasks.bagel.cases import TASK

    assert {"t2i-1024-nocfg", "t2i-1024-cfg3", "i2i-1024", "i2t-1024"} <= {c.name for c in TASK.cases}
    for profile in ("smoke", "nightly", "full"):
        assert not any("1024" in r.case for r in registry.expand(TASK, profile=profile, repeats=1))


def test_a_task_that_does_not_fit_is_reported_and_the_rest_of_the_plan_continues(capsys, monkeypatch):
    monkeypatch.setattr(hardware, "resolve_config", lambda path: Path(path))     # as if the machine had no overlay configs
    try:
        hardware.set_current("rtx5880")
        status = runner.main_run(["baseline-qwen-t2s", "baseline-cosmos3"], run_id="01", protocol="back-to-back", gap_s=0.0,
                                 arms=None, execute_it=False, hardware_name="rtx5880")
        out = capsys.readouterr()
        assert status == 1
        assert "cannot be planned here" in out.out and "does not fit one rtx5880" in out.out
        assert "BASELINE_COSMOS3_MAIN_01_B2B" in out.out and "cells," in out.out   # Cosmos3 still planned, back-to-back by default
    finally:
        hardware.reset()


# --- protocol decisions ------------------------------------------------------------------------------------------------------
def test_decision_rule_three_outcomes(monkeypatch):
    from exps import protocols

    try:
        hardware.set_current("rtx5880")
        same = protocols.decide({"mstar": 1.02, "vllm-omni": 0.98})
        assert same["kind"] == "insensitive" and same["protocols"] == ["back-to-back"]      # stable cooling: the cheap one
        hardware.set_current("h200nvl")
        # throttling chassis: insensitive only counts as "use the cheap one" with evidence the clock held under back-to-back
        assert protocols.decide({"mstar": 1.02, "vllm-omni": 0.98})["protocols"] == ["gated"]
        held = protocols.decide({"mstar": 1.02, "vllm-omni": 0.98}, clock_ratios={"mstar": 0.995, "vllm-omni": 0.99})
        assert held["protocols"] == ["back-to-back"] and "clock held" in held["reason"]
        dropped = protocols.decide({"mstar": 1.02, "vllm-omni": 0.98}, clock_ratios={"mstar": 0.90, "vllm-omni": 0.99})
        assert dropped["protocols"] == ["gated"] and "throttling" in dropped["reason"]
        together = protocols.decide({"mstar": 0.80, "sglang-core": 0.81, "vllm-omni": 0.82})
        assert together["kind"] == "shifts-together" and len(together["protocols"]) == 1
        # Pi0.5 on the H200: b2b / gated = 0.94, 0.86, 0.67 -> the frameworks move by different amounts: report both
        pi05 = protocols.decide({"mstar": 0.94, "sglang-core": 0.86, "vllm-omni": 0.67})
        assert pi05["kind"] == "framework-dependent" and pi05["protocols"] == ["gated", "back-to-back"]
        with pytest.raises(ValueError):
            protocols.decide({})
    finally:
        hardware.reset()


def test_baselines_refuse_to_plan_without_a_decision_and_follow_it_when_there_is_one(monkeypatch, tmp_path):
    from exps import protocols

    monkeypatch.setattr(protocols, "DECISION_DIR", tmp_path / "protocol")
    try:
        hardware.set_current("h200nvl")
        baseline = catalogue.get("baseline-pi05")
        with pytest.raises(SystemExit, match="protocol-ab-pi05"):
            runner.arm_runs(baseline)
        # an explicit protocol is allowed (and is the operator's responsibility)
        assert runner.arm_runs(baseline, protocol="gated")[0].protocol == "gated"
        # non-baselines never need one
        assert runner.arm_runs(catalogue.get("mstar-ablation-pi05"))[0].protocol == "gated"
        protocols.save({"pi05": protocols.record("pi05", {"mstar": 0.94, "sglang-core": 0.86, "vllm-omni": 0.67}, ["x"])})
        both = runner.arm_runs(baseline)
        assert [r.protocol for r in both] == ["gated", "back-to-back"] and len({r.campaign for r in both}) == 2
        # ablations (M* against itself) take the hardware default even when the baseline needs both protocols
        assert [r.protocol for r in runner.arm_runs(catalogue.get("mstar-ablation-pi05"))] == ["gated"] * 7
        protocols.save({"pi05": protocols.record("pi05", {"mstar": 1.0, "vllm-omni": 1.01}, ["x"])})
        assert [r.protocol for r in runner.arm_runs(baseline)] == ["gated"]
        # a hand-run cell follows a single-protocol decision too
        # decisions are per hardware profile: another machine has none, so it falls back to its own default
        hardware.set_current("rtx5880")
        assert protocols.default_for("pi05") == "back-to-back"
    finally:
        hardware.reset()


def test_decide_protocol_reads_finished_protocol_ab_runs(monkeypatch, tmp_path):
    from exps import protocols
    from exps.suite import decide

    monkeypatch.setattr(paths, "RUNS", tmp_path / "runs")
    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(protocols, "DECISION_DIR", tmp_path / "protocol")
    try:
        hardware.set_current("h200nvl")
        experiment = catalogue.get("protocol-ab-orpheus")   # mstar and voxserve, one case, no gap arm
        runs = {r.arm.name: r for r in runner.arm_runs(experiment)}
        for repeat in ("r01", "r02"):
            for fw, gated, b2b in (("mstar", 100.0, 99.0), ("voxserve", 150.0, 112.0)):   # voxserve alone speeds up without the gate
                _fabricate(tmp_path / "runs", runs["gated"].campaign, "orpheus", fw, "b1", repeat, [gated] * 5)
                _fabricate(tmp_path / "runs", runs["b2b"].campaign, "orpheus", fw, "b1", repeat, [b2b] * 5, back_to_back=True)
        _, messages = decide.decide_all(tasks=["orpheus"], run_id="01", tolerance=0.05)
        saved = protocols.load()["orpheus"]
        assert saved["protocols"] == ["gated", "back-to-back"] and saved["kind"] == "framework-dependent"
        assert saved["ratios_b2b_over_gated"] == {"mstar": 0.99, "voxserve": 0.747}
        assert (tmp_path / "results" / "protocol-decision_h200nvl.md").is_file() and any("orpheus" in m for m in messages)
        # missing evidence leaves the task undecided instead of guessing
        protocols.save({})
        _, messages = decide.decide_all(tasks=["cosmos3"], run_id="01", tolerance=0.05)
        assert protocols.load() == {} and any("not decided" in m for m in messages)
    finally:
        hardware.reset()


def test_suite_repeats_and_measured_rescale_every_arm_and_gate_seconds_follow_the_hardware():
    experiment = catalogue.get("mstar-ablation-bagel")
    argv = runner.arm_runs(experiment, repeats=1, measured=3)[0].argv
    assert argv[argv.index("--repeats") + 1] == "1" and argv[argv.index("--measured") + 1] == "3"
    try:
        hardware.set_current("rtx5880")
        config = cli.measurement_config(argparse.Namespace(), output_dir=Path("/tmp/x"), task="bagel", case="t2i-512-nocfg")
        assert config.loaded_idle_continuous_s == 15 and "hardware(rtx5880)" in config.calibration_source
        hardware.set_current("h200nvl")
        config = cli.measurement_config(argparse.Namespace(), output_dir=Path("/tmp/x"), task="bagel", case="t2i-512-nocfg")
        assert config.loaded_idle_continuous_s == 60
    finally:
        hardware.reset()


def test_auto_select_takes_exactly_the_free_gpus_a_deployment_needs(monkeypatch):
    """A shared machine: cards with a compute process or resident memory are someone else's; take as many as needed, no more."""
    # no allowlist on rtx5880: card 1 is idle, so it is a candidate like any other idle card (lowest index first)
    state = {0: (21_500.0, 1), 1: (18.0, 0), 2: (18.0, 0), 3: (18.0, 0), 4: (18.0, 0), 5: (10_000.0, 1), 6: (24_000.0, 1), 7: (18.0, 0)}
    monkeypatch.setattr(hardware, "_all_gpu_state", lambda: state)
    try:
        hw = hardware.set_current("rtx5880")
        assert hw.auto_select_gpus and hardware.free_gpus() == [1, 2, 3, 4, 7]
        assert hardware.select_gpus("qwen_t2s", "mstar").visible_devices == "1,2,3"        # 3 needed, 4+ free: takes the 3 lowest
        assert hardware.select_gpus("vjepa_ac", "mstar").visible_devices == "1"             # one card for a one-card task
        assert hardware.current().primary_gpu == 1
        # a card that fills up between cells is skipped by the next cell
        state[2] = (30_000.0, 1)
        assert hardware.select_gpus("qwen_txt", "vllm-omni").visible_devices == "1,3,4"
        # not enough free cards: refuse (naming the shortfall) rather than start on someone else's GPU
        for index in (3, 4):
            state[index] = (9_000.0, 1)
        with pytest.raises(RuntimeError, match="needs 3 free GPU"):
            hardware.select_gpus("qwen_a2t", "sglang-omni")
        # an explicit --gpus pins the list and switches selection off
        pinned = hardware.set_current("rtx5880", "4,5")
        assert not pinned.auto_select_gpus and hardware.select_gpus("qwen_t2s", "mstar").visible_devices == "4,5"
        # profiles without auto selection are untouched
        assert hardware.set_current("h200nvl") is hardware.select_gpus("qwen_t2s", "mstar")
    finally:
        hardware.reset()


def test_gpus_per_task_prefers_the_task_framework_entry():
    hw = hardware.Hardware(name="x", gpus_per_task={"a": 2, "a:big": 4})
    assert hw.gpus_needed("a") == 2 and hw.gpus_needed("a", "small") == 2 and hw.gpus_needed("a", "big") == 4
    assert hw.gpus_needed("b", "big") == 1


def test_rtx5880_bagel_overlays_cap_the_mstar_kv_pool_and_place_cfg_parallel_ranks():
    import json

    profile = json.loads((paths.REPO_ROOT / "exps" / "profiles" / "hardware" / "rtx5880.json").read_text())
    mapping, overlay = profile["config_overlay_files"], paths.CONFIGS / "hardware" / "rtx5880"
    for default in ("mstar/bagel_single_gpu.yaml", "mstar/bagel_cfg_parallel.yaml", "vllm_omni/bagel_cfg_parallel.yaml"):
        assert (paths.CONFIGS / default).is_file() and (overlay / mapping[default]).is_file(), default
    single = yaml.safe_load((overlay / mapping["mstar/bagel_single_gpu.yaml"]).read_text())
    parallel = yaml.safe_load((overlay / mapping["mstar/bagel_cfg_parallel.yaml"]).read_text())
    # the default pool (2048 pages) does not fit beside the 29 GB of weights on 48 GiB; the cap leaves room for activations
    assert single["resources"]["kv"]["max_num_pages"] == 512 == parallel["resources"]["kv"]["max_num_pages"]
    ranks = {tuple(g["node_names"]): g["ranks"] for g in parallel["node_groups"]}
    assert ranks[("LLM", "combine_cfg")] == [0] and ranks[("LLM_cfg_text",)] == [1] and ranks[("LLM_cfg_img",)] == [2]
    two_stage = yaml.safe_load((overlay / mapping["vllm_omni/bagel_cfg_parallel.yaml"]).read_text())
    assert two_stage["stages"][0]["devices"] == "0" and two_stage["stages"][1]["devices"] == "1,2,3"
    assert two_stage["stages"][1]["cfg_parallel_size"] == 3
    single_stage = yaml.safe_load((paths.CONFIGS / "vllm_omni" / "bagel_single_stage_cfg_parallel.yaml").read_text())
    assert single_stage["stages"][0]["devices"] == "0,1,2" and single_stage["stages"][0]["cfg_parallel_size"] == 3
    assert profile["gpus_per_task"]["bagel:vllm-omni-two-stage:cfg-parallel"] == 4


def test_gpus_needed_follows_the_topology_option():
    from exps.core import overrides

    hw = hardware.Hardware(name="x", gpus_per_task={"bagel:big:cfg-parallel": 4})
    overrides.set_current(overrides.build(options=["bagel.topology=cfg-parallel"]))
    try:
        assert hw.gpus_needed("bagel", "big") == 4 and hw.gpus_needed("bagel", "small") == 3
    finally:
        overrides.reset()
    assert hw.gpus_needed("bagel", "small") == 1


def test_multi_gpu_experiments_are_skipped_not_failed_on_a_single_gpu_machine(capsys):
    experiment = catalogue.get("baseline-bagel-cfgpar")
    assert experiment.min_gpus == 3
    status = runner.main_run(["baseline-bagel-cfgpar", "baseline-bagel"], run_id="01", protocol="back-to-back", gap_s=0.0,
                        arms=None, execute_it=False, hardware_name="h200nvl")
    out = capsys.readouterr()
    assert "skipped: needs 3 GPUs" in out.out and "baseline-bagel-cfgpar" in out.out
    assert "cfgpar" not in out.err and status == 0, out.err


def test_results_file_names_carry_the_hardware_profile(monkeypatch, tmp_path):
    """Two machines writing into one results folder must not collide: `<name>_<protocol>_<hardware>.md`, post outputs too."""
    from exps.suite import analysis

    monkeypatch.setattr(paths, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(analysis, "collect", lambda *a, **k: [])
    monkeypatch.setattr(analysis, "render", lambda *a, **k: "# table\n")
    experiment = catalogue.get("baseline-cosmos3")
    written = {}
    for name in ("h200nvl", "rtx5880"):
        hardware.set_current(name)
        try:
            written[name] = analysis.analyze(experiment, run_id="01", protocol="back-to-back", gap_s=0.0, arms=None)[0]
        finally:
            hardware.reset()
    assert written["h200nvl"].name == "baseline-cosmos3_b2b_h200nvl.md"
    assert written["rtx5880"].name == "baseline-cosmos3_b2b_rtx5880.md" and written["h200nvl"] != written["rtx5880"]
    for experiment in catalogue.ALL.values():
        for template in experiment.post:
            assert "{results}/" not in template or "_{hardware}.md" in template, (experiment.name, template)


# --- suite CLI: list -------------------------------------------------------------------------------------------------------
def test_list_prints_every_catalogue_group_and_entry(capsys):
    from exps.suite import __main__ as suite_cli

    assert suite_cli.cmd_list(argparse.Namespace(group=None)) == 0
    out = capsys.readouterr().out
    for group in {e.group for e in catalogue.ALL.values()}:
        assert f"[{group}]" in out, group
    for name in catalogue.ALL:
        assert f"  {name} " in out or f"  {name}\n" in out, name


def test_catalogue_groups_keep_known_order_and_append_unknown_groups(monkeypatch):
    from types import SimpleNamespace

    from exps.suite import __main__ as suite_cli

    monkeypatch.setattr(catalogue, "ALL", {
        "a": SimpleNamespace(group="fair"), "b": SimpleNamespace(group="brand-new"), "c": SimpleNamespace(group="baseline"),
    })
    assert suite_cli._catalogue_groups() == ["baseline", "fair", "brand-new"]
