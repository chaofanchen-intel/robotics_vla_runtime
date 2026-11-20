"""CPU tests for the round-8 tools: thermal audit and probe, the BAGEL sampler hook, the cuDNN pin hook, the script."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exps" / "tools"))

import analyze_mstar_logstats as alog  # noqa: E402
import analyze_thermal_history as ath  # noqa: E402
import gpu_thermal_probe as probe  # noqa: E402
import analyze_followups as followups  # noqa: E402
import analyze_nsys_gemm_launches as gemm_launches  # noqa: E402

MSTAR_PYTHON = ROOT / "envs" / "mstar" / "bin" / "python"
SAMPLER_HOOK = ROOT / "exps" / "tools" / "mstar_bagel_sampler_hook"
CUDNN_HOOK = ROOT / "exps" / "tools" / "mstar_cudnn_hook"


# ----- thermal audit -----------------------------------------------------------------------------------------------------

def _write_cell(runs: Path, campaign: str, framework: str, case: str, rows: list[dict]) -> None:
    cell = runs / campaign / "BAGEL" / framework / case / "r01"
    cell.mkdir(parents=True)
    (cell / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def _request(when: datetime, e2e: float, *, temp: float = 60.0, clock_mean: float = 1785.0, clock_min: float = 1785.0,
             clean: bool = True, reasons: list[str] | None = None) -> dict:
    return {"phase": "measured", "success": True, "started_at_utc": when.isoformat(), "e2e_ms": e2e,
            "temperature_c_before": temp, "thermally_clean": clean, "throttle_reasons_active": reasons or [],
            "cooldown_waited_s": 31, "cooldown_timed_out": False,
            "thermal_monitor": {"sm_clock_mhz_min": clock_min, "sm_clock_mhz_mean": clock_mean, "nominal_sm_mhz": 1785.0,
                                "temperature_c_max": temp + 10, "power_w_mean": 300.0, "event_reasons_seen": reasons or []}}


def test_audit_separates_clean_and_throttled_cells_and_splits_at_the_change(tmp_path):
    t0 = datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc)
    _write_cell(tmp_path, "OLD", "mstar", "t2i-768", [
        _request(t0 + timedelta(minutes=i), 10000, clock_mean=1400, clock_min=1050, clean=False, reasons=["sw_power_cap"])
        for i in range(3)])
    _write_cell(tmp_path, "NEW", "mstar", "t2i-768", [
        _request(t0 + timedelta(days=1, minutes=i), 8000) for i in range(3)])
    rows = ath.load(tmp_path, None)
    assert len(rows) == 6
    text = ath.report(rows, t0 + timedelta(hours=12), cells=True)
    before, after = [line for line in text.splitlines() if line.startswith(("| before", "| after")) and line.count("|") > 8]
    assert "0/3" in before and "1050" in before and "sw_power_cap" in before
    assert "3/3" in after and "1785" in after and "none" in after
    assert "| NEW | mstar | t2i-768 | 3 | 8000 | 1785 |" in text


def test_same_cell_comparison_pairs_cases_present_on_both_sides_and_honours_exclusions(tmp_path):
    t0 = datetime(2026, 10, 26, 10, 0, tzinfo=timezone.utc)
    _write_cell(tmp_path, "OLD", "mstar", "t2i-512", [_request(t0 + timedelta(minutes=i), 4800, clock_mean=1490) for i in range(3)])
    _write_cell(tmp_path, "OLD", "mstar", "only-before", [_request(t0 + timedelta(minutes=i), 1) for i in range(3)])
    _write_cell(tmp_path, "NEW", "mstar", "t2i-512", [_request(t0 + timedelta(days=1, minutes=i), 4900, clock_mean=1480) for i in range(3)])
    _write_cell(tmp_path, "HOOK", "mstar", "t2i-512", [_request(t0 + timedelta(days=1, hours=1, minutes=i), 100) for i in range(3)])
    rows = ath.load(tmp_path, None)
    split = t0 + timedelta(hours=12)
    lines = ath.same_cell(rows, split, exclude=("HOOK",))
    assert any(line.startswith("| mstar | t2i-512 | 3 / 3 | 4800 -> 4900 | 1490 -> 1480 |") for line in lines)
    assert not any("only-before" in line for line in lines)
    assert any("| 3 / 6 |" in line for line in ath.same_cell(rows, split))                # without the exclusion the hook arm leaks in
    whitelisted = ath.same_cell(rows, split, matched=("OLD", "NEW"))                       # a whitelist keeps only the named campaigns
    assert any("| 3 / 3 |" in line for line in whitelisted) and not any("only-before" in line for line in whitelisted)


def _trace(path: Path, start: datetime, temps: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["wall_clock,utilization_percent,memory_used_mib,temperature_c,memory_temperature_c,clocks_sm_mhz,clocks_max_sm_mhz,"
             "power_draw_w,any_event_reason_active,passes"]
    for i, temp in enumerate(temps):
        lines.append(f"{(start + timedelta(seconds=10 * i)).replace(tzinfo=None).isoformat()},0,5e4,{temp},60,1785,1785,115,False,False")
    path.write_text("\n".join(lines))


def test_cooldown_table_measures_the_time_to_the_gate_temperature(tmp_path):
    local = datetime(2026, 10, 26, 10, 0).astimezone()
    for campaign, day, per_step in (("OLD", 0, 1.0), ("NEW", 1, 2.0)):
        temps = [80 - per_step * i for i in range(30)]                                  # 80 C falling per_step C every 10 s
        _trace(tmp_path / campaign / "BAGEL" / "mstar" / "t2i-768" / "r01" / "cooldowns" / "measured_002.csv",
               local + timedelta(days=day), temps)
    _trace(tmp_path / "OLD" / "BAGEL" / "mstar" / "t2i-768" / "r01" / "cooldowns" / "measured_003.csv", local, [70.0] * 12)  # never hot
    rows = ath.cooldowns(tmp_path)
    assert [round(r["seconds"]) for r in rows] == [100, 50] or sorted(round(r["seconds"]) for r in rows) == [50, 100]
    split = (local + timedelta(hours=12)).astimezone(timezone.utc)
    table = "\n".join(ath.cooldown_table(rows, split))
    assert "| mstar | t2i-768 | 1 / 1 | 80 -> 80 | 100 -> 50 | 10 -> 5 |" in table


def test_cfg_ladder_view_counts_branches_per_engine_and_pairs_the_equal_work_cells(tmp_path, monkeypatch):
    from exps.core import paths
    monkeypatch.setattr(paths, "RUNS", tmp_path)
    t0 = datetime(2026, 10, 28, tzinfo=timezone.utc)

    def cell(campaign, framework, case, e2e):
        _write_cell(tmp_path, campaign, framework, case, [_request(t0 + timedelta(minutes=i), e2e) for i in range(5)])
    cell("L", "mstar", "t2i-768-nocfg", 4000)
    cell("L", "mstar", "t2i-768", 10400)
    cell("L", "mstar", "t2i-768-cfg3", 10500)
    cell("L", "vllm-omni-single-stage", "t2i-768-nocfg", 3500)
    cell("L", "vllm-omni-single-stage", "t2i-768", 8000)
    cell("L", "vllm-omni-single-stage", "t2i-768-cfg3", 10300)
    cell("L", "vllm-omni-single-stage", "i2i-768", 1)             # not a ladder case: ignored
    text = followups.ladder(["L"])
    assert "| 768 | mstar | default | 2 |" in text and "| 768 | vllm-omni-single-stage | default | 2 |" in text
    assert "| 768 | mstar | cfg3 | 3 | 1 | 10,500 [10,500–10,500] |" in text
    assert "| 768 | vllm-omni-single-stage | 10,500 | 10,300 | 0.98x |" in text            # M* cfg3 vs vLLM cfg3: the three-branch equal-work ratio
    assert "| 768 | vllm-omni-single-stage | 4,000 | 3,500 | 0.88x |" in text             # one branch: same work and same output semantics
    assert "| 768 | mstar | default | 2 | 1 | 10,400 [10,400–10,400] | 1,785 | 128.0 |" in text          # (10,400 - 4,000) / 1 / 50: two branches in current M*
    assert "| 768 | vllm-omni-single-stage | default | 2 | 1 | 8,000 [8,000–8,000] | 1,785 | 90.0 |" in text  # (8,000 - 3,500) / 1 / 50
    assert "i2i" not in text


def _nsys_db(path: Path, kernels: list[tuple[str, int, int]]) -> Path:
    import sqlite3
    con = sqlite3.connect(path)
    con.execute("create table StringIds (id integer primary key, value text)")
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (shortName integer, start integer, end integer)")
    for i, (name, launches, duration_ns) in enumerate(kernels, 1):
        con.execute("insert into StringIds values (?, ?)", (i, name))
        for j in range(launches):
            con.execute("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?)", (i, j * 10_000_000, j * 10_000_000 + duration_ns))
    con.commit()
    con.close()
    return path


def test_gemm_launch_table_lists_heaviest_gemms_and_skips_memsets_and_non_gemms(tmp_path):
    db = _nsys_db(tmp_path / "a.sqlite", [("nvjet_tst_192x192_64x4_2x1_v_b_51", 4, 2_900_000), ("nvjet_tst_256x128_64x4", 10, 800_000),
                                           ("Memset (Device)", 50, 10_000_000), ("vectorized_elementwise_kernel", 100, 20_000)])
    rows, busy = gemm_launches.top_gemms(db, 5)
    assert [r["name"] for r in rows] == ["nvjet_tst_192x192_64x4_2x1_v_b_51", "nvjet_tst_256x128_64x4"]
    assert rows[0]["launches"] == 4 and round(rows[0]["mean_us"]) == 2900 and busy > rows[0]["total_ms"]
    text = gemm_launches.markdown([db], 2)
    assert "| 1 | `nvjet_tst_192x192_64x4_2x1_v_b_51` | 4 | 2,900 |" in text


def test_audit_ignores_requests_without_telemetry_and_failed_ones(tmp_path):
    t0 = datetime(2026, 10, 26, tzinfo=timezone.utc)
    bad = _request(t0, 100) | {"success": False}
    bare = {"phase": "measured", "success": True, "started_at_utc": t0.isoformat(), "e2e_ms": 5}
    _write_cell(tmp_path, "C", "mstar", "case", [bad, bare, _request(t0, 100)])
    assert len(ath.load(tmp_path, None)) == 1


def test_sensitivity_needs_a_temperature_spread_and_says_so(tmp_path):
    t0 = datetime(2026, 10, 26, tzinfo=timezone.utc)
    flat = [_request(t0 + timedelta(minutes=i), 1000 + i, temp=60.0) for i in range(8)]
    _write_cell(tmp_path, "FLAT", "mstar", "case", flat)
    slope, n, _span = ath.sensitivity(ath.load(tmp_path, None))
    assert slope is None and n == 0                                       # a constant start temperature shows no slope: not zero
    hot = [_request(t0 + timedelta(minutes=i), 1000 * (1 + 0.01 * (60 + i - 63)), temp=60.0 + i) for i in range(8)]
    _write_cell(tmp_path, "HOT", "mstar", "case2", hot)
    slope, n, span = ath.sensitivity(ath.load(tmp_path, ["HOT"]))
    assert n == 8 and span == 7 and slope == pytest.approx(1.0, abs=0.05)  # +1% E2E per degree


# ----- thermal probe -----------------------------------------------------------------------------------------------------

def _samples(path: Path, burn_clock: float, reason: str = "", seconds: int = 60) -> None:
    path.mkdir(parents=True)
    fields = ["t_s", "phase", *probe.GPU_FIELDS, *probe.CPU_FIELDS]
    t = 0.0
    with open(path / "samples.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for phase, clock, temp, power in (("idle", 345, 48, 75), ("burn", burn_clock, 80, 450), ("mixed", burn_clock, 75, 300),
                                          ("cool", 345, 70, 75)):
            for i in range(seconds):
                temp_now = temp - (i * 0.3 if phase == "cool" else 0)
                writer.writerow({"t_s": t, "phase": phase, "utilization_percent": 99, "temperature_c": temp_now,
                                 "memory_temperature_c": temp_now - 4, "clocks_sm_mhz": clock, "clocks_max_sm_mhz": 1785,
                                 "power_draw_w": power, "enforced_power_limit_w": 600, "any_event_reason_active": bool(reason),
                                 "reasons": reason if phase == "burn" else "", "cpu_mhz_top8_mean": 3500,
                                 "cpu_mhz_max": 3900, "cpu_pkg_temp_c_max": 40})
                t += 1
    (path / "meta.json").write_text(json.dumps({"note": "test", "host_before": {"cpu_throttle_counters": {"core": 0, "package": 0}},
                                                "host_after": {"cpu_throttle_counters": {"core": 0, "package": 3}}}))


def test_probe_verdict_clean_when_the_clock_holds(tmp_path):
    _samples(tmp_path / "ok", 1785)
    rows, meta = probe.load(tmp_path / "ok")
    text, stats = probe.report(rows, meta, "ok")
    assert "CLEAN" in text and "never" in text and "package thermal-throttle counter: 0 -> 3 (+3" in text
    assert "back to <= 70 C after 0 s" in text and "back to <= 60 C after" in text


def test_probe_verdict_throttled_when_a_reason_is_active_or_the_clock_falls(tmp_path):
    _samples(tmp_path / "bad", 1200, reason="sw_thermal_slowdown")
    rows, meta = probe.load(tmp_path / "bad")
    text, _ = probe.report(rows, meta, "bad")
    assert "THROTTLED" in text and "sw_thermal_slowdown 25%" not in text and "sw_thermal_slowdown 100%" in text
    _samples(tmp_path / "mid", 1700)                       # 95% of nominal, no reason: neither clean nor throttled
    rows, meta = probe.load(tmp_path / "mid")
    assert "MARGINAL" in probe.report(rows, meta, "mid")[0]


def test_probe_analyze_compares_with_a_baseline(tmp_path):
    _samples(tmp_path / "new", 1785)
    _samples(tmp_path / "old", 1300, reason="sw_power_cap")
    out = subprocess.run([sys.executable, str(ROOT / "exps/tools/gpu_thermal_probe.py"), "analyze", str(tmp_path / "new"),
                          "--baseline", str(tmp_path / "old")], capture_output=True, text=True, check=True).stdout
    assert "burn median SM clock (MHz): 1300 -> 1785" in out


def test_probe_dry_run_prints_the_plan_without_touching_a_gpu(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "exps/tools/gpu_thermal_probe.py"), "run", "--dry-run",
                          "--out", str(tmp_path / "p"), "--burn", "10"], capture_output=True, text=True, check=True).stdout
    assert "burn 10s" in out and "would run" in out and not (tmp_path / "p").exists()


def test_probe_refuses_to_overwrite_an_existing_run(tmp_path):
    (tmp_path / "p").mkdir()
    (tmp_path / "p" / "samples.csv").write_text("x")
    # the guard comes after the (lazy) gpu import; an environment without nvidia-smi fails earlier, which is also a refusal
    result = subprocess.run([sys.executable, str(ROOT / "exps/tools/gpu_thermal_probe.py"), "run", "--out", str(tmp_path / "p")],
                            capture_output=True, text=True)
    assert result.returncode != 0 and (tmp_path / "p" / "samples.csv").read_text() == "x"


# ----- the two hooks (fake mstar modules; torch is real and CPU-only) ---------------------------------------------------

def _fake_bagel(tmp_path: Path) -> Path:
    pkg = tmp_path / "fake" / "mstar" / "model" / "bagel"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "fake" / "mstar", tmp_path / "fake" / "mstar" / "model", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "submodules.py").write_text(
        "class LLMSubmodule:\n    def _sample_per_request(self, request_ids, logits):\n"
        "        return {rid: {'new_token': [1]} for rid in request_ids}\n")
    return tmp_path / "fake"


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
@pytest.mark.parametrize("armed, applied", [("1", True), ("0", False)])
def test_sampler_hook_wraps_only_when_armed_and_always_logs(tmp_path, armed, applied):
    log = tmp_path / "proof.jsonl"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SAMPLER_HOOK), str(_fake_bagel(tmp_path))]),
           "MSTAR_BAGEL_SAMPLER_EAGER": armed, "MSTAR_BAGEL_SAMPLER_LOG": str(log)}
    probe_code = ("from mstar.model.bagel.submodules import LLMSubmodule as L\n"
                  "print(bool(getattr(L._sample_per_request, '_torchdynamo_disable', False)), L()._sample_per_request(['a'], None))")
    out = subprocess.run([str(MSTAR_PYTHON), "-c", probe_code], env=env, cwd=tmp_path, capture_output=True, text=True, check=True).stdout
    assert out.startswith(str(applied)) and "'a': {'new_token': [1]}" in out        # behaviour unchanged, only the dynamo flag differs
    line = json.loads(log.read_text().splitlines()[0])
    assert line["applied"] is applied and line["armed"] == armed


_FAKE_FULL = """
import torch
_LOGITS = "logits"


class LLMSubmodule(torch.nn.Module):
    def _forward_decode_batched(self, input_ids, requires_cfg=False):
        return {_LOGITS: [input_ids.float() @ torch.ones(4, 8)]}

    def _forward_prefill_text(self, input_ids, requires_cfg=False, sample_prefill_token=True, **kwargs):
        return {_LOGITS: [input_ids.float() @ torch.ones(4, 8)]}

    def _forward_prefill_vit(self, input_embeds, requires_cfg=False, sample_prefill_token=False, **kwargs):
        return {}

    def _sample_tokens(self, request_ids, logits):
        return logits.argmax(-1).clone()

    def _sample_per_request(self, request_ids, logits):
        tokens = self._sample_tokens(request_ids, logits)
        return {rid: {"new_token": [t]} for rid, t in zip(request_ids, tokens.split(1), strict=True)}

    def forward_batched(self, graph_walk, engine_inputs, input_ids=None, input_embeds=None, sample_token=False,
                        requires_cfg=False, **kwargs):
        request_ids = engine_inputs.request_ids

        if graph_walk == "decode":
            out = self._forward_decode_batched(input_ids=input_ids, requires_cfg=requires_cfg)
        elif graph_walk == "prefill_text":
            out = self._forward_prefill_text(input_ids=input_ids, requires_cfg=requires_cfg, sample_prefill_token=True, **kwargs)
        elif graph_walk == "prefill_vit":
            out = self._forward_prefill_vit(input_embeds=input_embeds, requires_cfg=requires_cfg,
                                            sample_prefill_token=sample_token, **kwargs)
        else:
            raise ValueError(graph_walk)
        if _LOGITS in out:
            return self._sample_per_request(request_ids, out.pop(_LOGITS)[0])
        return out
"""

_RECOMPILE_DRIVER = """
import types, uuid, sys, torch
from mstar.model.bagel.submodules import LLMSubmodule
s = LLMSubmodule()
f = torch.compile(s.forward_batched, backend="eager", fullgraph=False, dynamic=None)   # what Engine._compile_submodules does
x = torch.randn(1, 4)
for _ in range(6):
    r = f("decode", types.SimpleNamespace(request_ids=[str(uuid.uuid4())]), input_ids=x)
print("TOKEN", list(r.values())[0]["new_token"][0].item())
"""


def _fake_full(tmp_path: Path) -> Path:
    pkg = tmp_path / "fakef" / "mstar" / "model" / "bagel"
    pkg.mkdir(parents=True, exist_ok=True)
    for d in (tmp_path / "fakef" / "mstar", tmp_path / "fakef" / "mstar" / "model", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "submodules.py").write_text(_FAKE_FULL)
    return tmp_path / "fakef"


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
@pytest.mark.parametrize("armed, at_most, at_least", [("0", 99, 5), ("1", 99, 5), ("2", 0, 0)])
def test_sampler_hook_variant_2_stops_the_per_request_recompile(tmp_path, armed, at_most, at_least):
    """Variant 1 (sampler only) leaves the forward_batched resume frame guarding on the request id; variant 2 removes it."""
    log = tmp_path / "proof.jsonl"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SAMPLER_HOOK), str(_fake_full(tmp_path))]),
           "MSTAR_BAGEL_SAMPLER_EAGER": armed, "MSTAR_BAGEL_SAMPLER_LOG": str(log), "TORCH_LOGS": "recompiles"}
    done = subprocess.run([str(MSTAR_PYTHON), "-c", _RECOMPILE_DRIVER], env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    recompiles = (done.stdout + done.stderr).count("Recompiling function")
    assert at_least <= recompiles <= at_most, done.stderr[-2000:]
    assert "TOKEN" in done.stdout
    assert json.loads(log.read_text().splitlines()[0])["applied"] is (armed != "0")


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
def test_sampler_hook_variant_2_gives_the_same_tokens_as_the_original(tmp_path):
    outs = []
    for armed in ("0", "2"):
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SAMPLER_HOOK), str(_fake_full(tmp_path))]), "MSTAR_BAGEL_SAMPLER_EAGER": armed}
        outs.append(subprocess.run([str(MSTAR_PYTHON), "-c", _RECOMPILE_DRIVER], env=env, cwd=tmp_path, capture_output=True,
                                   text=True, check=True).stdout.strip().splitlines()[-1])
    assert outs[0] == outs[1]


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
def test_sampler_hook_variant_2_refuses_when_the_source_differs(tmp_path):
    log = tmp_path / "proof.jsonl"
    fake = _fake_bagel(tmp_path)        # a forward_batched-less module: not what variant 2 was written against
    (fake / "mstar" / "model" / "bagel" / "submodules.py").write_text(
        "_LOGITS = 'logits'\nclass LLMSubmodule:\n    def forward_batched(self):\n        return 1\n"
        "    def _sample_per_request(self, r, l):\n        return 1\n")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(SAMPLER_HOOK), str(fake)]), "MSTAR_BAGEL_SAMPLER_EAGER": "2",
           "MSTAR_BAGEL_SAMPLER_LOG": str(log)}
    subprocess.run([str(MSTAR_PYTHON), "-c", "from mstar.model.bagel.submodules import LLMSubmodule as L; print(L().forward_batched())"],
                   env=env, cwd=tmp_path, capture_output=True, text=True, check=True)
    line = json.loads(log.read_text().splitlines()[0])
    assert line["applied"] is False and "differs" in line["refused"]


def test_sampler_hook_is_inert_without_the_environment(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MSTAR_BAGEL_SAMPLER")}
    env["PYTHONPATH"] = os.pathsep.join([str(SAMPLER_HOOK), str(_fake_bagel(tmp_path))])
    out = subprocess.run([sys.executable, "-c", "import sys; print(any(type(f).__module__ == 'sitecustomize' for f in sys.meta_path))"],
                         env=env, cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def _fake_qwen(tmp_path: Path) -> Path:
    pkg = tmp_path / "fakeq" / "mstar" / "model" / "qwen3_omni"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "fakeq" / "mstar", tmp_path / "fakeq" / "mstar" / "model", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "qwen3_omni_model.py").write_text(
        "import torch\n\ndef create_code2wav():\n    torch.backends.cudnn.benchmark = True   # what M* does\n")
    return tmp_path / "fakeq"


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
@pytest.mark.parametrize("pin, expect", [("0", "False"), ("1", "True")])
def test_cudnn_hook_pins_the_flag_against_the_models_assignment(tmp_path, pin, expect):
    log = tmp_path / "proof.jsonl"
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(CUDNN_HOOK), str(_fake_qwen(tmp_path))]),
           "MSTAR_CUDNN_BENCHMARK": pin, "MSTAR_CUDNN_LOG": str(log)}
    code = ("import torch\nfrom mstar.model.qwen3_omni import qwen3_omni_model as m\n"
            "m.create_code2wav()\nprint(torch.backends.cudnn.benchmark)")
    out = subprocess.run([str(MSTAR_PYTHON), "-c", code], env=env, cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert out == expect
    events = [json.loads(line)["event"] for line in log.read_text().splitlines()]
    assert events[0] == "pin installed"
    assert ("assignment overridden" in events) is (pin == "0")      # the control arm is not overridden: the model's True is the pin


def test_cudnn_hook_is_inert_without_the_environment(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MSTAR_CUDNN")}
    env["PYTHONPATH"] = str(CUDNN_HOOK)
    out = subprocess.run([sys.executable, "-c", "import sys; print(any(type(f).__module__ == 'sitecustomize' for f in sys.meta_path))"],
                         env=env, cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


# ----- logstats series --------------------------------------------------------------------------------------------------

def test_logstats_series_lists_one_step_per_request_in_order():
    requests = [{"rid": f"r{i}", "segments": {}, "walks": {"audio_encoder.prefill_audio": {"n": 1, "all": a, "fwd": a - 1, "pre": 0.1, "post": 0.1}}}
                for i, a in enumerate((4254.9, 129.5, 17.0), 1)]
    requests.append({"rid": "r4", "segments": {}, "walks": {}})
    text = alog.markdown("x", requests, "audio_encoder.prefill_audio")
    rows = [line for line in text.splitlines() if line.startswith("| ") and line.split("|")[1].strip().isdigit()]
    assert rows[:3] == ["| 1 | 4254.9 | 4253.9 |", "| 2 | 129.5 | 128.5 |", "| 3 | 17.0 | 16.0 |"]
    assert any(row.startswith("| 4 | n/a") for row in rows)


# ----- the script -------------------------------------------------------------------------------------------------------











def test_pi05_graph_option_drops_the_enforce_eager_flag_only_with_the_option():
    import contextlib
    import io

    from exps import run as run_module
    from exps.core import overrides, paths

    def plan(*extra):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            assert run_module.main(["pi05", "--framework", "vllm-omni", "--profile", "smoke", "--dry-run", "--campaign", "T",
                                    *extra]) == 0
        overrides.reset()
        paths.set_campaign(None)
        return json.loads(buffer.getvalue())["plans"][0]

    eager = plan()["service_command"]
    graph = plan("--opt", "pi05.vllm_graph=1")
    assert "--enforce-eager" in eager and "pi05_bfloat16.yaml" in eager and "graph.yaml" not in eager
    assert "--enforce-eager" not in graph["service_command"] and "pi05_bfloat16_graph.yaml" in graph["service_command"]
    assert graph["overrides"]["options"] == {"pi05.vllm_graph": "1"}




def test_vjepa_native_precision_context_changes_compute_dtype():
    import torch
    from exps.tasks.vjepa_ac.meta_native_client import precision_context

    layer = torch.nn.Linear(8, 8)
    sample = torch.ones(1, 8)
    with precision_context(torch, "float32", "cpu"):
        assert layer(sample).dtype == torch.float32
    with precision_context(torch, "bfloat16", "cpu"):
        assert layer(sample).dtype == torch.bfloat16


def test_vllm_extra_env_reaches_the_serve_command():
    sys.path.insert(0, str(ROOT))
    import exps.frameworks as fw
    from exps.core import overrides
    overrides.reset()
    plain = fw.vllm_omni_serve(model_path=Path("/m"), deploy_config=Path("/c.yaml"), port_number=1)
    overrides.set_current(overrides.build(env=["vllm-omni:PYTHONPATH=/hook", "vllm-omni:VLLM_BAGEL_CPROFILE_OUT=/out"]))
    try:
        hooked = fw.vllm_omni_serve(model_path=Path("/m"), deploy_config=Path("/c.yaml"), port_number=1)
        # a task recipe env (BAGEL's tuned MoT folder) is overridden by the same key from the command line
        overrides.set_current(overrides.build(env=["vllm-omni:VLLM_TUNED_CONFIG_FOLDER=/other"]))
        both = fw.vllm_omni_serve(model_path=Path("/m"), deploy_config=Path("/c.yaml"), port_number=1,
                                  env={"VLLM_TUNED_CONFIG_FOLDER": "/recipe"})
    finally:
        overrides.reset()
    assert "PYTHONPATH" not in plain and "PYTHONPATH=/hook VLLM_BAGEL_CPROFILE_OUT=/out" in hooked
    assert hooked.replace(" PYTHONPATH=/hook VLLM_BAGEL_CPROFILE_OUT=/out", "") == plain
    assert "VLLM_TUNED_CONFIG_FOLDER=/other" in both and "/recipe" not in both


def _fake_vllm_bagel(tmp_path: Path) -> Path:
    pkg = tmp_path / "fake" / "vllm_omni" / "diffusion" / "models" / "bagel"
    pkg.mkdir(parents=True)
    for d in (tmp_path / "fake" / "vllm_omni", tmp_path / "fake" / "vllm_omni" / "diffusion", tmp_path / "fake" / "vllm_omni" / "diffusion" / "models", pkg):
        (d / "__init__.py").write_text("")
    (pkg / "bagel_transformer.py").write_text(
        "class Bagel:\n    def forward(self, x, scale=1):\n        return sum(range(1000)) + x * scale\n")
    return tmp_path / "fake"


def test_vllm_cprofile_hook_profiles_the_counted_forward_calls_only(tmp_path):
    out = tmp_path / "prof"
    hook = ROOT / "exps" / "tools" / "vllm_bagel_cprofile_hook"
    probe = ("from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel\n"
             "b = Bagel()\n"
             "assert [b.forward(1, scale=2) for _ in range(6)] == [sum(range(1000)) + 2] * 6\n")   # skip 2, profile 3, pass one through
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(hook), str(_fake_vllm_bagel(tmp_path))]), "VLLM_BAGEL_CPROFILE_OUT": str(out),
           "VLLM_BAGEL_CPROFILE_SKIP": "2", "VLLM_BAGEL_CPROFILE_COUNT": "3"}
    subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path, check=True)
    assert sorted(p.name for p in out.glob("*.pstats")) == [f"Bagel.forward.{k}.pstats" for k in (1, 2, 3)]
    assert [json.loads(line)["call"] for line in (out / "hook_proof.jsonl").read_text().splitlines()] == [3, 4, 5]
    quiet = {k: v for k, v in env.items() if k != "VLLM_BAGEL_CPROFILE_OUT"}
    name = subprocess.run([sys.executable, "-c", "from vllm_omni.diffusion.models.bagel.bagel_transformer import Bagel; print(Bagel.forward.__qualname__)"],
                          env=quiet, cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert name == "Bagel.forward"                                  # not wrapped without the environment variable


def test_c2w_graph_yaml_differs_from_the_async_yaml_only_in_stage_2_eager():
    import yaml
    d = ROOT / "exps" / "configs" / "vllm_omni"
    a = yaml.safe_load((d / "qwen3_omni_full_1gpu_default_async.yaml").read_text())
    b = yaml.safe_load((d / "qwen3_omni_full_1gpu_default_async_c2wgraph.yaml").read_text())
    assert a["stages"][2].pop("enforce_eager") is True and b["stages"][2].pop("enforce_eager") is False
    assert a == b and b["async_chunk"] is True
