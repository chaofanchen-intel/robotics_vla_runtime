"""CPU tests for the round-7 tools: the Engine.exec cProfile hook, its analyzer, the GEMM comparison, the script."""
from __future__ import annotations

import cProfile
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exps" / "tools"))

import analyze_cprofile as acp  # noqa: E402

HOOK = ROOT / "exps" / "tools" / "mstar_cprofile_hook"


def _fake_mstar(tmp_path: Path) -> Path:
    pkg = tmp_path / "fake" / "mstar" / "engine"
    pkg.mkdir(parents=True)
    (tmp_path / "fake" / "mstar" / "__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "engine.py").write_text(
        "class Engine:\n    def exec(self, batch):\n        return sum(range(1000)) + batch.n\n")
    return tmp_path / "fake"


def test_hook_profiles_only_the_target_calls_after_the_skip(tmp_path):
    out = tmp_path / "prof"
    probe = (
        "from types import SimpleNamespace as N\n"
        "from mstar.engine.engine import Engine\n"
        "e = Engine()\n"
        "mk = lambda node, walk: N(node_name=node, step_context=N(graph_walk=walk), n=1)\n"
        "for _ in range(3): e.exec(mk('LLM', 'decode'))\n"          # other target: never profiled
        "for _ in range(5): e.exec(mk('LLM', 'prefill_text'))\n"    # skip 2, profile 2, pass the last one through
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(HOOK), str(_fake_mstar(tmp_path))]),
           "MSTAR_CPROFILE_TARGET": "LLM.prefill_text", "MSTAR_CPROFILE_OUT": str(out),
           "MSTAR_CPROFILE_SKIP": "2", "MSTAR_CPROFILE_COUNT": "2"}
    subprocess.run([sys.executable, "-c", probe], env=env, cwd=tmp_path, check=True)
    assert sorted(p.name for p in out.glob("*.pstats")) == ["LLM.prefill_text.1.pstats", "LLM.prefill_text.2.pstats"]
    proof = [json.loads(line) for line in (out / "hook_proof.jsonl").read_text().splitlines()]
    assert [p["call"] for p in proof] == [3, 4]


def test_hook_is_inert_without_the_environment(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("MSTAR_CPROFILE")}
    env["PYTHONPATH"] = os.pathsep.join([str(HOOK), str(_fake_mstar(tmp_path))])
    out = subprocess.run([sys.executable, "-c", "from mstar.engine.engine import Engine; print(Engine.exec.__name__)"],
                         env=env, cwd=tmp_path, capture_output=True, text=True, check=True).stdout.strip()
    assert out == "exec"


def _dump(directory: Path, fn, calls: int) -> None:
    directory.mkdir(parents=True)
    for k in range(1, calls + 1):
        profile = cProfile.Profile()
        profile.enable()
        fn()
        profile.disable()
        profile.dump_stats(str(directory / f"t.{k}.pstats"))
    (directory / "hook_proof.jsonl").write_text("\n".join(json.dumps({"wall_ms": 10.0 * k}) for k in range(1, calls + 1)))


def _slow():
    return sum(i * i for i in range(200_000))


def _fast():
    return sum(range(10))


def test_analyzer_names_the_function_that_the_switch_added(tmp_path):
    _dump(tmp_path / "on", lambda: (_slow(), _fast()), 2)
    _dump(tmp_path / "off", _fast, 2)
    text = acp.markdown({"on": tmp_path / "on", "off": tmp_path / "off"}, top=5)
    assert "2 profiled calls" in text and "on minus off" in text
    delta_section = text.split("on minus off")[1]
    assert "_slow" in delta_section.splitlines()[5] or "_slow" in delta_section.splitlines()[4]


def test_analyzer_reports_a_group_without_dumps(tmp_path):
    (tmp_path / "empty").mkdir()
    assert "hook did not write" in acp.markdown({"x": tmp_path / "empty"}, top=3)


def _gemm(ms):
    return {"torch": "t", "cuda": "c", "device": "d",
            "rows": [{"M": 6912, "layer": "gate_up", "N": 1, "K": 1, "ms": ms, "tflops": 1.0, "kernel": "k"}]}


def test_gemm_compare_reports_the_ratio(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps(_gemm(2.0)))
    (tmp_path / "b.json").write_text(json.dumps(_gemm(1.0)))
    out = subprocess.run([sys.executable, str(ROOT / "exps" / "tools" / "bench_gemm_across_envs.py"), "compare",
                          str(tmp_path / "a.json"), str(tmp_path / "b.json")], capture_output=True, text=True, check=True)
    assert "| 6912 | gate_up | 2.000 | 1.000 | 2.00 |" in out.stdout


def test_gemm_compare_does_not_present_a_memset_as_the_gemm_kernel(tmp_path):
    # round 7 recorded the first CUDA event, which is the split-K workspace Memset; that must not read as a kernel name
    a = _gemm(1.0)
    a["rows"][0]["kernel"] = "Memset (Device)"
    a["rows"][0]["sm_clock_mhz"] = 1785.0
    (tmp_path / "a.json").write_text(json.dumps(a))
    (tmp_path / "b.json").write_text(json.dumps(_gemm(1.0)))
    out = subprocess.run([sys.executable, str(ROOT / "exps" / "tools" / "bench_gemm_across_envs.py"), "compare",
                          str(tmp_path / "a.json"), str(tmp_path / "b.json")], capture_output=True, text=True, check=True)
    assert "Memset" not in out.stdout and "n/a (not recorded)" in out.stdout and "1785.0 / n/a" in out.stdout




def test_batch_config_override_reaches_the_serve_command():
    from exps.core import overrides
    from exps.tasks import qwen_common

    overrides.reset()
    default = qwen_common.config_path("vllm-omni", batch_config=True)
    # the default concurrency recipe is the best one (async chunk + Code2Wav graph), not the plain sync yaml
    assert default.name == "qwen3_omni_full_1gpu_batch_async_c2wgraph.yaml"
    overrides.set_current(overrides.build(options=["qwen.vllm_batch_config=exps/configs/vllm_omni/qwen3_omni_full_1gpu_batch.yaml"]))
    try:
        assert qwen_common.config_path("vllm-omni", batch_config=True).name == "qwen3_omni_full_1gpu_batch.yaml"
        assert qwen_common.config_path("vllm-omni", batch_config=False).name == qwen_common.VLLM_CONFIG.name  # C=1 recipe unaffected
    finally:
        overrides.reset()


def test_batch_async_yaml_differs_only_in_async_chunk():
    base = (ROOT / "exps/configs/vllm_omni/qwen3_omni_full_1gpu_batch.yaml").read_text().splitlines()
    tuned = (ROOT / "exps/configs/vllm_omni/qwen3_omni_full_1gpu_batch_async.yaml").read_text().splitlines()
    body = [line for line in tuned if not line.startswith("# qwen3_omni_full_1gpu_batch.yaml with ONE") and
            not line.startswith("# the sync batch yaml is what")]
    changed = [(a, b) for a, b in zip(base, body) if a != b]
    assert changed == [("async_chunk: false", "async_chunk: true")] and len(base) == len(body)


MSTAR_PYTHON = ROOT / "envs" / "mstar" / "bin" / "python"


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
def test_logstats_parser_reads_what_mstar_itself_writes(tmp_path):
    """Generate the file with M*'s own `pretty_print_profile`, so the parser cannot drift from the real format."""
    log = tmp_path / "stats.txt"
    gen = (
        "from mstar.profile.display import pretty_print_profile\n"
        "from mstar.profile.format import RequestProfile, RequestTiming, GraphTiming\n"
        f"log = {str(log)!r}\n"
        "for i, scale in enumerate((1.0, 2.0)):\n"
        "    t = RequestTiming(recv_time=0.0, preprocess_finish_time=0.01, conductor_ingest_time=0.012, first_chunk_time=0.3*scale,\n"
        "                      last_chunk_time=0.5*scale, conductor_finish_time=0.5*scale, finish_time=0.51*scale)\n"
        "    g = [GraphTiming('Talker', 'talker_decode', 48, 0.37*scale, 0.30*scale, 0.03*scale, 0.04*scale),\n"
        "         GraphTiming('Thinker', 'thinker_decode', 13, 0.08*scale, 0.06*scale, 0.01*scale, 0.01*scale)]\n"
        "    pretty_print_profile(RequestProfile(rid=f'r{i}', timing=t, graph_timings=g), filename=log)\n"
    )
    subprocess.run([str(MSTAR_PYTHON), "-c", gen], cwd=tmp_path, check=True,
                   env={**os.environ, "PYTHONPATH": str(ROOT / "external" / "mstar")})
    import analyze_mstar_logstats as als

    requests = als.parse(log.read_text())
    assert [r["rid"] for r in requests] == ["r0", "r1"]
    walk = requests[1]["walks"]["Talker.talker_decode"]
    assert walk["n"] == 48 and walk["all"] == pytest.approx(740.0, abs=1) and walk["pre"] == pytest.approx(60.0, abs=1)
    assert requests[0]["segments"]["total"] == pytest.approx(510.0, abs=1)
    text = als.markdown("x", requests)
    assert "Talker.talker_decode" in text and "ratio" in text
    assert "No `Request profile`" in als.markdown("x", [])


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
def test_mstar_serve_accepts_the_logstats_flags():
    """Lesson of the --enable-nvtx slip: a dry-run only prints the command, the real argparse decides."""
    out = subprocess.run([str(ROOT / "envs/mstar/bin/mstar"), "serve", "--help"], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0 and "--log-stats-file" in out.stdout and "--log-stats" in out.stdout
