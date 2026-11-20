"""CPU checks of the tools behind the `qwen-async-sched`, `bagel-async-sched`, `cosmos3-decompose`, `bagel-i2t-nsys` experiments."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
from pathlib import Path

from exps.suite import catalogue, spec
from exps.tools import analyze_decode_itl, analyze_nsys_burst_gaps, analyze_nsys_step_kernels

REPO = Path(__file__).resolve().parents[2]
HOOK = REPO / "exps" / "tools" / "mstar_cosmos3_cfg_hook"


def _write_cell(root: Path, framework: str, case: str, itl: float, ttft: float, tokens: int = 4) -> None:
    cell = root / "TASK" / framework / case / "r01"
    cell.mkdir(parents=True)
    rows = [{"phase": "precondition", "success": True, "e2e_ms": 9e9, "ttft_ms": 9e9, "itl_ms": [999.0], "output_tokens": tokens}]
    for _ in range(3):
        rows.append({"phase": "measured", "success": True, "e2e_ms": ttft + itl * (tokens - 1), "ttft_ms": ttft,
                     "itl_ms": [0.1] + [itl] * (tokens - 1), "output_tokens": tokens})
    (cell / "requests.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


def test_decode_itl_ignores_preconditions_and_chunk_mates(tmp_path):
    _write_cell(tmp_path / "CAMP", "mstar", "txt-1k", itl=6.0, ttft=150.0)
    [(camp, fw, case, stats)] = analyze_decode_itl.collect([tmp_path / "CAMP"])
    assert (camp, fw, case) == ("CAMP", "mstar", "txt-1k")
    assert stats["n"] == 3 and stats["itl"] == 6.0 and stats["ttft"] == 150.0 and stats["tokens"] == 4
    assert "| 6.00 |" in analyze_decode_itl.render([(camp, fw, case, stats)])


def test_burst_gaps_split_at_idle_gaps(tmp_path):
    db = tmp_path / "x.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer)")
    ms = 1_000_000
    kernels = [(i * 10 * ms, i * 10 * ms + 5 * ms) for i in range(300)]                 # burst 1: 3 s of 5 ms kernels
    base = 3_000 * ms + 700 * ms                                                           # 700 ms idle
    kernels += [(base + i * 10 * ms, base + i * 10 * ms + 5 * ms) for i in range(300)]    # burst 2
    con.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?)", kernels)
    con.commit()
    first, second = analyze_nsys_burst_gaps.bursts(db, gap_ms=40.0, min_launches=200)
    assert first["gap_ms"] is None and first["launches"] == 300
    assert 695 <= second["gap_ms"] <= 705 and abs(second["busy_ms"] - 1500) < 1


def test_burst_gaps_keep_two_stage_processes_apart_and_report_the_handoff(tmp_path):
    """Both stage processes of a vLLM-Omni server are started with their own CUDA_VISIBLE_DEVICES, so nsys reports deviceId 0 for both: the tables must
    split by process, and --handoff must line the Thinker's last kernel up with the DiT's first."""
    db = tmp_path / "two.sqlite"
    con = sqlite3.connect(db)
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (start integer, end integer, deviceId integer, globalPid integer, shortName integer, demangledName integer)")
    con.execute("create table StringIds (id integer, value text)")
    con.execute("create table CUPTI_ACTIVITY_KIND_MEMCPY (start integer, end integer, bytes integer, copyKind integer, globalPid integer)")
    con.executemany("insert into StringIds values (?, ?)", [(1, "gemm"), (2, "void kernel_unified_attention<float>")])
    ms = 1_000_000
    thinker, dit = 11 << 24, 22 << 24
    rows = [(i * 2 * ms, i * 2 * ms + ms, 0, thinker, 1, 1) for i in range(500)]            # Thinker: 1 s, 500 kernels (the last one is attention)
    rows[-1] = (rows[-1][0], rows[-1][1], 0, thinker, 2, 2)
    rows += [(1300 * ms + i * 2 * ms, 1300 * ms + i * 2 * ms + ms, 0, dit, 1, 1) for i in range(1000)]   # DiT starts 300 ms after the Thinker's last kernel
    con.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?)", rows)
    con.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?)", (1010 * ms, 1020 * ms, 340_000_000, 2, thinker))    # KV out of the Thinker
    con.execute("insert into CUPTI_ACTIVITY_KIND_MEMCPY values (?, ?, ?, ?, ?)", (1200 * ms, 1201 * ms, 6_000_000, 1, dit))          # KV into the DiT
    con.commit()
    found = analyze_nsys_burst_gaps.streams(db)
    assert sorted(pid for _, pid in found) == [11, 22]
    assert [b["launches"] for b in analyze_nsys_burst_gaps.bursts(db, 40.0, 100, found[0])] != [1500]       # not merged into one table
    result = analyze_nsys_burst_gaps.handoff(db, dit_min_launches=500)
    [row] = result["rows"]
    assert row["thinker_launches"] == 500 and abs(row["thinker_busy_ms"] - 500) < 1 and abs(row["handoff_ms"] - 301) < 2
    assert abs(row["kv_out_to_load_ms"] - 180) < 2 and abs(row["load_to_kernel_ms"] - 100) < 2


def test_step_kernel_union_counts_overlap_once():
    assert analyze_nsys_step_kernels.union_ms([(0, 10), (5, 12), (20, 25)]) == 17


def test_cosmos3_hook_flips_the_class_attribute_and_logs_proof(tmp_path):
    package = tmp_path / "pkg" / "mstar" / "model" / "cosmos3"
    package.mkdir(parents=True)
    for directory in (package.parent.parent, package.parent, package):
        (directory / "__init__.py").write_text("")
    (package / "submodules.py").write_text("class Cosmos3DiTSubmodule:\n    batched_cfg = True\n")
    log = tmp_path / "proof.jsonl"
    code = textwrap.dedent("""
        import mstar.model.cosmos3.submodules as m
        print(m.Cosmos3DiTSubmodule.batched_cfg)
    """)
    env = {"PYTHONPATH": f"{HOOK}:{tmp_path / 'pkg'}", "MSTAR_COSMOS3_BATCHED_CFG": "0", "MSTAR_COSMOS3_CFG_LOG": str(log),
           "PATH": "/usr/bin:/bin"}
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
    [line] = [json.loads(x) for x in log.read_text().splitlines()]
    assert line["applied"] is True and line["before"] is True and line["after"] is False
    inert = {**env, "MSTAR_COSMOS3_BATCHED_CFG": "1"}
    out = subprocess.run([sys.executable, "-c", code], env=inert, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"


def test_new_experiments_are_registered_and_their_arms_are_checkable():
    for name in ("qwen-async-sched", "bagel-async-sched", "cosmos3-decompose", "bagel-i2t-nsys", "bagel-prefix-cache", "cosmos3-nsys",
                 "bagel-vllm-prefix-cache", "bagel-i2i-steps", "bagel-i2i-nsys",
                 "qwen-fair-txt", "qwen-fair-a2t", "qwen-t2s-fit", "qwen-ttft-txt", "qwen-ttft-a2t", "bagel-sampling-parity"):
        experiment = catalogue.get(name)
        assert isinstance(experiment, spec.Experiment) and experiment.arms
    decompose = catalogue.get("cosmos3-decompose")
    hook_arms = [a for a in decompose.arms if a.expect.proof == "hook_applied"]
    assert {a.name for a in hook_arms} == {"m-seqcfg", "m-seqcfg-nograph", "m-all-off"}
    for arm in hook_arms:
        assert any("MSTAR_COSMOS3_BATCHED_CFG=0" in e for e in arm.env)
    from exps import hardware
    from exps.suite import runner

    for machine in ("rtx5880", "h200nvl"):               # each machine's own file for the counterfactual, both with the switch off
        hardware.set_current(machine)
        for name in ("qwen-async-sched", "bagel-async-sched"):
            off = catalogue.get(name).arm("vllm-async-off")
            [option] = runner.expand_all(off.opts, campaign="C")
            path = REPO / option.split("=", 1)[1]
            assert path.is_file() and "async_scheduling: false" in path.read_text(), (machine, name, path)
            assert f"configs/hardware/{machine}/" in str(path)


def test_bagel_prefix_cache_switch_reaches_only_mstar_requests():
    from exps.tasks.bagel import client

    assert "prefix_cache" not in client.model_kwargs_for("i2t", 512, 128)
    assert client.model_kwargs_for("i2t", 512, 128, prefix_cache=False)["prefix_cache"] is False


def test_qwen_lookahead_eligible_requests_drop_only_the_fields_that_gate_sglang(monkeypatch):
    """SGLang-Omni's lookahead_eligible rejects a batch with seed / min_new_tokens; the counterfactual drops exactly those."""
    from exps.tasks.qwen_a2t import client as a2t
    from exps.tasks.qwen_txt import client as txt

    monkeypatch.delenv("EXPS_QWEN_SAMPLING", raising=False)
    pinned = txt.sampling_kwargs(256, "sglang-omni")
    assert pinned["seed"] == 42 and pinned["min_tokens"] == 256 and a2t.sampling_kwargs(128)["seed"] == 42
    monkeypatch.setenv("EXPS_QWEN_SAMPLING", "lookahead-eligible")
    eligible = txt.sampling_kwargs(256, "sglang-omni")
    assert "seed" not in eligible and "min_tokens" not in eligible
    assert eligible["ignore_eos"] is True and eligible["max_tokens"] == 256 and eligible["temperature"] == 0.0
    assert "seed" not in a2t.sampling_kwargs(128)


def test_fair_experiments_use_configs_that_flip_only_the_named_switch():
    """Each counterfactual yaml differs from its default in exactly the switch the arm names."""
    hw = REPO / "exps/configs/hardware/rtx5880"

    def changed(a: Path, b: Path) -> set[str]:
        from collections import Counter

        x = Counter(l.strip() for l in a.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#"))
        y = Counter(l.strip() for l in b.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#"))
        return set((x - y) + (y - x))

    assert changed(hw / "vllm_omni/qwen3_omni_3gpu_thinker_tp2_async_c2wgraph.yaml",
                   hw / "vllm_omni/qwen3_omni_3gpu_thinker_tp2_prefix_cache.yaml") == {"enable_prefix_caching: false", "enable_prefix_caching: true"}
    assert changed(hw / "sglang_omni/qwen3_omni_3gpu_thinker_tp2_partial_start.yaml",
                   hw / "sglang_omni/qwen3_omni_3gpu_thinker_tp2_no_radix.yaml") == {"disable_radix_cache: true"}
    assert changed(hw / "sglang_omni/qwen3_omni_3gpu_thinker_tp2_partial_start.yaml",
                   hw / "sglang_omni/qwen3_omni_3gpu_thinker_tp2_no_async_decode.yaml") == {"enable_async_decode: false"}
    assert changed(hw / "mstar/qwen3omni_3gpu_thinker_tp2.yaml", hw / "mstar/qwen3omni_3gpu_thinker_tp2_prefix_cache.yaml") == {
        "cpu_offload_pages: 1024", "cpu_offload_pages: 0", "prefix_cache_tp: true", "prefix_cache_salt: qwen3-omni-30b-a3b-instruct"}
    # the chunk-budget counterfactuals add exactly the budget (the cache + chunk probe also flips the cache, like the plain cache config)
    assert changed(hw / "mstar/qwen3omni_3gpu_thinker_tp2.yaml", hw / "mstar/qwen3omni_3gpu_thinker_tp2_chunk4096.yaml") == {
        "max_batch_tokens:", "Thinker:", "thinker_mixed: 4096"}
    assert changed(hw / "mstar/qwen3omni_3gpu_thinker_tp2_prefix_cache.yaml", hw / "mstar/qwen3omni_3gpu_thinker_tp2_prefix_cache_chunk4096.yaml") == {
        "max_batch_tokens:", "Thinker:", "thinker_mixed: 4096"}
    # the ladder counterfactual adds only a Thinker prefill graph ladder on top of the chunk-4096 config
    assert changed(hw / "mstar/qwen3omni_3gpu_thinker_tp2_chunk4096.yaml", hw / "mstar/qwen3omni_3gpu_thinker_tp2_chunk4096_bucket1152.yaml") == {
        "model_kwargs:", "prefill_token_buckets: [128, 256, 512, 1024, 1152, 2048]"}
    assert changed(hw / "mstar/bagel_single_gpu_kv512.yaml", hw / "mstar/bagel_single_gpu_kv512_chunk8192.yaml") == {"max_batch_tokens:", "LLM: 8192"}
    talker = changed(hw / "vllm_omni/qwen3_omni_3gpu_thinker_tp2_async_c2wgraph.yaml", hw / "vllm_omni/qwen3_omni_3gpu_thinker_tp2_talker_greedy.yaml")
    assert talker == {"temperature: 0.9", "top_k: 50", "temperature: 0.0", "top_p: 1.0", "top_k: 1"}
    for name in ("qwen-fair-txt", "qwen-fair-a2t"):
        experiment = catalogue.get(name)
        # qwen-fair-txt has no M* `on` arm: at TP=2 the cache-on config crashes the second identical request (see its caveats)
        for arm in ("sglang-off", "sglang-on", "vllm-off", "vllm-on", "mstar-off", *(("mstar-on",) if name == "qwen-fair-a2t" else ())):
            assert "qwen.sampling=lookahead-eligible" in experiment.arm(arm).opts, (name, arm)


def test_rows_cycle_makes_the_precondition_phase_serve_the_measured_rows(monkeypatch):
    from exps.tasks.qwen_t2s import client as t2s

    selected = t2s.build_requests("verbatim", "en", 8, cycle=4)[1]
    ids = [r["fixture_id"] for r in selected]
    assert ids[:4] == ids[4:8] and len(set(ids[:4])) == 4
    sequential = [r["fixture_id"] for r in t2s.build_requests("verbatim", "en", 8)[1]]
    assert len(set(sequential)) == 8


def test_t2s_fit_and_ttft_parts_tools_on_synthetic_cells(tmp_path):
    import json
    from exps.tools import analyze_t2s_fit, analyze_ttft_parts

    def cell(root, fw, case, rows):
        d = root / "TASK" / fw / case / "r01"
        d.mkdir(parents=True)
        (d / "requests.jsonl").write_text("\n".join(json.dumps({"phase": "measured", "success": True, **r}) for r in rows))

    camp = tmp_path / "C"
    cell(camp, "fw", "verbatim-en", [{"e2e_ms": 100 + 70 * s, "ttfa_ms": 150.0, "audio_duration_s": s, "fixture_index": i}
                                     for i, s in enumerate((2.0, 3.0, 4.0, 5.0))] + [{"e2e_ms": 900.0, "ttfa_ms": 700.0, "audio_duration_s": 3.0, "fixture_index": 9}])
    [(_, _, _, s)] = analyze_t2s_fit.collect([camp])
    assert s["stalls"] == 1 and abs(s["fit_steady"][1] - 70) < 1e-6 and abs(s["fit_steady"][0] - 100) < 1e-6

    tc = tmp_path / "T"
    cell(tc, "fw", "txt-tiny-1tok", [{"e2e_ms": 50.0, "ttft_ms": 49.0, "input_tokens": 20}] * 3)
    cell(tc, "fw", "txt-1k-1tok", [{"e2e_ms": 150.0, "ttft_ms": 149.0, "input_tokens": 1020}] * 3)
    cell(tc, "fw", "a2t-en-1tok", [{"e2e_ms": 130.0, "ttft_ms": 129.0, "input_tokens": 520}] * 3)
    text = analyze_ttft_parts.render(analyze_ttft_parts.collect([tc]))
    # the tool reads first-token time (M* returns two tokens for a 1-token cap, so E2E is not equal work): 49 / 149 / 129 ms
    assert "| fw | 49 | 149 | 1,020 | 100.0 | 129 | 520 | 30 |" in text


def test_connector_configs_change_only_the_kv_edge_and_the_handoff_log_is_parsed(tmp_path):
    import yaml
    from exps.tools import analyze_kv_handoff

    hw = REPO / "exps/configs/hardware/rtx5880/vllm_omni"
    base = yaml.safe_load((hw / "bagel_two_stage_2gpu.yaml").read_text())
    for name, connector in (("mooncake_tcp", "MooncakeTransferEngineConnector"), ("nixl", "NixlConnector")):
        cfg = yaml.safe_load((hw / f"bagel_two_stage_2gpu_{name}.yaml").read_text())
        assert cfg["connectors"] == {"kv_connector": {"name": connector, "extra": cfg["connectors"]["kv_connector"]["extra"]}}
        assert cfg["stages"][0]["output_connectors"] == {"to_stage_1": "kv_connector"}
        assert cfg["stages"][1]["input_connectors"] == {"from_stage_0": "kv_connector"}
        for a, b in zip(base["stages"], cfg["stages"]):
            strip = lambda st: {k: v for k, v in st.items() if k not in ("input_connectors", "output_connectors")}
            assert strip(a) == strip(b)
        assert cfg.get("async_chunk") == base.get("async_chunk")
    assert cfg["connectors"]["kv_connector"]["extra"]["backends"] == ["UCX"]
    assert yaml.safe_load((hw / "bagel_two_stage_2gpu_mooncake_tcp.yaml").read_text())["connectors"]["kv_connector"]["extra"]["protocol"] == "tcp"
    exp = catalogue.get("bagel-connector")
    for arm in ("mooncake-tcp", "nixl"):
        assert any(e.startswith("vllm-omni:PYTHONPATH=") for e in exp.arm(arm).env)

    log = ("INFO [kv_transfer_manager.py:1162] KV cache serialized for chatcmpl-a in 467.5 ms\n"
           "INFO [kv_transfer_manager.py:1176] KV transfer OK: chatcmpl-a, 340457838 bytes across 1 key(s), 0.359s, 903.3 MB/s\n"
           "INFO KV cache serialized for chatcmpl-a__cfg_text in 463.8 ms\n"
           "INFO KV transfer OK: chatcmpl-a__cfg_text, 339998985 bytes across 1 key(s), 0.356s, 910.7 MB/s\n"
           "INFO KV cache serialized for chatcmpl-a__cfg_img in 1.1 ms\n"
           "INFO KV transfer OK: chatcmpl-a__cfg_img, 463230 bytes across 1 key(s), 0.001s, 753.6 MB/s\n")
    parsed = analyze_kv_handoff.parse_log(log)
    assert list(parsed) == ["chatcmpl-a"]
    assert abs(parsed["chatcmpl-a"]["serialize_ms"] - 932.4) < 0.01 and abs(parsed["chatcmpl-a"]["transfer_ms"] - 716.0) < 0.01


def test_connector_timeline_experiment_and_analyzer_on_synthetic_ms_log(tmp_path):
    import json
    from datetime import datetime, timezone

    from exps.tools import analyze_i2i_timeline

    exp = catalogue.get("bagel-connector-timeline")
    assert [a.name for a in exp.arms] == ["shm", "mooncake-tcp", "nixl", "nixl-cuda-ipc", "nixl-tcp", "mooncake-tcp-prefix"]
    from exps import hardware
    from exps.suite import runner

    for machine in ("rtx5880", "h200nvl"):
        hardware.set_current(machine)
        for arm in exp.arms:  # every arm carries the millisecond logging switch, so the arms differ only in the connector / transport
            env = runner.expand_all(arm.env, campaign="C")
            assert any(e.startswith("vllm-omni:VLLM_LOGGING_CONFIG_PATH=") and e.endswith("ms_timestamps.json") for e in env), (machine, arm.name)
            assert any(f"configs/hardware/{machine}/vllm_omni/logging/ms_timestamps.json" in e for e in env)
    hardware.set_current("h200nvl")
    assert "vllm-omni:UCX_TLS=cuda_copy,cuda_ipc,tcp" in exp.arm("nixl-cuda-ipc").env
    assert "vllm-omni:UCX_TLS=cuda_copy,tcp" in exp.arm("nixl-tcp").env
    assert not any(e.startswith("vllm-omni:UCX_TLS") for e in exp.arm("nixl").env)
    logging_cfg = json.loads((REPO / "exps/configs/hardware/rtx5880/vllm_omni/logging/ms_timestamps.json").read_text())
    assert "%(msecs)03d" in logging_cfg["formatters"]["ms"]["format"] and {"vllm", "vllm_omni"} <= set(logging_cfg["loggers"])

    start = datetime(2026, 11, 8, 9, 0, 0, tzinfo=timezone.utc)  # client start, UTC
    local = start.astimezone()
    stamp = lambda dt_ms: (datetime.fromtimestamp(start.timestamp() + dt_ms / 1000)).strftime("%m-%d %H:%M:%S.") + f"{int(round(dt_ms)) % 1000:03d}"
    # engine has it at +0.30 s, first branch serialised +1.50 s, last branch sent +1.60 s, DiT add request +1.70 s, client end +4.50 s
    log = (f"(APIServer pid=1) INFO {stamp(300)} [async_omni_engine.py:887] [AsyncOmniEngine] CFG expansion for req chatcmpl-a-1: 2 companions\n"
           f"(Stage pid=2) INFO {stamp(1500)} [kv_transfer_manager.py:1162] KV cache serialized for chatcmpl-a-1 in 0.8 ms\n"
           f"(Stage pid=2) INFO {stamp(1550)} [kv_transfer_manager.py:1176] KV transfer OK: chatcmpl-a-1, 340457725 bytes across 1 key(s), 0.013s, 24348.9 MB/s\n"
           f"(Stage pid=2) INFO {stamp(1580)} [kv_transfer_manager.py:1162] KV cache serialized for chatcmpl-a-1__cfg_text in 0.8 ms\n"
           f"(Stage pid=2) INFO {stamp(1600)} [kv_transfer_manager.py:1176] KV transfer OK: chatcmpl-a-1__cfg_text, 339999086 bytes across 1 key(s), 0.013s, 1.0 MB/s\n"
           f"(APIServer pid=1) INFO {stamp(1700)} [stage_diffusion_client.py:347] [StageDiffusionClient] stage-1 [rep-0] add request: chatcmpl-a-1\n"
           "(APIServer pid=1) INFO 11-08 18:03:08 [x.py:1] a line with a one-second stamp only: ignored\n")
    cell = tmp_path / "CAMP_SHM_01_B2B" / "BAGEL" / "vllm-omni-two-stage" / "i2i-512-steps5" / "r01"
    cell.mkdir(parents=True)
    (cell / "server.log").write_text(log)
    (cell / "requests.jsonl").write_text(json.dumps({"phase": "measured", "success": True, "e2e_ms": 4500.0, "started_at_utc": start.isoformat()}) + "\n")
    rows = analyze_i2i_timeline.collect([tmp_path / "CAMP_SHM_01_B2B"])
    assert len(rows) == 1 and rows[0]["n"] == 1
    r = rows[0]
    assert (round(r["pre"]), round(r["stage0"]), round(r["send"]), round(r["gap"]), round(r["dit"])) == (300, 1200, 100, 100, 2800)
    assert round(r["pre"] + r["stage0"] + r["send"] + r["gap"] + r["dit"]) == 4500 == round(r["e2e"])
    assert "| CAMP_SHM_01_B2B | vllm-omni-two-stage | i2i-512-steps5 | 1 | 4,500 | 300 | 1,200 | 100 | 100 | 2,800 |" in analyze_i2i_timeline.render(rows)
    assert local.year == 2026


def test_log_expectations_accept_regex_and_sglang_cache_state_is_read_from_cached_tokens():
    from exps.suite import analysis, catalogue

    off = "Init Unified Radix Cache. Components: ...\nPrefill batch, #new-token: 1072, #cached-token: 0, token usage: 0.03\n"
    on = off + "Prefill batch, #new-token: 1, #cached-token: 1071, token usage: 0.03\n"
    # the radix object is built even when disabled, so the plain line is no evidence; the counter is
    assert analysis._in_log("Init Unified Radix Cache", off) and not analysis._in_log(catalogue.SGLANG_PREFIX_HIT, off)
    assert analysis._in_log(catalogue.SGLANG_PREFIX_HIT, on) and analysis._in_log("plain substring", "a plain substring here")


def test_one_token_probe_accepts_mstars_two_tokens_but_longer_caps_stay_exact():
    import pytest
    from exps.core.codecs import ContractViolation
    from exps.tasks.qwen_txt import client

    client.make_validator(1)(None, {"output_tokens": 2}, 0)
    client.make_validator(1)(None, {"output_tokens": 1}, 0)
    with pytest.raises(ContractViolation):
        client.make_validator(1)(None, {"output_tokens": 3}, 0)
    with pytest.raises(ContractViolation):
        client.make_validator(256)(None, {"output_tokens": 257}, 0)


def test_nsys_analyser_reports_a_capture_without_kernels_instead_of_a_traceback(tmp_path):
    """H200, 2026-11-13: the SGLang-core Pi0.5 capture held the server's and its worker's runtime calls and no kernel table, and the analyser died
    with `no such table: CUPTI_ACTIVITY_KIND_KERNEL` hours after the run. It must say what the file holds, keep the other reports, and fail."""
    for name, kernels in (("broken", 0), ("good", 3)):
        con = sqlite3.connect(tmp_path / f"{name}.sqlite")
        con.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (globalTid integer)")
        con.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?)", [((4242 << 24) | 7,)] * 564)
        if kernels:
            con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (deviceId integer, globalPid integer, start integer, end integer, shortName integer)")
            con.execute("create table StringIds (id integer, value text)")
            con.execute("insert into StringIds values (1, 'k')")
            con.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (0, ?, ?, ?, 1)", [(4242 << 24, i * 1000, i * 1000 + 500) for i in range(kernels)])
        con.commit()
        con.close()
    assert analyze_nsys_burst_gaps.has_kernels(tmp_path / "good.sqlite") and not analyze_nsys_burst_gaps.has_kernels(tmp_path / "broken.sqlite")
    text = analyze_nsys_burst_gaps.report(tmp_path, gap_ms=10.0, min_launches=1, last=4, top=0)
    assert "### broken: NO CUDA KERNEL RECORDS" in text and "pid 4242: 564 runtime calls" in text and "failed capture" in text
    assert "### good" in text                                   # one broken report does not hide the others
    run = subprocess.run([sys.executable, str(REPO / "exps/tools/analyze_nsys_burst_gaps.py"), str(tmp_path), "--min-launches", "1", "--top", "0"],
                         capture_output=True, text=True)
    assert run.returncode != 0 and "no CUDA kernel records in: broken.sqlite" in run.stderr and "Traceback" not in run.stderr
    # a directory whose only capture is broken is a failed run: it must not leave a results file that reads like a result
    only_broken = tmp_path / "only_broken"
    only_broken.mkdir()
    (tmp_path / "broken.sqlite").replace(only_broken / "broken.sqlite")
    out = tmp_path / "result.md"
    run = subprocess.run([sys.executable, str(REPO / "exps/tools/analyze_nsys_burst_gaps.py"), str(only_broken), "--top", "0", "--markdown", str(out)],
                         capture_output=True, text=True)
    assert run.returncode != 0 and not out.exists()
    run = subprocess.run([sys.executable, str(REPO / "exps/tools/analyze_nsys_burst_gaps.py"), str(tmp_path), "--min-launches", "1", "--top", "0",
                          "--markdown", str(out)], capture_output=True, text=True)
    assert run.returncode == 0 and out.exists() and "NO CUDA KERNEL RECORDS" not in out.read_text()


def test_per_arm_nsys_options_reach_the_command_and_existing_arms_keep_the_defaults():
    from exps.suite import runner

    diag = {r.arm.name: " ".join(r.argv) for r in runner.arm_runs(catalogue.get("pi05-nsys-sglang-diag"), protocol="back-to-back")}
    assert "--nsys-graph-trace graph" in diag["graph-trace-graph"] and "--nsys-flush-ms 1000" in diag["graph-trace-graph"]
    assert "--nsys-flush-ms 100 " in diag["flush-100ms"] + " " and "--nsys-graph-trace" not in diag["flush-100ms"]
    assert "pi05_pipeline_graph_off.json" in diag["no-cuda-graph"]
    from exps import hardware

    hardware.set_current("rtx5880")       # the sglang-core arm of pi05-nsys keeps the 1,000 ms default here; H200 flushes at 100 ms (test_hardware_variants)
    stock = " ".join(runner.arm_runs(catalogue.get("pi05-nsys"), protocol="back-to-back")[1].argv)
    assert "--nsys-flush-ms 1000" in stock and "--nsys-graph-trace" not in stock
