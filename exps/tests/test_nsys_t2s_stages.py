"""CPU-only checks for the per-stage T2S timeline analysis (synthetic nsys sqlite)."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from exps.tools import analyze_nsys_t2s_stages as stages

ORIGIN_NS = 1_700_000_000_000_000_000


def _gid(pid: int, tid: int = 0) -> int:
    return (pid << 24) | tid


def _make_db(path, kernels, runtime=(), nvtx=()):
    con = sqlite3.connect(path)
    con.execute("create table TARGET_INFO_SESSION_START_TIME (utcEpochNs integer)")
    con.execute("insert into TARGET_INFO_SESSION_START_TIME values (?)", (ORIGIN_NS,))
    con.execute("create table CUPTI_ACTIVITY_KIND_KERNEL (globalPid integer, correlationId integer, streamId integer,"
                " start integer, end integer)")
    con.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (?,?,?,?,?)", kernels)
    con.execute("create table CUPTI_ACTIVITY_KIND_RUNTIME (correlationId integer, globalTid integer, start integer)")
    con.executemany("insert into CUPTI_ACTIVITY_KIND_RUNTIME values (?,?,?)", runtime)
    con.execute("create table NVTX_EVENTS (start integer, end integer, text text, globalTid integer)")
    con.executemany("insert into NVTX_EVENTS values (?,?,?,?)", nvtx)
    con.commit()
    con.close()


def _requests(path, e2e_ms=100.0):
    started = datetime.fromtimestamp(ORIGIN_NS / 1e9 + 1.0, tz=timezone.utc).isoformat()
    path.write_text(json.dumps({"phase": "measured", "success": True, "started_at_utc": started,
                                "e2e_ms": e2e_ms, "audio_duration_s": 2.0}) + "\n")


def test_stage_map_from_vllm_and_sglang_logs():
    text = ("(StageEngineCoreProc_stage0_replica0 pid=11) x\n(StageEngineCoreProc_stage1_replica0 pid=22) y\n"
            "StageGroup talker_ar: spawned 1 process(es) (pids=[33])\n")
    assert stages.stage_map_from_log(text) == {11: "thinker", 22: "talker", 33: "talker_ar"}


def test_intersect_ns_counts_only_simultaneous_time():
    assert stages.intersect_ns([(0, 10), (20, 30)], [(5, 25)]) == 10
    assert stages.intersect_ns([(0, 10)], [(10, 20)]) == 0


def test_serial_processes_have_zero_overlap(tmp_path):
    t = 1_000_000_000  # request starts 1 s after the session origin
    ms = 1_000_000
    kernels = [(_gid(1), 1, 7, t + 10 * ms, t + 40 * ms), (_gid(2), 2, 7, t + 50 * ms, t + 90 * ms)]
    _make_db(tmp_path / "a.sqlite", kernels)
    _requests(tmp_path / "r.jsonl")
    rows = stages.per_request(tmp_path / "a.sqlite", tmp_path / "r.jsonl",
                              "(StageEngineCoreProc_stage0_replica0 pid=1)\n(StageEngineCoreProc_stage1_replica0 pid=2)\n")
    pair = rows[0]["pairs"]["talker|thinker"]
    assert pair["co_busy_ms"] == 0 and pair["span_overlap_ms"] == 0
    assert rows[0]["groups"]["thinker"]["busy_ms"] == 30


def test_nvtx_nodes_label_kernels_in_one_process(tmp_path):
    t, ms, tid = 1_000_000_000, 1_000_000, _gid(5, 9)
    kernels = [(_gid(5), 1, 7, t + 10 * ms, t + 20 * ms), (_gid(5), 2, 7, t + 15 * ms, t + 30 * ms),
               (_gid(5), 3, 7, t + 60 * ms, t + 70 * ms)]
    runtime = [(1, tid, t + 9 * ms), (2, tid, t + 14 * ms), (3, tid, t + 59 * ms)]
    nvtx = [(t + 8 * ms, t + 12 * ms, "engine.thinker.decode.bs1", tid),
            (t + 13 * ms, t + 17 * ms, "engine.talker.decode.bs1", tid),
            (t + 58 * ms, t + 62 * ms, "engine.thinker.decode.bs1", tid),
            (t + 1 * ms, t + 2 * ms, "engine.prepare_inputs.bs1", tid)]
    _make_db(tmp_path / "m.sqlite", kernels, runtime, nvtx)
    _requests(tmp_path / "r.jsonl")
    rows = stages.per_request(tmp_path / "m.sqlite", tmp_path / "r.jsonl", "")
    groups = rows[0]["groups"]
    assert set(groups) == {"node:thinker", "node:talker"}
    assert groups["node:thinker"]["launches"] == 2
    assert rows[0]["pairs"]["node:talker|node:thinker"]["co_busy_ms"] == 5  # 15..20 ms overlap


def test_mstar_nvtx_gaps_and_gpu_busy(tmp_path):
    from exps.tools import analyze_nsys_mstar_nvtx as nvtx_tool

    t, ms, tid = 1_000_000_000, 1_000_000, _gid(5, 9)
    # two LLM forwards 10 ms long with a 4 ms gap between them, one vae forward after a 6 ms gap
    nvtx = [(t + 10 * ms, t + 20 * ms, "engine.LLM.image_gen.bs1", tid),
            (t + 24 * ms, t + 34 * ms, "engine.LLM.image_gen.bs1", tid),
            (t + 40 * ms, t + 45 * ms, "engine.vae_decoder.decode.bs1", tid)]
    kernels = [(_gid(5), 1, 7, t + 12 * ms, t + 18 * ms), (_gid(5), 2, 7, t + 41 * ms, t + 44 * ms)]
    runtime = [(1, tid, t + 11 * ms), (2, tid, t + 41 * ms)]
    _make_db(tmp_path / "m.sqlite", kernels, runtime, nvtx)
    _requests(tmp_path / "r.jsonl")
    rows = nvtx_tool.per_request(tmp_path / "m.sqlite", tmp_path / "r.jsonl")
    assert rows[0]["gaps"]["n"] == 2 and rows[0]["gaps"]["total_ms"] == 10
    assert rows[0]["nodes"]["LLM.image_gen"]["calls"] == 2
    assert rows[0]["nodes"]["LLM.image_gen"]["gpu_busy_ms"] == 6
    assert rows[0]["nodes"]["vae_decoder.decode"]["gpu_busy_ms"] == 3
    assert "No `engine." in nvtx_tool.markdown("x", [])
    assert rows[0]["nodes"]["LLM.image_gen"]["batch_sizes"] == {1: 2}
    assert "bs1: 2" in nvtx_tool.markdown("x", rows)


def test_mstar_nvtx_counts_kernels_matching_a_name_per_node(tmp_path):
    from exps.tools import analyze_nsys_mstar_nvtx as nvtx_tool

    t, ms, tid = 1_000_000_000, 1_000_000, _gid(5, 9)
    nvtx = [(t + 10 * ms, t + 20 * ms, "engine.thinker.prefill.bs1", tid),
            (t + 30 * ms, t + 40 * ms, "engine.thinker.decode.bs1", tid)]
    kernels = [(_gid(5), cid, 7, t + (11 + cid) * ms, t + (11 + cid) * ms + 1000) for cid in range(1, 4)]
    kernels.append((_gid(5), 4, 7, t + 32 * ms, t + 33 * ms))
    runtime = [(cid, tid, t + (11 + cid) * ms) for cid in range(1, 4)] + [(4, tid, t + 32 * ms)]
    _make_db(tmp_path / "m.sqlite", kernels, runtime, nvtx)
    con = sqlite3.connect(tmp_path / "m.sqlite")
    con.execute("alter table CUPTI_ACTIVITY_KIND_KERNEL add column shortName integer")
    con.execute("create table StringIds (id integer, value text)")
    con.executemany("insert into StringIds values (?,?)", [(1, "moe_sum_reduce_kernel"), (2, "nvjet_gemm")])
    con.executemany("update CUPTI_ACTIVITY_KIND_KERNEL set shortName=? where correlationId=?",
                    [(1, 1), (1, 2), (2, 3), (1, 4)])
    con.commit()
    con.close()
    _requests(tmp_path / "r.jsonl")
    rows = nvtx_tool.per_request(tmp_path / "m.sqlite", tmp_path / "r.jsonl", match="moe_sum_reduce")
    assert rows[0]["nodes"]["thinker.prefill"]["matched"] == 2
    assert rows[0]["nodes"]["thinker.decode"]["matched"] == 1
    assert "`moe_sum_reduce` launches" in nvtx_tool.markdown("x", rows, "moe_sum_reduce")
    # without --match nothing needs StringIds
    assert nvtx_tool.per_request(tmp_path / "m.sqlite", tmp_path / "r.jsonl")[0]["nodes"]["thinker.prefill"]["matched"] == 0


def test_mstar_nvtx_graph_replay_is_one_launch_for_many_kernels(tmp_path):
    """A CUDA-graph replay is a single correlationId carrying many kernels; round-5 traces crashed the analyzer
    (placeholders built for the kernel count, bound once per distinct id) and would have under-counted them."""
    from exps.tools import analyze_nsys_mstar_nvtx as nvtx_tool

    t, ms, tid = 1_000_000_000, 1_000_000, _gid(5, 9)
    nvtx = [(t + 10 * ms, t + 20 * ms, "engine.thinker.decode.bs1", tid)]
    kernels = [(_gid(5), 1, 7, t + (11 + i) * ms, t + (11 + i) * ms + 1000) for i in range(6)]  # one id, six kernels
    kernels.append((_gid(5), 2, 7, t + 18 * ms, t + 19 * ms))
    runtime = [(1, tid, t + 11 * ms), (2, tid, t + 18 * ms)]
    _make_db(tmp_path / "m.sqlite", kernels, runtime, nvtx)
    con = sqlite3.connect(tmp_path / "m.sqlite")
    con.execute("alter table CUPTI_ACTIVITY_KIND_KERNEL add column shortName integer")
    con.execute("create table StringIds (id integer, value text)")
    con.execute("insert into StringIds values (1, 'gemm')")
    con.execute("update CUPTI_ACTIVITY_KIND_KERNEL set shortName=1")
    con.commit()
    con.close()
    _requests(tmp_path / "r.jsonl")
    rows = nvtx_tool.per_request(tmp_path / "m.sqlite", tmp_path / "r.jsonl", match="gemm")
    node = rows[0]["nodes"]["thinker.decode"]
    assert node["matched"] == 7                                    # all six graph kernels plus the plain one
    assert node["gpu_busy_ms"] == pytest.approx(1.0 + 6 * 0.001)   # merged interval union, not one kernel per id
