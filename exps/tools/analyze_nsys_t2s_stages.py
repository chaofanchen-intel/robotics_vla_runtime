#!/usr/bin/env python3
"""Per-stage GPU activity timeline inside one Qwen3-Omni T2S request (nsys, C=1).

Question: do Thinker, Talker and Code2Wav run one after another, or do they overlap in time?
A deck claim ("M*: 3 partitions overlap; vLLM-Omni: strictly serial; SGLang-Omni: Thinker -> Talker
serial, Talker || Code2Wav") was derived from config and source. This tool measures it.

Attribution of a kernel to a stage
  * vLLM-Omni / SGLang-Omni: one OS process per stage (SGLang: Talker and Code2Wav share a process).
    The process -> stage map is parsed from the cell's server.log.
  * M*: one worker process, one engine thread. Needs `mstar serve --enable-nvtx` (the engine wraps
    every forward in `engine.<node>.<walk>.bs<N>`); a kernel is attributed to the innermost such
    range around the CUDA runtime call that launched it. Without NVTX, M* shows as one group.

Per measured request (client UTC window against the nsys session start) and per group it reports the
first kernel start, last kernel end (relative to request start), GPU busy time, and, for every pair of
groups, the time during which both had a kernel executing ("co-busy") and the overlap of their
[first, last] spans. Medians over the measured requests.

    envs/mstar/bin/python exps/tools/analyze_nsys_t2s_stages.py            # QWEN_T2S_PROFILE_01, 3 engines
    envs/mstar/bin/python exps/tools/analyze_nsys_t2s_stages.py \\
        --cells QWEN_T2S_STAGE_01/mstar QWEN_T2S_STAGE_02/vllm-omni QWEN_T2S_STAGE_03/vllm-omni
"""
from __future__ import annotations

import argparse
import bisect
import json
import re
import sqlite3
import statistics
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths

NS = 1e6
NVTX_NODE = re.compile(r"^engine\.([^.]+)\.([^.]+)\.bs\d+$")
VLLM_STAGE = {"0": "thinker", "1": "talker", "2": "code2wav"}


def pid_of(global_id: int) -> int:
    return (global_id >> 24) & 0xFFFFFF


def stage_map_from_log(text: str) -> dict[int, str]:
    """OS pid -> stage label, from the framework's own start-up log lines."""
    out: dict[int, str] = {}
    for stage, pid in re.findall(r"StageEngineCoreProc_stage(\d+)_replica\d+ pid=(\d+)", text):
        out[int(pid)] = VLLM_STAGE.get(stage, f"stage{stage}")
    for name, pids in re.findall(r"StageGroup (\w+): spawned \d+ process\(es\) \(pids=\[([\d, ]+)\]\)", text):
        for pid in pids.split(","):
            out[int(pid)] = name
    return out


def union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(intervals):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def intersect_ns(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    """Total length where both interval lists (already merged) are active."""
    i = j = total = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            total += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def kernels_in_window(con: sqlite3.Connection, t0: float, t1: float):
    q = ("select globalPid, correlationId, streamId, start, end from CUPTI_ACTIVITY_KIND_KERNEL "
         "where start between ? and ? order by start")
    return con.execute(q, (t0, t1)).fetchall()


def nvtx_ranges(con: sqlite3.Connection) -> dict[int, tuple[list[int], list[tuple[int, str]]]]:
    """Per OS thread: sorted range starts and (end, node) for `engine.<node>.<walk>.bs<N>` ranges."""
    tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
    if "NVTX_EVENTS" not in tables:
        return {}
    per: dict[int, list[tuple[int, int, str]]] = defaultdict(list)
    for start, end, text, gtid in con.execute(
        "select start, end, text, globalTid from NVTX_EVENTS where text like 'engine.%' and end is not null"
    ):
        m = NVTX_NODE.match(text or "")
        if m:
            per[gtid].append((start, end, m.group(1)))
    out = {}
    for gtid, rows in per.items():
        rows.sort()
        out[gtid] = ([r[0] for r in rows], [(r[1], r[2]) for r in rows])
    return out


def launch_threads(con: sqlite3.Connection, correlation_ids: list[int]) -> dict[int, tuple[int, int]]:
    """correlationId -> (globalTid, runtime-call start) for the launches that produced the kernels."""
    out: dict[int, tuple[int, int]] = {}
    for i in range(0, len(correlation_ids), 500):
        chunk = correlation_ids[i:i + 500]
        marks = ",".join("?" * len(chunk))
        for cid, gtid, start in con.execute(
            f"select correlationId, globalTid, start from CUPTI_ACTIVITY_KIND_RUNTIME where correlationId in ({marks})",
            chunk,
        ):
            out[cid] = (gtid, start)
    return out


def group_of_kernel(pid: int, cid: int, pid_stage: dict[int, str], nvtx, launches) -> str:
    stage = pid_stage.get(pid)
    if stage:
        return stage
    if nvtx and cid in launches:
        gtid, t = launches[cid]
        starts, ends = nvtx.get(gtid, ([], []))
        k = bisect.bisect_right(starts, t) - 1
        if k >= 0 and ends[k][0] >= t:
            return f"node:{ends[k][1]}"
        return "outside-node"
    return f"pid{pid}"


def per_request(db: Path, requests_file: Path, log_text: str, pad_s: float = 0.05) -> list[dict]:
    con = sqlite3.connect(db)
    origin = con.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    pid_stage = stage_map_from_log(log_text)
    nvtx = nvtx_ranges(con)
    requests = [json.loads(line) for line in requests_file.read_text().splitlines()]
    requests = [r for r in requests if r["phase"] == "measured" and r.get("success")]
    out = []
    for req in requests:
        t0 = datetime.fromisoformat(req["started_at_utc"]).timestamp() * 1e9 - origin
        t1 = t0 + req["e2e_ms"] * 1e6
        ks = kernels_in_window(con, t0 - pad_s * 1e9, t1 + pad_s * 1e9)
        launches = launch_threads(con, [k[1] for k in ks]) if nvtx else {}
        groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for pid_gid, cid, _stream, s, e in ks:
            groups[group_of_kernel(pid_of(pid_gid), cid, pid_stage, nvtx, launches)].append((s, e))
        merged = {g: union(v) for g, v in groups.items()}
        info = {}
        for g, iv in merged.items():
            info[g] = {
                "first_ms": (iv[0][0] - t0) / NS, "last_ms": (iv[-1][1] - t0) / NS,
                "busy_ms": sum(e - s for s, e in iv) / NS, "launches": len(groups[g]),
            }
        names = sorted(info)
        pairs = {}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                both = intersect_ns(merged[a], merged[b]) / NS
                span = max(0.0, min(info[a]["last_ms"], info[b]["last_ms"]) - max(info[a]["first_ms"], info[b]["first_ms"]))
                pairs[f"{a}|{b}"] = {"co_busy_ms": both, "span_overlap_ms": span}
        out.append({"e2e_ms": req["e2e_ms"], "audio_s": req.get("audio_duration_s"),
                    "ttfa_ms": req.get("ttfa_ms"), "groups": info, "pairs": pairs,
                    "silent_stages": sorted(set(pid_stage.values()) - {g for g in info})})
    return out


def summarize(rows: list[dict]) -> dict:
    med = statistics.median
    groups = sorted({g for r in rows for g in r["groups"]})
    pairs = sorted({p for r in rows for p in r["pairs"]})
    return {
        "n": len(rows),
        "e2e_ms": med(r["e2e_ms"] for r in rows),
        "audio_s": med(r["audio_s"] for r in rows if r["audio_s"]) if any(r["audio_s"] for r in rows) else None,
        "groups": {g: {k: med(r["groups"][g][k] for r in rows if g in r["groups"])
                       for k in ("first_ms", "last_ms", "busy_ms", "launches")} for g in groups},
        "pairs": {p: {k: med(r["pairs"][p][k] for r in rows if p in r["pairs"])
                      for k in ("co_busy_ms", "span_overlap_ms")} for p in pairs},
        "silent_stages": sorted(set.intersection(*(set(r["silent_stages"]) for r in rows))),
    }


def markdown(results: dict[str, dict]) -> str:
    lines = ["# T2S per-stage GPU timeline (nsys, C=1; medians over measured requests)", ""]
    for name, s in results.items():
        if "error" in s:
            lines += [f"## {name}", "", s["error"], ""]
            continue
        lines += [f"## {name}", "", f"requests: {s['n']}, client E2E (nsys-run) {s['e2e_ms']:.0f} ms"
                  + (f", audio {s['audio_s']:.2f} s" if s["audio_s"] else ""), "",
                  "| group | first kernel ms | last kernel ms | GPU busy ms | launches |", "|---|---:|---:|---:|---:|"]
        for g, v in sorted(s["groups"].items(), key=lambda kv: kv[1]["first_ms"]):
            lines.append(f"| {g} | {v['first_ms']:.0f} | {v['last_ms']:.0f} | {v['busy_ms']:.0f} | {v['launches']:.0f} |")
        if s["silent_stages"]:
            lines += ["", f"No kernels recorded for: {', '.join(s['silent_stages'])} (process traced but silent; "
                          "cannot say whether it ran concurrently)."]
        if s["pairs"]:
            lines += ["", "| pair | both executing ms | [first,last] span overlap ms |", "|---|---:|---:|"]
            for p, v in s["pairs"].items():
                lines.append(f"| {p.replace('|', ' / ')} | {v['co_busy_ms']:.1f} | {v['span_overlap_ms']:.0f} |")
        lines.append("")
    lines.append("Limits: nsys inflates host time (BAGEL +6-18%, A2T +34%, T2S x1.6-2.2 and most for M*, the engine with the most launches; gaps and overlap fractions are structural, not absolute), one start per cell; \"both executing\" "
                 "needs two stages to have a kernel on the GPU at the same instant (processes time-slice the GPU "
                 "without MPS, one-stream engines cannot), so span overlap is the structural indicator.")
    return "\n".join(lines) + "\n"


def find_cell(spec: str, case: str) -> tuple[Path, Path, Path]:
    """`CAMPAIGN/framework` -> (nsys sqlite, requests.jsonl, server.log) of that campaign's first start."""
    campaign, framework = spec.split("/", 1)
    root = paths.campaign_root(campaign)
    cell = next(iter(sorted((root / "QWEN_T2S" / framework / case).glob("r*"))), None)
    if cell is None:
        raise FileNotFoundError(f"no cell directory {root / 'QWEN_T2S' / framework / case}")
    db = root / "nsys" / f"{framework}-{case}-{cell.name}.sqlite"
    rep = db.with_suffix(".nsys-rep")
    if (not db.exists() or db.stat().st_size == 0) and rep.exists():
        import subprocess
        subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o", str(db), str(rep)],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not db.exists():
        raise FileNotFoundError(f"no nsys report {rep}")
    return db, cell / "requests.jsonl", cell / "server.log"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", nargs="+",
                    default=[f"QWEN_T2S_PROFILE_01/{fw}" for fw in ("mstar", "sglang-omni", "vllm-omni")],
                    help="CAMPAIGN/framework, one nsys cell each (a variant lives in its own campaign).")
    ap.add_argument("--case", default="verbatim-en")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    results = {}
    for name in args.cells:
        try:
            db, requests_file, log_file = find_cell(name, args.case)
            rows = per_request(db, requests_file, log_file.read_text(errors="replace") if log_file.exists() else "")
            ok = sum(1 for line in requests_file.read_text().splitlines()
                     if line.strip() and json.loads(line).get("phase") == "measured" and json.loads(line).get("success"))
            reason = ("every measured request failed (see requests.jsonl), so there is no window to attribute kernels to"
                      if not ok else "no kernels inside any measured-request window")
            results[name] = summarize(rows) if rows else {"error": reason}
        except FileNotFoundError as exc:
            results[name] = {"error": str(exc)}
    text = markdown(results)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
