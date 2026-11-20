#!/usr/bin/env python3
"""GPU activity bursts and the idle gaps between them in nsys reports (CPU only; sqlite exports).

Why: two engines can decode at the same per-token cost and still differ in time to first token. nsys cannot see the HTTP
arrival of a request, but with back-to-back requests (no cool-down) the idle time between the last kernel of request N
and the first kernel of request N+1 is exactly what a request spends before the GPU starts: client round trip, front-end
preprocessing, scheduling, stage hand-off. This tool splits each report's kernel stream at idle gaps longer than
`--gap-ms` (default 40 ms; decode steps are 17 ms apart in the cells it was written for), keeps bursts with at least
`--min-launches` launches, and prints the last `--last` ones: gap before, span, GPU busy (union of the kernel intervals) and
launches. A burst is not a request (a request can have several: ViT + prefill, then decode), so read the table as a
timeline.

    envs/mstar/bin/python exps/tools/analyze_nsys_burst_gaps.py exps/runs/BAGEL_I2T_NSYS_NSYS_01_B2B/nsys --markdown out.md

nsys inflates host time (most for M*): the gaps are structure, not benchmark numbers.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
import subprocess
from collections import defaultdict
from pathlib import Path

NS = 1e6


def union_ms(intervals: list[tuple[int, int]]) -> float:
    total, cur_s, cur_e = 0, None, None
    for start, end in sorted(intervals):
        if cur_s is None or start > cur_e:
            if cur_s is not None:
                total += cur_e - cur_s
            cur_s, cur_e = start, end
        else:
            cur_e = max(cur_e, end)
    return (total + (cur_e - cur_s if cur_s is not None else 0)) / NS


Stream = tuple[int, int | None] | None  # (deviceId, OS pid) of one process's kernel stream; None: the table has no deviceId column


def streams(db: Path) -> list[Stream]:
    """Kernel streams in the report, one per (deviceId, process). Grouping by deviceId alone is wrong for a multi-process server: every
    stage process is started with its own CUDA_VISIBLE_DEVICES, so the Thinker (card 0) and the DiT (card 1) both report deviceId 0 and
    their bursts, which overlap in time, would be merged into one table."""
    con = sqlite3.connect(db)
    try:
        found = [(row[0], row[1]) for row in con.execute(
            "select deviceId, (globalPid >> 24) & 16777215 from CUPTI_ACTIVITY_KIND_KERNEL group by deviceId, globalPid order by min(start)")]
    except sqlite3.OperationalError:
        try:
            found = [(row[0], None) for row in con.execute("select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL order by deviceId")]
        except sqlite3.OperationalError:
            return [None]
    return found or [None]


def has_kernels(db: Path) -> bool:
    """False when the capture holds no CUDA kernel records at all (the table is only created when there is one)."""
    con = sqlite3.connect(db)
    try:
        return con.execute("select 1 from CUPTI_ACTIVITY_KIND_KERNEL limit 1").fetchone() is not None
    except sqlite3.OperationalError:
        return False
    finally:
        con.close()


def no_kernels_note(db: Path) -> str:
    """Say what a kernel-less capture does hold, so it reads as a failed capture and not as a bug in this analysis.

    H200, 2026-11-13: the SGLang-core Pi0.5 capture held 1,128 runtime calls (564 from the server, 564 from its worker) and no kernel, against
    195,186 runtime calls and 68,613 kernels in the M* capture of the same request stream: the worker's GPU activity was never written."""
    con = sqlite3.connect(db)
    try:
        rows = con.execute("select (globalTid >> 24) & 16777215, count(*) from CUPTI_ACTIVITY_KIND_RUNTIME group by 1").fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        con.close()
    held = ", ".join(f"pid {pid}: {count:,} runtime calls" for pid, count in rows) or "no CUDA runtime calls either"
    return (f"The capture holds no CUDA kernel records, so there is no GPU timeline to read ({held}). The profiler lost, or never started, the "
            "GPU-activity buffers of the process that did the work (a worker that exits without flushing is the usual cause), so this is a failed "
            "capture, not an analysis problem. A failed cell needs a new campaign id: rerun the arm with a new `--run-id`.")


def _where(stream: Stream, alias: str = "") -> str:
    if stream is None:
        return ""
    device, pid = stream
    cond = f"{alias}deviceId = {int(device)}"
    if pid is not None:
        cond += f" and (({alias}globalPid >> 24) & 16777215) = {int(pid)}"
    return f" where {cond}"


def bursts(db: Path, gap_ms: float, min_launches: int, device: Stream = None) -> list[dict]:
    con = sqlite3.connect(db)
    where = _where(device)
    rows = con.execute(f"select start, end from CUPTI_ACTIVITY_KIND_KERNEL{where} order by start").fetchall()
    out, cur, last_end = [], [], None
    for start, end in rows:
        if last_end is not None and (start - last_end) / NS > gap_ms:
            out.append(cur)
            cur = []
        cur.append((start, end))
        last_end = end if last_end is None else max(last_end, end)
    if cur:
        out.append(cur)
    kept, prev_end = [], None
    for burst in out:
        end = max(e for _, e in burst)
        if len(burst) >= min_launches:
            kept.append({
                "gap_ms": None if prev_end is None else (burst[0][0] - prev_end) / NS,
                "span_ms": (end - burst[0][0]) / NS,
                "busy_ms": union_ms(burst),
                "launches": len(burst),
            })
            prev_end = end
    return kept


def _pid(global_pid_col: str = "globalPid") -> str:
    return f"(({global_pid_col} >> 24) & 16777215)"


def handoff(db: Path, gap_ms: float = 40.0, dit_min_launches: int = 5000, thinker_kernel: str = "kernel_unified_attention") -> dict | None:
    """Two-process (Thinker + DiT) request timeline from one nsys report: per DiT burst, the Thinker's kernels in the window since the previous
    DiT burst ended, then the hand-off (Thinker's last kernel -> DiT's first kernel), split by the KV copies when the report has them.

    The Thinker is the process that runs `thinker_kernel` (vLLM's paged-attention kernel, absent from the DiT); the DiT is the other one.
    Returns None for a report with fewer than two processes. `thinker_trace_ends_ms` / `dit_trace_ends_ms` (relative to the first DiT burst)
    expose a truncated trace: nsys has been seen to lose the tail of one process's kernels, and a row whose Thinker window is empty or cut
    must not be read as 'no Thinker work'."""
    found = streams(db)
    if len(found) < 2 or any(s is None or s[1] is None for s in found):
        return None
    con = sqlite3.connect(db)
    tk_names = {row[0] for row in con.execute("select id from StringIds where value like ?", (f"%{thinker_kernel}%",))}
    if not tk_names:
        return None
    marks = ",".join(str(i) for i in tk_names)
    thinker_pids = {row[0] for row in con.execute(
        f"select distinct {_pid()} from CUPTI_ACTIVITY_KIND_KERNEL where coalesce(demangledName, shortName) in ({marks})")}
    if len(thinker_pids) != 1:
        return None
    th = thinker_pids.pop()
    dit = next(pid for _, pid in found if pid != th)
    data = {pid: con.execute(f"select start, end from CUPTI_ACTIVITY_KIND_KERNEL where {_pid()} = {pid} order by start").fetchall() for pid in (th, dit)}

    def split(kernels):
        out, cur, last = [], [], None
        for start, end in kernels:
            if last is not None and (start - last) / NS > gap_ms:
                out.append(cur)
                cur = []
            cur.append((start, end))
            last = end if last is None else max(last, end)
        return out + [cur] if cur else out

    dit_bursts = [(b[0][0], max(e for _, e in b), len(b), union_ms(b)) for b in split(data[dit]) if len(b) >= dit_min_launches]
    if not dit_bursts:
        return None
    try:
        d2h = con.execute(f"select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where {_pid()} = {th} and copyKind = 2 and bytes > 100000000 order by start").fetchall()
        h2d = con.execute(f"select start from CUPTI_ACTIVITY_KIND_MEMCPY where {_pid()} = {dit} and copyKind = 1 and bytes > 1000000 order by start").fetchall()
    except sqlite3.OperationalError:
        d2h, h2d = [], []
    t0 = dit_bursts[0][0]
    rows = []
    for i, (d_start, d_end, launches, busy) in enumerate(dit_bursts):
        # the first burst has no previous one: look back 10 s only (the kernels before that are warm-up of earlier requests)
        window_start = dit_bursts[i - 1][1] if i else d_start - 10_000_000_000
        inside = [(s, e) for s, e in data[th] if window_start < s < d_start]
        row = {"dit_launches": launches, "dit_span_ms": (d_end - d_start) / NS, "dit_busy_ms": busy, "thinker_launches": len(inside)}
        if inside:
            first, last = min(s for s, _ in inside), max(e for _, e in inside)
            row.update({"pre_ms": None if i == 0 else (first - window_start) / NS, "thinker_span_ms": (last - first) / NS,
                        "thinker_busy_ms": union_ms(inside), "handoff_ms": (d_start - last) / NS})
            copies = [c for c in d2h if first <= c[1] <= d_start]
            loads = [c[0] for c in h2d if last - 1 <= c[0] <= d_start]
            if copies and loads:
                row.update({"kv_out_to_load_ms": (min(loads) - max(c[1] for c in copies)) / NS, "load_to_kernel_ms": (d_start - min(loads)) / NS})
        rows.append(row)
    return {"rows": rows, "thinker_pid": th, "dit_pid": dit,
            "thinker_trace_ends_ms": (max(e for _, e in data[th]) - t0) / NS, "dit_trace_ends_ms": (dit_bursts[-1][1] - t0) / NS}


def render_handoff(db: Path, result: dict) -> str:
    def f(row, key):
        return "-" if row.get(key) is None else f"{row[key]:,.0f}"
    lines = [f"### {db.stem}: Thinker (process {result['thinker_pid']}) and DiT (process {result['dit_pid']}), one row per DiT burst", "",
             "| DiT burst | to Thinker's first kernel | Thinker first..last | Thinker GPU busy | Thinker launches | Thinker last -> DiT first | of which KV D2H done -> first H2D | first H2D -> first DiT kernel | DiT span | DiT busy | DiT launches |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, row in enumerate(result["rows"]):
        lines.append(f"| {i} | {f(row, 'pre_ms')} | {f(row, 'thinker_span_ms')} | {f(row, 'thinker_busy_ms')} | {row['thinker_launches']:,} | {f(row, 'handoff_ms')} | "
                     f"{f(row, 'kv_out_to_load_ms')} | {f(row, 'load_to_kernel_ms')} | {f(row, 'dit_span_ms')} | {f(row, 'dit_busy_ms')} | {row['dit_launches']:,} |")
    if result["thinker_trace_ends_ms"] < result["dit_trace_ends_ms"] - 500:
        lines += ["", f"**Truncated Thinker trace**: its last kernel is {result['dit_trace_ends_ms'] - result['thinker_trace_ends_ms']:,.0f} ms before the DiT's last, "
                      "so the later rows have a missing or partial Thinker window; read only the rows that are followed by a complete one."]
    return "\n".join(lines) + "\n"


def nsys_candidates() -> list[str]:
    """nsys binaries to try, newest first: $NSYS, user installs under ~/.local/opt, then PATH (a distro nsys cannot read newer reports; an export without -o can silently write nothing)."""
    found = [os.environ["NSYS"]] if os.environ.get("NSYS") else []
    found += [str(p) for p in sorted(Path.home().glob(".local/opt/nsys-*/opt/nvidia/nsight-systems-cli/*/target-linux-x64/nsys"), reverse=True)]
    if shutil.which("nsys"):
        found.append(shutil.which("nsys"))
    return found


def export_sqlite(rep: Path) -> None:
    candidates = nsys_candidates()
    if not candidates:
        raise SystemExit(f"nsys is not available; cannot export {rep.name} (nsys export --type sqlite {rep})")
    for nsys in candidates:
        done = subprocess.run([nsys, "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(rep.with_suffix(".sqlite")), str(rep)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if done.returncode == 0 and rep.with_suffix(".sqlite").is_file() and rep.with_suffix(".sqlite").stat().st_size > 0:
            return
    raise SystemExit(f"no nsys in {candidates} could export {rep.name} (set NSYS=<path of the nsys that wrote it>)")


def ensure_sqlite(nsys_dir: Path) -> list[Path]:
    """One .sqlite per .nsys-rep (exported on demand: the suite leaves only the reports; an empty .sqlite is a failed export)."""
    dbs = []
    for rep in sorted(nsys_dir.glob("*.nsys-rep")):
        db = rep.with_suffix(".sqlite")
        if not db.is_file() or db.stat().st_size == 0 or db.stat().st_mtime < rep.stat().st_mtime:
            export_sqlite(rep)
        dbs.append(db)
    dbs += [d for d in sorted(nsys_dir.glob("*.sqlite")) if d not in dbs]
    if not dbs:
        raise SystemExit(f"{nsys_dir}: no .nsys-rep or .sqlite files")
    return dbs


def top_kernels(db: Path, gap_ms: float, min_launches: int, top: int, device: Stream = None) -> list[tuple[str, float, int]]:
    """Kernel time by name inside the last burst that has at least `min_launches` launches (one request in a back-to-back run)."""
    con = sqlite3.connect(db)
    where = _where(device, "k.")
    rows = con.execute(
        "select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k "
        f"join StringIds s on s.id = coalesce(k.demangledName, k.shortName){where} order by k.start").fetchall()
    out, cur, last_end = [], [], None
    for row in rows:
        if last_end is not None and (row[0] - last_end) / NS > gap_ms:
            out.append(cur)
            cur = []
        cur.append(row)
        last_end = row[1] if last_end is None else max(last_end, row[1])
    out.append(cur)
    kept = [b for b in out if len(b) >= min_launches]
    if not kept:
        return []
    agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
    for start, end, name in kept[-1]:
        agg[name][0] += (end - start) / NS
        agg[name][1] += 1
    return [(n, t, int(c)) for n, (t, c) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]]


def report(nsys_dir: Path, gap_ms: float, min_launches: int, last: int, top: int = 8) -> str:
    """One section per report; a report that holds the kernels of several processes (vLLM-Omni two-stage: Thinker process on one card, DiT
    process on the other) gets one table per process, because bursts of different cards overlap in time and must not be merged.
    Processes are listed in order of their first kernel (the Thinker first)."""
    lines = []
    for db in ensure_sqlite(nsys_dir):
        if not has_kernels(db):
            lines += [f"### {db.stem}: NO CUDA KERNEL RECORDS", "", no_kernels_note(db), ""]
            continue
        for device in streams(db):
            found = bursts(db, gap_ms, min_launches, device)[-last:]
            where = "" if device is None else (f" (GPU {device[0]})" if device[1] is None else f" (process {device[1]}, GPU {device[0]})")
            lines += [f"### {db.stem}{where}", "", "| gap before ms | span ms | GPU busy ms | launches |", "|---:|---:|---:|---:|"]
            for b in found:
                gap = "-" if b["gap_ms"] is None else f"{b['gap_ms']:,.0f}"
                lines.append(f"| {gap} | {b['span_ms']:,.0f} | {b['busy_ms']:,.0f} | {b['launches']:,} |")
            lines.append("")
            if top and found:
                lines += ["Top kernels of the last burst:", "", "| kernel ms | launches | kernel |", "|---:|---:|---|"]
                lines += [f"| {t:,.1f} | {c:,} | `{n[:90]}` |" for n, t, c in top_kernels(db, gap_ms, min_launches, top, device)]
                lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("nsys_dir", type=Path)
    parser.add_argument("--gap-ms", type=float, default=40.0)
    parser.add_argument("--min-launches", type=int, default=200)
    parser.add_argument("--last", type=int, default=12)
    parser.add_argument("--top", type=int, default=8, help="kernel names listed for the last burst (0: none)")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--handoff", action="store_true", help="two-stage reports: add the per-request Thinker / hand-off / DiT table")
    args = parser.parse_args()
    text = report(args.nsys_dir, args.gap_ms, args.min_launches, args.last, args.top)
    if args.handoff:
        for db in ensure_sqlite(args.nsys_dir):
            result = handoff(db, args.gap_ms)
            text += "\n" + (render_handoff(db, result) if result else f"### {db.stem}: no Thinker + DiT process pair found (single-process report)\n")
    print(text)
    dbs = ensure_sqlite(args.nsys_dir)
    empty = [db.name for db in dbs if not has_kernels(db)]
    # a directory where no capture has a kernel is a failed run, not a result: no file in exps/results for it to be mistaken for one
    if args.markdown and len(empty) < len(dbs):
        args.markdown.write_text(text)
    if empty:         # the table above names each one; the exit status makes `exps.suite analyze` report the arm as failed, not as done
        sys.exit(f"no CUDA kernel records in: {', '.join(empty)}")


if __name__ == "__main__":
    main()
