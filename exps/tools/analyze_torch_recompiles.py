#!/usr/bin/env python3
"""Count torch.compile recompiles and graph breaks in an M* server log, and place them against the measured requests.

Round 5 showed that with torch.compile on, M*'s BAGEL `LLM.prefill_text` forward costs ~360 ms in *every* measured request
(~80 ms with TORCHDYNAMO_DISABLE=1), i.e. it is a recurring per-request cost and not a one-off warm-up. The suspect is a
recompile per request (a new prompt length is a new shape). `TORCH_LOGS=recompiles,graph_breaks` makes dynamo print each
recompile with the guard that failed; this tool turns that log into numbers (CPU only):

    envs/mstar/bin/python exps/tools/analyze_torch_recompiles.py \
        --cell exps/runs/BAGEL_RECOMPILE_01/BAGEL/mstar/t2i-256/r01 [--markdown out.md]

Log lines carry a local wall-clock time (``V1025 20:21:11.234``); request times are UTC in requests.jsonl, converted with
the machine's current UTC offset. If the log has no timestamps the per-request split is skipped, not guessed.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

_TIME = r"^[VIWE](\d{2})(\d{2}) (\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?"
RECOMPILE = re.compile(_TIME + r".*?Recompiling function (\S+) in (\S+)")
BREAK = re.compile(_TIME + r".*?Graph break.*?at (\S+)")
GUARD = re.compile(r"triggered by the following guard failure")
# Every continuation line repeats the log prefix: "V1026 ... [22/12] [__recompiles]     - 22/11: <failed guard>"
FAILED = re.compile(r"\[__recompiles\]\s+-\s+\d+/\d+:\s+(.+)$")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _stamp(month: str, day: str, hh: str, mm: str, ss: str, fraction: str | None, year: int) -> datetime:
    """Keep the sub-second part: a request that starts at :02.286 owns an event logged at :02.327 (truncating the event to :02
    put it before the window and dropped one recompile per request from the count)."""
    micro = int((fraction or "0").ljust(6, "0")[:6])
    return datetime(year, int(month), int(day), int(hh), int(mm), int(ss), micro)


def reason_key(text: str) -> str:
    """Fold request ids and numbers so that the same kind of guard failure counts once."""
    return re.sub(r"\d+", "N", UUID.sub("<uuid>", text.split("  #")[0].strip()))[:120]


def parse(lines: list[str], year: int) -> dict:
    recompiles, breaks, reasons = [], [], Counter()
    pending = None
    for line in lines:
        m = RECOMPILE.search(line)
        if m:
            when = _stamp(*m.group(1, 2, 3, 4, 5, 6), year)
            recompiles.append({"time": when, "function": m.group(7), "file": m.group(8)})
            pending = recompiles[-1]
            continue
        b = BREAK.search(line)
        if b:
            breaks.append({"time": _stamp(*b.group(1, 2, 3, 4, 5, 6), year), "site": b.group(7)})
            pending = None
            continue
        if pending is not None:
            f = FAILED.search(line)
            if f:
                # the first failure listed is the newest cache entry; all of them are guards that did not match this call
                pending.setdefault("reasons", []).append(f.group(1)[:200])
                if len(pending["reasons"]) == 1:
                    reasons[reason_key(f.group(1))] += 1
            elif line.strip() and not GUARD.search(line):
                pending = None
    return {"recompiles": recompiles, "graph_breaks": breaks, "reasons": reasons}


def windows(requests_file: Path) -> list[tuple[datetime, datetime]]:
    out = []
    for line in requests_file.read_text().splitlines():
        r = json.loads(line)
        if r.get("phase") != "measured" or not r.get("success"):
            continue
        start = datetime.fromisoformat(r["started_at_utc"]).astimezone().replace(tzinfo=None)
        out.append((start, start + timedelta(milliseconds=r["e2e_ms"])))
    return out


def e2e_series(requests_file: Path) -> list[float]:
    """E2E ms of the measured successful requests, in order (same selection as `windows`)."""
    rows = [json.loads(line) for line in requests_file.read_text().splitlines()]
    return [r["e2e_ms"] for r in rows if r.get("phase") == "measured" and r.get("success")]


def markdown(name: str, found: dict, spans: list[tuple[datetime, datetime]], e2e: list[float] | None = None) -> str:
    rec, brk = found["recompiles"], found["graph_breaks"]
    lines = [f"# {name}: torch.compile recompiles and graph breaks (server.log, TORCH_LOGS=recompiles,graph_breaks)", "",
             f"recompile events: {len(rec)}; graph breaks: {len(brk)}", ""]
    if rec:
        by_fn = Counter(r["function"] for r in rec)
        lines += ["| function | recompiles |", "|---|---:|"] + [f"| {fn} | {n} |" for fn, n in by_fn.most_common(10)] + [""]
    if found["reasons"]:
        lines += ["Newest failed guard of each recompile (request ids -> `<uuid>`, digits -> N):", "",
                  "| failed guard | count |", "|---|---:|"] + [
            f"| `{text}` | {n} |" for text, n in found["reasons"].most_common(8)] + [""]
    if spans and rec:
        inside = [[r for r in rec if lo <= r["time"] <= hi] for lo, hi in spans]
        before = sum(1 for r in rec if r["time"] < spans[0][0])
        lines += [f"recompiles before the first measured request: {before}",
                  "recompiles inside each measured request: " + ", ".join(str(len(x)) for x in inside),
                  "graph breaks inside each measured request: "
                  + ", ".join(str(sum(1 for g in brk if lo <= g["time"] <= hi)) for lo, hi in spans), ""]
        if e2e:
            lines += ["E2E ms of each measured request: " + ", ".join(f"{v:,.0f}" for v in e2e), ""]
        names = Counter(r["function"] for x in inside for r in x)
        if names:
            lines += ["Recompiled functions inside the measured requests:", "", "| function | events | per request |",
                      "|---|---:|---:|"] + [f"| {fn} | {n} | {n / len(spans):.1f} |" for fn, n in names.most_common(10)] + [""]
    elif rec:
        lines += ["No measured-request windows supplied: per-request split skipped.", ""]
    if not rec and not brk:
        lines += ["No recompile or graph-break lines. Either dynamo never recompiled (then the ~280 ms is not a recompile), "
                  "or TORCH_LOGS did not reach the worker (check the serve command in the manifest).", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", type=Path, required=True, help="cell directory holding server.log and requests.jsonl")
    ap.add_argument("--year", type=int, default=datetime.now().year)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    log = args.cell / "server.log"
    if not log.exists():
        print(f"no server.log under {args.cell}", file=sys.stderr)
        return 2
    found = parse(log.read_text(errors="replace").splitlines(), args.year)
    requests = args.cell / "requests.jsonl"
    spans = windows(requests) if requests.exists() else []
    text = markdown(args.cell.parent.name + "/" + args.cell.name, found, spans, e2e_series(requests) if requests.exists() else None)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
