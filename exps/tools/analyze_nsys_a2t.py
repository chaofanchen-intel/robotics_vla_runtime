#!/usr/bin/env python3
"""Per-request GPU activity for the Qwen3-Omni A2T nsys profile (3 engines, same clips).

A2T requests are short (~0.3 s) and the GPU is cool, so the per-request thermal gate does not insert
long idle gaps (requests are separated by 0.1-0.3 s of client time), and gap clustering merges or
splits them. Instead each measured request is matched to its kernels by wall-clock window (client UTC
start + e2e against the nsys session start), so busy time and launches are reported per request and per
output token.

    envs/mstar/bin/python exps/tools/analyze_nsys_a2t.py --campaign QWEN_A2T_PROFILE_01
    envs/mstar/bin/python exps/tools/analyze_nsys_a2t.py --campaign BAGEL_PROFILE_02 --task-dir BAGEL \\
        --case i2t-256-tokens64 --units 64 --frameworks mstar vllm-omni-single-stage vllm-omni-two-stage
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths
from exps.tools.analyze_nsys_requests import NS, clusters, load, union_ns

FRAMEWORKS = ("mstar", "sglang-omni", "vllm-omni")


def per_request(db: Path, requests_file: Path, pad_s: float = 0.05, units: int | None = None) -> list[dict]:
    """Attribute kernels to measured requests by wall-clock window.

    nsys kernel timestamps are relative to the session start, whose UTC time is recorded in
    TARGET_INFO_SESSION_START_TIME; the client records each request's UTC start and e2e. A kernel
    belongs to a request if it starts inside [start - pad, start + e2e + pad]. (Gap clustering is not
    usable here: A2T requests are back to back.)
    """
    import bisect
    import sqlite3
    from datetime import datetime

    con = sqlite3.connect(db)
    origin_ns = con.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME").fetchone()[0]
    rows, _ = load(db)
    starts = [r[0] for r in rows]
    measured = [json.loads(l) for l in requests_file.read_text().splitlines()]
    measured = [r for r in measured if r["phase"] == "measured"]
    out = []
    for req in measured:
        t0 = datetime.fromisoformat(req["started_at_utc"]).timestamp() * 1e9 - origin_ns - pad_s * 1e9
        t1 = t0 + (req["e2e_ms"] / 1e3 + 2 * pad_s) * 1e9
        lo, hi = bisect.bisect_left(starts, t0), bisect.bisect_right(starts, t1)
        c = rows[lo:hi]
        if not c:
            continue
        span = (max(e for _, e, _ in c) - c[0][0]) / NS
        busy = union_ns(c) / NS
        tokens = units if units is not None else (req.get("output_tokens") or 0)
        out.append({
            "tokens": tokens, "client_e2e_ms": req["e2e_ms"], "span_ms": span, "busy_ms": busy,
            "launches": len(c), "busy_per_token_ms": busy / tokens if tokens else None,
            "launches_per_token": len(c) / tokens if tokens else None,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default="QWEN_A2T_PROFILE_01")
    ap.add_argument("--case", default="a2t-en")
    ap.add_argument("--task-dir", default="QWEN_A2T", help="Task directory under the campaign (QWEN_A2T, BAGEL).")
    ap.add_argument("--frameworks", nargs="+", default=list(FRAMEWORKS))
    ap.add_argument("--units", type=int, default=None,
                    help="Work units per request (denoise steps / decode tokens); default: output_tokens.")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    root = paths.campaign_root(args.campaign)
    result = {}
    for fw in args.frameworks:
        db = root / "nsys" / f"{fw}-{args.case}-r01.sqlite"
        req = root / args.task_dir / fw / args.case / "r01" / "requests.jsonl"
        if not db.is_file() or db.stat().st_size == 0:
            rep = db.with_suffix(".nsys-rep")
            if rep.is_file():
                import subprocess
                subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o", str(db), str(rep)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if db.is_file() and req.is_file():
            result[fw] = per_request(db, req, units=args.units)
    lines = ["| framework | n | client e2e ms (nsys-run) | GPU busy ms | launches | busy ms/unit | launches/unit |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    med = lambda rows, k: statistics.median(r[k] for r in rows if r[k] is not None)  # noqa: E731
    for fw, rows in result.items():
        if not rows:
            lines.append(f"| {fw} | 0 | trace has no kernels inside any measured-request window (report truncated) | | | | |")
            continue
        lines.append(f"| {fw} | {len(rows)} | {med(rows, 'client_e2e_ms'):.0f} | {med(rows, 'busy_ms'):.0f} | "
                     f"{med(rows, 'launches'):.0f} | {med(rows, 'busy_per_token_ms'):.2f} | {med(rows, 'launches_per_token'):.0f} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
