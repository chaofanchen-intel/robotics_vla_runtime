#!/usr/bin/env python3
"""BAGEL two-stage: what the Thinker -> DiT KV hand-off costs per request, by connector (CPU only).

Reads each cell's `server.log`, where vLLM-Omni's KV transfer manager writes, per branch of a request (`<id>`, `<id>__cfg_text`, `<id>__cfg_img`),
`KV cache serialized for <id> in X ms` and `KV transfer OK: <id>, N bytes across K key(s), Ys, Z MB/s`, and the cell's `requests.jsonl` for the E2E.
The receiver side logs nothing, so the table also gives `E2E - single-stage E2E` when a single-stage campaign is passed: the hand-off cost seen by the client.

    envs/mstar/bin/python exps/tools/analyze_kv_handoff.py exps/runs/<CAMPAIGN>... --markdown exps/results/bagel-connector_rtx5880.md

Per (campaign, framework, case): median over requests of the sender-side serialise time, transfer time and bytes (summed over a request's branches), the achieved MB/s,
and the median E2E. The warm-up requests that the engine runs at start are ignored (they carry no `chatcmpl-` id with a CFG companion).
"""
from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

try:
    from exps.tools.analyze_decode_itl import cell_rows
except ImportError:  # pragma: no cover
    from analyze_decode_itl import cell_rows

ANSI = re.compile(r"\x1b\[[0-9;]*m")
SERIALIZED = re.compile(r"KV cache serialized for (\S+) in ([\d.]+) ms")
TRANSFERRED = re.compile(r"KV transfer OK: (\S+), (\d+) bytes across (\d+) key\(s\), ([\d.]+)s")


def parse_log(text: str) -> dict[str, dict[str, float]]:
    """request id (branch suffix stripped) -> {serialize_ms, transfer_ms, bytes}, summed over the request's branches."""
    out: dict[str, dict[str, float]] = {}
    for line in ANSI.sub("", text).splitlines():
        m = SERIALIZED.search(line)
        if m:
            entry = out.setdefault(m.group(1).split("__cfg")[0], {"serialize_ms": 0.0, "transfer_ms": 0.0, "bytes": 0.0})
            entry["serialize_ms"] += float(m.group(2))
            continue
        m = TRANSFERRED.search(line)
        if m:
            entry = out.setdefault(m.group(1).split("__cfg")[0], {"serialize_ms": 0.0, "transfer_ms": 0.0, "bytes": 0.0})
            entry["transfer_ms"] += float(m.group(4)) * 1000
            entry["bytes"] += float(m.group(2))
    return out


def collect(campaigns: list[Path]) -> list[dict]:
    rows = []
    for camp in campaigns:
        pooled: dict[tuple[str, str], dict] = {}
        for cell in sorted(camp.glob("*/*/*/r*")):
            if not cell.is_dir():
                continue
            fw, case = cell.parent.parent.name, cell.parent.name
            bucket = pooled.setdefault((fw, case), {"e2e": [], "serialize_ms": [], "transfer_ms": [], "bytes": []})
            bucket["e2e"] += [r["e2e_ms"] for r in cell_rows(cell)]
            log = cell / "server.log"
            if log.is_file():
                for entry in parse_log(log.read_text(errors="replace")).values():
                    if entry["bytes"] > 1e6:  # a real KV payload, not a warm-up or a 463 KB cfg_img branch alone
                        for key in ("serialize_ms", "transfer_ms", "bytes"):
                            bucket[key].append(entry[key])
        for (fw, case), b in pooled.items():
            if b["e2e"]:
                rows.append({"campaign": camp.name, "framework": fw, "case": case, "n": len(b["e2e"]), "e2e": statistics.median(b["e2e"]),
                             "requests_logged": len(b["bytes"]),
                             **{k: (statistics.median(b[k]) if b[k] else None) for k in ("serialize_ms", "transfer_ms", "bytes")}})
    return rows


def fmt(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def render(rows: list[dict]) -> str:
    lines = ["| campaign | framework | case | n | E2E ms | KV logged requests | KV MB per request | serialise ms | transfer ms | sender-side total ms | MB/s (bytes / transfer) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        total = r["serialize_ms"] + r["transfer_ms"] if r["serialize_ms"] is not None and r["transfer_ms"] is not None else None
        mbps = r["bytes"] / 1e6 / (r["transfer_ms"] / 1000) if r["bytes"] and r["transfer_ms"] else None
        lines.append(f"| {r['campaign']} | {r['framework']} | {r['case']} | {r['n']} | {fmt(r['e2e'])} | {r['requests_logged']} | "
                     f"{fmt(r['bytes'] / 1e6 if r['bytes'] else None)} | {fmt(r['serialize_ms'])} | {fmt(r['transfer_ms'])} | {fmt(total)} | {fmt(mbps)} |")
    lines += ["", "serialise / transfer are the sender-side times vLLM-Omni logs, summed over a request's branches; the receiver's time is not logged, so compare E2E "
              "(and the single-stage row, which has no hand-off) for the full cost. Single request at a time, one pair of cards without NVLink or RDMA."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    text = render(collect(args.campaigns))
    print(text)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
