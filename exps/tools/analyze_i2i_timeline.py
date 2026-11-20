#!/usr/bin/env python3
"""BAGEL I2I two-stage: where one request's wall time goes, from millisecond log stamps (CPU only).

vLLM-Omni's default log stamps have one-second resolution, which cannot split a 4.5 s request. The `bagel-connector-timeline` arms start the servers with
`VLLM_LOGGING_CONFIG_PATH=exps/configs/hardware/rtx5880/vllm_omni/logging/ms_timestamps.json`, which adds milliseconds, and this tool lines the markers up
with the client's own start and end (`requests.jsonl`):

    client start ──pre──▶ `CFG expansion for req <id>` (the engine has the request)
        ──stage0──▶ first `KV cache serialized for <id>` (the first branch's prefill is done)
        ──send──▶ last `KV transfer OK: <id>...` (all branches handed to the connector)
        ──gap──▶ `stage-1 [rep-0] add request: <id>` (the DiT stage has the request)
        ──dit──▶ client end   (receiver-side KV load + denoising + VAE + response; receiver-side time is not logged separately)

Single-stage has no KV markers: only `pre` and `engine` (= everything after the engine has the request) are reported for it.

    envs/mstar/bin/python exps/tools/analyze_i2i_timeline.py exps/runs/<CAMPAIGN>... --markdown exps/results/bagel-connector-timeline_rtx5880.md

Run the clock check first: the log has no time zone and no year, so the log's local time is matched to the client's UTC start through the machine's own zone
(client and server run on the same box).
"""
from __future__ import annotations

import argparse
import re
import statistics
from datetime import datetime
from pathlib import Path

try:
    from exps.tools.analyze_decode_itl import cell_rows
except ImportError:  # pragma: no cover
    from analyze_decode_itl import cell_rows

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"(\d\d)-(\d\d) (\d\d):(\d\d):(\d\d)\.(\d{3})")
REQ = r"(chatcmpl-[0-9a-f]+-[0-9a-f]+)"
MARKERS = {
    "engine_in": re.compile(r"CFG expansion for req " + REQ),
    "serialized": re.compile(r"KV cache serialized for " + REQ),
    "sent": re.compile(r"KV transfer OK: " + REQ),
    "stage1_in": re.compile(r"stage-1 \[rep-\d+\] add request: " + REQ),
}
SEGMENTS = ("pre", "stage0", "send", "gap", "dit")


def stamp_seconds(line: str, year: int) -> float | None:
    """Seconds since the epoch of the log line's local-time `MM-DD HH:MM:SS.mmm` stamp, or None when the line carries no millisecond stamp."""
    m = STAMP.search(line)
    if not m:
        return None
    month, day, hour, minute, second, ms = (int(x) for x in m.groups())
    return datetime(year, month, day, hour, minute, second, ms * 1000).timestamp()  # naive datetime = the machine's local zone


def parse_log(text: str, year: int) -> dict[str, dict[str, float]]:
    """parent request id -> {engine_in, first serialized, last sent, stage1_in} (epoch seconds, local clock)."""
    out: dict[str, dict[str, float]] = {}
    for line in ANSI.sub("", text).splitlines():
        t = stamp_seconds(line, year)
        if t is None:
            continue
        for name, pattern in MARKERS.items():
            m = pattern.search(line)
            if not m:
                continue
            entry = out.setdefault(m.group(1).split("__cfg")[0], {})
            if name == "serialized":
                entry.setdefault("serialized", t)  # first branch
            elif name == "sent":
                entry["sent"] = max(entry.get("sent", t), t)  # last branch
            else:
                entry.setdefault(name, t)
    return out


def segments(entry: dict[str, float], start: float, end: float) -> dict[str, float] | None:
    """Per-request segments in ms from the markers and the client's [start, end]; None when a marker is missing."""
    if "engine_in" not in entry:
        return None
    seg = {"pre": (entry["engine_in"] - start) * 1000, "engine": (end - entry["engine_in"]) * 1000}
    if all(k in entry for k in ("serialized", "sent", "stage1_in")):
        seg.update({"stage0": (entry["serialized"] - entry["engine_in"]) * 1000, "send": (entry["sent"] - entry["serialized"]) * 1000,
                    "gap": (entry["stage1_in"] - entry["sent"]) * 1000, "dit": (end - entry["stage1_in"]) * 1000})
    return seg


def cell_segments(cell: Path) -> list[dict[str, float]]:
    log = cell / "server.log"
    rows = cell_rows(cell)
    if not log.is_file() or not rows:
        return []
    first = datetime.fromisoformat(rows[0]["started_at_utc"]).astimezone()
    parsed = parse_log(log.read_text(errors="replace"), first.year)
    out = []
    for row in rows:
        start = datetime.fromisoformat(row["started_at_utc"]).timestamp()
        end = start + row["e2e_ms"] / 1000
        # the engine receives the request after the client sent it and before the client got its answer: take the first engine_in in that window
        candidates = sorted((e for e in parsed.values() if "engine_in" in e and start - 0.2 <= e["engine_in"] <= end), key=lambda e: e["engine_in"])
        if candidates:
            seg = segments(candidates[0], start, end)
            if seg:
                out.append({**seg, "e2e": row["e2e_ms"]})
    return out


def collect(campaigns: list[Path]) -> list[dict]:
    rows = []
    for camp in campaigns:
        pooled: dict[tuple[str, str], list[dict[str, float]]] = {}
        for cell in sorted(camp.glob("*/*/*/r*")):
            if cell.is_dir():
                pooled.setdefault((cell.parent.parent.name, cell.parent.name), []).extend(cell_segments(cell))
        for (fw, case), segs in pooled.items():
            if segs:
                keys = [k for k in ("e2e", "pre", "engine", *SEGMENTS) if all(k in s for s in segs)]
                rows.append({"campaign": camp.name, "framework": fw, "case": case, "n": len(segs), **{k: statistics.median(s[k] for s in segs) for k in keys}})
    return rows


def render(rows: list[dict]) -> str:
    def f(r, k):
        return "-" if k not in r else f"{r[k]:,.0f}"
    lines = ["| campaign | framework | case | n | E2E ms | pre (client to engine) | stage 0 (to first KV branch) | send (all branches) | gap (to DiT add request) | DiT side (KV load + denoise + VAE) | engine (all after pre) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['campaign']} | {r['framework']} | {r['case']} | {r['n']} | {f(r, 'e2e')} | {f(r, 'pre')} | {f(r, 'stage0')} | {f(r, 'send')} | "
                     f"{f(r, 'gap')} | {f(r, 'dit')} | {f(r, 'engine')} |")
    lines += ["", "Medians over the measured requests, matched to the log by time (the engine's `CFG expansion` marker falls inside the client's start..end). "
              "`DiT side` includes the receiver pulling the KV (NIXL's pull is asynchronous: its sender-side `transfer` is only the submit), so compare it across connectors "
              "rather than reading it as pure denoising time; the same denoising runs in every arm."]
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
