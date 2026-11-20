#!/usr/bin/env python3
"""TTFT / inter-token latency / E2E per (campaign, framework, case) from the client's requests.jsonl (CPU only).

Why: an E2E ratio mixes three things that have different causes: the work before the first token (front end, ViT,
prefill, stage hand-off), the cost of each decoded token, and the tail. At a pinned output length (the Qwen TXT and BAGEL
I2T cases decode a fixed number of tokens) the inter-token latency (ITL) is the per-step cost of the decode loop, and it
separates "the GPU does more work per token" (ITL moves with the kernels) from "the CPU leaves the GPU idle between
tokens" (ITL moves with scheduling) and from "the request spends time before the first token" (TTFT).

    envs/mstar/bin/python exps/tools/analyze_decode_itl.py exps/runs/BASELINE_QWEN_TXT_MAIN_01_B2B \\
        exps/runs/MSTAR_ABLATION_QWEN_TXT_S_01_B2B --markdown exps/results/decode-itl_rtx5880.md

Every positional argument is a campaign directory (`exps/runs/<CAMPAIGN>`); each cell `<TASK>/<framework>/<case>/rNN` is
read. ITL values under 0.5 ms are chunk-mates (several tokens delivered together) and are dropped; the median of the
remaining values is reported. A framework that does not stream (vLLM single-stage BAGEL I2T answers in one chunk) has
TTFT == E2E and no ITL.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def cell_rows(cell: Path) -> list[dict]:
    path = cell / "requests.jsonl"
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("phase") == "measured" and row.get("success"):
            rows.append(row)
    return rows


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def summarise(rows: list[dict]) -> dict:
    itl = [x for row in rows for x in (row.get("itl_ms") or []) if x > 0.5]
    tokens = [row["output_tokens"] for row in rows if row.get("output_tokens")]
    return {
        "n": len(rows),
        "e2e": median([row["e2e_ms"] for row in rows]),
        "ttft": median([row["ttft_ms"] for row in rows if row.get("ttft_ms")]),
        "itl": median(itl),
        "tokens": median(tokens),
    }


def collect(campaigns: list[Path]) -> list[tuple[str, str, str, dict]]:
    out = []
    for camp in campaigns:
        pooled: dict[tuple[str, str], list[dict]] = {}
        for cell in sorted(camp.glob("*/*/*/r*")):
            if not cell.is_dir():
                continue
            fw, case = cell.parent.parent.name, cell.parent.name
            pooled.setdefault((fw, case), []).extend(cell_rows(cell))
        for (fw, case), rows in pooled.items():
            if rows:
                out.append((camp.name, fw, case, summarise(rows)))
    return out


def fmt(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def render(rows: list[tuple[str, str, str, dict]]) -> str:
    lines = ["| campaign | framework | case | n | tokens | E2E ms | TTFT ms | ITL ms/token | (E2E - TTFT) / tokens |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for camp, fw, case, s in rows:
        tail = None
        if s["e2e"] and s["ttft"] and s["tokens"] and s["tokens"] > 1:
            tail = (s["e2e"] - s["ttft"]) / (s["tokens"] - 1)
        lines.append(f"| {camp} | {fw} | {case} | {s['n']} | {fmt(s['tokens'])} | {fmt(s['e2e'])} | {fmt(s['ttft'])} "
                     f"| {fmt(s['itl'], 2)} | {fmt(tail, 2)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("campaigns", nargs="+", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    text = render(collect(args.campaigns))
    print(text)
    if args.markdown:
        args.markdown.write_text(text)


if __name__ == "__main__":
    main()
