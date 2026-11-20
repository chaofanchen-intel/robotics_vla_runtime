#!/usr/bin/env python3
"""BAGEL attention-backend experiment: probe counters per cell, then the cross-framework ratio matrix (CPU only).

    envs/mstar/bin/python exps/tools/analyze_attn_probe.py \
        --campaigns BAGEL_ATTN_BACKEND_MSTAR_PROBE_01 BAGEL_ATTN_BACKEND_MSTAR_NO_FLASH_ATTN_01 \
        --vllm BAGEL_ATTN_BACKEND_VLLM_DEFAULT_01 BAGEL_ATTN_BACKEND_VLLM_SDPA_01 --markdown out.md

Part 1 lists, for every M* cell, what the worker-side probe counted: ViT flash-attn calls, ViT SDPA-fallback calls,
whether the pip package was importable, and which attention manager each attention resource resolved to. Part 2 pairs
every vLLM-Omni arm with every M* arm per case. A ratio that holds across all four pairings is not a flash-attn effect;
one that appears only against the `no-flash-attn` arm is.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths
from exps.suite import analysis


def load_cells(campaign: str, framework: str) -> dict[str, list[analysis.CellData]]:
    cells: dict[str, list[analysis.CellData]] = {}
    root = paths.campaign_root(campaign) / "BAGEL" / framework
    for requests in sorted(root.glob("*/r*/requests.jsonl")):
        cell_dir = requests.parent
        manifest = analysis.json.loads((cell_dir / "manifest.json").read_text())
        cells.setdefault(cell_dir.parent.name, []).append(
            analysis.CellData(framework, cell_dir.parent.name, cell_dir.name, cell_dir, manifest, analysis._jsonl(requests)))
    return cells


def report(mstar: list[str], vllm: list[str]) -> str:
    lines = ["# BAGEL attention backend: probe counters and ratio matrix", "",
             "## What the worker-side probe counted (M*)", "",
             "| campaign | case | start | probe procs | pip flash_attn importable | ViT flash calls | ViT SDPA-fallback calls | attention managers |",
             "|---|---|---|---:|---|---:|---:|---|"]
    p50: dict[tuple[str, str], float] = {}
    for campaign in mstar:
        for case, cells in load_cells(campaign, "mstar").items():
            stat = analysis.cell_stat(cells, "e2e_ms")
            if stat:
                p50[(campaign, case)] = stat.p50
            for cell in cells:
                probe = analysis.probe_for_cell(cell, campaign)
                lines.append(f"| {campaign} | {case} | {cell.repeat} | {probe['processes']} | {probe['flash_available']} | "
                             f"{probe['flash_calls']} | {probe['sdpa_calls']} | {', '.join(probe['managers']) or '-'} |")
    for campaign in vllm:
        for case, cells in load_cells(campaign, "vllm-omni-two-stage").items():
            stat = analysis.cell_stat(cells, "e2e_ms")
            if stat:
                p50[(campaign, case)] = stat.p50
    cases = sorted({case for (_c, case) in p50})
    lines += ["", "## E2E p50 (ms, pooled) and vLLM-Omni / M* for every pairing", "",
              "| case | " + " | ".join(mstar + vllm) + " |", "|---|" + "---:|" * (len(mstar) + len(vllm))]
    for case in cases:
        lines.append(f"| {case} | " + " | ".join(f"{p50[(c, case)]:,.0f}" if (c, case) in p50 else "-" for c in mstar + vllm) + " |")
    lines += ["", "| case | vLLM arm | M* arm | vLLM / M* |", "|---|---|---|---:|"]
    for case in cases:
        for v in vllm:
            for m in mstar:
                if (v, case) in p50 and (m, case) in p50:
                    lines.append(f"| {case} | {v} | {m} | {p50[(v, case)] / p50[(m, case)]:.2f}x |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaigns", nargs="+", required=True, help="M* campaigns (probe arms)")
    parser.add_argument("--vllm", nargs="*", default=[], help="vLLM-Omni two-stage campaigns")
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    text = report(args.campaigns, args.vllm)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
