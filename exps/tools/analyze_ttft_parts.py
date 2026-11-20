#!/usr/bin/env python3
"""Qwen3-Omni first-token latency split into a fixed per-request part, a per-prompt-token part and an audio part (CPU only).

Three cases per framework, all with the completion capped at ONE token (M* returns two; every number below is first-token time, which is equal work):
  txt-tiny-1tok   a 56-token (8 + the server's 48 template) prompt: the per-request fixed cost (HTTP, template, tokenisation, scheduling, a tiny forward, sampling)
  txt-1k-1tok     the 1,024-token prompt: fixed + prefill of 1,024 tokens
  a2t-en-1tok     FLEURS English clips: fixed + prefill of the (audio-token) prompt + the audio encoder

    envs/mstar/bin/python exps/tools/analyze_ttft_parts.py exps/runs/<TXT campaign> exps/runs/<A2T campaign> --markdown out.md

`fixed` = tiny TTFT. `per 1k tokens` = (1k TTFT - tiny TTFT) scaled to 1,000 prompt tokens (prefill, MoE tiles). `A2T excess` = A2T TTFT minus
(fixed + per-token x input_tokens) = what the audio path adds beyond text prefill of the same length. Comparing frameworks on each column says
whether a first-token gap is front-end overhead, prefill kernels, or the audio path.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

try:
    from exps.tools.analyze_decode_itl import cell_rows
except ImportError:  # pragma: no cover
    from analyze_decode_itl import cell_rows

CASES = ("txt-tiny-1tok", "txt-1k-1tok", "a2t-en-1tok")


def collect(campaigns: list[Path]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for camp in campaigns:
        for cell in sorted(camp.glob("*/*/*/r*")):
            case = cell.parent.name
            if case not in CASES or not cell.is_dir():
                continue
            rows = cell_rows(cell)
            bucket = out.setdefault(cell.parent.parent.name, {}).setdefault(case, {"e2e": [], "ttft": [], "tokens_in": []})
            # first-token time, not E2E: the 1-token probes return 1 token on vLLM / SGLang but 2 on M*, so only TTFT is equal work
            bucket["e2e"] += [r["ttft_ms"] if r.get("ttft_ms") is not None else r["e2e_ms"] for r in rows]
            bucket["ttft"] += [r["ttft_ms"] for r in rows if r.get("ttft_ms") is not None]
            bucket["tokens_in"] += [r["input_tokens"] for r in rows if r.get("input_tokens")]
    return out


def med(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fmt(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def render(data: dict[str, dict[str, dict]]) -> str:
    lines = ["| framework | tiny TTFT ms (fixed) | 1k TTFT ms | prompt tokens 1k | prefill, ms per 1k tokens | A2T TTFT ms | A2T input tokens | A2T excess over text prefill ms |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for fw, cases in sorted(data.items()):
        tiny, big, a2t = (med(cases.get(c, {}).get("e2e", [])) for c in CASES)
        t_tok = med(cases.get("txt-tiny-1tok", {}).get("tokens_in", []))
        b_tok = med(cases.get("txt-1k-1tok", {}).get("tokens_in", []))
        a_tok = med(cases.get("a2t-en-1tok", {}).get("tokens_in", []))
        per_token = (big - tiny) / (b_tok - t_tok) if None not in (tiny, big, t_tok, b_tok) and b_tok != t_tok else None
        excess = a2t - (tiny + per_token * (a_tok - (t_tok or 0))) if None not in (a2t, tiny, per_token, a_tok) else None
        lines.append(f"| {fw} | {fmt(tiny)} | {fmt(big)} | {fmt(b_tok)} | {fmt(per_token * 1000 if per_token is not None else None, 1)} | "
                     f"{fmt(a2t)} | {fmt(a_tok)} | {fmt(excess)} |")
    lines += ["", "Token counts are the server's reported prompt tokens when the client recorded them (`-` otherwise: the per-token columns need them). "
              "Single GPU set, C=1; a linear model of prefill in prompt length, adequate between ~15 and ~1,000 tokens."]
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
