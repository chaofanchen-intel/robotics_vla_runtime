#!/usr/bin/env python3
"""Qwen3-Omni T2S: E2E and time-to-first-audio against the audio duration produced, per framework (CPU only).

Why: the T2S ratio (ms per audio second) mixes a fixed start-up cost (Thinker, the Talker's first frames, the first Code2Wav chunk) with a
steady-state cost per second of audio (the Talker's frame loop). `E2E = a + b * audio_s` separates them: a gap in `a` is the front of the
pipeline, a gap in `b` is the per-frame loop. It also counts "stall" requests, whose time-to-first-audio is more than twice the cell's
median (vLLM-Omni JIT-compiles Triton MoE kernels for input shapes it has not seen; with a short precondition phase that lands in the
measured phase).

    envs/mstar/bin/python exps/tools/analyze_t2s_fit.py exps/runs/<CAMPAIGN>... --markdown exps/results/qwen-t2s-fit_rtx5880.md

Every positional argument is a campaign directory; cells `<TASK>/<framework>/<case>/rNN` are pooled per (campaign, framework, case). The
audio length differs a little between frameworks for the same sentence (their Talkers do not decode identically), which is why the fit
uses the realised duration rather than the sentence.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

try:  # run as a script from the repo root, or imported by the tests
    from exps.tools.analyze_decode_itl import cell_rows
except ImportError:  # pragma: no cover
    from analyze_decode_itl import cell_rows


def ols(x: list[float], y: list[float]) -> tuple[float, float, float] | None:
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx == 0:
        return None
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    a0 = my - b * mx
    total = sum((c - my) ** 2 for c in y)
    resid = sum((c - (a0 + b * a)) ** 2 for a, c in zip(x, y))
    return a0, b, (1 - resid / total) if total else 0.0


def summarise(rows: list[dict]) -> dict:
    rows = [r for r in rows if r.get("audio_duration_s") and r.get("ttfa_ms") is not None]
    if not rows:
        return {}
    ttfa_median = statistics.median(r["ttfa_ms"] for r in rows)
    steady = [r for r in rows if r["ttfa_ms"] <= 2 * ttfa_median]
    out = {"n": len(rows), "stalls": len(rows) - len(steady), "audio_s": statistics.median(r["audio_duration_s"] for r in rows),
           "e2e": statistics.median(r["e2e_ms"] for r in rows), "ttfa": statistics.median(r["ttfa_ms"] for r in rows),
           "ttfa_steady": statistics.median(r["ttfa_ms"] for r in steady) if steady else None}
    for name, subset in (("all", rows), ("steady", steady)):
        fit = ols([r["audio_duration_s"] for r in subset], [r["e2e_ms"] for r in subset])
        out[f"fit_{name}"] = fit
    tails = [(r["e2e_ms"] - r["ttfa_ms"]) / r["audio_duration_s"] for r in steady]
    out["after_first_audio_per_s"] = statistics.median(tails) if tails else None
    out["per_audio_s"] = statistics.median(r["e2e_ms"] / r["audio_duration_s"] for r in rows)
    out["rows_distinct"] = len({r.get("fixture_index") for r in rows})
    return out


def collect(campaigns: list[Path]) -> list[tuple[str, str, str, dict]]:
    out = []
    for camp in campaigns:
        pooled: dict[tuple[str, str], list[dict]] = {}
        for cell in sorted(camp.glob("*/*/*/r*")):
            if cell.is_dir():
                pooled.setdefault((cell.parent.parent.name, cell.parent.name), []).extend(cell_rows(cell))
        for (fw, case), rows in pooled.items():
            summary = summarise(rows)
            if summary:
                out.append((camp.name, fw, case, summary))
    return out


def fmt(value: float | None, digits: int = 0) -> str:
    return "-" if value is None else f"{value:,.{digits}f}"


def render(rows: list[tuple[str, str, str, dict]]) -> str:
    lines = ["| campaign | framework | case | n | rows | stalls | audio s | E2E ms | ms / audio s | TTFA ms (steady) | after first audio, ms / audio s | "
             "fit a ms (all / steady) | fit b ms per audio s (all / steady) | R² (steady) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|"]
    for camp, fw, case, s in rows:
        fa, fs = s["fit_all"], s["fit_steady"]
        lines.append(
            f"| {camp} | {fw} | {case} | {s['n']} | {s['rows_distinct']} | {s['stalls']} | {fmt(s['audio_s'], 2)} | {fmt(s['e2e'])} | "
            f"{fmt(s['per_audio_s'], 1)} | {fmt(s['ttfa_steady'])} | {fmt(s['after_first_audio_per_s'], 1)} | "
            f"{fmt(fa[0]) if fa else '-'} / {fmt(fs[0]) if fs else '-'} | {fmt(fa[1], 1) if fa else '-'} / {fmt(fs[1], 1) if fs else '-'} | "
            f"{fmt(fs[2], 2) if fs else '-'} |")
    lines += ["", "stall = a request whose time-to-first-audio exceeds twice its cell's median. `steady` excludes stalls. a = fixed latency, b = ms per "
              "second of audio produced. Single GPU set, C=1, one fixture family; localises the gap, does not name the kernel."]
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
