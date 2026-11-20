#!/usr/bin/env python3
"""Small request-level summaries for the round-4 follow-up experiments (CPU only).

    resolution   BAGEL E2E by resolution and engine (256 / 384 / 512 / 768), per-step cost and the gap to M*
    waves        T2S concurrency cells wave by wave: transient (first waves) vs steady state, per audio-second
    chunk-ab     T2S E2E / TTFA / audio seconds / chunk count per engine and campaign (async_chunk, partial start)
    compile      count `torch.compile applied` lines in M* server logs (NOT proof of anything: M* logs it even with
                 TORCHDYNAMO_DISABLE=1; the proof is zero `triton_*` kernels in an nsys report)
    ab           A/B of named campaign groups for one task: `ab --task-dir BAGEL on=C1,C2 off=C3,C4` (first group is
                 the reference); E2E per start, ratio to the reference, optionally per audio-second

Starts are never pooled: every number is the median over measured requests of one start, then the median over
starts (range in brackets).

    envs/mstar/bin/python exps/tools/analyze_followups.py resolution BAGEL_ABLATION_02 BAGEL_RES_SWEEP_01
    envs/mstar/bin/python exps/tools/analyze_followups.py waves CONCURRENCY_01 CONCURRENCY_02 CONCURRENCY_03
    envs/mstar/bin/python exps/tools/analyze_followups.py chunk-ab QWEN_T2S_CHUNK_TUNED_01 QWEN_T2S_VLLM_GREEDY_SYNC_01
    envs/mstar/bin/python exps/tools/analyze_followups.py compile BAGEL_NOCOMPILE_01
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths


def measured(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("phase") == "measured" and r.get("success")]


def cells(campaign: str, task_dir: str):
    root = paths.campaign_root(campaign)
    for req in sorted((root / task_dir).glob("*/*/r*/requests.jsonl")):
        yield req.parents[2].name, req.parents[1].name, req.parent.name, req


def spread(values: list[float]) -> str:
    return f"{statistics.median(values):,.0f} [{min(values):,.0f}–{max(values):,.0f}]"


def resolution(campaigns: list[str]) -> str:
    per: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for campaign in campaigns:
        for fw, case, _rep, req in cells(campaign, "BAGEL"):
            m = re.fullmatch(r"(t2i|i2i)-(\d+)", case)
            rows = measured(req)
            if m and rows:
                per[(int(m.group(2)), m.group(1) + "|" + fw, m.group(1))].append(statistics.median(r["e2e_ms"] for r in rows))
    lines = ["| task | resolution | latent tokens | engine | E2E p50 ms | ms / step (50) | vs M* |", "|---|---:|---:|---|---:|---:|---:|"]
    for task in ("t2i", "i2i"):
        for res in sorted({k[0] for k in per if k[2] == task}):
            ref = per.get((res, f"{task}|mstar", task))
            for fw in ("mstar", "vllm-omni-single-stage", "vllm-omni-two-stage"):
                values = per.get((res, f"{task}|{fw}", task))
                if not values:
                    continue
                mid = statistics.median(values)
                ratio = f"{mid / statistics.median(ref):.2f}x" if ref else "-"
                lines.append(f"| {task} | {res} | {(res // 16) ** 2:,} | {fw} ({len(values)} starts) | {spread(values)} | {mid / 50:.1f} | {ratio} |")
    return "\n".join(lines) + "\n\nLatent tokens = (side/16)^2; the card is gated to a cool start for every request.\n"


#: CFG branches actually computed per denoise step at t2i with cfg_text_scale=4.0, by engine and ladder variant. Both engines compute
#: the third (image-guidance) branch only for cfg_img_scale > 1.0: M* `LLMSubmodule.image_gen` runs two branches "when image guidance
#: is off (img scale <= 1)". (An older M* repeated the packed sequence three times; the measured default/cfg3 times on the RTX 5880,
#: 12.98 s vs 18.83 s at 512^2, show the current one does not.)
BRANCHES = {("mstar", "nocfg"): 1, ("mstar", "default"): 2, ("mstar", "cfg3"): 3,
            ("vllm-omni-single-stage", "nocfg"): 1, ("vllm-omni-single-stage", "default"): 2, ("vllm-omni-single-stage", "cfg3"): 3,
            ("vllm-omni-two-stage", "nocfg"): 1, ("vllm-omni-two-stage", "default"): 2, ("vllm-omni-two-stage", "cfg3"): 3}
ENGINES = ("mstar", "vllm-omni-single-stage", "vllm-omni-two-stage")


def ladder(campaigns: list[str]) -> str:
    """T2I E2E against the number of CFG branches: does equal work (three branches on both engines) give equal time?"""
    per: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    clock: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for campaign in campaigns:
        for fw, case, _rep, req in cells(campaign, "BAGEL"):
            m = re.fullmatch(r"t2i-(\d+)(?:-(nocfg|cfg3|default))?", case)
            rows = measured(req)
            if not m or not rows or (fw, m.group(2) or "default") not in BRANCHES:
                continue
            key = (int(m.group(1)), fw, m.group(2) or "default")
            per[key].append(statistics.median(r["e2e_ms"] for r in rows))
            means = [r["thermal_monitor"]["sm_clock_mhz_mean"] for r in rows if (r.get("thermal_monitor") or {}).get("sm_clock_mhz_mean")]
            if means:
                clock[key].append(statistics.mean(means))
    lines = ["| resolution | engine | variant | CFG branches | starts | E2E p50 ms [min-max] | mean SM clock MHz | "
             "ms per extra branch per step |", "|---:|---|---|---:|---:|---:|---:|---:|"]
    for res in sorted({k[0] for k in per}):
        for fw in ENGINES:
            base = per.get((res, fw, "nocfg"))
            for variant in ("nocfg", "default", "cfg3"):
                values = per.get((res, fw, variant))
                if not values:
                    continue
                branches = BRANCHES[(fw, variant)]
                extra = "-" if variant == "nocfg" or not base else \
                    f"{(statistics.median(values) - statistics.median(base)) / (branches - 1) / 50:.1f}"
                clk = f"{statistics.mean(clock[(res, fw, variant)]):,.0f}" if clock.get((res, fw, variant)) else "n/a"
                lines.append(f"| {res} | {fw} | {variant} | {branches} | {len(values)} | {spread(values)} | {clk} | {extra} |")
    ratios = []
    for title, mkey in (("One branch (CFG off: same work AND same output semantics on both engines)", "nocfg"),
                        ("Three branches on both engines (vLLM with cfg_img_scale=2.0, so its output differs from the default request)", "cfg3")):
        ratios += ["", f"{title}, vLLM / M*:", "", "| resolution | vLLM deployment | M* ms | vLLM ms | vLLM / M* |", "|---:|---|---:|---:|---:|"]
        for res in sorted({k[0] for k in per}):
            a = per.get((res, "mstar", mkey))
            for fw in ENGINES[1:]:
                b = per.get((res, fw, mkey))
                if a and b:
                    ratios.append(f"| {res} | {fw} | {statistics.median(a):,.0f} | {statistics.median(b):,.0f} | {statistics.median(b) / statistics.median(a):.2f}x |")
    note = ("\nBoth engines run two branches on the `default` request (cfg_img_scale=1.0: text guidance only) and three on `cfg3`, so "
            "`default` is equal work too; `ms per extra branch per step` should agree between the two variants of one engine. "
            "Clocks are the per-request means from the harness; an engine that draws more power per step sits lower on the same card, so compare the clock column before reading "
            "a ratio as the engine's.\n")
    return "\n".join(lines + ratios) + "\n" + note


def waves(campaigns: list[str], transient: int) -> str:
    lines = [f"| campaign | engine | case | start | waves ms (first {transient} = transient window) | steady p50 ms | ms per audio-s (steady) |",
             "|---|---|---|---|---|---:|---:|"]
    for campaign in campaigns:
        for fw, case, rep, req in cells(campaign, "QWEN_T2S"):
            rows = [r for r in measured(req) if r.get("wave_wall_ms")]
            if not rows or "-b" not in case:
                continue
            wall = [r["wave_wall_ms"] for r in rows]
            audio = [sum(r.get("per_request_audio_duration_s") or [0]) for r in rows]
            steady = list(zip(wall, audio))[transient:] or list(zip(wall, audio))
            per_audio = [w / a for w, a in steady if a]
            shown = " ".join(f"{w:,.0f}" for w in wall)
            lines.append(f"| {campaign} | {fw} | {case} | {rep} | {shown} | {statistics.median(w for w, _ in steady):,.0f} | "
                         f"{statistics.median(per_audio):.0f} |" if per_audio else f"| {campaign} | {fw} | {case} | {rep} | {shown} | - | - |")
    return "\n".join(lines) + "\n"


def chunk_ab(campaigns: list[str]) -> str:
    lines = ["| campaign | engine | starts | E2E mean ms | TTFA mean ms | audio s | audio chunks | E2E per audio-s | contract ok |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for campaign in campaigns:
        grouped: dict[str, list[list[dict]]] = defaultdict(list)
        for fw, case, _rep, req in cells(campaign, "QWEN_T2S"):
            if case == "verbatim-en":
                grouped[fw].append(measured(req))
        for fw, starts in sorted(grouped.items()):
            starts = [s for s in starts if s]
            if not starts:
                continue
            mean = lambda key, rows: statistics.mean(r[key] for r in rows if r.get(key) is not None)  # noqa: E731
            e2e = [mean("e2e_ms", s) for s in starts if s]
            ttfa = [mean("ttfa_ms", s) for s in starts if s and any(r.get("ttfa_ms") for r in s)]
            audio = [mean("audio_duration_s", s) for s in starts if s]
            chunks = [statistics.mean((r.get("per_modality_chunks") or {}).get("audio") or 0 for r in s) for s in starts if s]
            ok = [statistics.mean(1.0 if r.get("contract_honoured") else 0.0 for r in s) for s in starts if s]
            ratio = [e / a for e, a in zip(e2e, audio) if a]
            lines.append(f"| {campaign} | {fw} | {len(e2e)} | {spread(e2e)} | {spread(ttfa) if ttfa else '-'} | {statistics.median(audio):.2f} | "
                         f"{statistics.median(chunks):.1f} | {statistics.median(ratio):.0f} | {100 * statistics.median(ok):.0f}% |")
    return "\n".join(lines) + "\n"


def compile_check(campaigns: list[str]) -> str:
    lines = ["| campaign | cell | `torch.compile applied` lines | `torch.compile disabled` lines |", "|---|---|---:|---:|"]
    for campaign in campaigns:
        root = paths.campaign_root(campaign)
        for log in sorted(root.glob("*/*/*/r*/server.log")):
            text = log.read_text(errors="replace")
            lines.append(f"| {campaign} | {'/'.join(log.parts[-4:-1])} | {text.count('torch.compile applied')} | "
                         f"{text.count('torch.compile disabled')} |")
    return "\n".join(lines) + "\n"


def ab(task_dir: str, groups: dict[str, list[str]], per_audio: bool) -> str:
    """Per (engine, case): start-level p50 for every group (all campaigns of a group pooled as starts)."""
    table: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for name, campaigns in groups.items():
        for campaign in campaigns:
            for fw, case, _rep, req in cells(campaign, task_dir):
                rows = measured(req)
                if not rows:
                    continue
                if per_audio:
                    rows = [r for r in rows if r.get("audio_duration_s")]
                    if not rows:
                        continue
                    value = statistics.median(r["e2e_ms"] / r["audio_duration_s"] for r in rows)
                else:
                    value = statistics.median(r["e2e_ms"] for r in rows)
                table[(fw, case)][name].append(value)
    reference = next(iter(groups))
    unit = "ms per audio-s" if per_audio else "E2E p50 ms"
    lines = [f"| engine | case | group | starts | {unit} | vs {reference} |", "|---|---|---|---:|---:|---:|"]
    for (fw, case), by_group in sorted(table.items()):
        ref = statistics.median(by_group[reference]) if by_group.get(reference) else None
        for name in groups:
            values = by_group.get(name)
            if not values:
                continue
            ratio = f"{statistics.median(values) / ref:.2f}x" if ref else "-"
            lines.append(f"| {fw} | {case} | {name} | {len(values)} | {spread(values)} | {ratio} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("view", choices=("resolution", "waves", "chunk-ab", "compile", "ab", "ladder"))
    ap.add_argument("campaigns", nargs="+", help="Campaign ids (for `ab`: name=C1,C2 groups, reference first).")
    ap.add_argument("--task-dir", default="BAGEL", help="ab: BAGEL, QWEN_T2S, QWEN_A2T.")
    ap.add_argument("--per-audio", action="store_true", help="ab: divide E2E by audio seconds (T2S).")
    ap.add_argument("--transient", type=int, default=3, help="waves: leading waves treated as warm-up transient.")
    ap.add_argument("--markdown", type=Path)
    args = ap.parse_args()
    text = {"resolution": lambda: resolution(args.campaigns), "ladder": lambda: ladder(args.campaigns), "waves": lambda: waves(args.campaigns, args.transient),
            "chunk-ab": lambda: chunk_ab(args.campaigns), "compile": lambda: compile_check(args.campaigns),
            "ab": lambda: ab(args.task_dir, {g.split("=", 1)[0]: g.split("=", 1)[1].split(",") for g in args.campaigns},
                             args.per_audio)}[args.view]()
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
