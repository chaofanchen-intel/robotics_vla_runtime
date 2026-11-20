#!/usr/bin/env python3
"""Decompose the BAGEL cross-engine gap into per-request and per-step cost.

Question this answers
---------------------
The measured BAGEL gap between M* and vLLM-Omni could come from

* **per-step / per-token cost** -- kernels, attention backend, how CFG branches are batched; or
* **per-request fixed cost**   -- prefill, Thinker->DiT KV hand-off between stages, VAE encode and
  decode, request handling, output encoding.

A single E2E number cannot separate them. For each engine this fits, over the ablation sweep,

    E2E_ms = a + b * x        x = denoise steps (T2I, I2I)  or  completion tokens (I2T)

``a`` is the fixed cost, ``b`` the marginal cost. It also reports the CFG-off ratio
``E2E(no CFG) / E2E(baseline)``.

Pre-registered reading (write it down before looking at data)
-------------------------------------------------------------
Let ``gap = E2E_other(x0) - E2E_mstar(x0)`` at the baseline ``x0`` (50 steps / 128 tokens).

* ``Δa / gap >= 0.6``  -> the gap is mostly **fixed per-request cost** (orchestration, hand-off,
  encode/decode). Supports "component/stage organisation matters"; still not proof the *graph
  abstraction* is the cause -- a stage-split M* placement would be needed to isolate that.
* ``Δb * x0 / gap >= 0.6`` -> mostly **per-step compute** (kernels, CFG batching). The abstraction
  is not the explanation.
* otherwise -> mixed; report both shares.
* ``R² < 0.98`` for an engine -> E2E is not linear in x there; do not report a/b for it.

What this cannot do: it localises *where* the gap lives, not *why*. It is a single-GPU, C=1,
256x256 experiment and says nothing about CFG parallelism across GPUs.

Usage::

    EXPS_CAMPAIGN_ID=BAGEL_ABLATION_01 \
      envs/mstar/bin/python exps/tools/analyze_bagel_ablation.py [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths
from exps.tasks.bagel.cases import (
    ABLATION_I2T_TOKENS,
    ABLATION_RESOLUTION,
    ABLATION_STEPS,
    GRAPH_I2T_TOKENS,
    I2T_MAX_TOKENS,
    SERVER_STEPS,
    TIMESTEP_POINTS,
)

RES = ABLATION_RESOLUTION
SWEEPS = {
    # task -> (knob name, baseline x, [(x, case name)])
    "t2i": ("denoise_steps", TIMESTEP_POINTS,
            [(s, f"t2i-{RES}-steps{s}") for s in ABLATION_STEPS] + [(TIMESTEP_POINTS, f"t2i-{RES}-default")]),
    "i2i": ("denoise_steps", TIMESTEP_POINTS,
            [(s, f"i2i-{RES}-steps{s}") for s in ABLATION_STEPS] + [(TIMESTEP_POINTS, f"i2i-{RES}")]),
    "i2t": ("completion_tokens", I2T_MAX_TOKENS,
            [(t, f"i2t-{RES}-tokens{t}") for t in ABLATION_I2T_TOKENS] + [(I2T_MAX_TOKENS, f"i2t-{RES}")]),
}
CFG_OFF_CASES = {"t2i": (f"t2i-{RES}-nocfg", f"t2i-{RES}-default"), "i2i": (f"i2i-{RES}-nocfg", f"i2i-{RES}")}
GRAPH_FW = "vllm-omni-single-stage+graph"  # virtual series: single-stage with enforce_eager=false


def _baseline_case(task: str) -> str:
    """The unmodified request for a sweep's x0: the default T2I request is `t2i-<res>-default` (the plain name is gone)."""
    return f"{task}-{RES}-default" if task == "t2i" else f"{task}-{RES}"


def series_cases(task: str, fw: str) -> tuple[str, list[tuple[int, str]]] | None:
    """(real framework, [(x, case)]) for one series of a sweep.

    * M* steps: request-level `-steps{N}` cases are inert (M* takes `num_timesteps` from the server
      config), so M* uses the server-side `-srv-steps{N}` cases plus the ordinary 50-step baseline.
    * `vllm-omni-single-stage+graph`: the I2T token sweep run with enforce_eager=false.
    """
    knob, x0, cases = SWEEPS[task]
    if fw == GRAPH_FW:
        if task != "i2t":
            return None
        return "vllm-omni-single-stage", [(t, f"i2t-{RES}-graph-tokens{t}") for t in GRAPH_I2T_TOKENS]
    if fw == "mstar" and task in ("t2i", "i2i"):
        return fw, [(s, f"{task}-{RES}-srv-steps{s}") for s in SERVER_STEPS] + [(TIMESTEP_POINTS, _baseline_case(task))]
    return fw, cases


LINEAR_R2_MIN = 0.98
SHARE_THRESHOLD = 0.6


def linear_fit(points: list[tuple[float, float]]) -> dict[str, float] | None:
    """Ordinary least squares ``y = a + b x`` with R²; needs >= 3 distinct x."""
    if len({x for x, _ in points}) < 3:
        return None
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    b = sum((x - mx) * (y - my) for x, y in points) / sxx
    a = my - b * mx
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in points)
    return {"a_ms": a, "b_ms_per_unit": b, "r2": 1.0 - ss_res / ss_tot if ss_tot else 1.0, "n_points": len(points)}


def collect_p50s(task_root: Path) -> dict[tuple[str, str], list[float]]:
    """(framework, case) -> per-start p50 E2E, completed cells only. Starts are never pooled."""
    found: dict[tuple[str, str], list[float]] = defaultdict(list)
    for manifest in sorted(task_root.glob("*/*/*/manifest.json")):
        cell = json.loads(manifest.read_text(encoding="utf-8"))
        summary_path = manifest.with_name("summary.json")
        if cell.get("status") != "completed" or not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        p50 = (summary.get("results", {}).get("metrics", {}).get("e2e_ms") or {}).get("p50")
        if p50 is not None:
            found[(cell["framework"], cell["case"])].append(float(p50))
    return found


def analyze(task_root: Path) -> dict[str, Any]:
    p50s = collect_p50s(task_root)
    out: dict[str, Any] = {"sweeps": {}, "cfg_off": {}, "campaign_root": str(task_root)}
    frameworks = sorted({fw for fw, _ in p50s}) + [GRAPH_FW]

    for task, (knob, x0, cases) in SWEEPS.items():
        per_fw: dict[str, Any] = {}
        for fw in frameworks:
            resolved = series_cases(task, fw)
            if resolved is None:
                continue
            real_fw, fw_cases = resolved
            points, starts, ranges = [], {}, {}
            for x, case in fw_cases:
                values = p50s.get((real_fw, case))
                if values:
                    points.append((float(x), statistics.median(values)))
                    starts[x] = len(values)
                    ranges[int(x)] = [min(values), max(values)]
            if not points:
                continue
            fit = linear_fit(points)
            per_fw[fw] = {
                "points_e2e_p50_ms": {int(x): y for x, y in points},
                "starts_per_point": starts,
                "start_p50_range_ms": ranges,
                "fit": fit,
                "fit_usable": bool(fit and fit["r2"] >= LINEAR_R2_MIN),
            }
        out["sweeps"][task] = {"knob": knob, "baseline_x": x0, "frameworks": per_fw,
                               "gap_decomposition": gap_decomposition(per_fw, x0)}

    # enforce_eager=false control for the single-stage recipe: same case, graph deploy config.
    for task in ("t2i", "i2i"):
        base, graph = p50s.get(("vllm-omni-single-stage", _baseline_case(task))), p50s.get(
            ("vllm-omni-single-stage", f"{task}-{RES}-graph"))
        if base and graph:
            out.setdefault("eager_control", {})[task] = {
                "eager_p50_ms": statistics.median(base), "graph_p50_ms": statistics.median(graph),
                "ratio_graph_over_eager": statistics.median(graph) / statistics.median(base),
                "starts": {"eager": len(base), "graph": len(graph)},
            }
    for task, (off_case, base_case) in CFG_OFF_CASES.items():
        rows = {}
        for fw in frameworks:
            off, base = p50s.get((fw, off_case)), p50s.get((fw, base_case))
            if off and base:
                rows[fw] = {
                    "e2e_cfg_off_ms": statistics.median(off),
                    "e2e_baseline_ms": statistics.median(base),
                    "ratio_off_over_baseline": statistics.median(off) / statistics.median(base),
                }
        if rows:
            out["cfg_off"][task] = rows
    return out


def gap_decomposition(per_fw: dict[str, Any], x0: float, reference: str = "mstar") -> dict[str, Any]:
    """Split the gap to the reference engine into fixed (Δa) and marginal (Δb·x0) parts."""
    ref = per_fw.get(reference)
    result: dict[str, Any] = {}
    if not ref or not ref["fit_usable"]:
        return {"status": "reference_fit_unusable"}
    for fw, row in per_fw.items():
        if fw == reference:
            continue
        if not row["fit_usable"]:
            result[fw] = {"status": "fit_unusable", "r2": (row["fit"] or {}).get("r2")}
            continue
        da = row["fit"]["a_ms"] - ref["fit"]["a_ms"]
        db_x0 = (row["fit"]["b_ms_per_unit"] - ref["fit"]["b_ms_per_unit"]) * x0
        gap = da + db_x0
        if gap <= 0:
            result[fw] = {"status": "reference_not_faster", "gap_ms": gap}
            continue
        fixed_share, marginal_share = da / gap, db_x0 / gap
        if fixed_share >= SHARE_THRESHOLD:
            reading = "mostly_fixed_per_request_cost"
        elif marginal_share >= SHARE_THRESHOLD:
            reading = "mostly_per_step_or_per_token_cost"
        else:
            reading = "mixed"
        result[fw] = {"status": "ok", "gap_ms_at_baseline": gap, "delta_a_ms": da,
                      "delta_b_times_x0_ms": db_x0, "fixed_share": fixed_share,
                      "marginal_share": marginal_share, "reading": reading}
    return result


def to_markdown(report: dict[str, Any]) -> str:
    lines = ["# BAGEL ablation: fixed vs marginal cost", ""]
    for task, sweep in report["sweeps"].items():
        lines += [f"## {task.upper()} — E2E p50 vs {sweep['knob']} (median over starts, ms)", "",
                  "| framework | points | a (fixed ms) | b (ms/unit) | R² | usable |", "|---|---|---:|---:|---:|---|"]
        for fw, row in sweep["frameworks"].items():
            fit = row["fit"] or {}
            rng = row.get("start_p50_range_ms", {})
            pts = ", ".join(
                f"{k}:{v:,.0f}" + (f" [{rng[k][0]:,.0f}–{rng[k][1]:,.0f}]" if k in rng and rng[k][0] != rng[k][1] else "")
                for k, v in row["points_e2e_p50_ms"].items())
            lines.append(f"| {fw} | {pts} | {fit.get('a_ms', float('nan')):,.0f} | "
                         f"{fit.get('b_ms_per_unit', float('nan')):,.1f} | {fit.get('r2', float('nan')):.3f} | "
                         f"{'yes' if row['fit_usable'] else 'no'} |")
        lines += ["", "Gap to M* at baseline:", ""]
        for fw, d in sweep["gap_decomposition"].items():
            if isinstance(d, dict) and d.get("status") == "ok":
                lines.append(f"- {fw}: gap {d['gap_ms_at_baseline']:,.0f} ms = fixed {d['delta_a_ms']:,.0f} "
                             f"({d['fixed_share']:.0%}) + marginal {d['delta_b_times_x0_ms']:,.0f} "
                             f"({d['marginal_share']:.0%}) → **{d['reading']}**")
            else:
                lines.append(f"- {fw}: {d}")
        lines.append("")
    if report["cfg_off"]:
        lines += ["## CFG off / baseline E2E ratio", ""]
        for task, rows in report["cfg_off"].items():
            for fw, r in rows.items():
                lines.append(f"- {task} {fw}: {r['e2e_cfg_off_ms']:,.0f} / {r['e2e_baseline_ms']:,.0f} "
                             f"= {r['ratio_off_over_baseline']:.2f}")
        lines.append("")
    if report.get("eager_control"):
        lines += ["## vLLM single-stage: enforce_eager=false vs true (same case, 50 steps)", ""]
        for task, r in report["eager_control"].items():
            lines.append(f"- {task}: eager {r['eager_p50_ms']:,.0f} ms, graph {r['graph_p50_ms']:,.0f} ms "
                         f"(ratio {r['ratio_graph_over_eager']:.2f}; starts {r['starts']['eager']}/{r['starts']['graph']})")
        lines.append("")
    lines += ["Limits: single GPU, C=1, 256x256; localises the gap, does not prove the graph abstraction "
              "causes it. See the module docstring for the pre-registered reading."]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    task_root = paths.campaign_root(args.campaign) / "BAGEL"
    if not task_root.is_dir():
        print(f"no BAGEL cells under {task_root}", file=sys.stderr)
        return 2
    report = analyze(task_root)
    text = to_markdown(report)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        args.markdown.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
