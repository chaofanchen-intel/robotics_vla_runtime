#!/usr/bin/env python3
"""Aggregate the Level 1 runs: <run-dir>/<framework>/level1_r<repeat>.json -> level1_summary.md and level1_summary.json.

Every number is the median over repeats of that repeat's statistic; the bracket is the smallest and largest repeat.
A repeat is one server start and one pass over the cells, so the bracket includes restart-to-restart and host-load noise.

    envs/mstar/bin/python scripts/level1_report.py --run-dir runs/<date>_level1
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))            # exps/
sys.path.insert(0, str(REPO / "src"))   # runtime, backends, metrics, scenarios

import backends  # noqa: E402

FRAMEWORKS = backends.LEVEL1_ALL
NAMES = {n: b.label for n, b in backends.ALL.items()}
CELL_ORDER = ["period-100ms", "period-200ms", "period-300ms", "openloop-8", "threshold-0.3s", "threshold-0.5s",
              "threshold-0.8s", "openloop-8-naive", "threshold-0.5s@lockstep", "period-200ms@lockstep"]
STREAMS = ("state", "ext1", "wrist", "ext2")


def episode_bounds() -> list[float]:
    """End time (s) of each episode on the replay timeline, from the episode lengths of droid_100 at 15 Hz."""
    import pyarrow.parquet as pq

    from runtime.dataset import DATASET, DEFAULT_EPISODES, NATIVE_HZ
    eps = pq.read_table(DATASET / "data" / "chunk-000" / "file-000.parquet", columns=["episode_index"]).to_pandas()
    counts = eps["episode_index"].value_counts()
    ends, acc = [], 0.0
    for e in DEFAULT_EPISODES:
        acc += counts[e] / NATIVE_HZ
        ends.append(acc)
    return ends


_BOUNDS: list[float] = []


def e2e_by_episode(requests: list[dict]) -> dict[int, list[float]]:
    if not _BOUNDS:
        _BOUNDS.extend(episode_bounds())
    out: dict[int, list[float]] = {}
    for r in requests:
        ep = next((i for i, b in enumerate(_BOUNDS) if r["t_snap"] < b), len(_BOUNDS))
        out.setdefault(ep, []).append(r["e2e_ms"])
    return out


def load(run_dir: Path, only: dict[str, set[str]] | None = None) -> dict[str, dict[str, list[dict]]]:
    """Cells of every framework in `run_dir`. `only` limits the labels per framework (used for the overriding runs)."""
    data: dict[str, dict[str, list[dict]]] = {}
    for fw in FRAMEWORKS:
        for f in sorted((run_dir / fw).glob("level1_r*.json")):
            doc = json.loads(f.read_text())
            for cell in doc["cells"]:
                if only is not None and cell["label"] not in only.get(fw, ()):
                    continue
                data.setdefault(fw, {}).setdefault(cell["label"], []).append(
                    {"repeat": doc["repeat_index"], "summary": cell["summary"], "load": cell.get("host_loadavg_1m"),
                     "skipped_triggers": cell["skipped_triggers"],
                     "e2e_by_episode": e2e_by_episode(cell["requests"])})
    return data


def get(summary: dict, path: str):
    cur = summary
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def agg(cells: list[dict], path: str):
    vals = [v for v in (get(c["summary"], path) for c in cells) if isinstance(v, (int, float))]
    if not vals:
        return None
    return statistics.median(vals), min(vals), max(vals), len(vals)


def fmt(a, digits: int = 0, pct: bool = False) -> str:
    if a is None:
        return "—"
    m, lo, hi, n = a
    k = 100.0 if pct else 1.0
    u = "%" if pct else ""
    if n == 1:
        return f"{m * k:.{digits}f}{u}"
    return f"{m * k:.{digits}f}{u} [{lo * k:.{digits}f}–{hi * k:.{digits}f}]"


def table(data, cells: list[str], columns: list[tuple[str, str, int, bool]], title: str) -> list[str]:
    lines = [f"### {title}", "", "| 框架 | cell | " + " | ".join(c[0] for c in columns) + " |",
             "|---|---|" + "|".join("---" for _ in columns) + "|"]
    for label in cells:
        for fw in FRAMEWORKS:
            reps = data.get(fw, {}).get(label)
            if not reps:
                continue
            row = [fmt(agg(reps, path), d, pct) for _h, path, d, pct in columns]
            lines.append(f"| {NAMES[fw]} | {label} | " + " | ".join(row) + " |")
    lines.append("")
    return lines


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--extra-dir", type=Path, nargs="*", default=[],
                   help="further run directories; their cells replace the same (framework, cell) of --run-dir")
    args = p.parse_args()
    data = load(args.run_dir)
    replaced: list[str] = []
    for extra in args.extra_dir:
        for fw, cells in load(extra).items():
            for label, reps_ in cells.items():
                if label in data.get(fw, {}):
                    replaced.append(f"{NAMES[fw]} {label} ← {extra.name}")
                data.setdefault(fw, {})[label] = reps_
    reps = {fw: sorted({c["repeat"] for cells in data.get(fw, {}).values() for c in cells}) for fw in FRAMEWORKS}
    loads = [x for fw in data.values() for cells in fw.values() for c in cells for x in (c["load"] or [])]
    lines = [f"# Level 1 汇总：{args.run_dir.name}", "",
             "每个数是各轮统计量的中位数，方括号是最小到最大的一轮。一轮 = 一次服务启动 + 一遍全部 cell，所以范围包含启动之间的差别和主机负载噪声。",
             f"轮数：" + "，".join(f"{NAMES[fw]} {len(reps[fw])}" for fw in FRAMEWORKS) + "。"
             + (f" 主机 1 分钟 load average 在 {min(loads):.0f}–{max(loads):.0f} 之间（256 核，其他用户的作业在同一台机器上）。" if loads else ""),
             ""]
    if args.extra_dir:
        lines += ["补测目录：" + "，".join(d.name for d in args.extra_dir) + "。"
                  + (f" 被替换的 cell：" + "；".join(replaced) + "。" if replaced else ""), ""]
    main_cells = [c for c in CELL_ORDER if "@" not in c and "naive" not in c]
    lines += table(data, main_cells, [
        ("请求/s", "requests.per_s", 1, False), ("跳过的触发", "requests.skipped_triggers", 0, False),
        ("E2E p50 (ms)", "e2e_ms.p50", 0, False), ("E2E p95 (ms)", "e2e_ms.p95", 0, False),
        ("重规划间隔 p50 (ms)", "replan_interval_ms.p50", 0, False)], "触发策略：请求与端到端（多频输入，对齐执行）")
    lines += table(data, main_cells, [
        ("动作年龄 p50 (ms)", "action_age_ms.p50", 0, False), ("动作年龄 p95 (ms)", "action_age_ms.p95", 0, False),
        ("最旧输入的陈旧度 p50 (ms)", "staleness_at_execution_ms.oldest.p50", 0, False),
        ("最旧输入的陈旧度 p95 (ms)", "staleness_at_execution_ms.oldest.p95", 0, False),
        ("欠载 tick 占比", "underrun.fraction", 1, True), ("最长欠载 (ms)", "underrun.longest_ms", 0, False)],
        "触发策略：动作新鲜度与欠载（多频输入，对齐执行）")
    lines += table(data, ["openloop-8", "openloop-8-naive"], [
        ("动作年龄 p50 (ms)", "action_age_ms.p50", 0, False), ("预定时刻滞后 p50 (ms)", "intent_lag_ms.p50", 0, False),
        ("预定时刻滞后 p95 (ms)", "intent_lag_ms.p95", 0, False), ("欠载 tick 占比", "underrun.fraction", 1, True)],
        "对齐执行与从动作 0 开始（OpenPI 参考客户端的做法）")
    lines += table(data, ["threshold-0.5s", "threshold-0.5s@lockstep", "period-200ms", "period-200ms@lockstep"], [
        ("快照时最旧输入的年龄 p50 (ms)", "input_age_at_snapshot_ms.oldest.p50", 0, False),
        ("快照时最旧输入的年龄 p95 (ms)", "input_age_at_snapshot_ms.oldest.p95", 0, False),
        ("执行时最旧输入的陈旧度 p50 (ms)", "staleness_at_execution_ms.oldest.p50", 0, False),
        ("执行时最旧输入的陈旧度 p95 (ms)", "staleness_at_execution_ms.oldest.p95", 0, False)],
        "多频输入与同步输入（lockstep）：输入年龄")
    lines += table(data, ["threshold-0.5s"], [(f"{s} 年龄 p50 / p95 (ms)", f"input_age_at_snapshot_ms.{s}.p50", 0, False)
                                              for s in STREAMS], "快照时每一路输入的年龄 p50（threshold-0.5s，多频输入）")
    lines += table(data, ["threshold-0.5s"], [(f"{s} 年龄 p95 (ms)", f"input_age_at_snapshot_ms.{s}.p95", 0, False)
                                              for s in STREAMS], "快照时每一路输入的年龄 p95（threshold-0.5s，多频输入）")
    lines += table(data, CELL_ORDER, [("控制 tick 迟到 p95 (ms)", "tick_lateness_ms.p95", 1, False),
                                      ("控制 tick 迟到 最大 (ms)", "tick_lateness_ms.max", 1, False),
                                      ("打包 p50 (ms)", "pack_ms.p50", 2, False)], "客户端自检：控制时钟与打包开销")
    import pyarrow.parquet as pq
    from runtime.dataset import DATASET, DEFAULT_EPISODES
    tasks = pq.read_table(DATASET / "meta" / "tasks.parquet").to_pandas()
    task_of = {int(v): str(k) for k, v in zip(tasks.index, tasks["task_index"])}
    pool: dict[str, dict[int, list[float]]] = {}
    for fw, cells in data.items():
        for reps_ in cells.values():
            for c in reps_:
                for ep, v in c["e2e_by_episode"].items():
                    pool.setdefault(fw, {}).setdefault(int(ep), []).extend(v)
    eps_shown = sorted({ep for d in pool.values() for ep in d})
    lines += ["### E2E p50 按请求所在的 episode（所有 cell 与各轮合并，ms）", "",
              "每个 episode 的语言指令不同，状态也不同，因此送给模型的提示词长度不同。数值随 episode 变化，说明延迟与请求内容有关。", "",
              "| 框架 | " + " | ".join(f"episode {DEFAULT_EPISODES[e]}" for e in eps_shown if e < len(DEFAULT_EPISODES)) + " |",
              "|---|" + "|".join("---" for e in eps_shown if e < len(DEFAULT_EPISODES)) + "|"]
    for fw in FRAMEWORKS:
        if fw in pool:
            lines.append(f"| {NAMES[fw]} | " + " | ".join(
                f"{statistics.median(pool[fw][e]):.0f} (n={len(pool[fw][e])})" if e in pool[fw] else "—"
                for e in eps_shown if e < len(DEFAULT_EPISODES)) + " |")
    lines.append("")
    text = "\n".join(lines)
    (args.run_dir / "level1_summary.md").write_text(text + "\n")
    out = {fw: {label: {"repeats": [c["repeat"] for c in cells], "loadavg_1m": [c["load"] for c in cells],
                        "summaries": [c["summary"] for c in cells]} for label, cells in d.items()}
           for fw, d in data.items()}
    (args.run_dir / "level1_summary.json").write_text(json.dumps(out))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
