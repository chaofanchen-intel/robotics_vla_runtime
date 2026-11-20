"""Level 1 metrics of one cell, from its request records and control-tick records.

Per request: E2E, server time, the age of every input at the snapshot, the replan interval, the actions skipped on
arrival. Per control tick: queue depth, underrun (no action left, the robot holds), the age of the executed action since
its snapshot, how late it runs against the time it was planned for (intent lag), and the staleness of each input when
its action is executed. All times are milliseconds.
"""
from __future__ import annotations

from metrics.stats import runs as _runs
from metrics.stats import stats
from runtime.observation import AGE_STREAMS


def summarize(result, dt: float, unused_superseded: int = 0) -> dict:
    reqs, ticks = result.requests, result.ticks
    st = stats
    s: dict = {"control_dt_ms": dt * 1000.0, "ticks": len(ticks)}
    s["requests"] = {"n": len(reqs), "skipped_triggers": result.skipped_triggers,
                     "per_s": len(reqs) / max(1e-9, result.config["duration_s"]),
                     "applied": sum(1 for r in reqs if r["applied"]), "unused_superseded": unused_superseded}
    s["e2e_ms"] = st([r["e2e_ms"] for r in reqs])
    s["infer_ms"] = st([r["infer_ms"] for r in reqs if r["infer_ms"] is not None])
    s["pack_ms"] = st([r["pack_ms"] for r in reqs])
    s["snap_to_send_ms"] = st([r["snap_to_send_ms"] for r in reqs])
    s["input_age_at_snapshot_ms"] = {n: st([r["age_ms"][n] for r in reqs]) for n in AGE_STREAMS}
    s["input_age_at_snapshot_ms"]["oldest"] = st([max(r["age_ms"].values()) for r in reqs])
    applied = sorted((r for r in reqs if r["applied"]), key=lambda r: r["t_snap"])
    s["replan_interval_ms"] = st([(b["t_snap"] - a["t_snap"]) * 1000.0 for a, b in zip(applied, applied[1:])])
    skipped = [r["skipped_at_first_use"] for r in reqs if r["skipped_at_first_use"] is not None]
    s["skipped_actions_at_first_use"] = st([float(x) for x in skipped])
    exec_ticks = [t for t in ticks if not t["underrun"]]
    under = [t["underrun"] for t in ticks]
    runs = _runs(under)
    s["underrun"] = {"ticks": int(sum(under)), "fraction": (sum(under) / len(ticks)) if ticks else None,
                     "episodes": len(runs), "longest_ms": (max(runs) * dt * 1000.0) if runs else 0.0}
    s["action_age_ms"] = st([t["action_age_ms"] for t in exec_ticks])
    s["intent_lag_ms"] = st([t["intent_lag_ms"] for t in exec_ticks])
    s["staleness_at_execution_ms"] = {n: st([t[f"stale_{n}_ms"] for t in exec_ticks]) for n in AGE_STREAMS}
    s["staleness_at_execution_ms"]["oldest"] = st([max(t[f"stale_{n}_ms"] for n in AGE_STREAMS) for t in exec_ticks])
    s["queue_remaining_actions"] = st([float(t["remaining"]) for t in ticks])
    s["tick_lateness_ms"] = st([t["late_ms"] for t in ticks])
    return s
