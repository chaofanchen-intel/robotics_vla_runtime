"""Percentile summaries shared by the probes and the Level 1 metrics."""
from __future__ import annotations

import statistics


def stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round(p / 100 * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "n": len(ordered),
        "p50": statistics.median(ordered),
        "p95": pct(95),
        "p99": pct(99),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def runs(flags: list[bool]) -> list[int]:
    """Lengths of the consecutive True runs in `flags`."""
    out, n = [], 0
    for f in flags:
        if f:
            n += 1
        elif n:
            out.append(n)
            n = 0
    if n:
        out.append(n)
    return out
