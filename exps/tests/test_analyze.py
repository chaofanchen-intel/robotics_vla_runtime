"""Cross-start comparisons use valid starts, not pooled requests."""
from __future__ import annotations

import json

from exps.core.analyze import analyze
from exps.tasks.pi05.cases import TASK


def write_cell(root, framework, repeat, p50, temp, *, drift=False):
    target = root / "PI05" / framework / "droid-46row" / repeat
    target.mkdir(parents=True)
    (target / "manifest.json").write_text(json.dumps({
        "cell": f"pi05/{framework}/droid-46row/{repeat}",
        "case": "droid-46row", "framework": framework, "status": "completed",
    }))
    (target / "summary.json").write_text(json.dumps({
        "status": "completed",
        "results": {
            "headline_requests": 5,
            "intra_start_drift": {"drifting": drift},
            "metrics": {"e2e_ms": {"p50": p50}},
            "thermal_start_state": {"temperature_c": {"mean": temp}},
        },
    }))


def test_analysis_uses_median_of_starts_and_labels_temperature(tmp_path):
    write_cell(tmp_path, "mstar", "r01", 100, 60)
    write_cell(tmp_path, "mstar", "r02", 200, 61)
    write_cell(tmp_path, "vllm-omni", "r01", 50, 65)
    write_cell(tmp_path, "vllm-omni", "r02", 10, 66, drift=True)
    result = analyze(TASK, root=tmp_path)
    frameworks = result["cases"]["droid-46row"]["frameworks"]
    assert frameworks["mstar"]["headline_p50"] == 150
    assert frameworks["vllm-omni"]["headline_p50"] == 30
    comparison = result["cases"]["droid-46row"]["comparisons"]["vllm-omni"]
    assert comparison["speedup"] == 5
    assert comparison["start_temperature_delta_c"] == 5
    assert "unequal_thermal_start" in comparison["labels"]
    assert "suspect_intra_start_drift" in comparison["labels"]
    assert "preliminary_single_start" not in comparison["labels"]
    assert result["excluded_cells"] == []
