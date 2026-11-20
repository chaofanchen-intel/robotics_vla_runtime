"""Calibration never recommends an unmeasured target."""
from __future__ import annotations

import json

import exps.calibrate as calibrate
from exps.calibrate import recommendation


def sample(start, p50, *, drift=False):
    return {
        "status": "measured", "requests": 3,
        "delta_temperature_per_request_c": 2,
        "cooldown_waited_s": [120, 150, 110],
        "start_temperature_c": start, "p50_e2e_ms": p50,
        "drift": {"drifting": drift},
    }


def test_recommendation_uses_observed_stable_target():
    data = {
        55: sample(55, 100),
        60: sample(60, 102),
        65: sample(65, 108),
        70: sample(70, 110),
        75: sample(75, 115),
    }
    result = recommendation(data, 3)
    assert result["cooldown_target_c"] == 60
    assert result["measured"] == 3
    assert result["cooldown_timeout_s"] >= 900
    assert result["sensitivity"]["slope_ms_per_c"] is not None


def test_recommendation_refuses_insufficient_temperature_range():
    data = {55: sample(55, 100), 60: sample(60, 101, drift=True), 65: sample(65, 102)}
    assert recommendation(data, 3)["status"] == "insufficient_evidence"


def test_calibration_finishes_priority_task_before_next_task(monkeypatch, tmp_path):
    calls = []
    progress = []
    output = tmp_path / "calibrated.json"

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs["env"]["EXPS_CAMPAIGN_ID"]))
        if argv[argv.index("--tasks") + 1] == "qwen_t2s":
            progress.append(json.loads(output.read_text()))
        return type("Completed", (), {"returncode": 0})()

    monkeypatch.setattr(calibrate.subprocess, "run", fake_run)
    monkeypatch.setattr("sys.argv", [
        "calibrate.py", "--execute", "--tasks", "bagel", "qwen_t2s",
        "--targets", "60", "65", "70", "--output", str(output),
    ])
    assert calibrate.main() == 1  # no GPU results were written by the fake subprocess
    assert [
        (argv[argv.index("--tasks") + 1], argv[argv.index("--cooldown-target-c") + 1])
        for argv, _ in calls
    ] == [
        ("bagel", "60.0"), ("bagel", "65.0"), ("bagel", "70.0"),
        ("qwen_t2s", "60.0"), ("qwen_t2s", "65.0"), ("qwen_t2s", "70.0"),
    ]
    assert len({campaign for _, campaign in calls}) == 6
    assert progress[0]["status"] == "in_progress"
    assert progress[0]["completed_tasks"] == ["bagel"]
    assert list(progress[0]["evidence"]) == ["bagel"]
    final = json.loads(output.read_text())
    assert final["status"] == "no_recommendations"
    assert final["completed_tasks"] == ["bagel", "qwen_t2s"]
    assert list(final["campaigns"]) == ["bagel", "qwen_t2s"]
