"""CPU tests for the round-6 tools: torch recompile log parser and the M* codec-chunk hook."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "exps" / "tools"))

import analyze_torch_recompiles as art  # noqa: E402

P = "V1025 20:21:{sec} 123 torch/_dynamo/guards.py:4181] [{n}] [__recompiles]"
LOG = "\n".join([
    P.format(sec="11.234000", n="22/1") + " Recompiling function sample in /x/sampler.py:40",
    P.format(sec="11.234000", n="22/1") + "     triggered by the following guard failure(s):",
    P.format(sec="11.234000", n="22/1") + "     - 22/0: request_ids[0] == '51029c01-201d-4d68-8a7b-fb527c66d862'  # rid: {...}  # x.py:657",
    P.format(sec="11.234000", n="22/1") + "     - 22/0: len(request_ids) == 16   # for rid, token in zip(...)",
    "I1025 20:21:12.000000 123 other.py:1] unrelated line",
    P.format(sec="15.100000", n="22/2") + " Recompiling function sample in /x/sampler.py:40",
    P.format(sec="15.100000", n="22/2") + "     triggered by the following guard failure(s):",
    P.format(sec="15.100000", n="22/2") + "     - 22/1: request_ids[0] == '7f6c6587-a4c9-4d03-8e45-4258c261656a'  # rid: {...}  # x.py:657",
    "V1025 20:21:16.000000 123 torch/_dynamo/symbolic_convert.py:611] [0/0] [__graph_breaks] Graph break in user code at /x/llm.py:120",
])


def test_parse_counts_recompiles_reasons_and_breaks():
    found = art.parse(LOG.splitlines(), 2026)
    assert [r["function"] for r in found["recompiles"]] == ["sample", "sample"]
    assert len(found["graph_breaks"]) == 1
    (reason, count), = found["reasons"].most_common(1)
    assert reason.startswith("request_ids[N] == '<uuid>'") and count == 2     # request ids folded: one reason for both
    assert len(found["recompiles"][0]["reasons"]) == 2


def test_event_in_the_same_second_as_the_request_start_is_inside_the_window():
    # regression: events were truncated to whole seconds, so an event at :11.234 fell before a request that started at :11.000
    # + 0.1 s and the real log reported "0 recompiles" for requests that recompiled every time
    found = art.parse(LOG.splitlines(), 2026)
    span = (datetime(2026, 10, 25, 20, 21, 11, 100000), datetime(2026, 10, 25, 20, 21, 13))
    assert art.markdown("cell", found, [span]).count("inside each measured request: 1") == 1


def test_markdown_splits_recompiles_by_measured_request():
    found = art.parse(LOG.splitlines(), 2026)
    spans = [(datetime(2026, 10, 25, 20, 21, 10), datetime(2026, 10, 25, 20, 21, 13)),
             (datetime(2026, 10, 25, 20, 21, 14), datetime(2026, 10, 25, 20, 21, 17))]
    text = art.markdown("cell", found, spans)
    assert "recompile events: 2; graph breaks: 1" in text
    assert "recompiles inside each measured request: 1, 1" in text
    assert "graph breaks inside each measured request: 0, 1" in text
    assert "| sample | 2 | 1.0 |" in text


def test_markdown_without_lines_says_what_it_means():
    text = art.markdown("cell", art.parse([], 2026), [])
    assert "never recompiled" in text and "TORCH_LOGS did not reach" in text


def test_windows_reads_measured_successes_only(tmp_path):
    f = tmp_path / "requests.jsonl"
    rows = [{"phase": "warmup", "success": True, "started_at_utc": "2026-10-25T10:00:00+00:00", "e2e_ms": 5},
            {"phase": "measured", "success": True, "started_at_utc": "2026-10-25T10:00:01+00:00", "e2e_ms": 1000},
            {"phase": "measured", "success": False, "started_at_utc": "2026-10-25T10:00:02+00:00", "e2e_ms": 1000}]
    f.write_text("\n".join(json.dumps(r) for r in rows))
    spans = art.windows(f)
    assert len(spans) == 1 and (spans[0][1] - spans[0][0]).total_seconds() == 1.0


MSTAR_PYTHON = ROOT / "envs" / "mstar" / "bin" / "python"


@pytest.mark.skipif(not MSTAR_PYTHON.exists(), reason="M* interpreter missing")
def test_codec_chunk_hook_changes_the_real_code2wav_config(tmp_path):
    log = tmp_path / "hook.jsonl"
    hook = ROOT / "exps" / "tools" / "mstar_codec_chunk_hook"
    probe = ("from mstar.model.qwen3_omni.config import Code2WavConfig as C; "
             "c = C.from_dict({}); print(c.codec_chunk_frames, c.codec_left_context_frames)")

    def run(**env):
        merged = {k: v for k, v in os.environ.items() if not k.startswith("MSTAR_QWEN_CODEC")}
        merged["PYTHONPATH"] = os.pathsep.join([str(ROOT / "external" / "mstar"), *([str(hook)] if env else [])])
        merged.update(env)
        return subprocess.run([str(MSTAR_PYTHON), "-c", probe], capture_output=True, text=True, cwd=tmp_path,
                              env=merged, check=True).stdout.split()

    assert run() == ["25", "25"]                                       # untouched default
    assert run(MSTAR_QWEN_CODEC_CHUNK_FRAMES="8", MSTAR_QWEN_CODEC_LOG=str(log)) == ["8", "8"]   # left <= chunk
    assert run(MSTAR_QWEN_CODEC_CHUNK_FRAMES="100") == ["100", "25"]
    assert json.loads(log.read_text().splitlines()[0])["codec_chunk_frames"] == 8


REPO = ROOT




