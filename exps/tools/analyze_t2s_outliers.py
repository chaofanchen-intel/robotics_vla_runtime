#!/usr/bin/env python3
"""Qwen3-Omni T2S: which responses fall outside the speech-rate band, per framework and batch size.

Meant for a campaign run with ``--opt qwen.contract=record`` (responses are flagged, not voided). For every
cell (framework x case) it takes each response's audio duration, divides the fixture sentence's character count
by it, and compares that with the contract band in ``exps/tasks/qwen_t2s/cases.py``. It prints the outlier rate
per cell and lists every outlier with its phase, wave, position and fixture row, so a repeat on the same row
(prompt-specific) can be told apart from a spread over rows (state or sampling).

    envs/mstar/bin/python exps/tools/analyze_t2s_outliers.py --campaign T2S_CONC_OUTLIERS_RECORD_01_B2B [--markdown out.md]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from exps.tasks.qwen_t2s.cases import CHARS_PER_SECOND  # noqa: E402


def _cells(campaign_dir: Path):
    for summary in sorted(campaign_dir.glob("*/*/*/r*/summary.json")):
        yield summary.parent


def _rows(fixture: Path) -> list[dict]:
    return [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]


def analyse_cell(cell: Path) -> dict | None:
    summary = json.loads((cell / "summary.json").read_text())
    requests = cell / "requests.jsonl"
    if not requests.exists():
        return None
    fixture = _rows(REPO / summary["fixture"])
    used = summary.get("fixture_rows_used") or []  # fixture_id of every request in order; exact under any row order (shuffle)
    text_by_id = {r.get("fixture_id"): r["synthesis_text"] for r in fixture if r.get("fixture_id")}
    low_rate, high_rate = CHARS_PER_SECOND[summary.get("language", "en")]
    batch = int(summary.get("batch_size") or 1)
    items = []
    wave_no = {"precondition": 0, "measured": 0}
    for line in requests.read_text().splitlines():
        row = json.loads(line)
        if not row.get("success", True) or row.get("phase") not in wave_no:
            continue
        wave_no[row["phase"]] += 1
        offset = int(row.get("fixture_index") or 0)
        if batch == 1:
            durations = [row.get("audio_duration_s")]
        else:
            durations = row.get("per_request_audio_duration_s") or []
        for position, duration in enumerate(durations):
            slot = offset + position
            if slot < len(used) and used[slot] in text_by_id:
                index, text = used[slot], text_by_id[used[slot]]
            else:
                index = (slot) % len(fixture)
                text = fixture[index]["synthesis_text"]
            if not duration:
                items.append((row["phase"], wave_no[row["phase"]], position, index, len(text), None, None, "no audio"))
                continue
            cps = len(text) / duration
            verdict = "ok"
            if cps < low_rate:
                verdict = "too long"
            elif cps > high_rate:
                verdict = "too short"
            items.append((row["phase"], wave_no[row["phase"]], position, index, len(text), duration, cps, verdict))
    return {
        "framework": summary["framework"], "case": summary["case"], "batch": batch,
        "status": summary.get("status"), "band_cps": [low_rate, high_rate], "items": items,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--campaign", required=True, help="campaign directory name under exps/runs")
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    campaign_dir = REPO / "exps" / "runs" / args.campaign
    cells = [c for c in (analyse_cell(p) for p in _cells(campaign_dir)) if c]
    if not cells:
        print(f"no cells with requests.jsonl under {campaign_dir}", file=sys.stderr)
        return 2

    lines = [
        "# Qwen3-Omni T2S: responses outside the speech-rate band (record mode)",
        "",
        "Band = characters of the target sentence per second of audio. `too long` = audio much longer than the sentence "
        "needs (runaway Talker), `too short` = truncated or implausibly fast.",
        "",
        "| framework | case | B | responses | outliers | rate | too long | too short | duration range s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    detail: list[str] = []
    served: Counter = Counter()
    failed: Counter = Counter()
    for c in cells:
        for i in c["items"]:
            served[(c["framework"], i[3], c["batch"])] += 1
            if i[7] != "ok":
                failed[(c["framework"], i[3], c["batch"])] += 1
    for cell in sorted(cells, key=lambda c: (c["framework"], c["batch"])):
        items = cell["items"]
        bad = [i for i in items if i[7] != "ok"]
        durations = [i[5] for i in items if i[5]]
        lines.append(
            f"| {cell['framework']} | {cell['case']} | {cell['batch']} | {len(items)} | {len(bad)} | "
            f"{(100 * len(bad) / len(items)) if items else 0:.1f}% | {sum(1 for i in bad if i[7] == 'too long')} | "
            f"{sum(1 for i in bad if i[7] == 'too short')} | "
            f"{(f'{min(durations):.1f}-{max(durations):.1f}') if durations else '-'} |"
        )
        for phase, wave, position, index, chars, duration, cps, verdict in bad:
            detail.append(
                f"| {cell['framework']} | {cell['case']} | {phase} | {wave} | {position} | {index} | {chars} | "
                f"{'-' if duration is None else f'{duration:.2f}'} | {'-' if cps is None else f'{cps:.1f}'} | {verdict} |"
            )
    if detail:
        lines += [
            "", "## Outliers", "",
            "| framework | case | phase | wave | position | fixture row | chars | audio s | chars/s | verdict |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
            *detail,
        ]
        repeats = Counter((c["framework"], i[3]) for c in cells for i in c["items"] if i[7] != "ok")
        again = {k: v for k, v in repeats.items() if v > 1}
        bad_rows = sorted({(fw, row) for (fw, row, _b) in failed})
        batches = sorted({c["batch"] for c in cells})
        lines += ["", "## Outlier rows across batch sizes (failed / served)", "",
                  "| framework | fixture row | " + " | ".join(f"B={b}" for b in batches) + " |", "|---|---|" + "---|" * len(batches)]
        for fw, row in bad_rows:
            cols = [f"{failed[(fw, row, b)]} / {served[(fw, row, b)]}" for b in batches]
            lines.append(f"| {fw} | {row} | " + " | ".join(cols) + " |")
        lines += ["", "Fixture rows that failed more than once (same framework): "
                  + (", ".join(f"{fw} row {row} x{n}" for (fw, row), n in sorted(again.items())) or "none")]
    else:
        lines += ["", "No response was outside the band."]
    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
