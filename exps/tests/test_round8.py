"""Round 8: T2S contract band / record mode, and the BAGEL max_num_seqs concurrency counterfactual."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from exps.core.codecs import ContractViolation
from exps.suite import catalogue
from exps.tasks.qwen_t2s import contract
from exps.tasks.qwen_t2s.cases import CHARS_PER_SECOND

REPO = Path(__file__).resolve().parents[2]
CONFIGS = REPO / "exps/configs/hardware/rtx5880/vllm_omni"


class _Result:  # what adapters.response_text reads is patched below
    pass


def _verify(monkeypatch, text, duration):
    monkeypatch.setattr(contract, "spoken_text", lambda result: text)
    row = {"synthesis_text": text}
    return contract.verify_verbatim(_Result(), row, language="en", audio_duration_s=duration)


def test_english_band_admits_the_vllm_fast_reading_that_voided_a_cell(monkeypatch):
    sentence = "How about you kick the damn thing straight?"  # 43 chars, vLLM read it in 1.419 s = 30.3 chars/s
    assert CHARS_PER_SECOND["en"][1] >= 36.0
    assert _verify(monkeypatch, sentence, 1.419)["duration_in_band"] is True


def test_long_audio_for_a_short_sentence_still_voids(monkeypatch):
    sentence = "She leaves abruptly during the meeting, and is never seen again."  # 64 chars, M* returned 20.5 s
    with pytest.raises(ContractViolation, match="outside the"):
        _verify(monkeypatch, sentence, 20.5)


def test_contract_modes_are_enforce_and_record():
    from exps.tasks import qwen_common

    assert qwen_common.CONTRACT_MODES == ("enforce", "record")


def test_seqs_configs_differ_from_the_prefix_cache_recipe_only_in_the_thinker_cap():
    base = (CONFIGS / "bagel_two_stage_2gpu_prefix_cache.yaml").read_text().splitlines()
    for n in (8, 16):
        variant = (CONFIGS / f"bagel_two_stage_2gpu_prefix_cache_seqs{n}.yaml").read_text().splitlines()
        changed = [(a, b) for a, b in zip(base, variant) if a != b and not a.startswith("#")]
        assert changed == [("    max_num_seqs: 3", f"    max_num_seqs: {n}")]
        assert len(base) == len(variant)


def test_new_experiments_are_in_the_catalogue_and_parse():
    seqs = catalogue.get("bagel-concurrency-seqs")
    assert [a.name for a in seqs.arms] == ["m-cache-on", "vllm-seqs3", "vllm-seqs8", "vllm-seqs16"]
    out = catalogue.get("t2s-conc-outliers")
    assert "qwen.contract=record" in out.arms[0].opts


def test_outlier_tool_flags_the_runaway_and_the_repeat(tmp_path):
    campaign = tmp_path / "C" / "QWEN_T2S" / "mstar" / "verbatim-en-b4" / "r01"
    campaign.mkdir(parents=True)
    fixture = tmp_path / "fx.jsonl"
    fixture.write_text("\n".join(json.dumps({"synthesis_text": "x" * 60}) for _ in range(4)))
    (campaign / "summary.json").write_text(json.dumps(
        {"framework": "mstar", "case": "verbatim-en-b4", "batch_size": 4, "language": "en", "status": "completed",
         "fixture": str(fixture)}))
    waves = [
        {"phase": "measured", "success": True, "fixture_index": 0, "per_request_audio_duration_s": [4.0, 20.0, 4.0, 4.0]},
        {"phase": "measured", "success": True, "fixture_index": 4, "per_request_audio_duration_s": [4.0, 21.0, 4.0, 4.0]},
    ]
    (campaign / "requests.jsonl").write_text("\n".join(json.dumps(w) for w in waves))
    # the tool resolves the fixture relative to the repo; an absolute path works with Path division
    proc = subprocess.run([sys.executable, str(REPO / "exps/tools/analyze_t2s_outliers.py"), "--campaign", str(tmp_path / "C")],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 0, proc.stderr
    assert "| mstar | verbatim-en-b4 | 4 | 8 | 2 | 25.0% | 2 | 0 |" in proc.stdout
    assert "mstar row 1 x2" in proc.stdout  # position 1 of waves at offsets 0 and 4 is fixture row 1 both times (4 rows)


def test_shuffled_rows_cover_each_row_per_pass_and_differ_between_passes():
    from exps.tasks.qwen_t2s.client import shuffled_rows

    rows = [{"fixture_id": f"r{i}"} for i in range(10)]
    picked = [r["fixture_id"] for r in shuffled_rows(rows, 25)]
    assert sorted(picked[:10]) == sorted(r["fixture_id"] for r in rows)
    assert sorted(picked[10:20]) == sorted(r["fixture_id"] for r in rows)
    assert picked[:10] != picked[10:20]
    assert picked == [r["fixture_id"] for r in shuffled_rows(rows, 25)]  # seeded
    from exps.tasks import qwen_common

    assert "shuffle" in qwen_common.ROWS_MODES


def test_i2i_concurrency_cases_and_experiment():
    from exps.tasks.bagel import cases

    names = {c.name for c in cases._all_cases()}
    assert {"i2i-512-steps5-b4", "i2i-512-steps5-b16", "i2i-512-srv-steps5-b4", "i2i-512-srv-steps5-b16"} <= names
    exp = catalogue.get("bagel-i2i-concurrency")
    assert [a.name for a in exp.arms] == ["m", "vllm-seqs3", "vllm-seqs16"]
    assert "qwen.rows=shuffle" in catalogue.get("t2s-conc-shuffle").arms[0].opts
    a, b = (REPO / "exps/configs/hardware/rtx5880/vllm_omni" / n for n in (
        "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache.yaml", "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16.yaml"))
    diff = [(x, y) for x, y in zip(a.read_text().splitlines(), b.read_text().splitlines()[1:]) if x != y]
    assert diff == [("    max_num_seqs: 3", "    max_num_seqs: 16")]


def test_probe_rows_put_the_probe_once_per_wave_at_a_rotating_position():
    from exps.tasks.qwen_t2s.client import probe_rows

    rows = [{"fixture_id": f"r{i}", "synthesis_text": "x" * (i + 1)} for i in range(30)]
    for fill in ("short", "long", "same", "random"):
        picked = [r["fixture_id"] for r in probe_rows(rows, 4 * 6, 4, "r7", fill)]
        waves = [picked[i:i + 4] for i in range(0, len(picked), 4)]
        assert all(w.count("r7") >= 1 for w in waves)
        if fill != "same":
            assert [w.index("r7") for w in waves][:5] == [0, 1, 2, 3, 0]
            assert all(w.count("r7") == 1 for w in waves)
    short = [r["fixture_id"] for r in probe_rows(rows, 4, 4, "r7", "short")]
    assert set(short) == {"r7", "r0", "r1", "r2"} or set(short) == {"r7", "r0", "r1", "r2"}
    long_ = [r["fixture_id"] for r in probe_rows(rows, 4, 4, "r7", "long")]
    assert "r29" in long_ and "r28" in long_


def test_kvtimeout_yamls_differ_only_by_the_stage1_kv_block_and_experiments_plan():
    base = REPO / "exps/configs/hardware/rtx5880/vllm_omni"
    for src, dst in (("bagel_two_stage_2gpu_mooncake_tcp_prefix_cache.yaml", "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_kvtimeout900.yaml"),
                     ("bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16.yaml", "bagel_two_stage_2gpu_mooncake_tcp_prefix_cache_seqs16_kvtimeout900.yaml")):
        a = (base / src).read_text().splitlines()
        b = (base / dst).read_text().splitlines()[1:]
        added = [l for l in b if l not in a]
        assert added == ["    omni_kv_config:", "      need_recv_cache: true", "      recv_timeout: 900"] or [l.strip() for l in added] == ["omni_kv_config:", "need_recv_cache: true", "recv_timeout: 900"]
        assert [l for l in a if l not in b][:1] == [] or a[0].startswith("#")
    assert [a.name for a in catalogue.get("bagel-t2i-concurrency").arms] == ["m", "vllm-seqs3", "vllm-seqs3-t900"]
    assert len(catalogue.get("t2s-conc-probe").arms) == 8
