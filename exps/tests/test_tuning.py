"""`exps.tuning status`: finds the exact table each framework will look up, and flags near-misses that are silently ignored."""
from __future__ import annotations

import dataclasses
from pathlib import Path

from exps import hardware, tuning
from exps.core import paths


def _setup(monkeypatch, tmp_path, gpu="NVIDIA RTX 5880 Ada Generation", triton="3.8.0", thinker_tp=1):
    monkeypatch.setattr(hardware, "gpu_name", lambda index=0: gpu)
    monkeypatch.setattr(tuning, "VLLM_TUNED_DIR", tmp_path / "tuned")
    monkeypatch.setattr(tuning, "VLLM_WHEEL_MOE_DIR", tmp_path / "wheel")
    monkeypatch.setattr(tuning, "MSTAR_TABLE_DIR", tmp_path / "mstar_tables")      # the real folder holds this machine's sweeps
    monkeypatch.setattr(tuning, "triton_version", lambda site=None: triton)
    monkeypatch.setattr(paths, "CONFIGS", tmp_path / "configs")
    for d in ("tuned", "wheel", "configs/sglang_omni/moe_rtx5880/configs"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    hw = hardware.set_current("rtx5880")
    if thinker_tp is not None:       # the file-matching tests below are about names, not topology: use the TP=1 shapes
        hardware._CURRENT = dataclasses.replace(hw, qwen_thinker_tp=thinker_tp)  # noqa: SLF001


def test_everything_missing_on_a_new_gpu_and_mstar_is_marked_untunable(monkeypatch, tmp_path):
    try:
        _setup(monkeypatch, tmp_path)
        items = tuning.status()
        by = {(i.framework, i.name): i for i in items}
        assert all(i.ok is False and not i.stale for i in items)
        mstar = [i for i in items if i.framework == "mstar"]
        assert len(mstar) == 2 and all(i.ok is False for i in mstar)          # M* has a tuner too: its sweep is the "how"
        assert "benchmark.kernels.sweep moe_gemm" in mstar[0].how and "E=128,N=1536,K=2048,top_k=8" in mstar[0].how
        mot = next(i for i in items if i.name == "BAGEL MoT GEMM")
        assert "device_name=RTX5880AdaGeneration,dtype=w16a16.json" in mot.expected
        text = tuning.render(items)
        assert "MISSING" in text and "python -m exps.suite run bagel-mot-tune" in text
    finally:
        hardware.reset()


def test_tables_are_found_by_the_exact_device_name_and_stale_triton_versions_are_flagged(monkeypatch, tmp_path):
    try:
        _setup(monkeypatch, tmp_path)
        (tmp_path / "tuned" / "device_name=RTX5880AdaGeneration,dtype=w16a16.json").write_text("{}")
        (tmp_path / "tuned" / "E=128,N=768,device_name=NVIDIA_RTX_5880_Ada_Generation.json").write_text("{}")
        old = tmp_path / "configs/sglang_omni/moe_rtx5880/configs/triton_3_7_1"
        old.mkdir(parents=True)
        (old / "E=128,N=768,device_name=NVIDIA_RTX_5880_Ada_Generation.json").write_text("{}")
        items = {(i.framework, i.name): i for i in tuning.status()}
        assert items[("vllm-omni", "BAGEL MoT GEMM")].ok is True
        assert items[("vllm-omni", "fused-MoE E=128,N=768 (Qwen3-Omni thinker)")].ok is True
        assert items[("vllm-omni", "fused-MoE E=128,N=384 (Qwen3-Omni talker)")].ok is False
        sg = items[("sglang-omni", "fused-MoE E=128,N=768 (Qwen3-Omni thinker)")]
        # SGLang falls back to the other Triton version's table with a warning: usable, but stale for the installed Triton
        assert sg.ok is True and sg.stale and "triton_3_7_1" in sg.how and "STALE" in sg.how and "triton_3_8_0" in sg.expected
        assert "STALE" in tuning.render(list(items.values()))
        # moving to the Triton version the table was tuned under makes it findable
        monkeypatch.setattr(tuning, "triton_version", lambda site=None: "3.7.1")
        exact = {(i.framework, i.name): i for i in tuning.status()}[("sglang-omni", sg.name)]
        assert exact.ok is True and not exact.stale
    finally:
        hardware.reset()


def test_h200_family_falls_back_to_the_name_vllm_actually_uses(monkeypatch, tmp_path):
    try:
        _setup(monkeypatch, tmp_path, gpu="NVIDIA H200 NVL")
        hardware.set_current("h200nvl")
        (tmp_path / "wheel" / "E=128,N=768,device_name=NVIDIA_H200.json").write_text("{}")
        items = {(i.framework, i.name): i for i in tuning.status()}
        assert items[("vllm-omni", "fused-MoE E=128,N=768 (Qwen3-Omni thinker)")].ok is True
    finally:
        hardware.reset()


def test_mstar_tables_match_the_exact_torch_device_name_and_shape(monkeypatch, tmp_path):
    import json

    try:
        _setup(monkeypatch, tmp_path, gpu="NVIDIA H200 NVL")
        hardware.set_current("h200nvl")
        monkeypatch.setattr(tuning, "MSTAR_TABLE_DIR", tmp_path / "mstar_tables")
        monkeypatch.setattr(tuning, "MSTAR_PACKAGE_TABLES", tmp_path / "package")
        # the package ships NVIDIA_H200 (SXM): the H200 NVL's torch name is NVIDIA H200 NVL, so it does not apply
        package = tmp_path / "package" / "moe_gemm"
        package.mkdir(parents=True)
        thinker = tuning.MSTAR_MOE_SHAPES["thinker"]
        (package / "NVIDIA_H200.json").write_text(json.dumps({"entries": [{"dtype": "bfloat16", "shape": thinker}]}))
        items = [i for i in tuning.status() if i.framework == "mstar"]
        assert all(i.ok is False for i in items)
        # a sweep on this machine writes NVIDIA_H200_NVL.json under the harness table dir: found, for the shapes it covers
        ours = tmp_path / "mstar_tables" / "moe_gemm"
        ours.mkdir(parents=True)
        (ours / "NVIDIA_H200_NVL.json").write_text(json.dumps({"entries": [{"dtype": "bfloat16", "shape": thinker}]}))
        by = {i.name: i for i in tuning.status() if i.framework == "mstar"}
        assert by["moe_gemm E=128,N=1536,K=2048,top_k=8 (Qwen3-Omni thinker, TP1)"].ok is True
        assert by["moe_gemm E=128,N=768,K=1024,top_k=6 (Qwen3-Omni talker, TP1)"].ok is False
    finally:
        hardware.reset()


def test_tuned_shapes_follow_the_thinker_tensor_parallel_size_of_the_overlays(monkeypatch, tmp_path):
    """The RTX 5880 overlays run the Thinker at TP=2, so its per-rank MoE shape is half the TP=1 one the tuners used to write."""
    import yaml

    try:
        _setup(monkeypatch, tmp_path, thinker_tp=None)
        assert hardware.current().qwen_thinker_tp == 2
        assert tuning.qwen_moe_shapes() == {"thinker": (128, 384), "talker": (128, 384)}
        assert tuning.mstar_moe_shapes()["thinker"] == {"E": 128, "N": 768, "K": 2048, "top_k": 8}
        assert tuning.mstar_moe_shapes()["talker"] == {"E": 128, "N": 768, "K": 1024, "top_k": 6}
        hardware.set_current("h200nvl")
        assert tuning.qwen_moe_shapes()["thinker"] == (128, 768) and tuning.mstar_moe_shapes()["thinker"]["N"] == 1536
    finally:
        hardware.reset()
    # the profile's number must be the TP size the overlay really configures
    overlay = Path(__file__).resolve().parents[1] / "configs" / "hardware" / "rtx5880" / "mstar" / "qwen3omni_3gpu_thinker_tp2.yaml"
    thinker = next(g for g in yaml.safe_load(overlay.read_text())["node_groups"] if g["node_names"] == ["Thinker"])
    assert thinker["tp_size"] == hardware.profiles()["rtx5880"].qwen_thinker_tp


def test_tune_commands_use_the_thinker_tp_and_the_talker_at_tp1():
    from exps.suite import catalogue, runner

    try:
        hardware.set_current("rtx5880")
        for name, needle in (("vllm-moe-tune", "--tp-size ${spec##*:}"), ("sglang-moe-tune", "--tp-size ${spec##*:}")):
            text = " ".join(runner.expand_text(part, campaign="") for part in catalogue.get(name).tool)
            assert "for spec in thinker:2;" in text and needle in text, name      # same N=384 file as the Talker: tune it once
        text = " ".join(runner.expand_text(part, campaign="") for part in catalogue.get("mstar-moe-tune").tool)
        assert "E=128,N=768,K=2048,top_k=8:thinker:2" in text and "E=128,N=768,K=1024,top_k=6:talker:1" in text
        hardware.set_current("h200nvl")
        text = " ".join(runner.expand_text(part, campaign="") for part in catalogue.get("mstar-moe-tune").tool)
        assert "E=128,N=1536,K=2048,top_k=8:thinker:1" in text
        text = " ".join(runner.expand_text(part, campaign="") for part in catalogue.get("sglang-moe-tune").tool)
        assert "for spec in talker:1 thinker:1;" in text          # TP=1: two different N, both tuned, as before
    finally:
        hardware.reset()
