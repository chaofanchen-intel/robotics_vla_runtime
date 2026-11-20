"""Regression tests for Qwen adapter request contracts."""
from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace

import pytest

from exps.core import adapters
from exps.core.codecs import ContractViolation
from exps.core.measure import MeasurementConfig
from exps.core.registry import CellRef
from exps.tasks import qwen_common
from exps.tasks.qwen_a2t import cell as qwen_a2t_cell
from exps.tasks.qwen_a2t import client as qwen_a2t_client
from exps.tasks.qwen_a2t.cases import TASK as QWEN_A2T_TASK
from exps.tools import qwen_a2t_audio_encoder_probe
from exps.tools.qwen_a2t_stage_trace import (
    _model_forward_phase as model_forward_phase,
    _wrap as wrap_stage_trace,
)
from exps.tasks.qwen_a2t.audio_io import read_audio_wav, write_float_wav
from exps.tasks.qwen_t2s import client as qwen_t2s_client
from exps.tasks.qwen_txt import client as qwen_txt_client
from exps.tools.analyze_qwen_a2t_stage_trace import analyze_cell


def test_qwen_a2t_probe_selects_exact_prewarm_then_measured_paths():
    short = {"audio_relpath": "en_us/short.wav", "id": "short"}
    long = {"audio_relpath": "en_us/long.wav", "id": "long"}

    selected = qwen_a2t_client.select_probe_samples(
        [short, long],
        precondition_count=1,
        measured_count=3,
        precondition_audio_relpaths=("en_us/short.wav",),
        measured_audio_relpaths=("en_us/long.wav",),
    )

    assert [sample["id"] for sample in selected] == [
        "short",
        "long",
        "long",
        "long",
    ]


def test_qwen_a2t_probe_supports_no_request_precondition():
    long = {"audio_relpath": "en_us/long.wav", "id": "long"}

    selected = qwen_a2t_client.select_probe_samples(
        [long],
        precondition_count=0,
        measured_count=2,
        precondition_audio_relpaths=(),
        measured_audio_relpaths=("en_us/long.wav",),
    )

    assert [sample["id"] for sample in selected] == ["long", "long"]


def test_qwen_a2t_probe_rejects_paths_outside_selected_fixture():
    sample = {"audio_relpath": "en_us/long.wav"}

    with pytest.raises(ValueError, match="not in the selected FLEURS config"):
        qwen_a2t_client.select_probe_samples(
            [sample],
            precondition_count=1,
            measured_count=1,
            precondition_audio_relpaths=("en_us/other.wav",),
            measured_audio_relpaths=("en_us/long.wav",),
        )


def test_qwen_a2t_different_audio_precondition_is_trimmed_to_measured_shape(
    monkeypatch,
    tmp_path,
):
    import hashlib
    import numpy as np

    audio_root = tmp_path / "fixture_audio"
    audio_root.mkdir()
    source = audio_root / "en_us" / "other.wav"
    write_float_wav(source, np.linspace(-0.1, 0.1, 12, dtype=np.float32), 16_000)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(qwen_a2t_client, "AUDIO_ROOT", audio_root)
    sample = {
        "id": 42,
        "audio_relpath": "en_us/other.wav",
        "audio_sha256": source_sha,
        "num_samples": 12,
        "duration_seconds": 12 / 16_000,
    }

    adjusted, generated = qwen_a2t_client.materialize_shape_matched_precondition(
        sample,
        target_num_samples=10,
        target_sample_rate=16_000,
        output_dir=tmp_path / "run",
        ordinal=1,
    )
    waveform, sample_rate = read_audio_wav(generated)

    assert sample_rate == 16_000
    assert len(waveform) == 10
    assert adjusted["probe_source_audio_relpath"] == "en_us/other.wav"
    assert adjusted["probe_audio_transform"] == "trim_right"
    assert adjusted["probe_precondition_audio_sha256"] == hashlib.sha256(
        generated.read_bytes()
    ).hexdigest()


def test_qwen_a2t_probe_cell_passes_exact_audio_path_selection(tmp_path):
    case = QWEN_A2T_TASK.case("a2t-en-long-short-precondition")
    config = MeasurementConfig(
        output_dir=tmp_path,
        precondition_requests=1,
        measured_requests=3,
    )
    plan = qwen_a2t_cell.plan(
        CellRef(
            task="qwen_a2t",
            framework="mstar",
            case=case.name,
            repeat="r01",
        ),
        case,
        config,
    )

    precondition_flag = plan.client_argv.index("--precondition-audio-relpaths")
    measured_flag = plan.client_argv.index("--measured-audio-relpaths")
    assert plan.client_argv[precondition_flag + 1] == "en_us/00003_1728.wav"
    assert plan.client_argv[measured_flag + 1] == "en_us/00505_1901.wav"
    assert plan.context["probe_kind"] == "long-audio-precondition"


def test_qwen_a2t_shape_match_and_stage_cases_are_probe_only():
    normal_cases = {
        case.name for case in QWEN_A2T_TASK.cases_for_profile("nightly")
    }
    assert normal_cases == {"a2t-en", "a2t-zh"}
    assert QWEN_A2T_TASK.case(
        "a2t-en-long-shape-matched-other-audio-precondition"
    ).params["precondition_match_measured_shape"] is True
    stage_case = QWEN_A2T_TASK.case("a2t-en-stage-timing")
    assert stage_case.params["probe_kind"] == "stage-timing"
    assert stage_case.params["precondition_match_measured_shape"] is True
    assert (
        stage_case.params["precondition_audio_relpaths"]
        != stage_case.params["measured_audio_relpaths"]
    )


def test_qwen_a2t_cell_selects_shape_matching_precondition(tmp_path):
    case = QWEN_A2T_TASK.case(
        "a2t-en-long-shape-matched-other-audio-precondition"
    )
    plan = qwen_a2t_cell.plan(
        CellRef(
            task="qwen_a2t",
            framework="mstar",
            case=case.name,
            repeat="r01",
        ),
        case,
        MeasurementConfig(
            output_dir=tmp_path,
            precondition_requests=1,
            measured_requests=3,
        ),
    )

    assert "--precondition-match-measured-shape" in plan.client_argv
    assert plan.client_argv[
        plan.client_argv.index("--precondition-audio-relpaths") + 1
    ] == "en_us/00352_2008.wav"


@pytest.mark.parametrize(
    ("framework", "expected_config"),
    [
        ("mstar", "qwen3omni_1gpu.yaml"),
        ("sglang-omni", "qwen3_omni_colocated_h200_stage_timing.yaml"),
        ("vllm-omni", "qwen3_omni_full_1gpu_stage_timing.yaml"),
    ],
)
def test_qwen_a2t_stage_profile_sets_trace_environment_and_profile_config(
    framework,
    expected_config,
    tmp_path,
):
    case = QWEN_A2T_TASK.case("a2t-en-stage-timing")
    plan = qwen_a2t_cell.plan(
        CellRef(
            task="qwen_a2t",
            framework=framework,
            case=case.name,
            repeat="r01",
        ),
        case,
        MeasurementConfig(output_dir=tmp_path, precondition_requests=1, measured_requests=1),
    )

    assert plan.context["stage_timing_profile"] == "eager_no_compile_no_cuda_graph"
    assert plan.service.env["QWEN_A2T_STAGE_TRACE_FRAMEWORK"] == framework
    assert plan.service.env["QWEN_A2T_STAGE_TRACE_PATH"] == str(
        tmp_path / "stage_timings.jsonl"
    )
    assert plan.context["config"].endswith(expected_config)
    assert "qwen_a2t_stage_trace_hook" in plan.service.env["PYTHONPATH"]
    assert "--precondition-match-measured-shape" in plan.client_argv
    assert plan.client_argv[
        plan.client_argv.index("--precondition-audio-relpaths") + 1
    ] != plan.client_argv[
        plan.client_argv.index("--measured-audio-relpaths") + 1
    ]
    if framework == "sglang-omni":
        assert plan.service.env["SGLANG_OMNI_TRACE_ENCODER_CACHE"] == "1"
    else:
        assert "SGLANG_OMNI_TRACE_ENCODER_CACHE" not in plan.service.env
    if framework == "sglang-omni":
        assert plan.service.env["SGLANG_OMNI_TRACE_ENCODER_CACHE"] == "1"
    else:
        assert "SGLANG_OMNI_TRACE_ENCODER_CACHE" not in plan.service.env


def test_qwen_a2t_stage_trace_wrapper_records_wall_span(monkeypatch, tmp_path):
    import json

    trace_path = tmp_path / "stage_timings.jsonl"
    monkeypatch.setenv("QWEN_A2T_STAGE_TRACE_PATH", str(trace_path))

    def preprocess(value: int) -> int:
        return value + 1

    traced = wrap_stage_trace(preprocess, "audio_preprocess_total")

    assert traced(2) == 3
    row = json.loads(trace_path.read_text(encoding="utf-8").strip())
    assert row["stage"] == "audio_preprocess_total"
    assert row["wall_ms"] >= 0
    assert row["cuda_event_ms"] is None
    assert row["error"] is None


def test_qwen_a2t_stage_trace_analyzer_joins_components_to_measured_request(tmp_path):
    import json
    from datetime import datetime, timedelta, timezone

    cell_dir = tmp_path / "cell"
    cell_dir.mkdir()
    start = datetime(2026, 10, 19, 1, 0, tzinfo=timezone.utc)
    request = {
        "phase": "measured",
        "sequence": 1,
        "success": True,
        "started_at_utc": start.isoformat(),
        "finished_at_utc": (start + timedelta(seconds=1)).isoformat(),
        "e2e_ms": 1000.0,
        "ttft_ms": 300.0,
        "output_tokens": 3,
        "thermally_clean": True,
        "cooldown_timed_out": False,
    }
    manifest = {
        "cell": "qwen_a2t/mstar/a2t-en-stage-timing/r01",
        "status": "completed",
        "framework": "mstar",
        "case": "a2t-en-stage-timing",
        "repeat": "r01",
        "probe_kind": "stage-timing",
    }

    def event(stage, offset_ms, duration_ms, phase=None):
        start_ns = int(start.timestamp() * 1e9 + offset_ms * 1e6)
        return {
            "stage": stage,
            "phase": phase,
            "started_unix_ns": start_ns,
            "finished_unix_ns": start_ns + int(duration_ms * 1e6),
            "wall_ms": duration_ms,
            "cuda_event_ms": duration_ms / 2,
        }

    (cell_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (cell_dir / "requests.jsonl").write_text(
        json.dumps(request) + "\n", encoding="utf-8"
    )
    (cell_dir / "stage_timings.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in (
                event("request_preprocessor_total", 100, 100),
                event("audio_feature_extractor", 110, 40),
                event("audio_encoder", 220, 120),
                event("thinker_transformer_core", 400, 350, "prefill"),
                event("thinker_transformer_core", 800, 20),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    result = analyze_cell(cell_dir)
    measured = result["measured_requests"][0]

    assert result["trace_status"] == "captured"
    assert measured["stage_spans"]["request_preprocessor_inclusive_wall_ms"] == 100
    assert measured["stage_spans"]["audio_feature_extractor_wall_ms"] == 40
    assert measured["stage_spans"]["audio_encoder_cuda_event_ms"] == 60
    assert measured["stage_spans"]["thinker_prefill_cuda_event_ms"] == 175
    assert measured["stage_spans"]["thinker_transformer_core_wall_ms"] == 370
    assert measured["stage_spans"]["thinker_transformer_core_cuda_event_ms"] == 185
    assert measured["stage_spans"]["thinker_other_phase_wall_ms"] == 20


def test_qwen_a2t_stage_trace_classifies_mstar_thinker_embeds():
    assert model_forward_phase({"input_embeds": SimpleNamespace(shape=(43, 2048))}) == "prefill"
    assert model_forward_phase({"input_embeds": SimpleNamespace(shape=(1, 2048))}) == "decode"


def test_qwen_a2t_stage_trace_analyzer_handles_failed_cell_without_request_log(tmp_path):
    import json

    cell_dir = tmp_path / "vllm-omni" / "failed-cell"
    cell_dir.mkdir(parents=True)
    failure = "ServiceStartupError: service exited with code 1 before becoming ready."
    (cell_dir / "manifest.json").write_text(
        json.dumps(
            {
                "cell": "qwen_a2t/vllm-omni/a2t-en-stage-timing/r01",
                "status": "failed",
                "failure": failure,
                "framework": "vllm-omni",
                "case": "a2t-en-stage-timing",
                "probe_kind": "stage-timing",
                "repeat": "r01",
            }
        ),
        encoding="utf-8",
    )
    (cell_dir / "stage_timings.jsonl").write_text(
        json.dumps({"stage": "trace_module_patched", "module": "test.module"}) + "\n",
        encoding="utf-8",
    )

    result = analyze_cell(cell_dir)

    assert result["status"] == "failed"
    assert result["failure_summary"] == failure
    assert result["request_log_status"] == "missing"
    assert result["trace_status"] == "missing_request_file"
    assert result["measured_requests"] == []


def test_qwen_a2t_diagnostic_cases_are_excluded_from_regular_profiles():
    assert [
        case.name for case in QWEN_A2T_TASK.cases_for_profile("nightly")
    ] == ["a2t-en", "a2t-zh"]
    assert [
        case.name for case in QWEN_A2T_TASK.cases_for_profile("full")
    ] == ["a2t-en", "a2t-zh"]


def test_qwen_a2t_fa3_adapter_uses_packed_qkv_and_returns_transformers_layout(
    monkeypatch,
):
    import sys
    import types
    from types import SimpleNamespace

    import torch

    captured = {}

    def fake_flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        **kwargs,
    ):
        captured.update(
            {
                "query": query,
                "key": key,
                "value": value,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "kwargs": kwargs,
            }
        )
        return query + 1

    fa3_module = types.ModuleType("fa3_fwd_interface")
    fa3_module.flash_attn_varlen_func = fake_flash_attn_varlen_func
    monkeypatch.setitem(sys.modules, "fa3_fwd_interface", fa3_module)
    query = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    cu_seqlens = torch.tensor([0, 2, 3], dtype=torch.int64)

    output, weights = qwen_a2t_audio_encoder_probe._fa3_varlen_attention_forward(
        SimpleNamespace(is_causal=False),
        query,
        query,
        query,
        attention_mask=None,
        scaling=0.5,
        is_causal=False,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=torch.tensor(2),
        max_length_k=torch.tensor(2),
    )

    assert tuple(captured["query"].shape) == (3, 2, 4)
    assert captured["cu_seqlens_q"].dtype == torch.int32
    assert captured["cu_seqlens_k"].dtype == torch.int32
    assert captured["max_seqlen_q"] == 2
    assert captured["max_seqlen_k"] == 2
    assert captured["kwargs"] == {
        "softmax_scale": 0.5,
        "causal": False,
        "softcap": 0.0,
    }
    assert torch.equal(output, query.transpose(1, 2) + 1)
    assert weights is None


def test_qwen_a2t_fa3_adapter_registers_as_a_flash_attention_interface():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    implementation = qwen_a2t_audio_encoder_probe._register_fa3_attention_interface()

    assert implementation == qwen_a2t_audio_encoder_probe.MSTAR_FA3_ATTENTION_IMPLEMENTATION
    assert implementation in ALL_ATTENTION_FUNCTIONS.valid_keys()


class _NotFoundResponse:
    status = 404

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def text(self):
        return "synthetic 404"


class _RecordingSession:
    def __init__(self):
        self.payloads = []

    def post(self, _url, **kwargs):
        self.payloads.append(kwargs["json"])
        return _NotFoundResponse()


def test_qwen_txt_validator_accepts_exact_reported_length():
    validator = qwen_txt_client.make_validator(max_output_tokens=256)

    audit = validator(
        SimpleNamespace(response_chunks={}),
        {"output_tokens": 256},
        0,
    )

    assert audit == {
        "completion_length_verified": True,
        "requested_output_tokens": 256,
        "realised_output_tokens": 256,
    }


@pytest.mark.parametrize("output_tokens", [None, 2, 255, 257])
def test_qwen_txt_validator_rejects_unverified_or_mismatched_length(output_tokens):
    validator = qwen_txt_client.make_validator(max_output_tokens=256)

    with pytest.raises(ContractViolation):
        validator(
            SimpleNamespace(response_chunks={"text": 1}),
            {"output_tokens": output_tokens},
            0,
        )


@pytest.mark.parametrize("arm", ["verbatim", "voice_clone"])
def test_qwen_t2s_validator_tags_the_fixture_used_for_drift_analysis(monkeypatch, arm):
    monkeypatch.setattr(
        qwen_t2s_client.contract,
        "verify_verbatim" if arm == "verbatim" else "verify_voice_clone",
        lambda *_args, **_kwargs: {"contract_honoured": True},
    )
    validator = qwen_t2s_client.make_validator(
        arm,
        "en",
        [{"fixture_id": "fixture-a"}, {"fixture_id": "fixture-b"}],
    )

    audit = validator(None, {}, 1)

    assert audit["drift_workload_key"] == "fixture-b"


@pytest.mark.parametrize(
    ("framework", "adapter_name"),
    [("vllm-omni", "VLLMOmni"), ("sglang-omni", "SGLangOmni")],
)
@pytest.mark.parametrize(
    ("request_type_name", "audio_input"),
    [("text_to_text", False), ("audio_to_text", True)],
)
def test_qwen_text_payloads_use_served_name_modalities_and_request_kwargs(
    framework, adapter_name, request_type_name, audio_input, tmp_path
):
    request = adapters.load_benchmark_module("benchmark.request")
    audio_path = tmp_path / "input.wav"
    if audio_input:
        audio_path.write_bytes(b"test audio")
    request_kwargs = {
        "seed": 42,
        "top_p": 1.0,
        "top_k": 1,
        "max_tokens": 256,
    }
    if request_type_name == "text_to_text":
        request_kwargs.update({"min_tokens": 256, "ignore_eos": True})
    req_input = request.RequestInput(
        req_type=adapters.request_type(request_type_name),
        prompt="Transcribe this." if audio_input else "Reply with text.",
        audio_path=str(audio_path) if audio_input else None,
        model_kwargs=request_kwargs,
    )
    model = adapters.model_instance(
        "qwen3omni", **qwen_common.client_model_instance_kwargs(framework)
    )
    session = _RecordingSession()

    asyncio.run(
        getattr(request, adapter_name)().send_request(
            session, req_input, "http://localhost", 0, model
        )
    )

    payload = session.payloads[0]
    assert payload["model"] == qwen_common.SERVED_MODEL_NAME
    assert payload["modalities"] == ["text"]
    for key, value in request_kwargs.items():
        assert payload[key] == value


def test_qwen_txt_sampling_pins_total_output_length_per_runtime() -> None:
    mstar_params = qwen_txt_client.sampling_kwargs(256, "mstar")
    openai_params = qwen_txt_client.sampling_kwargs(256, "vllm-omni")

    assert mstar_params["max_tokens"] == 256
    assert mstar_params["max_output_tokens"] == 256
    assert mstar_params["min_tokens"] == 256
    assert mstar_params["ignore_eos"] is True
    assert openai_params["max_tokens"] == 256
    assert openai_params["max_output_tokens"] == 256
    assert openai_params["min_tokens"] == 256
    assert openai_params["ignore_eos"] is True


def test_qwen_txt_mstar_request_cap_is_the_total_completion_length():
    """M* counts the prefill-sampled token toward the cap; SMOKE_01 asked 255 and got 255."""
    request_input = qwen_txt_client.build_requests(
        "fixed prompt", 1, 256, "mstar"
    )[0]

    assert request_input.model_kwargs["max_tokens"] == 256
    assert request_input.model_kwargs["max_output_tokens"] == 256
    assert request_input.model_kwargs["ignore_eos"] is True


def test_response_text_decodes_bytes_and_str_chunks():
    """M* keeps text chunks as bytes, the OpenAI adapters as str (SMOKE_01 TypeError)."""
    from exps.core import adapters

    snowman = "☃".encode("utf-8")
    as_bytes = SimpleNamespace(_text_chunks=[b"hello ", snowman[:1], snowman[1:]])
    as_str = SimpleNamespace(_text_chunks=["hello ", "☃"])
    empty = SimpleNamespace(_text_chunks=[])
    assert adapters.response_text(as_bytes) == "hello ☃"
    assert adapters.response_text(as_str) == "hello ☃"
    assert adapters.response_text(empty) == ""


@pytest.mark.parametrize(
    ("client_name", "framework", "expected_model_instance_kwargs"),
    [
        ("qwen_t2s", "mstar", {}),
        ("qwen_t2s", "sglang-omni", {"served_model_name": "qwen3_omni"}),
        ("qwen_t2s", "vllm-omni", {"served_model_name": "qwen3_omni"}),
        ("qwen_txt", "mstar", {}),
        ("qwen_txt", "sglang-omni", {"served_model_name": "qwen3_omni"}),
        ("qwen_txt", "vllm-omni", {"served_model_name": "qwen3_omni"}),
        ("qwen_a2t", "mstar", {}),
        ("qwen_a2t", "sglang-omni", {"served_model_name": "qwen3_omni"}),
        ("qwen_a2t", "vllm-omni", {"served_model_name": "qwen3_omni"}),
    ],
)
def test_qwen_clients_pass_served_name_only_to_openai_runtimes(
    client_name, framework, expected_model_instance_kwargs, monkeypatch, tmp_path
):
    clients = {
        "qwen_t2s": qwen_t2s_client,
        "qwen_txt": qwen_txt_client,
        "qwen_a2t": qwen_a2t_client,
    }
    client = clients[client_name]
    captured = {}
    monkeypatch.setattr(
        client.benchmark_client, "run", lambda **kwargs: captured.update(kwargs) or 0
    )
    extra_args = []
    if client_name == "qwen_t2s":
        monkeypatch.setattr(client, "build_requests", lambda *_args: ([], []))
        extra_args = ["--arm", "verbatim", "--language", "en"]
    elif client_name == "qwen_txt":
        monkeypatch.setattr(client, "load_prompt", lambda: ("prompt", 1024, "fixture-hash"))
        monkeypatch.setattr(client, "build_requests", lambda *_args: [])
    else:
        monkeypatch.setattr(client, "build_requests", lambda *_args: ([], []))
        extra_args = ["--language", "en", "--fleurs-config", "en_us"]

    monkeypatch.setattr(
        "sys.argv",
        [
            f"{client_name}.client",
            "--framework",
            framework,
            "--case",
            {"qwen_t2s": "verbatim-en", "qwen_txt": "txt-1k", "qwen_a2t": "a2t-en"}[
                client_name
            ],
            "--base-url",
            "http://localhost",
            "--output-dir",
            str(tmp_path),
            *extra_args,
        ],
    )

    assert client.main() == 0
    assert captured["model_instance_kwargs"] == expected_model_instance_kwargs
    if client_name == "qwen_txt":
        assert captured["context"]["input_tokens"] == 1024
        assert captured["context"]["prompt_sha256"] == "fixture-hash"


def test_qwen_t2s_canary_passes_served_name(monkeypatch, tmp_path):
    captured = {}
    monkeypatch.setattr(qwen_t2s_client.adapters, "inference_system", lambda _framework: object())
    monkeypatch.setattr(
        qwen_t2s_client.adapters,
        "model_instance",
        lambda _model_type, **kwargs: captured.update(kwargs) or object(),
    )
    monkeypatch.setattr(
        qwen_t2s_client.adapters, "send_sequential", lambda *_args, **_kwargs: object()
    )
    monkeypatch.setattr(
        qwen_t2s_client.adapters, "to_metric_row", lambda _result: {"success": True}
    )
    args = argparse.Namespace(
        framework="sglang-omni",
        case="verbatim-en",
        base_url="http://localhost",
        output_dir=tmp_path,
        request_timeout_s=1.0,
    )

    assert qwen_t2s_client.run_canary(
        args, [object()], [{}], lambda *_args: {"contract_honoured": True}
    ) == 0
    assert captured == {"served_model_name": qwen_common.SERVED_MODEL_NAME}
