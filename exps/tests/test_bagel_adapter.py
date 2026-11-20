"""Inspect the actual M* vLLM-Omni adapter payload without starting a server."""
from __future__ import annotations

import asyncio

import pytest

from exps.core import adapters
from exps.tasks.bagel import client as bagel_client
from exps.tasks.bagel.client import model_kwargs_for


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


@pytest.mark.parametrize("task", ["t2i", "i2i"])
def test_bagel_image_request_uses_per_request_dimensions_and_served_name(task):
    request = adapters.load_benchmark_module("benchmark.request")
    session = _RecordingSession()
    model = adapters.model_instance("bagel", image_preprocess="vllm", served_model_name="bagel")
    req_input = request.RequestInput(
        req_type=adapters.request_type("text_to_image" if task == "t2i" else "image_to_image"),
        prompt="A test image",
        _image_bytes=b"input-image" if task == "i2i" else None,
        model_kwargs=model_kwargs_for(task, 256, 128),
    )
    asyncio.run(request.VLLMOmni().send_request(session, req_input, "http://localhost", 0, model))
    payload = session.payloads[0]
    assert payload["extra_body"]["width"] == payload["extra_body"]["height"] == 256
    assert payload["extra_body"]["num_inference_steps"] == 50
    assert payload["extra_body"]["seed"] == 42
    assert payload["model"] == "bagel"


def test_bagel_text_request_uses_served_name_and_per_request_overrides():
    request = adapters.load_benchmark_module("benchmark.request")
    session = _RecordingSession()
    model = adapters.model_instance("bagel", served_model_name="bagel")
    req_input = request.RequestInput(
        req_type=adapters.request_type("image_to_text"),
        prompt="Describe this",
        _image_bytes=b"input-image",
        model_kwargs=model_kwargs_for("i2t", 256, 128),
    )
    asyncio.run(request.VLLMOmni().send_request(session, req_input, "http://localhost", 0, model))
    payload = session.payloads[0]
    assert payload["model"] == "bagel"
    assert payload["max_output_tokens"] == payload["max_tokens"] == 128
    assert payload["min_tokens"] == 128
    assert payload["ignore_eos"] is True


def test_unconfigured_model_name_keeps_hf_fallback():
    request = adapters.load_benchmark_module("benchmark.request")
    session = _RecordingSession()
    model = adapters.model_instance("bagel")
    req_input = request.RequestInput(
        req_type=adapters.request_type("text_to_image"),
        prompt="A test image",
        model_kwargs={"width": 256, "height": 256},
    )
    asyncio.run(request.VLLMOmni().send_request(session, req_input, "http://localhost", 0, model))
    assert session.payloads[0]["model"] == model.get_hf_url()


@pytest.mark.parametrize(
    ("framework", "expected_alias"),
    [
        ("vllm-omni-single-stage", "bagel"),
        ("vllm-omni-two-stage", "bagel"),
        ("mstar", None),
    ],
)
def test_bagel_client_configures_alias_only_for_vllm(
    monkeypatch, tmp_path, framework, expected_alias
):
    captured = {}

    def capture(**kwargs):
        captured.update(kwargs)
        return 0

    monkeypatch.setattr(bagel_client, "load_fixture", lambda: {"fixture_version": "test"})
    monkeypatch.setattr(bagel_client, "build_requests", lambda *_args: ["test request"])
    monkeypatch.setattr(bagel_client.benchmark_client, "run", capture)
    monkeypatch.setattr("sys.argv", [
        "bagel.client", "--framework", framework, "--case", "t2i-256",
        "--base-url", "http://localhost", "--output-dir", str(tmp_path),
        "--task", "t2i", "--resolution", "256",
    ])
    assert bagel_client.main() == 0
    assert captured["model_instance_kwargs"].get("served_model_name") == expected_alias
