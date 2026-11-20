"""vLLM-Omni v0.32.0 (pinned/vllm-omni): OpenPI WebSocket at /v1/realtime/robot/openpi."""
from __future__ import annotations

import json
import shlex

import exps.frameworks as frameworks
from exps.tasks.pi05 import cell, transport

from backends.base import Backend


class VllmOmni(Backend):
    name, family, label, port = "vllm-omni", "vllm-omni", "vLLM-Omni", 18102

    def endpoint(self, base_url: str) -> str:
        return base_url.rstrip("/") + "/v1/realtime/robot/openpi"

    def payload(self, example, images, session, steps, seed) -> dict:
        return transport._vllm_payload(example, images, session, steps, seed)  # noqa: SLF001

    def serve_command(self, checkpoint, port: int | None = None) -> str:
        extra = ("--enforce-eager --robot-openpi-idle-timeout 1800 --custom-pipeline-args "
                 + shlex.quote(json.dumps({"tokenizer": str(cell.TOKENIZER)})))
        return frameworks.vllm_omni_serve(model_path=checkpoint.snapshot, deploy_config=checkpoint.config[self.family],
                                          port_number=port or self.port, extra=extra)
