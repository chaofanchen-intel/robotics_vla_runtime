"""SGLang core 0.5.22 multimodal_gen (envs/sglang-omni): OpenPI WebSocket at /openpi/policy."""
from __future__ import annotations

import exps.frameworks as frameworks
from exps.tasks.pi05 import transport

from backends.base import Backend


class SglangCore(Backend):
    name, family, label, port = "sglang-core", "sglang-core", "SGLang core", 18103

    def endpoint(self, base_url: str) -> str:
        return base_url.rstrip("/") + "/openpi/policy"

    def payload(self, example, images, session, steps, seed) -> dict:
        # SGLang's request has no session id or seed; the key names differ from the other two
        return transport._sglang_payload(example, images, steps)  # noqa: SLF001

    def serve_command(self, checkpoint, port: int | None = None) -> str:
        return frameworks.sglang_core_serve(model_path=checkpoint.snapshot, pipeline="Pi05Pipeline",
                                            pipeline_config=checkpoint.config[self.family], port_number=port or self.port)


class SglangCoreBucketed(SglangCore):
    """SGLang core with `prompt_token_buckets: [200]` (configs/sglang_pi05_droid_buckets.json).

    By default the CUDA graphs are keyed by the exact prompt length, the denoise graph cache holds 4 signatures, and
    "new signatures run eagerly" once it is full. Pi0.5's prompt carries the discretised state as text, so its length
    varies from request to request. SGLang's config offers prompt buckets for this (padding changes reduction shapes, so
    it is opt-in); one bucket at max_token_len puts every request on one graph signature.
    """

    name, family, label, port = "sglang-core-bucketed", "sglang-core-bucketed", "SGLang core (prompt buckets)", 18105
