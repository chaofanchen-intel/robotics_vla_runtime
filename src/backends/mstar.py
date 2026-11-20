"""M* (external/mstar, 2c8ffde8): OpenPI WebSocket at "/", same port as HTTP /generate."""
from __future__ import annotations

import exps.frameworks as frameworks
from exps.tasks.pi05 import transport

from backends.base import REPO, Backend


class MStarRust(Backend):
    """Default frontend: the Rust `mstar-server` (build/bin/mstar-server, see build/README.md)."""

    name, family, label, port, http_arm = "mstar-rust", "mstar", "M* (Rust)", 18104, True
    server_bin = REPO / "build" / "bin" / "mstar-server"

    def endpoint(self, base_url: str) -> str:
        return base_url.rstrip("/") + "/"

    def payload(self, example, images, session, steps, seed) -> dict:
        return transport._vllm_payload(example, images, session, steps, seed)  # noqa: SLF001

    def frontend_flag(self) -> str:
        return f"--rust-frontend-bin {self.server_bin}"

    def serve_command(self, checkpoint, port: int | None = None) -> str:
        # envs/mstar is installed editable from external/mstar and carries the mstar_rust built from external/mstar/rust,
        # so no PYTHONPATH override is needed.
        return frameworks.mstar_serve(model="pi05", config=checkpoint.config[self.family], port_number=port or self.port,
                                      tag="robot-probe", extra=self.frontend_flag())


class MStarPython(MStarRust):
    """The deprecated Python frontend (--python-frontend), kept as a comparison."""

    name, label, port = "mstar", "M* (Python frontend)", 18101

    def frontend_flag(self) -> str:
        return "--python-frontend"
