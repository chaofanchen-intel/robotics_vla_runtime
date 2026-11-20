"""One Pi0.5 inference framework behind the OpenPI WebSocket, as the runtime sees it.

A backend knows four things that differ between frameworks: where its WebSocket is, how a request is laid out, which
command starts it, and its port. Everything else (the runtime, the probes, the metrics) is shared. The message layout,
msgpack-numpy framing and the reply decoding are the ones of exps/tasks/pi05/transport.py, so the engines see the same
requests as in the exps campaigns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from websockets.asyncio.client import connect

from exps.tasks.pi05 import transport

REPO = Path(__file__).resolve().parents[2]
ACTION_DIM = 32


class Backend:
    name: str = ""            # key used on the command line and in result files
    family: str = ""          # mstar | vllm-omni | sglang-core: selects the checkpoint's config file
    label: str = ""           # display name
    port: int = 0
    http_arm: bool = False    # also serves HTTP /generate on the same port (M*)

    def endpoint(self, base_url: str) -> str:
        raise NotImplementedError

    def payload(self, example: dict, images: dict, session: str, steps: int, seed: int) -> dict:
        """Request content of one observation. `example`: {state_normalized_32, task}; `images`: slot -> HxWx3 uint8."""
        raise NotImplementedError

    def serve_command(self, checkpoint, port: int | None = None) -> str:
        raise NotImplementedError

    # ---- shared protocol ------------------------------------------------------------------------------------------
    @staticmethod
    def pack(payload: dict) -> bytes:
        return transport._pack_msgpack(payload)  # noqa: SLF001

    @staticmethod
    def parse_reply(raw: bytes, horizon: int) -> tuple[np.ndarray, dict[str, Any]]:
        """(actions as horizon x 32, server_timing). Raises on an error frame or a wrong chunk length."""
        response = transport._unpack_msgpack(raw)  # noqa: SLF001
        if isinstance(response, dict) and response.get("type") == "error":
            raise RuntimeError(str(response.get("message", response)))
        action = np.asarray(transport._extract_action(response), dtype=np.float32)  # noqa: SLF001
        if action.size != horizon * ACTION_DIM:
            raise RuntimeError(f"reply has {action.size} values, expected {horizon}x{ACTION_DIM}")
        timing = response.get("server_timing") if isinstance(response, dict) else None
        return action.reshape(horizon, ACTION_DIM), timing or {}

    async def connect(self, base_url: str):
        """Open the WebSocket and read the metadata frame. Returns (connection, metadata)."""
        ws = await connect(self.endpoint(base_url), max_size=None, compression=None, ping_interval=None,
                           open_timeout=300, close_timeout=10)
        meta = transport._unpack_msgpack(await ws.recv())  # noqa: SLF001
        return ws, meta
