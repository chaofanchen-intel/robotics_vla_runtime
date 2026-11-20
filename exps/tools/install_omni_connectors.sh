#!/usr/bin/env bash
# Install the optional vLLM-Omni inter-stage connectors (Mooncake Transfer Engine, NIXL) into a USER directory, not into envs/vllm-omni.
#
# Why a separate directory: vLLM-Omni's default BAGEL two-stage deploy moves the Thinker's KV cache through SharedMemoryConnector
# (serialise to bytes, copy through shared memory, copy back: ~0.47 s + ~0.36 s per 340 MB branch). Mooncake and NIXL carry the GPU tensor
# directly. Keeping them out of the baseline environment means no baseline arm can pick them up: only an arm that sets
# `--env vllm-omni:PYTHONPATH=<this directory>` (see `bagel-connector` in exps/suite/catalogue.py) imports them.
#
#   exps/tools/install_omni_connectors.sh [TARGET_DIR]        (default: $HOME/.local/opt/omni-connectors/site)
#
# Versions are pinned to what was smoke-tested here (RTX 5880 box, no RDMA NIC, no NVLink):
#   mooncake-transfer-engine-cuda13 0.3.14   (the plain `mooncake-transfer-engine` wheel is built against CUDA 12 and fails to import with
#                                              torch cu130: "undefined symbol: cudaMemcpyBatchAsync, version libcudart.so.12")
#   nixl 1.5.2 + nixl-cu13 1.5.2             (the `nixl` wheel is a shim; the native part is nixl-cu13)
# `--no-deps`: both need only packages envs/vllm-omni already has (aiohttp, requests, msgpack, numpy); nothing in the baseline env changes.
set -euo pipefail
cd "$(dirname "$0")/../.."
TARGET="${1:-$HOME/.local/opt/omni-connectors/site}"
PY=envs/vllm-omni/bin/python
mkdir -p "$TARGET"
"$PY" -m pip install --no-deps --upgrade --target "$TARGET" \
  "mooncake-transfer-engine-cuda13==0.3.14" "nixl==1.5.2" "nixl-cu13==1.5.2"
PYTHONPATH="$TARGET" "$PY" - <<'PYEOF'
import torch  # the vLLM-Omni stage processes load torch's CUDA 13 runtime before the connectors
from mooncake.engine import TransferEngine
from nixl._api import nixl_agent, nixl_agent_config
engine = TransferEngine()
assert engine.initialize("127.0.0.1", "P2PHANDSHAKE", "tcp", "") == 0, "Mooncake TransferEngine failed to initialise (tcp)"
agent = nixl_agent("install_probe", nixl_agent_config(backends=["UCX"]))
print("OK: Mooncake TransferEngine (tcp) and a NIXL UCX agent both initialise; PYTHONPATH =", __import__("os").environ["PYTHONPATH"])
PYEOF
