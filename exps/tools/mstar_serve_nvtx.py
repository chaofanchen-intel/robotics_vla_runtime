#!/usr/bin/env python3
"""`mstar serve ...` with the engine's NVTX ranges on.

The low-level server (`mstar.api_server.entrypoint`) has `--enable-nvtx`, but the `mstar serve` wrapper does
not forward it ("unrecognized arguments"), so this launcher appends the flag to the arguments the wrapper
hands to the server and otherwise behaves exactly like the `mstar` console script. `external/mstar` is untouched.

    envs/mstar/bin/python exps/tools/mstar_serve_nvtx.py serve qwen3_omni --config ... --port 19311
"""
from __future__ import annotations

from mstar.api_server import entrypoint

_serve_main = entrypoint.main


def _main_with_nvtx(argv=None):
    return _serve_main([*(argv or []), "--enable-nvtx"])


entrypoint.main = _main_with_nvtx  # `_serve` imports `main` from the module at call time

if __name__ == "__main__":
    from mstar.cli.main import main

    main()
