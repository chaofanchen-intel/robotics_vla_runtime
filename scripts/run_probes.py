#!/usr/bin/env python3
"""Start one Pi0.5 server per framework on a free GPU, run the protocol probes, stop it.

Usage (from the repository root):
    envs/mstar/bin/python scripts/run_probes.py --framework mstar-rust --run-dir runs/<date>_<name>
    envs/mstar/bin/python scripts/run_probes.py --framework vllm-omni  --run-dir runs/<date>_<name> --checkpoint pi05_droid

The server commands are the ones the exps harness builds (exps/frameworks, exps/tasks/pi05/cell.py), so the engines,
configs and cache directory are the same as the existing Pi0.5 campaigns. The GPU is the one named by the serve command
(CUDA_VISIBLE_DEVICES=0 in exps/profiles/hardware/rtx5880.json); check `nvidia-smi` first.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))            # exps/
sys.path.insert(0, str(REPO / "src"))   # runtime, backends, metrics, scenarios

import backends  # noqa: E402
from backends.serving import Server  # noqa: E402

MODES = ("e2e", "period", "burst", "cancel", "concurrency")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", choices=tuple(backends.ALL), required=True)
    p.add_argument("--checkpoint", choices=tuple(backends.CHECKPOINTS), default="pi05_base")
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--run-dir", type=Path, required=True,
                   help="output directory, e.g. runs/<date>_<name>; required so that existing results are not overwritten")
    args = p.parse_args()

    backend, checkpoint = backends.get(args.framework), backends.CHECKPOINTS[args.checkpoint]
    out_dir = args.run_dir / args.framework
    with Server(backend, checkpoint, out_dir) as server:
        print(f"{args.framework} healthy after {server.ready_s:.0f} s on port {backend.port}", flush=True)
        for mode in args.modes:
            cmd = [str(REPO / "envs" / "mstar" / "bin" / "python"), str(REPO / "src" / "scenarios" / "protocol_probes.py"),
                   "--framework", args.framework, "--mode", mode, "--url", server.url,
                   "--out", str(out_dir / f"{mode}.json"), "--horizon", str(checkpoint.horizon)]
            if backend.http_arm and mode == "e2e":
                cmd += ["--http", server.http_url]       # M* serves HTTP /generate and the OpenPI socket on one port
            print(f"== {args.framework} {mode}", flush=True)
            rc = subprocess.run(cmd, cwd=REPO).returncode
            server.snapshot(f"after_{mode}")
            if rc != 0:
                print(f"mode {mode} failed with exit code {rc}", flush=True)
                return rc
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
