#!/usr/bin/env python3
"""Start one Pi0.5 server per framework on a free GPU, run the probes, stop it.

Usage (from the repository root):
    envs/mstar/bin/python harness/run_all.py --framework mstar
    envs/mstar/bin/python harness/run_all.py --framework mstar-rust
    envs/mstar/bin/python harness/run_all.py --framework vllm-omni
    envs/mstar/bin/python harness/run_all.py --framework sglang-core

The server commands are the ones the exps harness builds (exps/frameworks, exps/tasks/pi05/cell.py), so the
engines, configs and cache directory are the same as the existing Pi0.5 campaigns. The only difference is the
M* config: configs/mstar_pi05_openpi.yaml adds an `openpi:` block so the
M* OpenPI WebSocket serves the same request shape as the other two. The GPU is the one named by the serve
command (CUDA_VISIBLE_DEVICES=0 in exps/profiles/hardware/rtx5880.json); check `nvidia-smi` first.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import exps.frameworks as frameworks  # noqa: E402
from exps.tasks.pi05 import cell  # noqa: E402

HARNESS = Path(__file__).resolve().parent
OPENPI_CONFIG = REPO / "configs" / "mstar_pi05_openpi.yaml"
PORTS = {"mstar": 18101, "vllm-omni": 18102, "sglang-core": 18103, "mstar-rust": 18104}
MSTAR_LATEST = REPO / "external" / "mstar"
# mstar_rust built from external/mstar/rust (maturin build --release); the unpacked wheel is kept here, see build/README.md
MSTAR_RUST_LATEST = REPO / "build" / "mstar_rust"
# mstar-server (the default M* frontend): cargo build --release --manifest-path rust/server/Cargo.toml,
# copy kept in build/bin; see build/README.md
MSTAR_SERVER_BIN = REPO / "build" / "bin" / "mstar-server"
MODES = ("e2e", "period", "burst", "cancel", "concurrency")


def serve_command(framework: str, port: int) -> str:
    if framework in ("mstar", "mstar-rust"):
        # The campaign's M* (external/mstar, 9c6f1ad) predates the OpenPI WebSocket, so `/` answers 403 there.
        # The OpenPI socket is in external/mstar (2c8ffde8); its code is imported through PYTHONPATH, with
        # the same envs/mstar dependencies, except mstar_rust, rebuilt from the latest rust sources (the installed one
        # predates the source-hash check, which the latest graph runtime enforces).
        # "mstar" uses the deprecated Python frontend (--python-frontend); "mstar-rust" uses the default mstar-server.
        frontend = "--python-frontend" if framework == "mstar" else f"--rust-frontend-bin {MSTAR_SERVER_BIN}"
        command = frameworks.mstar_serve(model="pi05", config=OPENPI_CONFIG, port_number=port, tag="robot-probe",
                                             extra=frontend)
        needle = "exec envs/mstar/bin/mstar serve"
        assert command.count(needle) == 1, "unexpected mstar serve command shape"
        return command.replace(needle, f"exec env PYTHONPATH={MSTAR_RUST_LATEST}:{MSTAR_LATEST} envs/mstar/bin/mstar serve")
    if framework == "vllm-omni":
        extra = ("--enforce-eager --robot-openpi-idle-timeout 1800 --custom-pipeline-args "
                 + shlex.quote(json.dumps({"tokenizer": str(cell.TOKENIZER)})))
        return frameworks.vllm_omni_serve(model_path=cell.SNAPSHOT, deploy_config=cell.CONFIGS["vllm-omni"],
                                          port_number=port, extra=extra)
    if framework == "sglang-core":
        return frameworks.sglang_core_serve(model_path=cell.SNAPSHOT, pipeline="Pi05Pipeline",
                                            pipeline_config=cell.CONFIGS["sglang-core"], port_number=port)
    raise ValueError(framework)


def gpu_line() -> str:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu",
         "--format=csv,noheader"], capture_output=True, text=True).stdout
    return out


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float = 1800.0) -> float:
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see server.log")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    return time.time() - t0
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)
    raise TimeoutError(f"server not healthy after {timeout_s} s")


def stop(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def probe_args(framework: str, mode: str, port: int, out: Path, http_port: int | None) -> list[str]:
    args = [str(REPO / "envs" / "mstar" / "bin" / "python"), str(HARNESS / "probes.py"),
            "--framework", framework, "--mode", mode, "--url", f"ws://127.0.0.1:{port}",
            "--out", str(out)]
    if framework.startswith("mstar") and mode == "e2e" and http_port is not None:
        args += ["--http", f"http://127.0.0.1:{http_port}"]
    return args


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--framework", choices=tuple(PORTS), required=True)
    p.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    p.add_argument("--run-dir", type=Path, default=REPO / "runs" / "2026-11-20_ws-probe")
    args = p.parse_args()

    port = PORTS[args.framework]
    out_dir = args.run_dir / args.framework
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gpu_before.txt").write_text(gpu_line())
    command = serve_command(args.framework, port)
    (out_dir / "serve_command.txt").write_text(command + "\n")

    log = (out_dir / "server.log").open("w")
    proc = subprocess.Popen(["bash", "-c", command], cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    try:
        ready_s = wait_ready(port, proc)
        print(f"{args.framework} healthy after {ready_s:.0f} s on port {port}", flush=True)
        (out_dir / "ready_seconds.txt").write_text(f"{ready_s:.1f}\n")
        for mode in args.modes:
            target = out_dir / f"{mode}.json"
            # M* serves its HTTP /generate and its OpenPI WebSocket on the same port
            cmd = probe_args(args.framework, mode, port, target, port)
            print(f"== {args.framework} {mode}", flush=True)
            rc = subprocess.run(cmd, cwd=REPO).returncode
            (out_dir / f"gpu_after_{mode}.txt").write_text(gpu_line())
            if rc != 0:
                print(f"mode {mode} failed with exit code {rc}", flush=True)
                return rc
    finally:
        stop(proc)
        log.close()
        (out_dir / "gpu_after_stop.txt").write_text(gpu_line())
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
