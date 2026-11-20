"""Start a backend's server on the GPU, wait for it to be healthy, stop it; keep the logs and snapshots next to the results.

    with Server(backend, checkpoint, out_dir, tag="_r0") as server:
        ...                      # server.ready_s, server.snapshot("after_e2e")
"""
from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from pathlib import Path

from backends.base import REPO, Backend
from backends.checkpoints import Checkpoint


def gpu_line() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu,clocks.sm,temperature.gpu",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout


def wait_ready(port: int, proc: subprocess.Popen, timeout_s: float = 1800.0) -> float:
    t0 = time.time()
    url = f"http://127.0.0.1:{port}/health"
    while time.time() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with code {proc.returncode}; see the server log")
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


class Server:
    def __init__(self, backend: Backend, checkpoint: Checkpoint, out_dir: Path, tag: str = ""):
        self.backend, self.checkpoint, self.out_dir, self.tag = backend, checkpoint, Path(out_dir), tag
        self.proc: subprocess.Popen | None = None
        self.ready_s: float | None = None
        self.url = f"ws://127.0.0.1:{backend.port}"
        self.http_url = f"http://127.0.0.1:{backend.port}"

    def _write(self, name: str, text: str) -> None:
        (self.out_dir / name).write_text(text)

    def snapshot(self, name: str) -> None:
        """nvidia-smi snapshot as gpu_<name><tag>.txt"""
        self._write(f"gpu_{name}{self.tag}.txt", gpu_line())

    def __enter__(self) -> "Server":
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.snapshot("before")
        self._write(f"loadavg_before{self.tag}.txt", Path("/proc/loadavg").read_text())
        command = self.backend.serve_command(self.checkpoint)
        self._write(f"serve_command{self.tag}.txt", command + "\n")
        self._log = (self.out_dir / f"server{self.tag}.log").open("w")
        self.proc = subprocess.Popen(["bash", "-c", command], cwd=REPO, stdout=self._log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        try:
            self.ready_s = wait_ready(self.backend.port, self.proc)
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self._write(f"ready_seconds{self.tag}.txt", f"{self.ready_s:.1f}\n")
        return self

    def __exit__(self, *exc) -> None:
        if self.proc is not None:
            stop(self.proc)
        self._log.close()
        self._write(f"loadavg_after{self.tag}.txt", Path("/proc/loadavg").read_text())
        self.snapshot("after_stop")
