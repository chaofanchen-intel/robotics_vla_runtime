#!/usr/bin/env python3
"""Level 1 measurement for the three frameworks, interleaved in time.

For every repeat, each framework gets a fresh server start and one pass over the cells (src/scenarios/level1_periodic.py).
The order of the frameworks rotates with the repeat, so slow changes on the shared host (CPU load from other jobs) do
not always hit the same framework. Output: <run-dir>/<framework>/level1_r<repeat>.json plus server logs, GPU snapshots
and load averages.

    envs/mstar/bin/python scripts/run_level1.py --run-dir runs/<date>_level1 --repeats 3
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


def one_pass(name: str, repeat: int, run_dir: Path, checkpoint: backends.Checkpoint, extra: list[str]) -> int:
    backend = backends.get(name)
    out_dir = run_dir / name
    with Server(backend, checkpoint, out_dir, tag=f"_r{repeat}") as server:
        print(f"[{name} r{repeat}] healthy after {server.ready_s:.0f} s", flush=True)
        cmd = [str(REPO / "envs" / "mstar" / "bin" / "python"), str(REPO / "src" / "scenarios" / "level1_periodic.py"),
               "--framework", name, "--url", server.url, "--horizon", str(checkpoint.horizon),
               "--repeat-index", str(repeat), "--out", str(out_dir / f"level1_r{repeat}.json"), *extra]
        rc = subprocess.run(cmd, cwd=REPO).returncode
        server.snapshot("after")
        return rc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--checkpoint", choices=tuple(backends.CHECKPOINTS), default="pi05_droid")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--frameworks", nargs="+", choices=backends.LEVEL1_ALL, default=list(backends.LEVEL1))
    p.add_argument("--duration", type=float, default=40.0)
    p.add_argument("--warmup", type=float, default=3.0)
    p.add_argument("--cells", nargs="+", default=None)
    args = p.parse_args()
    extra = ["--duration", str(args.duration), "--warmup", str(args.warmup)]
    if args.cells:
        extra += ["--cells", *args.cells]
    checkpoint = backends.CHECKPOINTS[args.checkpoint]
    failures = 0
    for r in range(args.repeats):
        k = r % len(args.frameworks)
        for name in args.frameworks[k:] + args.frameworks[:k]:
            if (args.run_dir / name / f"level1_r{r}.json").exists():
                print(f"[{name} r{r}] exists, skipped", flush=True)
                continue
            rc = one_pass(name, r, args.run_dir, checkpoint, extra)
            if rc != 0:
                failures += 1
                print(f"[{name} r{r}] FAILED rc={rc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
