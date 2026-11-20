#!/usr/bin/env python3
"""Build HF-cache directories that make M* serve BAGEL with a different `num_timesteps`.

Why
---
M* reads `num_timesteps` from the model's `config.json` at server start-up (`BagelModelConfig`), not
from the request. Request-level `num_timesteps` / `num_inference_steps` are ignored, so a per-request
step sweep leaves M* at 50 steps. To sweep M*'s step count the server must be started against a
config.json that carries the value.

How
---
For each N this creates ``models/cache_bagel_steps<N>/`` mirroring the real HF cache layout of
``ByteDance-Seed/BAGEL-7B-MoT`` at the pinned revision. Every file in the snapshot is a **symlink**
to the real blob (no weights are copied) except ``config.json``, which is a copy with the top-level
key ``"num_timesteps": N`` added. Nothing under ``external/`` or ``models/cache/`` is modified.

    envs/mstar/bin/python exps/tools/make_bagel_steps_cache.py            # 5 10 25
    envs/mstar/bin/python exps/tools/make_bagel_steps_cache.py --steps 5 10 25 --verify
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):  # pragma: no cover - direct invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from exps.core import paths

REPO_DIR = "models--ByteDance-Seed--BAGEL-7B-MoT"
REVISION = "5019f57d168e5816e8f3f701b17cc816bb7cf24b"


def build(steps: int, *, cache_root: Path | None = None) -> Path:
    cache_root = cache_root or paths.MODELS_CACHE
    src_repo = cache_root / REPO_DIR
    src_snapshot = src_repo / "snapshots" / REVISION
    if not (src_snapshot / "config.json").is_file():
        raise SystemExit(f"BAGEL snapshot not found: {src_snapshot}")
    dst_root = cache_root.parent / f"cache_bagel_steps{steps}"
    dst_repo = dst_root / REPO_DIR
    dst_snapshot = dst_repo / "snapshots" / REVISION
    dst_snapshot.mkdir(parents=True, exist_ok=True)
    (dst_repo / "refs").mkdir(exist_ok=True)
    (dst_repo / "refs" / "main").write_text(REVISION)

    for entry in sorted(src_snapshot.iterdir()):
        target = dst_snapshot / entry.name
        if target.is_symlink() or target.exists():
            target.unlink()
        if entry.name == "config.json":
            config = json.loads(entry.read_text(encoding="utf-8"))
            config["num_timesteps"] = steps
            target.write_text(json.dumps(config, indent=2), encoding="utf-8")
        else:
            # Resolve to the real blob so the link survives even if the source is itself a symlink.
            os.symlink(entry.resolve(), target)
    return dst_root


def verify(steps: int, *, cache_root: Path | None = None) -> list[str]:
    """Problems found in `models/cache_bagel_steps<steps>`; empty means ready."""
    cache_root = cache_root or paths.MODELS_CACHE
    snapshot = cache_root.parent / f"cache_bagel_steps{steps}" / REPO_DIR / "snapshots" / REVISION
    problems = []
    config = snapshot / "config.json"
    if not config.is_file():
        return [f"missing {config}"]
    if json.loads(config.read_text(encoding="utf-8")).get("num_timesteps") != steps:
        problems.append(f"{config} does not carry num_timesteps={steps}")
    for entry in snapshot.iterdir():
        if entry.name != "config.json" and not entry.resolve().exists():
            problems.append(f"dangling link {entry}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, nargs="+", default=[5, 10, 25])
    parser.add_argument("--verify", action="store_true", help="Only check existing cache dirs.")
    args = parser.parse_args()
    bad = 0
    for n in args.steps:
        if not args.verify:
            print(f"built {build(n)}")
        problems = verify(n)
        print(f"steps={n}: {'OK' if not problems else 'PROBLEMS'}")
        for p in problems:
            print("  -", p)
        bad += bool(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
