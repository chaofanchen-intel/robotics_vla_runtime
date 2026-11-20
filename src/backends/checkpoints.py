"""The Pi0.5 checkpoints the probes and the Level 1 runtime can serve, with the per-framework config of each."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from exps.tasks.pi05 import cell

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "models" / "cache"


@dataclass(frozen=True)
class Checkpoint:
    name: str
    snapshot: Path
    horizon: int                 # action chunk length the server returns
    config: dict[str, Path]      # backend family -> serving config


CHECKPOINTS = {
    # lerobot/pi05_base: 50-step chunk. The 2026-11-20 protocol probes (docs/design.md 2.6) used it.
    "pi05_base": Checkpoint("pi05_base", cell.SNAPSHOT, 50, {
        "mstar": REPO / "configs" / "mstar_pi05_openpi.yaml",
        "vllm-omni": cell.CONFIGS["vllm-omni"], "sglang-core": cell.CONFIGS["sglang-core"],
        "sglang-core-bucketed": REPO / "configs" / "sglang_pi05_base_buckets.json"}),
    # lerobot/pi05_droid: DROID-finetuned, 15-step chunk = 1 s at DROID's 15 Hz. The Level 1 runtime uses it.
    "pi05_droid": Checkpoint(
        "pi05_droid", CACHE / "models--lerobot--pi05_droid" / "snapshots" / "72824c0a93f00ce5bb8bedb7feb58953ba1da364", 15, {
            "mstar": REPO / "configs" / "mstar_pi05_droid_openpi.yaml",
            "vllm-omni": REPO / "configs" / "vllm_omni_pi05_droid.yaml",
            "sglang-core": REPO / "configs" / "sglang_pi05_droid.json",
            "sglang-core-bucketed": REPO / "configs" / "sglang_pi05_droid_buckets.json"}),
}
