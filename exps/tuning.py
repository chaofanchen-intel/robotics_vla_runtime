#!/usr/bin/env python3
"""Which tunable kernels have a tuned config for *this* GPU, and what to run for the ones that do not.

    envs/mstar/bin/python -m exps.tuning status [--hardware rtx5880] [--require]

A baseline running a kernel on conservative default tiles while M* runs tuned ones (or the reverse) is a configuration
difference, not an engine difference, so every framework is tuned wherever it can be. The table below is what exists:

* vLLM-Omni BAGEL MoT GEMM        tuner in external/vllm-omni, automated:  `python -m exps.suite run bagel-mot-tune --execute`
* M* fused-MoE (Qwen3-Omni)       tuner in external/mstar (`benchmark/kernels/sweep.py`), automated:
                                  `python -m exps.suite run mstar-moe-tune --execute`; tables are looked up by exact torch
                                  device name under `MSTAR_KERNEL_TABLES` (set by the harness) and then in M*'s package, which
                                  ships H100 / H200 / GB200 / B300 tables: an H200 *NVL* or an RTX 5880 matches none of them
* vLLM fused-MoE (Qwen3-Omni)     external/vllm (v0.32.0 checkout, matching the env): `python -m exps.suite run vllm-moe-tune --execute`
* SGLang fused-MoE Triton         external/sglang (v0.5.22 checkout, matching the env): `python -m exps.suite run sglang-moe-tune --execute`
                                  (writes under the installed Triton version's folder)

Tuned files are matched by device name, so a file for another GPU is not used (the framework falls back to default tiles with
a warning in its log). SGLang additionally keys the folder by Triton version and, when the exact version is missing, falls back
to the newest *other* version's table with a warning ("Fallback to triton version ..."): it works, but the table was tuned under
another Triton. This tool reports that as STALE: usable, not what a fresh tune on the installed version would give.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from exps import hardware
from exps.core import paths

MSTAR_TABLE_DIR = paths.CONFIGS / "mstar" / "kernel_tables"
MSTAR_PACKAGE_TABLES = paths.EXTERNAL / "mstar" / "mstar" / "kernels" / "tables"
#: M* `moe_gemm` shape keys for Qwen3-Omni: N = 2 x per-expert intermediate (gate and up), K = hidden size.
MSTAR_MOE_SHAPES = {"thinker": {"E": 128, "N": 1536, "K": 2048, "top_k": 8},
                    "talker": {"E": 128, "N": 768, "K": 1024, "top_k": 6}}
VLLM_TUNED_DIR = paths.CONFIGS / "vllm_omni" / "tuned"
VLLM_WHEEL_MOE_DIR = (paths.ENVS / "vllm-omni" / "lib" / "python3.12" / "site-packages" / "vllm" / "model_executor"
                      / "layers" / "fused_moe" / "configs")
SGLANG_SITE = paths.ENVS / "sglang-omni" / "lib" / "python3.12" / "site-packages"

#: Qwen3-Omni MoE shapes (E experts, N = per-expert intermediate size at tp=1): Thinker and Talker.
QWEN_MOE_SHAPES = {"thinker": (128, 768), "talker": (128, 384)}


def qwen_moe_shapes() -> dict[str, tuple[int, int]]:
    """The fused-MoE shapes vLLM / SGLang look up on this machine: the Thinker's N is split over its tensor-parallel ranks
    (RTX 5880: TP=2 -> N=384, the same file name as the TP=1 Talker, so those two share one table)."""
    experts, n = QWEN_MOE_SHAPES["thinker"]
    return {"thinker": (experts, n // hardware.current().qwen_thinker_tp), "talker": QWEN_MOE_SHAPES["talker"]}


def mstar_moe_shapes() -> dict[str, dict[str, int]]:
    """M* `moe_gemm` shape keys as the Thinker's tensor-parallel rank sees them (N = 2 x intermediate / TP)."""
    thinker = dict(MSTAR_MOE_SHAPES["thinker"])
    thinker["N"] //= hardware.current().qwen_thinker_tp
    return {"thinker": thinker, "talker": dict(MSTAR_MOE_SHAPES["talker"])}


@dataclass
class Item:
    name: str
    framework: str
    ok: bool | None          # None: not tunable / not applicable
    expected: str
    how: str
    stale: bool = False      # a table exists but under another Triton version (SGLang falls back to it with a warning)


def device_name_underscored() -> str:
    return (hardware.gpu_name(hardware.current().primary_gpu) or "unknown").replace(" ", "_")


def _name_candidates(raw: str) -> list[str]:
    parts = raw.split("_")
    return ["_".join(parts[:n]) for n in range(len(parts), 1, -1)]


def mstar_stem(raw_underscored: str) -> str:
    """M*'s table file stem: the torch device name with each run of other characters as one `_` (`NVIDIA H200 NVL` -> NVIDIA_H200_NVL)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", raw_underscored).strip("_")


def mstar_table_entry(raw_underscored: str, shape: dict[str, int], dtype: str = "bfloat16") -> Path | None:
    """The table file holding an entry for (dtype, shape) under this GPU's exact name (M* then tries the architecture's file
    `sm_NN`, which is not checked here: a hit there is fine, a miss only means the sweep is still to do)."""
    import json

    for folder in (MSTAR_TABLE_DIR, MSTAR_PACKAGE_TABLES):
        path = folder / "moe_gemm" / f"{mstar_stem(raw_underscored)}.json"
        if not path.is_file():
            continue
        try:
            entries = json.loads(path.read_text(encoding="utf-8")).get("entries", [])
        except (OSError, ValueError):
            continue
        if any(e.get("dtype") == dtype and e.get("shape") == shape for e in entries):
            return path
    return None


def triton_version(site: Path = SGLANG_SITE) -> str | None:
    for dist in site.glob("triton-*.dist-info"):
        return dist.name[len("triton-"):-len(".dist-info")]
    return None


def status() -> list[Item]:
    from exps.tasks.bagel import cell as bagel_cell

    hw = hardware.current()
    raw = device_name_underscored()
    items: list[Item] = []

    mot = bagel_cell.mot_config_filename()
    items.append(Item("BAGEL MoT GEMM", "vllm-omni", None if mot is None else (VLLM_TUNED_DIR / mot).is_file(),
                      str(VLLM_TUNED_DIR / (mot or "device_name=<GPU>,dtype=w16a16.json")),
                      "python -m exps.suite run bagel-mot-tune --execute   (needs `ray` in envs/vllm-omni; ~5 min)"))

    for role, (experts, n) in qwen_moe_shapes().items():
        found = None
        for folder in (VLLM_TUNED_DIR, VLLM_WHEEL_MOE_DIR):
            for candidate in _name_candidates(raw):
                path = folder / f"E={experts},N={n},device_name={candidate}.json"
                if path.is_file():
                    found = path
                    break
            if found:
                break
        items.append(Item(f"fused-MoE E={experts},N={n} (Qwen3-Omni {role})", "vllm-omni", found is not None,
                          str(found or VLLM_TUNED_DIR / f"E={experts},N={n},device_name={raw}.json"),
                          "python -m exps.suite run vllm-moe-tune --execute   (needs `ray` in envs/vllm-omni; "
                          "tuner: external/vllm/benchmarks/kernels/benchmark_moe.py)"))

    triton = triton_version()
    sglang_dir = paths.CONFIGS / "sglang_omni" / f"moe_{hw.name}"
    version_dir = f"triton_{triton.replace('.', '_')}" if triton else "triton_<version>"
    for role, (experts, n) in qwen_moe_shapes().items():
        path = sglang_dir / "configs" / version_dir / f"E={experts},N={n},device_name={raw}.json"
        stale = sorted(p.parent.name for p in (sglang_dir / "configs").glob(f"*/E={experts},N={n},device_name={raw}.json")
                       if p.parent.name != version_dir) if (sglang_dir / "configs").is_dir() else []
        how = ("python -m exps.suite run sglang-moe-tune --execute   (tuner: external/sglang/benchmark/kernels/fused_moe_triton/; "
               f"writes {sglang_dir}/configs/{version_dir}/)")
        if stale and not path.is_file():
            how = (f"STALE: a table exists under {stale} and SGLang falls back to it with a warning, but the installed Triton "
                   f"is {triton} ({version_dir}). Re-tune on the installed version for a fair baseline. " + how)
        items.append(Item(f"fused-MoE E={experts},N={n} (Qwen3-Omni {role})", "sglang-omni",
                          path.is_file() or bool(stale), str(path), how, stale=bool(stale) and not path.is_file()))

    for role, shape in mstar_moe_shapes().items():
        found = mstar_table_entry(raw, shape)
        items.append(Item(f"moe_gemm {','.join(f'{k}={v}' for k, v in shape.items())} (Qwen3-Omni {role}, TP{hardware.current().qwen_thinker_tp if role == 'thinker' else 1})", "mstar",
                          found is not None, str(found or MSTAR_TABLE_DIR / "moe_gemm" / f"{mstar_stem(raw)}.json"),
                          "cd external/mstar && ../../envs/mstar/bin/python -m benchmark.kernels.sweep moe_gemm --dtype bfloat16 "
                          f"--shape {','.join(f'{k}={v}' for k, v in shape.items())} --note 'Qwen3-Omni {role}, TP{hardware.current().qwen_thinker_tp if role == 'thinker' else 1}' "
                          f"--out {MSTAR_TABLE_DIR}   (or: python -m exps.suite run mstar-moe-tune --execute; needs an idle GPU)"))
    return items


def render(items: list[Item]) -> str:
    hw = hardware.current()
    lines = [f"hardware: {hw.name} (GPU {hw.primary_gpu}: {device_name_underscored()}), triton for SGLang: {triton_version()}", ""]
    for item in items:
        mark = "STALE  " if item.stale else {True: "OK     ", False: "MISSING", None: "n/a    "}[item.ok]
        lines.append(f"[{mark}] {item.framework:<10} {item.name}\n           file: {item.expected}")
        if item.ok is not True or item.stale:
            lines.append(f"           do:   {item.how}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("status",))
    parser.add_argument("--hardware", default=None)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--require", action="store_true", help="exit 1 if any tunable item is missing")
    args = parser.parse_args()
    hardware.set_current(args.hardware, args.gpus)
    items = status()
    print(render(items))
    return 1 if args.require and any(i.ok is False or i.stale for i in items) else 0


if __name__ == "__main__":
    sys.exit(main())
