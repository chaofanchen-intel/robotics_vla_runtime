#!/usr/bin/env python3
"""Build the one-card H200 versions of the counterfactual serve configs the experiments swap in.

The RTX 5880 box needs multi-card topologies (Qwen3-Omni: Thinker TP=2 on two cards, Talker on a third; BAGEL two-stage: one card
per stage), so every counterfactual config under `exps/configs/hardware/rtx5880/` is "the 5880 recipe plus one change" (prefix
cache on, Thinker `max_num_seqs` 16, a Mooncake connector, a chunked-prefill budget, ...). The H200 NVL (141 GB) runs the default
single-card recipes. Its counterfactual configs must make *the same one change*, on top of its own default:

    H200 variant  =  H200 default config  +  (5880 variant  -  5880 base)

This script computes that, structurally (PyYAML, lists of stages matched by `stage_id`), writes the files under
`exps/configs/hardware/h200nvl/`, and `--check` fails when a file is missing or stale. Edit the SPECS below, never the generated
files; a change to a 5880 variant shows up as a stale H200 file.

Two deliberate differences from a pure delta, both for M* Qwen (`mstar/qwen3omni_1gpu.yaml`):

* `resources` of the 5880 variants is ignored: it holds the 5880 memory budget (`max_num_pages` 512, `cpu_offload_pages`) and the
  TP=2 prefix-cache switches (`prefix_cache_tp`, a salt, offload 0). On one rank M* opens the Thinker prefix cache by itself.
* The chunk / bucket variants add `resources.thinker_kv.prefix_cache: false`. The 5880 experiments measure a cold prefill on every
  request *because* the cache is off at TP=2 (server.log: "prefix cache off at world size 2"); on one rank the same repeated prompt
  would be a cache hit (server.log: "whole prompt already cached True") and the experiment would measure nothing.
"""
from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
OUT = CONFIGS / "hardware" / "h200nvl"
ADA = CONFIGS / "hardware" / "rtx5880"

BAGEL_VLLM_BASE = ("vllm_omni/bagel.yaml", "vllm_omni/bagel_two_stage_2gpu.yaml")
QWEN_VLLM_BASE = ("vllm_omni/qwen3_omni_full_1gpu_default_async_c2wgraph.yaml", "vllm_omni/qwen3_omni_3gpu_thinker_tp2_async_c2wgraph.yaml")
QWEN_SGLANG_BASE = ("sglang_omni/qwen3_omni_colocated_h200_partial_start.yaml", "sglang_omni/qwen3_omni_3gpu_thinker_tp2_partial_start.yaml")
QWEN_MSTAR_BASE = ("mstar/qwen3omni_1gpu.yaml", "mstar/qwen3omni_3gpu_thinker_tp2.yaml")
BAGEL_MSTAR_BASE = ("mstar/bagel_single_gpu.yaml", "mstar/bagel_single_gpu_kv512.yaml")

#: M* on one rank: the Thinker prefix cache is on by default; the experiments that measure a cold prefill switch it off.
CACHE_OFF = {"resources": {"thinker_kv": {"prefix_cache": False}}}


@dataclass(frozen=True)
class Spec:
    target: str                        # file under configs/hardware/h200nvl/
    h200_base: str                     # the H200 default it modifies (relative to configs/)
    ada_base: str                      # the 5880 recipe the 5880 variant modifies (relative to configs/hardware/rtx5880/)
    ada_variant: str | None            # the 5880 variant whose change is copied (None: no change besides `extra`)
    purpose: str
    skip: tuple[str, ...] = ()         # top-level keys of the 5880 change that do not apply on one card
    extra: dict = field(default_factory=dict)


def _bagel_vllm(suffix: str, purpose: str) -> Spec:
    return Spec(f"vllm_omni/bagel_two_stage_1gpu_{suffix}.yaml", BAGEL_VLLM_BASE[0], BAGEL_VLLM_BASE[1],
                f"vllm_omni/bagel_two_stage_2gpu_{suffix}.yaml", purpose)


def _qwen_vllm(suffix: str, purpose: str) -> Spec:
    return Spec(f"vllm_omni/qwen3_omni_full_1gpu_{suffix}.yaml", QWEN_VLLM_BASE[0], QWEN_VLLM_BASE[1],
                f"vllm_omni/qwen3_omni_3gpu_thinker_tp2_{suffix}.yaml", purpose)


def _qwen_sglang(suffix: str, purpose: str) -> Spec:
    return Spec(f"sglang_omni/qwen3_omni_colocated_h200_{suffix}.yaml", QWEN_SGLANG_BASE[0], QWEN_SGLANG_BASE[1],
                f"sglang_omni/qwen3_omni_3gpu_thinker_tp2_{suffix}.yaml", purpose)


def _qwen_mstar(target: str, ada_variant: str | None, purpose: str) -> Spec:
    return Spec(f"mstar/{target}", QWEN_MSTAR_BASE[0], QWEN_MSTAR_BASE[1],
                f"mstar/{ada_variant}" if ada_variant else None, purpose, skip=("resources",), extra=CACHE_OFF)


SPECS: tuple[Spec, ...] = (
    # --- vLLM-Omni BAGEL two-stage: both stages stay on GPU 0 (Thinker 0.45 of the card, as in the default bagel.yaml) ---------------
    _bagel_vllm("thinker_sync_sched", "Thinker async scheduling off (bagel-async-sched)."),
    _bagel_vllm("prefix_cache", "Thinker prefix caching on (bagel-vllm-prefix-cache, bagel-concurrency-fair)."),
    _bagel_vllm("prefix_cache_seqs8", "Thinker prefix caching on, max_num_seqs 8 (bagel-concurrency-seqs)."),
    _bagel_vllm("prefix_cache_seqs16", "Thinker prefix caching on, max_num_seqs 16 (bagel-concurrency-seqs)."),
    _bagel_vllm("mooncake_tcp", "Thinker -> DiT edge on MooncakeTransferEngineConnector over TCP (bagel-connector*)."),
    _bagel_vllm("nixl", "Thinker -> DiT edge on NixlConnector / UCX (bagel-connector*)."),
    _bagel_vllm("mooncake_tcp_prefix_cache", "Mooncake TCP edge and Thinker prefix caching on (bagel-i2i-fair, bagel-i2i-concurrency)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_seqs16", "Mooncake TCP, prefix caching on, Thinker max_num_seqs 16 (bagel-i2i-concurrency)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_kvtimeout900", "Mooncake TCP, prefix caching on, DiT KV receive timeout 900 s (bagel-i2i-kvtimeout)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_seqs16_kvtimeout900",
                "Mooncake TCP, prefix caching on, Thinker max_num_seqs 16, DiT KV receive timeout 900 s (bagel-i2i-kvtimeout)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_pool16g", "Mooncake TCP, prefix caching on, sender pool 16 GiB instead of 4 GiB (bagel-i2i-pool)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_pool8g", "Mooncake TCP, prefix caching on, sender pool 8 GiB (bagel-i2i-pool-size: smallest pool that serves B = 16)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_pool12g", "Mooncake TCP, prefix caching on, sender pool 12 GiB (bagel-i2i-pool-size)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_pool16g_kvtimeout900", "Mooncake TCP, prefix caching on, sender pool 16 GiB, DiT KV receive timeout 900 s (bagel-i2i-pool)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_seqs16_pool16g", "Mooncake TCP, prefix caching on, Thinker max_num_seqs 16, sender pool 16 GiB (bagel-i2i-pool)."),
    _bagel_vllm("mooncake_tcp_prefix_cache_seqs16_pool16g_kvtimeout900",
                "Mooncake TCP, prefix caching on, Thinker max_num_seqs 16, sender pool 16 GiB, DiT KV receive timeout 900 s (bagel-i2i-pool)."),
    # --- vLLM-Omni Qwen3-Omni (the batched deploy already exists for one card: qwen3_omni_full_1gpu_batch_async_c2wgraph.yaml) --------
    _qwen_vllm("thinker_sync_sched", "Thinker async scheduling off (qwen-async-sched, qwen-async-sched-a2t)."),
    _qwen_vllm("prefix_cache", "Thinker prefix caching on (qwen-fair-*, qwen-a2t-cache-warm)."),
    _qwen_vllm("talker_greedy", "Talker decoding greedily with the yaml's repetition_penalty (qwen-t2s-fit)."),
    _qwen_vllm("codepred", "Code predictor K/V cache, short-KV attention and fused sampling on (qwen-t2s-vllm-opt; upstream's H200 profile sets the same three)."),
    _qwen_vllm("mrv2_codepred", "codepred plus the second-generation model runner (MRv2) on the one card (qwen-t2s-vllm-opt; upstream's qwen3_omni_moe_mrv2_h200.yaml)."),
    # --- SGLang-Omni Qwen3-Omni -----------------------------------------------------------------------------------------------------
    _qwen_sglang("no_radix", "Thinker radix cache off (every fair / prefill / ttft experiment's SGLang cache-off arm)."),
    _qwen_sglang("no_async_decode", "SGLang's lookahead / async decode off (the control arm of qwen-fair-*)."),
    # --- M* Qwen3-Omni (cache off on every one-rank variant, see the module docstring) ------------------------------------------------
    _qwen_mstar("qwen3omni_1gpu_prefix_cache_off.yaml", None,
                "The default recipe with the Thinker prefix cache off: the M* arms of the fair / ttft / prefill experiments (cold prefill each request)."),
    _qwen_mstar("qwen3omni_1gpu_chunk4096.yaml", "qwen3omni_3gpu_thinker_tp2_chunk4096.yaml",
                "Thinker chunked-prefill budget 4,096 instead of 1,024 (qwen-prefill-*)."),
    _qwen_mstar("qwen3omni_1gpu_chunk4096_bucket1152.yaml", "qwen3omni_3gpu_thinker_tp2_chunk4096_bucket1152.yaml",
                "Budget 4,096 and an extra 1,152 prefill graph rung (qwen-prefill-len*, qwen-fair-txt-b1152)."),
    _qwen_mstar("qwen3omni_1gpu_chunk4096_bucket1152_1536.yaml", "qwen3omni_3gpu_thinker_tp2_chunk4096_bucket1152_1536.yaml",
                "Budget 4,096 with 1,152 and 1,536 prefill graph rungs (qwen-prefill-len-dense)."),
    # --- M* BAGEL ----------------------------------------------------------------------------------------------------------------
    Spec("mstar/bagel_single_gpu_chunk8192.yaml", BAGEL_MSTAR_BASE[0], BAGEL_MSTAR_BASE[1],
         "mstar/bagel_single_gpu_kv512_chunk8192.yaml", "LLM chunked-prefill budget 8,192 instead of 1,024 (bagel-prefill-chunk)."),
    Spec("mstar/bagel_single_gpu_kv1024.yaml", BAGEL_MSTAR_BASE[0], BAGEL_MSTAR_BASE[1],
         "mstar/bagel_single_gpu_kv1024.yaml", "KV pool 1,024 pages instead of the default 2,048 (bagel-kv-pool, the same pool test on this card)."),
)

#: files copied unchanged (nothing in them depends on the topology)
COPIES = (("vllm_omni/logging/ms_timestamps.json", "vllm_omni/logging/ms_timestamps.json"),)


# --- structural delta ------------------------------------------------------------------------------------------------------------
_DEL = object()


def _stage_list(value) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(v, dict) and "stage_id" in v for v in value)


def delta(a, b):
    """What turns `a` into `b`: per dict key set / delete / nest; lists of stages matched by stage_id; any other list replaced whole."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for key in list(a) + [k for k in b if k not in a]:
            if key not in b:
                out[key] = ("del", None)
            elif key not in a:
                out[key] = ("set", copy.deepcopy(b[key]))
            elif a[key] != b[key]:
                nested = delta(a[key], b[key])
                out[key] = ("nest", nested) if nested is not None else ("set", copy.deepcopy(b[key]))
        return out
    if _stage_list(a) and _stage_list(b) and [s["stage_id"] for s in a] == [s["stage_id"] for s in b]:
        return {("stage", sa["stage_id"]): ("nest", delta(sa, sb)) for sa, sb in zip(a, b) if sa != sb}
    return None


def apply(target, changes, where: str = ""):
    """Apply `delta` output to a parsed config (in place); a key the target lacks where the delta edits or deletes it is an error."""
    for key, (kind, value) in changes.items():
        if isinstance(key, tuple):                          # a stage entry of a list
            stage = next((s for s in target if s.get("stage_id") == key[1]), None)
            if stage is None:
                raise SystemExit(f"{where}: the H200 base has no stage {key[1]} to change")
            apply(stage, value, f"{where}[stage {key[1]}]")
        elif kind == "set":
            target[key] = copy.deepcopy(value)
        elif kind == "del":
            target.pop(key, None)
        elif kind == "nest":
            if key not in target:
                # the 5880 recipe spells out a block (SGLang `engine:`) the one-card default leaves to its defaults: the change adds keys
                # to it, so create the block; a change that edits stages of a list the base lacks cannot be copied
                if any(isinstance(k, tuple) for k in value):
                    raise SystemExit(f"{where}.{key}: the H200 base lacks this list, so the 5880 change to its stages cannot be copied")
                target[key] = {}
            apply(target[key], value, f"{where}.{key}")
    return target


def _merge(target: dict, extra: dict) -> None:
    for key, value in extra.items():
        if isinstance(value, dict):
            _merge(target.setdefault(key, {}), value)
        else:
            target[key] = value


def _load(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build(spec: Spec) -> str:
    base = _load(CONFIGS / spec.h200_base)
    changes = {}
    if spec.ada_variant:
        changes = delta(_load(ADA / spec.ada_base), _load(ADA / spec.ada_variant))
        if changes is None:
            raise SystemExit(f"{spec.ada_variant}: not a mapping")
        for key in spec.skip:
            changes.pop(key, None)
    config = apply(copy.deepcopy(base), changes, spec.target)
    _merge(config, spec.extra)
    header = [
        f"# H200 NVL (one card) version of the {spec.ada_variant or spec.ada_base} counterfactual of exps/configs/hardware/rtx5880/.",
        f"# = exps/configs/{spec.h200_base} with the change that 5880 file makes over its own base ({spec.ada_base}): {spec.purpose}",
        "# GENERATED by exps/tools/h200_variants.py: edit the spec there, not this file (`python -m exps.tools.h200_variants --check`).",
    ]
    if spec.skip or spec.extra:
        header.append(f"# Not copied from the 5880 file: {', '.join(spec.skip) or 'nothing'}; set instead: {spec.extra or 'nothing'} (see the generator's docstring).")
    body = yaml.safe_dump(config, sort_keys=False, allow_unicode=True, width=140)
    return "\n".join(header) + "\n" + body


def outputs() -> dict[Path, str]:
    files = {OUT / spec.target: build(spec) for spec in SPECS}
    for source, target in COPIES:
        files[OUT / target] = (ADA / source).read_text(encoding="utf-8")
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if a generated file is missing or differs (writes nothing)")
    args = parser.parse_args(argv)
    stale = []
    for path, text in outputs().items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(path)
            if not args.check:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
    shown = ", ".join(str(p.relative_to(OUT)) for p in stale) or "none"
    if args.check:
        print(f"stale or missing: {shown}")
        return 1 if stale else 0
    print(f"wrote {len(stale)} file(s): {shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
