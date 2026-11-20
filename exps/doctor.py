#!/usr/bin/env python3
"""Read-only preflight for the thermal-isolated experiment harness."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from exps import hardware
from exps.core import manifest, paths, registry
from exps.frameworks import ALL


def _is_local_vllm_agent_skill_overlay(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    if len(parts) != 3 or parts[:2] != (".claude", "skills"):
        return False
    skill_dir = paths.EXTERNAL / "vllm-omni" / Path(*parts)
    if not skill_dir.is_dir() or skill_dir.is_symlink():
        return False
    probe = subprocess.run(
        ["git", "-C", str(paths.EXTERNAL / "vllm-omni"), "ls-tree", "HEAD", "--", relative_path],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return probe.stdout.startswith("120000 ")


def _matches_expected_patch_worktree(
    name: str,
    modified_files: list[str],
    expected_patch_files: set[str],
    patch_ok: bool,
) -> bool:
    if not patch_ok or name not in {"mstar", "sglang-omni", "vllm-omni"}:
        return False
    files = set(modified_files)
    if name == "sglang-omni":
        # CLAUDE.md is local agent metadata, not part of the serving runtime.
        files.discard("CLAUDE.md")
    return files == expected_patch_files


def cuda_toolchain_check(env_name: str) -> dict[str, str | bool]:
    """FlashInfer JIT-compiles sampling kernels at server start with the env's own nvcc against the env's own CUDA headers;
    an nvcc newer than the headers (a later `pip install` upgrades `nvidia-cuda-nvcc` on its own) fails every cell of that
    framework with "CUDA compiler and CUDA toolkit headers are incompatible"."""
    root = paths.ENVS / env_name / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13"
    name = f"{env_name} CUDA toolchain (nvcc vs headers)"
    try:
        out = subprocess.run([str(root / "bin" / "nvcc"), "--version"], capture_output=True, text=True, timeout=20).stdout
        compiler = re.search(r"release (\d+)\.(\d+), V([\d.]+)", out)
        header = re.search(r"#define\s+CUDART_VERSION\s+(\d+)", (root / "include" / "cuda_runtime_api.h").read_text())
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"check": name, "ok": False, "detail": str(exc)}
    if not compiler or not header:
        return {"check": name, "ok": False, "detail": "could not read the nvcc version or CUDART_VERSION"}
    headers = int(header.group(1))
    same = (int(compiler.group(1)), int(compiler.group(2))) == (headers // 1000, headers % 1000 // 10)
    detail = f"nvcc {compiler.group(3)}, headers CUDART_VERSION={headers}"
    if not same:
        detail += (f". Reinstall the pinned toolchain: envs/{env_name}/bin/pip install --no-deps "
                   "nvidia-cuda-nvcc==13.0.88 nvidia-nvvm==13.0.88 nvidia-cuda-crt==13.0.88")
    return {"check": name, "ok": same, "detail": detail}


def _compute_capability(gpu_index: int) -> float:
    """The GPU's CUDA compute capability (8.9 for an RTX 5880, 9.0 for an H100/H200); 9.0 when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", f"--id={gpu_index}", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip()) if out.returncode == 0 else 9.0
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return 9.0


# Files each managed overlay may change in its checkout: the targets of exps/patches/<repo>-*.diff. Keep in step with
# bootstrap.sh's verify_* allow-lists; exps/tests/test_doctor.py fails if a patch touches a file missing here.
PATCH_TOUCHED_FILES: dict[str, frozenset[str]] = {
    "mstar": frozenset({
        "benchmark/request.py",
        "mstar/model/orpheus/orpheus_model.py",
        "mstar/model/vjepa2/vjepa2_model.py",
        "mstar/model/components/code_predictor_kernels.py",
    }),
    "sglang-omni": frozenset({
        "sglang_omni/client/types.py",
        "sglang_omni/models/qwen3_omni/request_builders.py",
        "sglang_omni/serve/openai_api.py",
        "sglang_omni/serve/protocol.py",
        "tests/unit_test/qwen3_omni/test_request_builder_text_only.py",
        "tests/unit_test/serve/test_openai_api.py",
    }),
    "vllm-omni": frozenset({
        "vllm_omni/config/omni_config.py",
        "vllm_omni/config/stage_config.py",
        "tests/engine/test_stage_omni_kv_config.py",
        "vllm_omni/diffusion/models/bagel/bagel_transformer.py",
        "vllm_omni/diffusion/models/bagel/pipeline_bagel.py",
        "vllm_omni/entrypoints/openai/serving_chat.py",
        "tests/diffusion/models/bagel/test_step_execution.py",
        "tests/entrypoints/openai_api/test_serving_chat_sampling_params.py",
        "vllm_omni/distributed/omni_connectors/model_runner/omni_connector_runtime.py",
    }),
}


def diagnose() -> list[dict[str, str | bool]]:
    checks: list[dict[str, str | bool]] = []
    ledger = paths.PATCHES / "applied.json"
    if ledger.is_file():
        try:
            json.loads(ledger.read_text(encoding="utf-8"))["applied"]
        except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
            patch_ok, patch_detail = False, f"Invalid patch ledger: {exc}"
        else:
            probe = subprocess.run(
                ["bash", str(paths.PATCHES / "bootstrap.sh"), "--verify"],
                capture_output=True, text=True, timeout=20,
            )
            patch_ok = probe.returncode == 0
            patch_detail = (probe.stdout if patch_ok else probe.stderr).strip()
    else:
        patch_ok, patch_detail = False, "No patch ledger: baseline has not been bootstrapped"
    checks.append({"check": "pinned patch baseline", "ok": patch_ok, "detail": patch_detail})
    for name in manifest.PINNED_REFS:
        state = manifest.external_state(name)
        checks.append({
            "check": f"external/{name} pinned revision",
            "ok": state.get("at_pinned_ref") is True,
            "detail": f"HEAD={state.get('head')}, expected={state.get('pinned_ref')}",
        })
        expected_patch_files: set[str] = set(PATCH_TOUCHED_FILES.get(name, ()))
        if name == "vllm-omni":
            expected_patch_files.update(
                file for file in state.get("modified_files", [])
                if _is_local_vllm_agent_skill_overlay(file)
            )
        clean = state.get("worktree_clean") is True or _matches_expected_patch_worktree(
            name, state.get("modified_files", []), expected_patch_files, patch_ok
        )
        checks.append({
            "check": f"external/{name} working tree",
            "ok": clean,
            "detail": ", ".join(state.get("modified_files", [])) or state.get("status", "clean"),
        })

    for framework in ALL.values():
        if framework.name == "pytorch-eager":
            continue
        executable = framework.python
        checks.append({
            "check": f"{framework.name} interpreter",
            "ok": executable.is_file(),
            "detail": str(executable),
        })

    ucx_prefix = paths.ENVS / "sglang-omni" / "ucx-1.20.1" / "usr"
    ucx_info = ucx_prefix / "bin" / "ucx_info"
    if not ucx_info.is_file():
        checks.append({
            "check": "SGLang UCX 1.20 runtime",
            "ok": False,
            "detail": f"Missing user-local UCX bundle: {ucx_prefix}",
        })
    else:
        ucx_env = os.environ.copy()
        ucx_env["UCX_MODULE_DIR"] = str(ucx_prefix / "lib" / "ucx")
        ucx_lib_dirs = (ucx_prefix / "lib", ucx_prefix / "lib" / "ucx")
        ld_library_path = [str(path) for path in ucx_lib_dirs]
        if ucx_env.get("LD_LIBRARY_PATH"):
            ld_library_path.append(ucx_env["LD_LIBRARY_PATH"])
        ucx_env["LD_LIBRARY_PATH"] = ":".join(ld_library_path)
        try:
            probe = subprocess.run(
                [str(ucx_info), "-v"],
                capture_output=True,
                text=True,
                check=True,
                env=ucx_env,
                timeout=10,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            checks.append({
                "check": "SGLang UCX 1.20 runtime",
                "ok": False,
                "detail": str(exc),
            })
        else:
            version = next(
                (line.strip() for line in probe.stdout.splitlines()
                 if line.startswith("# Library version:")),
                "",
            )
            checks.append({
                "check": "SGLang UCX 1.20 runtime",
                "ok": version.startswith("# Library version: 1.20."),
                "detail": version or probe.stdout.strip(),
            })

    for name in registry.TASK_ORDER:
        spec = registry.load_task(name)
        runner = paths.TASKS / name / "cell.py"
        checks.append({"check": f"{name} cell runner", "ok": runner.is_file(), "detail": str(runner)})
        if not spec.cases:
            checks.append({"check": f"{name} cases", "ok": False, "detail": "no cases"})

    for relative in (
        "BAGEL/bagel_shared_v1.json", "PI05/fixture_46row/fixture.json",
        "COS-T2I/cosmos3_t2i_prompts_v1.json",
        "ORPHEUS/seedtts_en_targets_v1.jsonl",
        "VJ-AC/droid_paper_first50_local_subset_v1.jsonl",
    ):
        path = paths.INPUTS / relative
        checks.append({"check": f"fixture {relative}", "ok": path.is_file(), "detail": str(path)})

    flashinfer = (
        paths.ENVS / "voxserve" / "lib" / "python3.12" / "site-packages"
        / "flashinfer" / "jit" / "core.py"
    )
    if flashinfer.is_file():
        content = flashinfer.read_text(encoding="utf-8")
        checks.append({
            "check": "VoxServe flashinfer C++20",
            "ok": "-std=c++17" not in content and "-std=c++20" in content,
            "detail": f"{flashinfer}: Torch 2.8 requires -std=c++20",
        })
        # Only an sm_89 card (RTX 5880) needs it; on an H100/H200 the FP8 build works and the env is left as installed.
        needs_fp8_off = _compute_capability(hardware.current().primary_gpu) < 9.0
        checks.append({
            "check": "VoxServe flashinfer FP8 off" + ("" if needs_fp8_off else " (not needed: sm_90+)"),
            "ok": (not needs_fp8_off) or "-DFLASHINFER_ENABLE_FP8_E4M3" not in content,
            "detail": f"{flashinfer}: FP8 conversions in the JIT RoPE module need ptxas >= 12.1 on sm_89; the host nvcc is 12.0 "
                      "(rerun `bash envs/setup.sh voxserve`)",
        })
    else:
        checks.append({
            "check": "VoxServe flashinfer installation",
            "ok": False, "detail": str(flashinfer),
        })

    for env_name in ("mstar", "vllm-omni", "sglang-omni"):
        checks.append(cuda_toolchain_check(env_name))

    gpu_index = hardware.current().primary_gpu
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,persistence_mode",
             "--format=csv,noheader,nounits", "-i", str(gpu_index)],
            capture_output=True, text=True, check=True, timeout=15,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        checks.append({"check": f"GPU {gpu_index} telemetry", "ok": False, "detail": str(exc)})
    else:
        checks.append({
            "check": f"GPU {gpu_index} telemetry",
            "ok": bool(probe.stdout.strip()),
            "detail": probe.stdout.strip(),
        })
    return checks


def main() -> int:
    checks = diagnose()
    print(json.dumps({"ok": all(check["ok"] for check in checks), "checks": checks}, indent=2))
    return 0 if all(check["ok"] for check in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
