"""CPU-only tests for experiment preflight policy."""

import json
from pathlib import Path

import pytest

from exps.core import paths
from exps.doctor import PATCH_TOUCHED_FILES, _matches_expected_patch_worktree


def test_sglang_pi05_tokenizer_paths_are_repo_relative():
    config_dir = paths.CONFIGS / "sglang_core"
    configs = sorted(config_dir.glob("pi05_pipeline*.json"))
    assert configs
    for config_path in configs:
        tokenizer_name = json.loads(config_path.read_text(encoding="utf-8"))["tokenizer_name"]
        tokenizer_path = Path(tokenizer_name)
        assert not tokenizer_path.is_absolute(), config_path.name
        assert tokenizer_path.parts[:2] == ("models", "cache"), config_path.name


def test_active_exps_tree_has_no_retired_roots_or_host_paths():
    roots = (
        "core", "frameworks", "tasks", "configs", "profiles", "tools", "tests", "inputs",
    )
    text_suffixes = {".md", ".py", ".json", ".jsonl", ".yaml", ".yml", ".sh"}
    retired_fragments = (
        "exp" + "3/",
        "exps" + "2/",
        "experi" + "ments/",
        "/home/" + "chaofan" + "/workspaces/",
    )
    files = [
        paths.EXPS_ROOT / name
        for name in ("README.md", "GUIDE.md", "PROTOCOL.md")
        if (paths.EXPS_ROOT / name).is_file()
    ]
    for root_name in roots:
        root = paths.EXPS_ROOT / root_name
        if root.is_dir():
            files.extend(
                path for path in root.rglob("*")
                if path.is_file() and path.suffix.lower() in text_suffixes
            )

    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for fragment in retired_fragments:
            assert fragment not in text, f"{path.relative_to(paths.EXPS_ROOT)} contains {fragment!r}"


def test_sglang_claude_md_does_not_block_expected_patch_set():
    expected = {"sglang_omni/serve/protocol.py", "tests/unit_test/serve/test_openai_api.py"}

    assert _matches_expected_patch_worktree(
        "sglang-omni", [*expected, "CLAUDE.md"], expected, patch_ok=True
    )


def test_unexpected_sglang_runtime_change_still_blocks_preflight():
    expected = {"sglang_omni/serve/protocol.py"}

    assert not _matches_expected_patch_worktree(
        "sglang-omni", ["sglang_omni/serve/protocol.py", "CLAUDE.md", "other.py"],
        expected, patch_ok=True
    )


def test_unverified_patch_baseline_still_blocks_preflight():
    expected = {"sglang_omni/serve/protocol.py"}

    assert not _matches_expected_patch_worktree(
        "sglang-omni", ["sglang_omni/serve/protocol.py", "CLAUDE.md"],
        expected, patch_ok=False
    )


def test_voxserve_fp8_check_only_applies_to_sm89_gpus(monkeypatch, tmp_path):
    """envs/voxserve keeps its FP8 build on an H100/H200 (it works there); only an Ada card needs the macros removed."""
    from exps import doctor, hardware

    core = tmp_path / "envs" / "voxserve" / "lib" / "python3.12" / "site-packages" / "flashinfer" / "jit" / "core.py"
    core.parent.mkdir(parents=True)
    core.write_text('flags = ["-std=c++20", "-DFLASHINFER_ENABLE_FP8_E4M3"]\n')
    monkeypatch.setattr(doctor.paths, "ENVS", tmp_path / "envs")
    for capability, ok in ((8.9, False), (9.0, True)):
        monkeypatch.setattr(doctor, "_compute_capability", lambda index, c=capability: c)
        checks = {c["check"].split(" (")[0]: c for c in doctor.diagnose() if c["check"].startswith("VoxServe flashinfer FP8")}
        assert checks["VoxServe flashinfer FP8 off"]["ok"] is ok, capability


@pytest.mark.parametrize(("prefix", "repo"), [("mstar-", "mstar"), ("sglang-", "sglang-omni"), ("vllm-", "vllm-omni")])
def test_expected_worktree_covers_every_patch_target(prefix, repo):
    """A patch that touches a file missing from PATCH_TOUCHED_FILES makes the preflight refuse every GPU cell."""
    targets = set()
    for diff in paths.PATCHES.glob(f"{prefix}*.diff"):
        targets.update(line[len("+++ b/"):] for line in diff.read_text(encoding="utf-8").splitlines()
                       if line.startswith("+++ b/"))
    assert targets, f"no {prefix}*.diff targets found"
    assert targets <= PATCH_TOUCHED_FILES[repo], sorted(targets - PATCH_TOUCHED_FILES[repo])
