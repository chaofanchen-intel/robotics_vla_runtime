#!/usr/bin/env python3
"""Provenance capture and manifest writing.

Every cell records enough to answer, months later, "what exactly produced this
number?" without trusting anyone's memory:

* the pinned commit of each external runtime, **and** whether its working tree
  was clean at the time
* which harness patches were applied, by content hash, not by name
* the serve command verbatim, the config file, the fixture and its hash
* which protocol parameters were in force, and whether they came from the
  thermal calibration or from the conservative defaults

The last point matters more than it looks. A cell measured with
``calibration: default_not_measured`` and one measured from a real calibration
are not the same evidence, and the distinction has to survive into the
analysis rather than living in someone's shell history.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import paths

#: Pinned refs recorded in external/SOURCE-RECORD.md. `bootstrap.sh` enforces
#: these; the manifest records what was actually found, so a drift is visible
#: in the evidence rather than only in a preflight log.
PINNED_REFS = {
    "mstar": "9c6f1ad757bcc1c19ea47e98f18010199bc098dd",          # main, 2026-11-03
    "vllm-omni": "5917730a4b02aedb44d374173cdc1b10484ccd52",      # v0.32.0
    "sglang-omni": "e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657",    # v0.2.0
    "vjepa2": "204698b45b3712590f06245fbfba32d3be539812",
    "vox-serve": "ce212de19e5df651ad9a44e5e3719b5e8cdb8ee3",
}


def _git(repo: Path, *args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.rstrip("\n")
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def external_state(name: str) -> dict[str, Any]:
    """HEAD, cleanliness and pinned-ref agreement for one external checkout."""
    repo = paths.EXTERNAL / name
    if not repo.is_dir():
        return {"repository": name, "status": "missing", "path": paths.relative_to_repo(repo)}
    head = _git(repo, "rev-parse", "HEAD")
    status = _git(repo, "status", "--porcelain")
    dirty_files = [line[3:] for line in (status or "").splitlines() if line.strip()]
    expected = PINNED_REFS.get(name)
    return {
        "repository": name,
        "path": paths.relative_to_repo(repo),
        "head": head,
        "pinned_ref": expected,
        "at_pinned_ref": (head == expected) if expected and head else None,
        "worktree_clean": not dirty_files,
        "modified_files": dirty_files,
    }


def file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def applied_patches() -> list[dict[str, Any]]:
    """Harness patches currently present, identified by content hash.

    Reads the ledger `bootstrap.sh` maintains. A missing ledger means bootstrap
    has not run, which is itself recorded rather than assumed benign.
    """
    ledger = paths.PATCHES / "applied.json"
    if not ledger.is_file():
        return [{"status": "no_patch_ledger", "expected_path": paths.relative_to_repo(ledger)}]
    try:
        return json.loads(ledger.read_text(encoding="utf-8")).get("applied", [])
    except (json.JSONDecodeError, OSError) as exc:
        return [{"status": "unreadable_patch_ledger", "error": str(exc)}]


def provenance(frameworks: list[str] | None = None) -> dict[str, Any]:
    """The provenance block embedded in every cell manifest."""
    names = frameworks or sorted(PINNED_REFS)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "harness_commit": _git(paths.REPO_ROOT, "rev-parse", "HEAD"),
        "harness_worktree_clean": not (_git(paths.REPO_ROOT, "status", "--porcelain") or "").strip(),
        "external": [external_state(name) for name in names],
        "patches": applied_patches(),
    }


def fixture_record(path: Path) -> dict[str, Any]:
    """Identify an input fixture by path and content hash."""
    resolved = Path(path)
    return {
        "path": paths.relative_to_repo(resolved),
        "sha256": file_sha256(resolved),
        "bytes": resolved.stat().st_size if resolved.is_file() else None,
    }


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write pretty JSON with a trailing newline, creating parents as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def mark_cell_done(cell_dir: Path, status: str) -> Path:
    """Sentinel that lets `--resume` skip a finished cell without re-parsing it."""
    return write_json(
        Path(cell_dir) / ".cell_done",
        {"status": status, "finished_at_utc": datetime.now(timezone.utc).isoformat()},
    )


def cell_is_done(cell_dir: Path) -> bool:
    """True only for a cell that completed; a failed cell is retried on resume."""
    sentinel = Path(cell_dir) / ".cell_done"
    if not sentinel.is_file():
        return False
    try:
        return json.loads(sentinel.read_text(encoding="utf-8")).get("status") == "completed"
    except (json.JSONDecodeError, OSError):
        return False
