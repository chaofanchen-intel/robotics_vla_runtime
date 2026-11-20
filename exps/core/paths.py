#!/usr/bin/env python3
"""Canonical filesystem locations for the experiments harness.

Everything the harness reads or writes resolves through here, so a run
directory moved or a repo cloned elsewhere needs no edits scattered across
eight task packages.

The harness owns its code, configs, fixtures, and run layout under `exps/`.
Workspace-level dependencies are explicit: model weights and tokenizers live
under `MODELS_CACHE`, runtime environments under `ENVS`, and framework source
checkouts under `EXTERNAL`. No prior-round experiment directory is read.
"""
from __future__ import annotations

import os
from pathlib import Path

#: exps/
EXPS_ROOT = Path(__file__).resolve().parents[1]
#: Repository root (the directory holding exps/, external/, envs/, models/).
REPO_ROOT = EXPS_ROOT.parent

CORE = EXPS_ROOT / "core"
FRAMEWORKS = EXPS_ROOT / "frameworks"
TASKS = EXPS_ROOT / "tasks"
CONFIGS = EXPS_ROOT / "configs"
#: Analysis output (markdown tables). Tracked in git, unlike `runs/`.
RESULTS = EXPS_ROOT / "results"
INPUTS = EXPS_ROOT / "inputs"
PATCHES = EXPS_ROOT / "patches"
PROFILES = EXPS_ROOT / "profiles"
RUNS = EXPS_ROOT / "runs"
TESTS = EXPS_ROOT / "tests"

#: Shared, not copied: pinned by revision in every manifest instead.
MODELS_CACHE = REPO_ROOT / "models" / "cache"
ENVS = REPO_ROOT / "envs"
EXTERNAL = REPO_ROOT / "external"

#: One campaign per rebuild of this harness. Run directories are namespaced by
#: it so a later round never writes into an earlier round's tree.
DEFAULT_CAMPAIGN_ID = "exps-20261104"


_CAMPAIGN: str | None = None


def set_campaign(campaign: str | None) -> None:
    """Select the campaign (run directory name) for this process; `--campaign` on every CLI calls this."""
    global _CAMPAIGN
    _CAMPAIGN = campaign


def campaign_id() -> str:
    """Campaign id: `--campaign` if given, else `EXPS_CAMPAIGN_ID` (bookkeeping only, not an experiment setting)."""
    return _CAMPAIGN or os.environ.get("EXPS_CAMPAIGN_ID") or DEFAULT_CAMPAIGN_ID


def campaign_root(campaign: str | None = None) -> Path:
    return RUNS / (campaign or campaign_id())


def cell_dir(
    task: str,
    framework: str,
    case: str,
    repeat: str,
    *,
    campaign: str | None = None,
) -> Path:
    """One directory per (task, framework, case, repeat) -- the unit of work."""
    return campaign_root(campaign) / task.upper() / framework / case / repeat


def lifecycle_dir(task: str, framework: str, case: str, repeat: str, *, campaign: str | None = None) -> Path:
    return campaign_root(campaign) / "_service_lifecycles" / f"{task}_{framework}_{case}_{repeat}"


def campaign_logs(campaign: str | None = None) -> Path:
    return campaign_root(campaign) / "_campaign_logs"


def analysis_dir(campaign: str | None = None) -> Path:
    return campaign_root(campaign) / "_analysis"


def env_python(env_name: str) -> Path:
    """Interpreter for one of the per-runtime virtualenvs under envs/."""
    return ENVS / env_name / "bin" / "python"


def relative_to_repo(path: Path) -> str:
    """Repo-relative string for manifests, falling back to absolute when outside."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved)
