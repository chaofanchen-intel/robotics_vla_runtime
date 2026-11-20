"""Unit tests describe a fixed machine, not the one they run on.

Hardware is otherwise auto-detected from the GPU name, so the same test passed on the H200 and failed on the RTX 5880 (whose
profile is a multi-GPU, auto-selecting one). Every test starts on the H200 NVL profile; a test about another machine selects
it itself with `hardware.set_current(...)`. The tuned-kernel folders on disk are also machine state (they exist only after
the tuners ran), so the serve-command tests see them as absent unless they create them.
"""
from __future__ import annotations

import pytest

from exps import hardware
import exps.tasks.bagel.cell  # noqa: F401  - imported before any test patches paths.CONFIGS, which its constants capture


@pytest.fixture(autouse=True)
def _fixed_hardware(monkeypatch):
    monkeypatch.setattr(hardware, "detect", lambda: hardware.profiles()["h200nvl"])   # what `--hardware` unset resolves to
    hardware.set_current("h200nvl")
    from exps import frameworks

    monkeypatch.setattr(frameworks, "_mstar_tuned_env", lambda: {})
    monkeypatch.setattr(frameworks, "_tuned_env", lambda: {})
    yield
    hardware.reset()
