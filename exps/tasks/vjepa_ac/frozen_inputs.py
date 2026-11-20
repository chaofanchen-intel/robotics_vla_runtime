"""Freeze the encoded DROID source videos so every client of a paired run sees byte-identical inputs.

`DROIDDataset` re-encodes each episode's clip into `droid_local_cache/ep<N>.mp4` every time a client starts, and the
encoder is not byte-deterministic: in the BASELINE_VJEPA_AC_MAIN_02 run, 7 of 36 client starts produced a different mp4
for the larger episodes (ep1, ep4), so the paired-cohort audit rejected 4 of 6 starts -- and, worse, the two engines
had really been fed slightly different bitstreams. The first build of a (episode, horizon, frame count) is copied here
and every later client reads that copy instead.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from exps.core import paths

FROZEN_DIR = paths.INPUTS / "VJ-AC" / "frozen_videos"


def freeze(requests: list[Any], *, horizon: int, num_video_frames: int, directory: Path = FROZEN_DIR) -> list[Any]:
    """Point every request at its frozen copy of the video (creating the copy on first use)."""
    directory.mkdir(parents=True, exist_ok=True)
    for request in requests:
        source = Path(request.video_path)
        frozen = directory / f"{source.stem}_h{horizon}_f{num_video_frames}.mp4"
        if not frozen.is_file():
            fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
            os.close(fd)
            shutil.copyfile(source, tmp)
            os.replace(tmp, frozen)  # atomic: a concurrent client sees the whole file or none
        request.video_path = str(frozen)
    return requests
