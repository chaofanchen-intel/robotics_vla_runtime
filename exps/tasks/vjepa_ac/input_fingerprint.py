"""Fingerprint the DROID source video and action/state trajectory for paired runs."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


@lru_cache(maxsize=32)
def _video_digest(path: Path) -> bytes:
    digest = hashlib.sha256()
    with path.open("rb") as video:
        for chunk in iter(lambda: video.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def source_input_sha256(request: Any, *, horizon: int, num_video_frames: int) -> str:
    """Hash the *source* inputs; this does not establish equal decoded tensors."""
    import numpy as np

    video_path = Path(request.video_path)
    digest = hashlib.sha256(b"vjepa-ac-source-input-v1\0")
    digest.update(_video_digest(video_path))
    digest.update(json.dumps([horizon, num_video_frames]).encode("ascii"))
    for field in ("actions", "states"):
        values = np.ascontiguousarray(np.asarray(request.model_kwargs[field], dtype="<f4"))
        digest.update(field.encode("ascii"))
        digest.update(json.dumps(values.shape).encode("ascii"))
        digest.update(values.tobytes())
    return digest.hexdigest()
