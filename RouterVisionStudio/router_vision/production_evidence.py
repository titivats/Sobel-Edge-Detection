"""Durable image evidence and final source checks before a production decision is committed."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from .guard import check_write_target, protected_roots
from .machine_reference import sha256


def verify_sources(evidence):
    hashes = {**evidence.get("source_hashes", {}), **evidence.get("input_files", {})}
    for path, expected in hashes.items():
        if sha256(path) != expected:
            raise ValueError(f"Production source changed before recording: {Path(path).name}")


def decision_image(image, status, note):
    bar = np.zeros((34, image.shape[1], 3), dtype=np.uint8)
    colour = (80, 220, 95) if status == "GOOD" else (70, 80, 245)
    text = f"FINAL {status} | {note}"
    cv2.putText(bar, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return np.vstack((bar, image))


def save_image(path, image, cfg):
    path = Path(path)
    check_write_target(path, protected_roots(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    okay, encoded = cv2.imencode(".png", image)
    if not okay:
        raise OSError("Cannot encode production evidence.")
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(encoded.tobytes())
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
