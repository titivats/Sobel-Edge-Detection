"""Durable image evidence and final source checks before a production decision is committed."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import cv2
import numpy as np

from .guard import check_write_target, protected_roots
from .machine_reference import sha256
from .production_store import json_safe


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


def atomic_write(path, content, cfg):
    path = Path(path)
    check_write_target(path, protected_roots(cfg))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_image(path, image, cfg):
    path = Path(path)
    check_write_target(path, protected_roots(cfg))
    extension = path.suffix.casefold()
    if extension not in {".png", ".jpg", ".jpeg"}:
        raise ValueError("Production images must use PNG or JPEG.")
    if image is None or image.size == 0 or image.dtype != np.uint8:
        raise ValueError("Production evidence needs a nonempty uint8 image.")
    options = []
    if extension in {".jpg", ".jpeg"}:
        quality = getattr(cfg, "production_jpeg_quality", 100)
        if type(quality) is not int or not 90 <= quality <= 100:
            raise ValueError("Production JPEG quality must be an integer from 90 to 100.")
        options = [cv2.IMWRITE_JPEG_QUALITY, quality]
        if hasattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR"):
            options += [cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444]
    okay, encoded = cv2.imencode(extension, image, options)
    if not okay:
        raise OSError("Cannot encode production evidence.")
    atomic_write(path, encoded.tobytes(), cfg)


def save_json(path, payload, cfg):
    atomic_write(
        path,
        (
            json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8"),
        cfg,
    )
