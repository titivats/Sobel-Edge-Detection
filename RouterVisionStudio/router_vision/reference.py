"""Experimental 2-D edge deviations from a manually aligned reference.

Measurements use original image pixels, never the resized ViT input. No output
from this module participates in the production gate. There is no automatic
registration: the operator must align the reference on each image.
"""
from __future__ import annotations

import json
import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .guard import check_write_target


@dataclass
class EdgeProfile:
    valid: bool = False
    reason: str = "Draw a reference line."
    coverage: float = 0.0
    inward_px: float = 0.0
    protrusion_px: float = 0.0
    normal: np.ndarray = field(default_factory=lambda: np.zeros(2))
    anchors: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 2)))
    deviations: np.ndarray = field(default_factory=lambda: np.empty(0))
    accepted: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=bool))


def normal_scale(normal, x_mm_per_px: float, y_mm_per_px: float) -> float:
    """Convert signed image-normal offsets to physical perpendicular mm.

    Uses separate X/Y scales, including when the reference is diagonal.
    """
    if not all(math.isfinite(v) and v > 0 for v in (x_mm_per_px, y_mm_per_px)):
        raise ValueError("Both calibrated X/Y scales must be positive and finite.")
    normal = np.asarray(normal, dtype=float)
    if normal.shape != (2,) or not np.isfinite(normal).all() or not np.isclose(np.linalg.norm(normal), 1):
        raise ValueError("A unit reference normal is required.")
    return 1.0 / math.hypot(normal[0] / x_mm_per_px, normal[1] / y_mm_per_px)


def measure_reference(image: np.ndarray, line, *, side=1, radius=24,
                      min_contrast=5.0, polarity=1) -> EdgeProfile:
    """Find positive material transitions along normals within a search band.

    Positive displacement is INTO the marked PCB side (material missing).
    Negative displacement is material protruding into the slot. Ambiguous,
    weak, clipped or out-of-image scans are rejected, not treated as zero.
    Thresholds are development heuristics and need validation on real images.
    """
    out = EdgeProfile()
    ends = np.asarray(line, dtype=np.float64)
    if ends.shape != (2, 2) or not np.isfinite(ends).all():
        out.reason = "Reference coordinates are invalid."
        return out
    if image is None or image.ndim not in (2, 3) or image.size == 0:
        out.reason = "Image could not be read."
        return out
    vector = ends[1] - ends[0]
    length = float(np.linalg.norm(vector))
    if length < 12 or side not in (-1, 1) or polarity not in (-1, 1):
        out.reason = "Draw a line at least 12 pixels long and select the PCB side."
        return out
    if not 4 <= radius <= 200 or not math.isfinite(min_contrast) or min_contrast <= 0:
        out.reason = "Search width / edge contrast is invalid."
        return out
    h, w = image.shape[:2]
    if np.any(ends < 0) or np.any(ends[:, 0] > w - 1) or np.any(ends[:, 1] > h - 1):
        out.reason = "Reference endpoints must be inside the image."
        return out
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = cv2.GaussianBlur(gray.astype(np.float32), (3, 3), 0)
    normal = side * np.array([-vector[1], vector[0]]) / length
    # At most one-pixel spacing. Refuse huge lines instead of skipping defects.
    count = max(13, int(math.ceil(length)) + 1)
    if count > 4096:
        out.reason = "Reference is too long; measure a shorter segment."
        return out
    anchors = ends[0] + np.linspace(0, 1, count)[:, None] * vector
    offsets = np.arange(-int(radius), int(radius) + 1, dtype=np.float32)
    coords = anchors[:, None, :] + offsets[None, :, None] * normal
    inside = ((coords[:, :, 0] >= 1) & (coords[:, :, 0] <= w - 2)
              & (coords[:, :, 1] >= 1) & (coords[:, :, 1] <= h - 2)).all(axis=1)
    samples = cv2.remap(gray, coords[:, :, 0].astype(np.float32),
                        coords[:, :, 1].astype(np.float32), cv2.INTER_LINEAR)
    gradient = np.gradient(samples, axis=1) * polarity
    peaks = gradient.argmax(axis=1)
    rows = np.arange(count)
    strengths = gradient[rows, peaks]
    competitor = gradient.copy()
    competitor[np.abs(np.arange(len(offsets))[None, :] - peaks[:, None]) <= 3] = 0
    # Detect competing boundaries (copper, reflections, opposite slot edge).
    ambiguous = competitor.max(axis=1) >= strengths * 0.8
    accepted = inside & (strengths >= min_contrast) & ~ambiguous
    accepted &= (peaks > 2) & (peaks < len(offsets) - 3)
    # Parabolic peak interpolation; this is not a subpixel accuracy guarantee.
    middle = np.clip(peaks, 1, len(offsets) - 2)
    a, b, c = (gradient[rows, middle - 1], gradient[rows, middle], gradient[rows, middle + 1])
    denominator = a - 2 * b + c
    fractional = np.divide(0.5 * (a - c), denominator, out=np.zeros_like(b),
                           where=np.abs(denominator) > 1e-6)
    deviations = offsets[peaks] + np.clip(fractional, -0.5, 0.5)
    out.normal, out.anchors = normal, anchors
    out.deviations, out.accepted = deviations, accepted
    out.points = anchors + deviations[:, None] * normal
    out.coverage = float(accepted.mean())
    if accepted.any():
        out.inward_px = max(0.0, float(deviations[accepted].max()))
        out.protrusion_px = max(0.0, float(-deviations[accepted].min()))
    # No complete-measurement claim when any part of the reference is unseen.
    out.valid = bool(accepted.all())
    out.reason = ("Measured along the selected segment only." if out.valid else
                  "MEASUREMENT NOT VALID: weak, ambiguous or clipped edge; adjust the line / search band.")
    return out


def image_signature(path: str | Path) -> dict:
    stat = Path(path).stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class ReferenceStore:
    """Per-image prototypes only; no unsafe cross-board template reuse."""
    def __init__(self, path: Path, protected=()):
        self.path = path
        self.protected = list(protected)

    @staticmethod
    def key(image_path):
        return os.path.normcase(str(Path(image_path).resolve()))

    def _read(self):
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("images"), dict):
            raise ValueError("Reference file is invalid; it has not been overwritten.")
        return raw["images"]

    def load(self, image_path):
        record = self._read().get(self.key(image_path))
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError("Saved reference is invalid.")
        if record.get("source") != image_signature(image_path):
            raise ValueError("Image changed since the reference was saved; draw a new reference.")
        return record.get("settings")

    def save(self, image_path, settings, expected_source):
        check_write_target(self.path, self.protected)
        if image_signature(image_path) != expected_source:
            raise ValueError("Image changed while open; reopen it before saving.")
        records = self._read()
        records[self.key(image_path)] = {"source": expected_source, "settings": settings}
        payload = json.dumps({"version": 1, "images": records}, indent=2, allow_nan=False)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(payload, encoding="utf-8")
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
