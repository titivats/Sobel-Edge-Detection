"""Project a selected recipe segment using an explicitly supplied camera pose.

The recipe reader supplies candidate geometry, not verified finished PCB edges.
An operator must verify the segment, camera mapping and signed toolpath offset.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .toolpath import load_toolpath


def recipe_signature(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recipe_segments(path):
    """Keep every finite, nonzero consecutive segment; do not infer its role."""
    result = []
    for chain in load_toolpath(path).segments:
        for index, (a, b) in enumerate(zip(chain.points, chain.points[1:])):
            points = np.asarray([a, b], dtype=float)
            if np.isfinite(points).all() and np.linalg.norm(points[1] - points[0]) > 1e-5:
                result.append(
                    {
                        "chain_offset": chain.offset,
                        "edge_index": index,
                        "points_mm": points.tolist(),
                    }
                )
    return result


def project_segment(points_mm, mapping, image_shape, offset_mm=0):
    """Map machine XY to image pixels and clip only the visible straight portion.

    image_x_axis_deg is the angle of image +X in machine XY. With y_up=True,
    increasing machine Y maps upwards at zero rotation. Scales are positive
    physical mm per pixel along the rotated image axes.
    """
    points = np.asarray(points_mm, dtype=float)
    values = [
        mapping[k] for k in ("center_x_mm", "center_y_mm", "scale_x", "scale_y", "image_x_axis_deg")
    ]
    if (
        points.shape != (2, 2)
        or not np.isfinite(points).all()
        or not np.isfinite(values).all()
        or not np.isfinite(offset_mm)
    ):
        raise ValueError("Recipe geometry and camera mapping must be finite numbers.")
    cx, cy, sx, sy, angle = values
    if sx <= 0 or sy <= 0:
        raise ValueError("Enter verified positive X/Y camera scales; no default scale is assumed.")
    if type(mapping.get("y_up")) is not bool:
        raise ValueError("Select the camera Y direction.")
    h, w = image_shape[:2]
    if min(h, w) < 15:
        raise ValueError("Image is too small for reference measurement.")
    vector = points[1] - points[0]
    length = np.linalg.norm(vector)
    if length < 1e-5:
        raise ValueError("Recipe segment has zero length.")
    points = points + offset_mm * np.array([-vector[1], vector[0]]) / length
    angle = np.deg2rad(angle)
    rotation = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    projected = (points - [cx, cy]) @ rotation.T
    projected /= [sx, -sy if mapping["y_up"] else sy]
    projected += [(w - 1) / 2, (h - 1) / 2]
    if not np.isfinite(projected).all():
        raise ValueError("Camera mapping exceeds the supported numeric range.")
    # Liang-Barsky clipping preserves subpixel geometry and never rescales to fit.
    start, delta = projected[0], projected[1] - projected[0]
    lo, hi = 0.0, 1.0
    for axis, upper in ((0, w - 2), (1, h - 2)):
        if abs(delta[axis]) < 1e-12:
            if not 1 <= start[axis] <= upper:
                raise ValueError(
                    "Selected recipe segment is outside this camera image. Check the camera pose."
                )
        else:
            limits = sorted(((1 - start[axis]) / delta[axis], (upper - start[axis]) / delta[axis]))
            lo, hi = max(lo, limits[0]), min(hi, limits[1])
    if hi <= lo:
        raise ValueError(
            "Selected recipe segment is outside this camera image. Check the camera pose."
        )
    line = np.array([start + lo * delta, start + hi * delta])
    if np.linalg.norm(line[1] - line[0]) < 12:
        raise ValueError("Less than 12 pixels of this segment are visible in the image.")
    return line.tolist()


def validate_recipe_reference(record, image_shape):
    """Recheck provenance and recompute geometry instead of trusting saved pixels."""
    path = Path(record["path"])
    if recipe_signature(path) != record["sha256"]:
        raise ValueError(
            "Recipe changed since this reference was created. Select the recipe again."
        )
    candidates = recipe_segments(path)
    match = next(
        (
            c
            for c in candidates
            if c["chain_offset"] == record["segment"]["chain_offset"]
            and c["edge_index"] == record["segment"]["edge_index"]
        ),
        None,
    )
    if match is None or match != record["segment"]:
        raise ValueError("Saved segment does not match the recipe.")
    return project_segment(match["points_mm"], record["mapping"], image_shape, record["offset_mm"])
