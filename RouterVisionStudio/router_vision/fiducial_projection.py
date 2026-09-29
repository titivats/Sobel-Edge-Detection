"""Experimental rigid alignment and explicit camera projection, not a production gate.

Recipe fiducials, detected fiducials, paths and camera pose must use the same
declared physical XY axes in mm. Machine-specific axis signs must be resolved
by the caller. No camera pose, material side or calibration is inferred.
"""

from __future__ import annotations

import math

import numpy as np


def _points(value, shape, label):
    points = np.asarray(value, dtype=float)
    if points.shape != shape or not np.isfinite(points).all():
        raise ValueError(f"{label} must have shape {shape} and finite coordinates.")
    return points


def align_two_fiducials(recipe_mm, detected_mm):
    """Fit rotation+translation; report spacing disagreement, never infer scale."""
    recipe = _points(recipe_mm, (2, 2), "Recipe fiducials")
    detected = _points(detected_mm, (2, 2), "Detected fiducials")
    a, b = recipe[1] - recipe[0], detected[1] - detected[0]
    la, lb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if min(la, lb) < 1e-6:
        raise ValueError("Distinct fiducial points are required.")
    theta = math.atan2(b[1], b[0]) - math.atan2(a[1], a[0])
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    translation = detected.mean(axis=0) - rotation @ recipe.mean(axis=0)
    residual = recipe @ rotation.T + translation - detected
    return {
        "rotation_rad": math.atan2(s, c),
        "translation_mm": translation.tolist(),
        "fiducial_spacing_difference_mm": lb - la,
        "max_fiducial_residual_mm": float(np.max(np.linalg.norm(residual, axis=1))),
        "calibration_verified": False,
    }


def project_aligned_edge(
    path_mm,
    *,
    rotation_rad,
    translation_mm,
    material_side,
    tool_radius_mm,
    camera_center_mm,
    camera_center_px,
    camera_x_angle_rad,
    scale_mm_per_px,
    image_y_up,
):
    """Choose material side, align in physical XY, then subtract explicit camera pose.

    material_side=+1 is the left normal of directed recipe start->end in XY;
    -1 is its opposite. The nominal material edge is one tool radius from
    the cutter path on that side. Offset is applied once in physical space.
    """
    path = _points(path_mm, (2, 2), "Tool path")
    shift = _points(translation_mm, (2,), "Alignment translation")
    camera = _points(camera_center_mm, (2,), "Camera center in mm")
    pixel_center = _points(camera_center_px, (2,), "Camera center in pixels")
    scale = _points(scale_mm_per_px, (2,), "Camera scale")
    if material_side not in (-1, 1) or isinstance(material_side, bool):
        raise ValueError("Select material_side +1 or -1 explicitly.")
    if type(image_y_up) is not bool:
        raise ValueError("Specify image Y direction explicitly.")
    if not all(math.isfinite(v) for v in (rotation_rad, camera_x_angle_rad, tool_radius_mm)):
        raise ValueError("Angles and tool radius must be finite.")
    if tool_radius_mm < 0 or np.any(scale <= 0):
        raise ValueError("Radius must be nonnegative and scales positive.")
    vector = path[1] - path[0]
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        raise ValueError("Tool path endpoints must be distinct.")
    normal = material_side * np.array([-vector[1], vector[0]]) / length
    edge = path + tool_radius_mm * normal
    c, s = math.cos(rotation_rad), math.sin(rotation_rad)
    rotation = np.array([[c, -s], [s, c]])
    aligned = edge @ rotation.T + shift
    ca, sa = math.cos(camera_x_angle_rad), math.sin(camera_x_angle_rad)
    camera_axes = np.array([[ca, sa], [-sa, ca]])
    pixels = (aligned - camera) @ camera_axes.T
    pixels /= scale * [1, -1 if image_y_up else 1]
    pixels += pixel_center
    return {
        "nominal_edge_recipe_mm": edge.tolist(),
        "aligned_edge_mm": aligned.tolist(),
        "edge_px": pixels.tolist(),
        "material_side": material_side,
        "production_reference_verified": False,
    }
