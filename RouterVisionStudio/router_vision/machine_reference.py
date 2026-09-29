"""Read image-bound, conditional router references produced by the audit workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

SCHEMA = "nominal-router-reference-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path):
    """Reject stale source data; a matching hash is not physical calibration."""
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ValueError("Unsupported machine reference manifest")
    sources = data.get("source_files")
    if not isinstance(sources, list) or not sources or not isinstance(data.get("images"), list):
        raise ValueError("Machine reference sources / image records are missing")
    for source in sources:
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Machine reference source changed: {source['path']}")
    data["manifest_path"] = str(path)
    data["manifest_sha256"] = sha256(path)
    return data


def reference_for_image(manifest, image_path, edge="B"):
    """Select one capture and reconstruct its circle tangent without image fitting."""
    if edge not in ("A", "B"):
        raise ValueError("Select edge A or B")
    path = Path(image_path).resolve()
    matches = [r for r in manifest["images"] if r["name"] == path.name]
    if len(matches) != 1:
        raise ValueError("Image must match exactly one machine reference record")
    record = matches[0]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != record["sha256"]:
        raise ValueError("Image hash does not match the machine reference")
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or [image.shape[1], image.shape[0]] != record["size_px"]:
        raise ValueError("Image dimensions do not match the machine reference")
    p = record["projection"]
    scales = np.asarray(manifest["scale_mm_per_px_xy"], dtype=float)
    nc = np.asarray(p["reconstructed_nc_xy_mm"], dtype=float)
    camera = np.asarray(p["reconstructed_g87_camera_xy_mm"], dtype=float)
    diameter = float(p["bit_diameter_mm"])
    if (
        scales.shape != (2,)
        or not np.isfinite(scales).all()
        or (scales <= 0).any()
        or nc.shape != (2, 2)
        or not np.isfinite(nc).all()
        or camera.shape != (2,)
        or not np.isfinite(camera).all()
        or not np.isfinite(diameter)
        or diameter <= 0
    ):
        raise ValueError("Invalid machine reference geometry / scale")
    direction = nc[1] - nc[0]
    length = np.linalg.norm(direction)
    if length <= 0:
        raise ValueError("Machine reference has zero travel length")
    normal = np.array([-direction[1], direction[0]]) / length
    if abs(normal[1]) < 0.5:
        raise ValueError("Upper/lower A/B naming is unsupported for this orientation")
    if normal[1] < 0:
        normal = -normal
    centers = (nc - camera) / scales + np.asarray(record["size_px"]) / 2
    material_direction = normal * (1 if edge == "B" else -1)
    line = centers + material_direction * diameter / 2 / scales
    stored = np.asarray(p["lower_B_px" if edge == "B" else "upper_A_px"], dtype=float)
    if stored.shape != (2, 2) or not np.allclose(line, stored, rtol=0, atol=1e-8):
        raise ValueError("Stored tangent differs from reconstructed machine geometry")
    if (line < 0).any() or (line > np.asarray(record["size_px"]) - 1).any():
        raise ValueError("Machine reference lies outside the original image")
    vector = line[1] - line[0]
    left_normal = np.array([-vector[1], vector[0]])
    side = 1 if np.dot(left_normal, material_direction / scales) > 0 else -1
    return {
        "line": line.tolist(),
        "side": side,
        "scales": scales.tolist(),
        "provenance": {
            "manifest_path": manifest["manifest_path"],
            "manifest_sha256": manifest["manifest_sha256"],
            "image_sha256": record["sha256"],
            "selected_edge": edge,
            "csv_sn": record["csv_sn"],
            "database_sn": record["database_sn"],
            "cut_point": record["cut_point"],
            "acquisition_log_line": record["acquisition_log_line"],
            "mapping_basis": record["mapping_basis"],
            "projection": p,
            "source_files": manifest["source_files"],
            "camera_reaches_command_assumed": True,
            "calibration_verified": False,
            "production_reference_verified": False,
            "edge_identity_verified": False,
        },
    }
