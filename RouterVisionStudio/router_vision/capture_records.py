"""Normalize explicit post-alignment capture geometry; no inferred offsets or angles."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

CAPTURE_SCHEMA = "recorded-router-captures-v1"


def normalize_captures(data):
    scales = np.asarray(data["scale_mm_per_px_xy"], dtype=float)
    if scales.shape != (2,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("Capture pixel scales must be positive finite X/Y values.")
    sources = {str(Path(s["path"]).resolve()): s["sha256"] for s in data["source_files"]}
    for record in data["images"]:
        if type(record.get("cut_point")) is not int or record["cut_point"] <= 0:
            raise ValueError("Capture cut point must be a positive integer.")
        size = record.get("size_px")
        if (
            not isinstance(size, list)
            or len(size) != 2
            or any(type(v) is not int or v <= 0 for v in size)
        ):
            raise ValueError("Capture image dimensions must be positive integers.")
        context = record["reference_context"]
        if type(context.get("cut_point")) is not int:
            raise ValueError("Capture context cut point must be an integer.")
        result_path = Path(record["result_path"]).resolve()
        if str(result_path) not in sources or result_path.name != record["result_file"]:
            raise ValueError("Capture Result must be a hashed source of this record.")
        if context["recipe_sha256"] not in sources.values():
            raise ValueError("Capture recipe must be a hashed source of this record.")
        with result_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = [
                row
                for row in csv.DictReader(stream)
                if str(row.get("SN", "")).strip() == str(record["csv_sn"])
            ]
        if len(rows) != 1:
            raise ValueError("Capture must identify exactly one CSV panel row.")
        row = rows[0]
        if (
            row.get("ProductId", "").strip() != context["product_id"]
            or Path(row.get("Recipe_Name", "")).name.casefold() != context["recipe_name"].casefold()
            or record["cut_point"] != context["cut_point"]
        ):
            raise ValueError("Capture product / recipe / cut identity does not match Result.")
        table = (row.get("CutedTable") or row.get("Table") or "").strip()
        if table and table != context["table"]:
            raise ValueError("Capture table differs from Result.")
        motion = record["motion"]
        if motion.get("alignment_applied") is not True:
            raise ValueError("Capture needs explicit Start/End AFTER alignment.")
        offset = np.asarray(motion["offset_xy_mm"], dtype=float)
        nc = np.asarray(motion["start_end_xy_mm"], dtype=float)
        camera = np.asarray(motion["camera_xy_mm"], dtype=float)
        diameter, angle = float(motion["diameter_mm"]), float(motion["rotation_rad"])
        if (
            offset.shape != (2,)
            or nc.shape != (2, 2)
            or camera.shape != (2,)
            or not all(np.isfinite(a).all() for a in (offset, nc, camera))
            or not math.isfinite(angle)
            or not math.isfinite(diameter)
            or diameter <= 0
        ):
            raise ValueError("Capture geometry / alignment values are invalid.")
        for actual, field in zip(offset, ("OffsetX", "OffsetY")):
            if not math.isclose(actual, float(row[field]), rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(f"Capture {field} differs from this board's Result.")
        for field, actual in (("BitDiameter", diameter), ("RotateAngle", angle)):
            if row.get(field) and not math.isclose(
                actual, float(row[field]), rel_tol=1e-6, abs_tol=1e-8
            ):
                raise ValueError(f"Capture {field} differs from Result.")
        direction = nc[1] - nc[0]
        length = np.linalg.norm(direction)
        if length <= 0:
            raise ValueError("Capture has zero cut travel.")
        normal = np.array([-direction[1], direction[0]]) / length
        vertical = abs((direction / scales)[1]) > abs((direction / scales)[0])
        if normal[0 if vertical else 1] < 0:
            normal = -normal
        centers = (nc - camera) / scales + np.asarray(record["size_px"]) / 2
        record["projection"] = {
            "source_recipe_sha256": context["recipe_sha256"],
            "reconstructed_nc_xy_mm": nc.tolist(),
            "reconstructed_g87_camera_xy_mm": camera.tolist(),
            "bit_diameter_mm": diameter,
            "board_offset_xy_mm": offset.tolist(),
            "rotation_rad": angle,
            "edge_A_px": (centers - normal * diameter / 2 / scales).tolist(),
            "edge_B_px": (centers + normal * diameter / 2 / scales).tolist(),
            "alignment_applied": True,
            "method": "Recorded post-alignment Start/End and camera pose; offsets are not applied a second time.",
        }
        record.setdefault("database_sn", None)
        record.setdefault("acquisition_log_line", None)
        record.setdefault("mapping_basis", "Explicit captured panel and cut identity")
    data["capture_source_schema"] = CAPTURE_SCHEMA
    data["schema"] = "nominal-router-reference-v1"
    return data
