from __future__ import annotations

import cv2
import numpy as np

from realtime_config import NG_CLASSES, PASS_CLASSES
from realtime_product_info import ProductInfo
from sobel_edge_detection import create_sobel_edge_masks

BoxResult = tuple[int, int, int, int, str, float]


def status_from_classes(class_names: list[str]) -> str:
    class_set = {name.strip().casefold() for name in class_names}
    if class_set & NG_CLASSES:
        return "NG"
    if class_set & PASS_CLASSES:
        return "PASS"
    return "UNKNOWN"


def make_sobel_bgr(image_bgr: np.ndarray, threshold_ratio: float) -> np.ndarray:
    _edge_x, _edge_y, edge_all = create_sobel_edge_masks(image_bgr, threshold_ratio)
    return cv2.cvtColor(edge_all, cv2.COLOR_GRAY2BGR)


def draw_overlay(
    image_bgr: np.ndarray,
    boxes: list[BoxResult],
    status: str,
    point_number: int,
    product_info: ProductInfo,
) -> np.ndarray:
    canvas = image_bgr.copy()
    for x1, y1, x2, y2, label, confidence in boxes:
        color = (0, 0, 255) if label.strip().casefold() in NG_CLASSES else (0, 180, 0)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        cv2.putText(
            canvas,
            f"{label} {confidence:.2f}",
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    status_color = (
        (0, 0, 255) if status == "NG" else (0, 180, 0) if status == "PASS" else (0, 180, 255)
    )
    height, width = canvas.shape[:2]
    font_scale = max(2.0, min(width, height) / 260.0)
    thickness = max(3, int(font_scale * 2))
    text_size, _baseline = cv2.getTextSize(status, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    text_x = max(0, (width - text_size[0]) // 2)
    text_y = max(text_size[1] + 10, height // 2)
    cv2.putText(
        canvas,
        status,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 5,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        status_color,
        thickness,
        cv2.LINE_AA,
    )

    metadata_lines = [
        f"Point: {point_number}",
        f"CSV: {product_info.csv_path.name if product_info.csv_path else '-'}",
        f"SN: {product_info.sn or '-'}",
        f"Table: {product_info.cuted_table or '-'}",
        f"Product: {product_info.product_id or '-'}",
        f"Recipe: {product_info.recipe_name or '-'}",
        f"Result: {product_info.result or '-'}",
        f"End: {product_info.end_time or '-'}",
    ]
    y = 28
    for line in metadata_lines:
        cv2.putText(
            canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA
        )
        cv2.putText(
            canvas, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA
        )
        y += 28
    return canvas
