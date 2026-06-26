from __future__ import annotations

import cv2
import numpy as np

from realtime_product_info import ProductInfo
from sobel_edge_detection import create_sobel_edge_masks


def make_sobel_bgr(image_bgr: np.ndarray, threshold_ratio: float) -> np.ndarray:
    _edge_x, _edge_y, edge_all = create_sobel_edge_masks(image_bgr, threshold_ratio)
    return cv2.cvtColor(edge_all, cv2.COLOR_GRAY2BGR)


def draw_classification_overlay(
    image_bgr: np.ndarray,
    status: str,
    predicted_class: str,
    confidence: float,
    point_number: int,
    product_info: ProductInfo,
) -> np.ndarray:
    canvas = image_bgr.copy()
    status_color = (
        (0, 0, 255) if status == "NG" else (0, 180, 0) if status == "PASS" else (0, 180, 255)
    )
    height, width = canvas.shape[:2]
    font_scale = max(2.0, min(width, height) / 260.0)
    thickness = max(3, int(font_scale * 2))
    status_text = f"{status} {confidence:.1%}"
    text_size, _baseline = cv2.getTextSize(
        status_text,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        thickness,
    )
    text_x = max(0, (width - text_size[0]) // 2)
    text_y = max(text_size[1] + 10, height // 2)
    cv2.putText(
        canvas,
        status_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 5,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        status_text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        status_color,
        thickness,
        cv2.LINE_AA,
    )

    metadata_lines = [
        f"Class: {predicted_class}",
        f"Confidence: {confidence:.2%}",
        f"Point: {point_number}",
        f"CSV: {product_info.csv_path.name if product_info.csv_path else '-'}",
        f"SN: {product_info.sn or '-'}",
        f"Table: {product_info.cuted_table or '-'}",
        f"Product: {product_info.product_id or '-'}",
        f"Recipe: {product_info.recipe_name or '-'}",
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
