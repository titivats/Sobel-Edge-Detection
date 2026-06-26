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
