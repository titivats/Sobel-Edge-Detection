from __future__ import annotations

import cv2
import numpy as np

from .measurement_data_models import FocusZone, SideStats


def draw_horizontal_marker(
    image: np.ndarray,
    label: str,
    baseline_x: int,
    edge_x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    if baseline_x == edge_x:
        return
    height, width = image.shape[:2]
    cv2.arrowedLine(image, (baseline_x, y), (edge_x, y), color, 2, tipLength=0.25)
    cv2.circle(image, (edge_x, y), 4, color, -1)
    text_size, _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_w, text_h = text_size
    text_x = min(baseline_x, edge_x) + 4
    if text_x + text_w > width - 4:
        text_x = max(width - text_w - 4, 4)
    text_y = max(y - 8, text_h + 4)
    if text_y > height - 4:
        text_y = height - 4
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_vertical_marker(
    image: np.ndarray,
    label: str,
    x: int,
    baseline_y: int,
    edge_y: int,
    color: tuple[int, int, int],
) -> None:
    if baseline_y == edge_y:
        return
    height, width = image.shape[:2]
    cv2.arrowedLine(image, (x, baseline_y), (x, edge_y), color, 2, tipLength=0.25)
    cv2.circle(image, (x, edge_y), 4, color, -1)
    text_size, _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    text_w, text_h = text_size
    text_x = x + 4
    if text_x + text_w > width - 4:
        text_x = max(x - text_w - 8, 4)
    text_y = min(baseline_y, edge_y) + text_h + 4
    if text_y > height - 4:
        text_y = height - 4
    if text_y < text_h + 4:
        text_y = text_h + 4
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_focus_zone(
    image: np.ndarray,
    focus_zone: FocusZone,
    color: tuple[int, int, int] = (0, 255, 255),
) -> None:
    cv2.rectangle(
        image,
        (focus_zone.x_min, focus_zone.y_min),
        (focus_zone.x_max, focus_zone.y_max),
        color,
        3,
    )
    label = "Focus zone"
    text_size, _baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    text_w, text_h = text_size
    text_x = min(max(focus_zone.x_min + 8, 4), image.shape[1] - text_w - 4)
    text_y = max(focus_zone.y_min - 8, text_h + 4)
    cv2.putText(
        image,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def measurement_summary_lines(
    orientation: str,
    left_stats: SideStats,
    right_stats: SideStats,
    top_stats: SideStats,
    bottom_stats: SideStats,
) -> tuple[str, str]:
    if orientation == "vertical":
        return (
            f"Left min/max: {left_stats.min_px:.1f}/{left_stats.max_px:.1f}px",
            f"Right min/max: {right_stats.min_px:.1f}/{right_stats.max_px:.1f}px",
        )
    return (
        f"Top min/max: {top_stats.min_px:.1f}/{top_stats.max_px:.1f}px",
        f"Bottom min/max: {bottom_stats.min_px:.1f}/{bottom_stats.max_px:.1f}px",
    )
