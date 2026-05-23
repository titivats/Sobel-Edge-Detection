from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .annotated_image_renderer import (
    draw_focus_zone,
    draw_horizontal_marker,
    draw_vertical_marker,
    measurement_summary_lines,
)
from .measurement_data_models import (
    FocusZone,
    HorizontalSlotMeasurement,
    VerticalSlotMeasurement,
)


def save_measurement_images(
    output_dir: Path,
    image_stem: str,
    display_edge: np.ndarray,
    focus_zone: FocusZone,
    vertical: VerticalSlotMeasurement,
    horizontal: HorizontalSlotMeasurement,
    orientation: str,
) -> None:
    annotated = cv2.cvtColor(display_edge, cv2.COLOR_GRAY2BGR)
    draw_focus_zone(annotated, focus_zone)
    _highlight_selected_sobel_edges(annotated, vertical, horizontal)
    _draw_baselines(annotated, vertical, horizontal)
    _draw_max_markers(annotated, vertical, horizontal)
    _draw_min_markers(annotated, vertical, horizontal)
    _draw_summary_text(annotated, orientation, vertical, horizontal)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "annotated").mkdir(parents=True, exist_ok=True)
    (output_dir / "sobel_edges").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / "sobel_edges" / f"{image_stem}_sobel_edge.png"), display_edge)
    cv2.imwrite(
        str(output_dir / "annotated" / f"{image_stem}_sobel_intrusion.png"),
        annotated,
    )


def _highlight_selected_sobel_edges(
    annotated: np.ndarray,
    vertical: VerticalSlotMeasurement,
    horizontal: HorizontalSlotMeasurement,
) -> None:
    selected_edges = np.zeros(annotated.shape[:2], dtype=np.uint8)
    for y, left_edge, right_edge in zip(
        vertical.rows,
        vertical.left_edges,
        vertical.right_edges,
    ):
        selected_edges[y, left_edge] = 255
        selected_edges[y, right_edge] = 255

    for x, top_edge, bottom_edge in zip(
        horizontal.columns,
        horizontal.top_edges,
        horizontal.bottom_edges,
    ):
        selected_edges[top_edge, x] = 255
        selected_edges[bottom_edge, x] = 255

    selected_edges = cv2.dilate(
        selected_edges,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    annotated[selected_edges > 0] = (0, 255, 255)


def _draw_baselines(
    annotated: np.ndarray,
    vertical: VerticalSlotMeasurement,
    horizontal: HorizontalSlotMeasurement,
) -> None:
    height, width = annotated.shape[:2]
    if vertical.rows.size:
        cv2.line(
            annotated,
            (round(vertical.baseline_left), 0),
            (round(vertical.baseline_left), height - 1),
            (0, 255, 0),
            2,
        )
        cv2.line(
            annotated,
            (round(vertical.baseline_right), 0),
            (round(vertical.baseline_right), height - 1),
            (0, 255, 0),
            2,
        )

    if horizontal.columns.size:
        cv2.line(
            annotated,
            (0, round(horizontal.baseline_top)),
            (width - 1, round(horizontal.baseline_top)),
            (0, 255, 0),
            2,
        )
        cv2.line(
            annotated,
            (0, round(horizontal.baseline_bottom)),
            (width - 1, round(horizontal.baseline_bottom)),
            (0, 255, 0),
            2,
        )


def _draw_max_markers(
    annotated: np.ndarray,
    vertical: VerticalSlotMeasurement,
    horizontal: HorizontalSlotMeasurement,
) -> None:
    if vertical.rows.size and vertical.left_stats.max_index is not None:
        index = vertical.left_stats.max_index
        draw_horizontal_marker(
            annotated,
            "L max",
            round(vertical.baseline_left),
            int(vertical.left_edges[index]),
            int(vertical.rows[index]),
            (0, 0, 255),
        )

    if vertical.rows.size and vertical.right_stats.max_index is not None:
        index = vertical.right_stats.max_index
        draw_horizontal_marker(
            annotated,
            "R max",
            round(vertical.baseline_right),
            int(vertical.right_edges[index]),
            int(vertical.rows[index]),
            (0, 0, 255),
        )

    if horizontal.columns.size and horizontal.top_stats.max_index is not None:
        index = horizontal.top_stats.max_index
        draw_vertical_marker(
            annotated,
            "T max",
            int(horizontal.columns[index]),
            round(horizontal.baseline_top),
            int(horizontal.top_edges[index]),
            (0, 0, 255),
        )

    if horizontal.columns.size and horizontal.bottom_stats.max_index is not None:
        index = horizontal.bottom_stats.max_index
        draw_vertical_marker(
            annotated,
            "B max",
            int(horizontal.columns[index]),
            round(horizontal.baseline_bottom),
            int(horizontal.bottom_edges[index]),
            (0, 0, 255),
        )


def _draw_min_markers(
    annotated: np.ndarray,
    vertical: VerticalSlotMeasurement,
    horizontal: HorizontalSlotMeasurement,
) -> None:
    if vertical.rows.size and vertical.left_stats.min_index is not None:
        index = vertical.left_stats.min_index
        draw_horizontal_marker(
            annotated,
            "L min",
            round(vertical.baseline_left),
            int(vertical.left_edges[index]),
            int(vertical.rows[index]),
            (255, 0, 255),
        )

    if vertical.rows.size and vertical.right_stats.min_index is not None:
        index = vertical.right_stats.min_index
        draw_horizontal_marker(
            annotated,
            "R min",
            round(vertical.baseline_right),
            int(vertical.right_edges[index]),
            int(vertical.rows[index]),
            (255, 0, 255),
        )

    if horizontal.columns.size and horizontal.top_stats.min_index is not None:
        index = horizontal.top_stats.min_index
        draw_vertical_marker(
            annotated,
            "T min",
            int(horizontal.columns[index]),
            round(horizontal.baseline_top),
            int(horizontal.top_edges[index]),
            (255, 0, 255),
        )

    if horizontal.columns.size and horizontal.bottom_stats.min_index is not None:
        index = horizontal.bottom_stats.min_index
        draw_vertical_marker(
            annotated,
            "B min",
            int(horizontal.columns[index]),
            round(horizontal.baseline_bottom),
            int(horizontal.bottom_edges[index]),
            (255, 0, 255),
        )


def _draw_summary_text(
    annotated: np.ndarray,
    orientation: str,
    vertical: VerticalSlotMeasurement,
    horizontal: HorizontalSlotMeasurement,
) -> None:
    summary_line_1, summary_line_2 = measurement_summary_lines(
        orientation,
        vertical.left_stats,
        vertical.right_stats,
        horizontal.top_stats,
        horizontal.bottom_stats,
    )
    cv2.putText(
        annotated,
        f"Direction: {orientation.upper()}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        summary_line_1,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        annotated,
        summary_line_2,
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
