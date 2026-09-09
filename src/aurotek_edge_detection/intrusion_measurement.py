from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .dark_background_mask import make_dark_mask
from .focus_zone import apply_focus_zone_to_mask
from .measurement_data_models import (
    FocusZone,
    HorizontalSlotMeasurement,
    IntrusionMeasurement,
    Roi,
    SideStats,
    VerticalSlotMeasurement,
)
from .measurement_image_output import save_measurement_images
from .measurement_statistics import calculate_side_stats
from .slot_boundary_detection import (
    find_horizontal_slot_boundaries,
    find_vertical_slot_boundaries,
)
from .slot_orientation_roi import (
    create_slot_roi_from_dark_mask,
    detect_slot_orientation_from_dark_mask,
)
from .sobel_edge_detection import create_sobel_edge_masks, thicken_edge_for_display


def measure_intrusion(
    image_path: Path,
    output_dir: Path,
    mm_per_pixel: float,
    black_threshold: int,
    min_span: int,
    baseline_percentile: float,
    sobel_threshold_ratio: float,
    roi_margin: int,
    orientation: str,
    display_edge_thickness: int,
    black_edge_search_radius: int,
    focus_zone: FocusZone,
) -> IntrusionMeasurement:
    image = _read_image(image_path)

    dark_mask = make_dark_mask(image, black_threshold)
    focused_dark_mask = apply_focus_zone_to_mask(dark_mask, focus_zone)
    detected_orientation = _resolve_orientation(
        requested_orientation=orientation,
        focused_dark_mask=focused_dark_mask,
        focus_zone=focus_zone,
    )
    slot_roi = create_slot_roi_from_dark_mask(
        focused_dark_mask,
        focus_zone,
        roi_margin,
        detected_orientation,
    )

    sobel_x_edges, sobel_y_edges, _sobel_all_edges = create_sobel_edge_masks(
        image,
        sobel_threshold_ratio,
    )
    vertical = _measure_vertical_slot(
        sobel_x_edges=sobel_x_edges,
        slot_roi=slot_roi,
        min_span=min_span,
        baseline_percentile=baseline_percentile,
        enabled=detected_orientation == "vertical",
    )
    horizontal = _measure_horizontal_slot(
        sobel_y_edges=sobel_y_edges,
        focused_dark_mask=focused_dark_mask,
        slot_roi=slot_roi,
        min_span=min_span,
        search_radius=black_edge_search_radius,
        enabled=detected_orientation == "horizontal",
    )

    if vertical.rows.size == 0 and horizontal.columns.size == 0:
        raise ValueError(f"No usable Sobel boundaries found: {image_path}")

    max_side, max_intrusion_px = _largest_intrusion_side(
        detected_orientation,
        vertical,
        horizontal,
    )

    selected_sobel_edges = sobel_x_edges if detected_orientation == "vertical" else sobel_y_edges
    display_edge = thicken_edge_for_display(
        selected_sobel_edges,
        display_edge_thickness,
    )
    save_measurement_images(
        output_dir=output_dir,
        image_stem=image_path.stem,
        display_edge=display_edge,
        focus_zone=focus_zone,
        vertical=vertical,
        horizontal=horizontal,
        orientation=detected_orientation,
    )

    height_px, width_px = image.shape[:2]
    return IntrusionMeasurement(
        image_path=image_path,
        width_px=width_px,
        height_px=height_px,
        orientation=detected_orientation,
        baseline_left_px=vertical.baseline_left,
        baseline_right_px=vertical.baseline_right,
        baseline_top_px=horizontal.baseline_top,
        baseline_bottom_px=horizontal.baseline_bottom,
        min_slot_width_px=vertical.min_slot_width,
        min_slot_height_px=horizontal.min_slot_height,
        min_left_intrusion_px=vertical.left_stats.min_px,
        max_left_intrusion_px=vertical.left_stats.max_px,
        min_right_intrusion_px=vertical.right_stats.min_px,
        max_right_intrusion_px=vertical.right_stats.max_px,
        min_top_intrusion_px=horizontal.top_stats.min_px,
        max_top_intrusion_px=horizontal.top_stats.max_px,
        min_bottom_intrusion_px=horizontal.bottom_stats.min_px,
        max_bottom_intrusion_px=horizontal.bottom_stats.max_px,
        max_intrusion_px=max_intrusion_px,
        max_intrusion_mm=max_intrusion_px * mm_per_pixel,
        max_intrusion_side=max_side,
    )


def _read_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return image


def _resolve_orientation(
    requested_orientation: str,
    focused_dark_mask: np.ndarray,
    focus_zone: FocusZone,
) -> str:
    if requested_orientation != "auto":
        return requested_orientation
    return detect_slot_orientation_from_dark_mask(focused_dark_mask, focus_zone)


def _measure_vertical_slot(
    sobel_x_edges: np.ndarray,
    slot_roi: Roi,
    min_span: int,
    baseline_percentile: float,
    enabled: bool,
) -> VerticalSlotMeasurement:
    if not enabled:
        return _empty_vertical_measurement()

    rows, left_edges, right_edges = find_vertical_slot_boundaries(
        sobel_x_edges,
        slot_roi,
        min_span,
    )
    if rows.size == 0:
        return _empty_vertical_measurement()

    baseline_left = float(np.percentile(left_edges, baseline_percentile))
    baseline_right = float(np.percentile(right_edges, 100.0 - baseline_percentile))
    left_intrusion = np.maximum(left_edges - baseline_left, 0.0)
    right_intrusion = np.maximum(baseline_right - right_edges, 0.0)

    return VerticalSlotMeasurement(
        rows=rows,
        left_edges=left_edges,
        right_edges=right_edges,
        baseline_left=baseline_left,
        baseline_right=baseline_right,
        left_stats=calculate_side_stats(left_intrusion),
        right_stats=calculate_side_stats(right_intrusion),
        min_slot_width=float((right_edges - left_edges + 1).min()),
    )


def _measure_horizontal_slot(
    sobel_y_edges: np.ndarray,
    focused_dark_mask: np.ndarray,
    slot_roi: Roi,
    min_span: int,
    search_radius: int,
    enabled: bool,
) -> HorizontalSlotMeasurement:
    if not enabled:
        return _empty_horizontal_measurement()

    columns, top_edges, bottom_edges, baseline_top, baseline_bottom = (
        find_horizontal_slot_boundaries(
            sobel_y_edges,
            focused_dark_mask,
            slot_roi,
            min_span,
            search_radius,
        )
    )
    if columns.size == 0:
        return _empty_horizontal_measurement()

    top_intrusion = np.maximum(top_edges - baseline_top, 0.0)
    bottom_intrusion = np.maximum(baseline_bottom - bottom_edges, 0.0)

    return HorizontalSlotMeasurement(
        columns=columns,
        top_edges=top_edges,
        bottom_edges=bottom_edges,
        baseline_top=baseline_top,
        baseline_bottom=baseline_bottom,
        top_stats=calculate_side_stats(top_intrusion),
        bottom_stats=calculate_side_stats(bottom_intrusion),
        min_slot_height=float((bottom_edges - top_edges + 1).min()),
    )


def _largest_intrusion_side(
    orientation: str,
    vertical: VerticalSlotMeasurement,
    horizontal: HorizontalSlotMeasurement,
) -> tuple[str, float]:
    if orientation == "vertical":
        side_maxes = {
            "left": vertical.left_stats.max_px,
            "right": vertical.right_stats.max_px,
        }
    else:
        side_maxes = {
            "top": horizontal.top_stats.max_px,
            "bottom": horizontal.bottom_stats.max_px,
        }

    max_side = max(side_maxes, key=side_maxes.get)
    return max_side, float(side_maxes[max_side])


def _empty_vertical_measurement() -> VerticalSlotMeasurement:
    empty = np.array([])
    empty_stats = SideStats(0.0, 0.0, None, None)
    return VerticalSlotMeasurement(
        rows=empty,
        left_edges=empty,
        right_edges=empty,
        baseline_left=0.0,
        baseline_right=0.0,
        left_stats=empty_stats,
        right_stats=empty_stats,
        min_slot_width=0.0,
    )


def _empty_horizontal_measurement() -> HorizontalSlotMeasurement:
    empty = np.array([])
    empty_stats = SideStats(0.0, 0.0, None, None)
    return HorizontalSlotMeasurement(
        columns=empty,
        top_edges=empty,
        bottom_edges=empty,
        baseline_top=0.0,
        baseline_bottom=0.0,
        top_stats=empty_stats,
        bottom_stats=empty_stats,
        min_slot_height=0.0,
    )
