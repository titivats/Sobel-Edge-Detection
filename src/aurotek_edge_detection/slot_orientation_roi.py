from __future__ import annotations

import numpy as np

from .measurement_data_models import FocusZone, Roi


def _focus_zone_crop(mask: np.ndarray, focus_zone: FocusZone) -> np.ndarray:
    return mask[
        focus_zone.y_min : focus_zone.y_max + 1,
        focus_zone.x_min : focus_zone.x_max + 1,
    ]


def _mean_largest(values: np.ndarray, count: int) -> float:
    if values.size == 0:
        return 0.0
    count = min(count, values.size)
    return float(np.mean(np.sort(values)[-count:]))


def detect_slot_orientation_from_dark_mask(
    dark_mask: np.ndarray,
    focus_zone: FocusZone,
) -> str:
    """Decide whether the black router slot is horizontal or vertical.

    We use row/column projection instead of the largest connected component.
    This avoids fixture/pallet edges near the image side becoming the selected
    component when the actual router slot is a long horizontal black band.
    """
    crop = _focus_zone_crop(dark_mask, focus_zone)
    crop_height, crop_width = crop.shape
    if crop_height == 0 or crop_width == 0:
        raise ValueError("Focus zone is empty")

    dark_pixels_per_row = np.count_nonzero(crop, axis=1)
    dark_pixels_per_column = np.count_nonzero(crop, axis=0)

    horizontal_score = _mean_largest(dark_pixels_per_row, 5) / crop_width
    vertical_score = _mean_largest(dark_pixels_per_column, 5) / crop_height
    return "horizontal" if horizontal_score >= vertical_score else "vertical"


def create_slot_roi_from_dark_mask(
    dark_mask: np.ndarray,
    focus_zone: FocusZone,
    margin_px: int,
    orientation: str,
) -> Roi:
    """Create a Sobel search ROI around the black slot inside the focus zone."""
    crop = _focus_zone_crop(dark_mask, focus_zone)
    crop_height, crop_width = crop.shape
    if crop_height == 0 or crop_width == 0:
        raise ValueError("Focus zone is empty")

    if orientation == "horizontal":
        dark_pixels_per_row = np.count_nonzero(crop, axis=1)
        selected_rows = _select_strong_projection_indexes(
            counts=dark_pixels_per_row,
            full_span_px=crop_width,
            origin_px=focus_zone.y_min,
            orientation_name="horizontal",
        )
        return Roi(
            x_min=focus_zone.x_min,
            y_min=max(int(selected_rows.min()) - margin_px, focus_zone.y_min),
            x_max=focus_zone.x_max,
            y_max=min(int(selected_rows.max()) + margin_px, focus_zone.y_max),
            center_x=(focus_zone.x_min + focus_zone.x_max) // 2,
            center_y=int(round(float(np.median(selected_rows)))),
        )

    dark_pixels_per_column = np.count_nonzero(crop, axis=0)
    selected_columns = _select_strong_projection_indexes(
        counts=dark_pixels_per_column,
        full_span_px=crop_height,
        origin_px=focus_zone.x_min,
        orientation_name="vertical",
    )
    return Roi(
        x_min=max(int(selected_columns.min()) - margin_px, focus_zone.x_min),
        y_min=focus_zone.y_min,
        x_max=min(int(selected_columns.max()) + margin_px, focus_zone.x_max),
        y_max=focus_zone.y_max,
        center_x=int(round(float(np.median(selected_columns)))),
        center_y=(focus_zone.y_min + focus_zone.y_max) // 2,
    )


def _select_strong_projection_indexes(
    counts: np.ndarray,
    full_span_px: int,
    origin_px: int,
    orientation_name: str,
) -> np.ndarray:
    if counts.max() == 0:
        raise ValueError(f"No black background pixels found for {orientation_name} ROI")

    threshold = max(
        int(round(full_span_px * 0.18)),
        int(round(float(counts.max()) * 0.40)),
        1,
    )
    selected = np.flatnonzero(counts >= threshold) + origin_px
    if selected.size == 0:
        raise ValueError(f"No {orientation_name} black background band found in focus zone")
    return selected
