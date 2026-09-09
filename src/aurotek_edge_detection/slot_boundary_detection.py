from __future__ import annotations

import numpy as np

from .measurement_data_models import Roi
from .measurement_statistics import dominant_boundary_position


def find_vertical_slot_boundaries(
    sobel_x_edges: np.ndarray,
    roi: Roi,
    min_span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find left/right Sobel boundaries for a vertical black slot."""
    rows: list[int] = []
    left_edges: list[int] = []
    right_edges: list[int] = []

    for y in range(roi.y_min, roi.y_max + 1):
        edge_x_positions = (
            np.flatnonzero(sobel_x_edges[y, roi.x_min : roi.x_max + 1] > 0) + roi.x_min
        )
        if edge_x_positions.size < 2:
            continue

        left_candidates = edge_x_positions[edge_x_positions < roi.center_x]
        right_candidates = edge_x_positions[edge_x_positions > roi.center_x]
        if left_candidates.size == 0 or right_candidates.size == 0:
            continue

        left_edge = int(left_candidates.max())
        right_edge = int(right_candidates.min())
        if right_edge - left_edge + 1 >= min_span:
            rows.append(y)
            left_edges.append(left_edge)
            right_edges.append(right_edge)

    if not rows:
        return np.array([]), np.array([]), np.array([])
    return np.asarray(rows), np.asarray(left_edges), np.asarray(right_edges)


def find_horizontal_slot_boundaries(
    sobel_y_edges: np.ndarray,
    focused_dark_mask: np.ndarray,
    roi: Roi,
    min_span: int,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Find top/bottom Sobel boundaries for a horizontal black slot.

    First, columns that still contain visible black pixels guide the normal
    top/bottom baseline. Then fallback Sobel-only detection measures columns
    where the tab covers the black background completely.
    """
    guide_cols, guide_tops, guide_bottoms = _find_black_guided_horizontal_edges(
        sobel_y_edges,
        focused_dark_mask,
        roi,
        min_span,
        search_radius,
    )
    if guide_cols.size == 0:
        return np.array([]), np.array([]), np.array([]), 0.0, 0.0

    baseline_top = dominant_boundary_position(guide_tops)
    baseline_bottom = dominant_boundary_position(guide_bottoms)
    guided_edges_by_column = {
        int(col): (int(top), int(bottom))
        for col, top, bottom in zip(guide_cols, guide_tops, guide_bottoms)
    }

    cols: list[int] = []
    top_edges: list[int] = []
    bottom_edges: list[int] = []
    for x in range(roi.x_min, roi.x_max + 1):
        if x in guided_edges_by_column:
            top_edge, bottom_edge = guided_edges_by_column[x]
        else:
            top_edge, bottom_edge = _find_fallback_horizontal_edges_for_column(
                sobel_y_edges,
                roi,
                x,
                baseline_top,
                baseline_bottom,
                search_radius,
            )

        if top_edge is None or bottom_edge is None:
            continue
        if bottom_edge - top_edge + 1 >= min_span:
            cols.append(x)
            top_edges.append(top_edge)
            bottom_edges.append(bottom_edge)

    if not cols:
        return np.array([]), np.array([]), np.array([]), baseline_top, baseline_bottom
    return (
        np.asarray(cols),
        np.asarray(top_edges),
        np.asarray(bottom_edges),
        baseline_top,
        baseline_bottom,
    )


def _find_black_guided_horizontal_edges(
    sobel_y_edges: np.ndarray,
    focused_dark_mask: np.ndarray,
    roi: Roi,
    min_span: int,
    search_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cols: list[int] = []
    top_edges: list[int] = []
    bottom_edges: list[int] = []

    for x in range(roi.x_min, roi.x_max + 1):
        dark_y_positions = (
            np.flatnonzero(focused_dark_mask[roi.y_min : roi.y_max + 1, x] > 0) + roi.y_min
        )
        if dark_y_positions.size < 2:
            continue

        top_dark = int(dark_y_positions.min())
        bottom_dark = int(dark_y_positions.max())
        edge_y_positions = (
            np.flatnonzero(sobel_y_edges[roi.y_min : roi.y_max + 1, x] > 0) + roi.y_min
        )
        if edge_y_positions.size < 2:
            continue

        top_edge = _nearest_edge_in_window(edge_y_positions, top_dark, search_radius)
        bottom_edge = _nearest_edge_in_window(edge_y_positions, bottom_dark, search_radius)
        if top_edge is None or bottom_edge is None:
            continue
        if bottom_edge - top_edge + 1 >= min_span:
            cols.append(x)
            top_edges.append(top_edge)
            bottom_edges.append(bottom_edge)

    if not cols:
        return np.array([]), np.array([]), np.array([])
    return np.asarray(cols), np.asarray(top_edges), np.asarray(bottom_edges)


def _find_fallback_horizontal_edges_for_column(
    sobel_y_edges: np.ndarray,
    roi: Roi,
    x: int,
    baseline_top: float,
    baseline_bottom: float,
    search_radius: int,
) -> tuple[int | None, int | None]:
    edge_y_positions = np.flatnonzero(sobel_y_edges[roi.y_min : roi.y_max + 1, x] > 0) + roi.y_min
    if edge_y_positions.size < 2:
        return None, None

    slot_height = max(baseline_bottom - baseline_top, 1.0)
    baseline_gap = max(8, min(18, int(round(slot_height * 0.06))))
    edge_runs = [
        (start, end) for start, end in _consecutive_runs(edge_y_positions) if end - start + 1 >= 3
    ]

    top_search_limit = baseline_top + min(slot_height * 0.45, 90.0)
    top_candidates = [
        (start, end)
        for start, end in edge_runs
        if end >= baseline_top + baseline_gap
        and start <= baseline_bottom - baseline_gap
        and start <= top_search_limit
    ]
    if top_candidates:
        top_edge = int(top_candidates[0][1])
    else:
        top_edge = _nearest_edge_in_window(
            edge_y_positions,
            int(round(baseline_top)),
            search_radius,
        )

    bottom_candidates = [
        (start, end)
        for start, end in edge_runs
        if start <= baseline_bottom - baseline_gap and end >= baseline_top + baseline_gap
    ]
    if bottom_candidates:
        bottom_edge = int(bottom_candidates[-1][0])
    else:
        bottom_edge = _nearest_edge_in_window(
            edge_y_positions,
            int(round(baseline_bottom)),
            search_radius,
        )

    return top_edge, bottom_edge


def _nearest_edge_in_window(
    edge_positions: np.ndarray,
    target: int,
    radius: int,
) -> int | None:
    if edge_positions.size == 0:
        return None

    nearby_edges = edge_positions[np.abs(edge_positions - target) <= radius]
    if nearby_edges.size:
        return int(nearby_edges[np.argmin(np.abs(nearby_edges - target))])
    return int(edge_positions[np.argmin(np.abs(edge_positions - target))])


def _consecutive_runs(values: np.ndarray) -> list[tuple[int, int]]:
    if values.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = previous = int(values[0])
    for value in values[1:]:
        current = int(value)
        if current <= previous + 1:
            previous = current
            continue
        runs.append((start, previous))
        start = previous = current
    runs.append((start, previous))
    return runs
