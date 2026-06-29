from __future__ import annotations

import numpy as np

from .measurement_data_models import SideStats


def calculate_side_stats(intrusion_values: np.ndarray) -> SideStats:
    """Return min/max positive intrusion values and their source indexes."""
    positive_values = intrusion_values[intrusion_values > 0]
    if positive_values.size == 0:
        return SideStats(0.0, 0.0, None, None)

    positive_indexes = np.flatnonzero(intrusion_values > 0)
    min_positive_offset = int(np.argmin(positive_values))
    max_index = int(np.argmax(intrusion_values))
    return SideStats(
        min_px=float(positive_values[min_positive_offset]),
        max_px=float(intrusion_values[max_index]),
        min_index=int(positive_indexes[min_positive_offset]),
        max_index=max_index,
    )


def dominant_boundary_position(edge_positions: np.ndarray) -> float:
    """Estimate the normal boundary using the most common edge position."""
    if edge_positions.size == 0:
        return 0.0

    rounded_positions = np.rint(edge_positions).astype(int)
    offset = int(rounded_positions.min())
    counts = np.bincount(rounded_positions - offset)
    mode_position = int(np.argmax(counts) + offset)

    nearby_positions = edge_positions[np.abs(edge_positions - mode_position) <= 2]
    if nearby_positions.size:
        return float(np.median(nearby_positions))
    return float(mode_position)
