from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Roi:
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    center_x: int
    center_y: int


@dataclass(frozen=True)
class FocusZone:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


@dataclass(frozen=True)
class SideStats:
    min_px: float
    max_px: float
    min_index: int | None
    max_index: int | None


@dataclass(frozen=True)
class VerticalSlotMeasurement:
    rows: np.ndarray
    left_edges: np.ndarray
    right_edges: np.ndarray
    baseline_left: float
    baseline_right: float
    left_stats: SideStats
    right_stats: SideStats
    min_slot_width: float


@dataclass(frozen=True)
class HorizontalSlotMeasurement:
    columns: np.ndarray
    top_edges: np.ndarray
    bottom_edges: np.ndarray
    baseline_top: float
    baseline_bottom: float
    top_stats: SideStats
    bottom_stats: SideStats
    min_slot_height: float


@dataclass(frozen=True)
class IntrusionMeasurement:
    image_path: Path
    width_px: int
    height_px: int
    orientation: str
    baseline_left_px: float
    baseline_right_px: float
    baseline_top_px: float
    baseline_bottom_px: float
    min_slot_width_px: float
    min_slot_height_px: float
    min_left_intrusion_px: float
    max_left_intrusion_px: float
    min_right_intrusion_px: float
    max_right_intrusion_px: float
    min_top_intrusion_px: float
    max_top_intrusion_px: float
    min_bottom_intrusion_px: float
    max_bottom_intrusion_px: float
    max_intrusion_px: float
    max_intrusion_mm: float
    max_intrusion_side: str
