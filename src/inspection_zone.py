from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class InspectionZone:
    x_min: int
    x_max: int
    y_min: int
    y_max: int


def inspection_zone_from_ratios(
    image_shape: tuple[int, int],
    x_min_ratio: float,
    x_max_ratio: float,
    y_min_ratio: float,
    y_max_ratio: float,
) -> InspectionZone:
    """Build an image-relative preview zone."""
    height, width = image_shape
    return InspectionZone(
        x_min=max(0, min(width - 1, int(round(width * x_min_ratio)))),
        x_max=max(0, min(width - 1, int(round(width * x_max_ratio)))),
        y_min=max(0, min(height - 1, int(round(height * y_min_ratio)))),
        y_max=max(0, min(height - 1, int(round(height * y_max_ratio)))),
    )


def apply_inspection_zone_to_mask(
    mask: np.ndarray,
    inspection_zone: InspectionZone,
) -> np.ndarray:
    """Keep mask pixels only inside the configured preview zone."""
    focused_mask = np.zeros_like(mask)
    focused_mask[
        inspection_zone.y_min : inspection_zone.y_max + 1,
        inspection_zone.x_min : inspection_zone.x_max + 1,
    ] = mask[
        inspection_zone.y_min : inspection_zone.y_max + 1,
        inspection_zone.x_min : inspection_zone.x_max + 1,
    ]
    return focused_mask
