from __future__ import annotations

import numpy as np

from .measurement_data_models import FocusZone


def focus_zone_from_ratios(
    image_shape: tuple[int, int],
    x_min_ratio: float,
    x_max_ratio: float,
    y_min_ratio: float,
    y_max_ratio: float,
) -> FocusZone:
    """Build the yellow measurement focus zone from image-relative ratios."""
    height, width = image_shape
    return FocusZone(
        x_min=max(0, min(width - 1, int(round(width * x_min_ratio)))),
        x_max=max(0, min(width - 1, int(round(width * x_max_ratio)))),
        y_min=max(0, min(height - 1, int(round(height * y_min_ratio)))),
        y_max=max(0, min(height - 1, int(round(height * y_max_ratio)))),
    )


def apply_focus_zone_to_mask(mask: np.ndarray, focus_zone: FocusZone) -> np.ndarray:
    """Keep mask pixels only inside the configured focus zone."""
    focused_mask = np.zeros_like(mask)
    focused_mask[
        focus_zone.y_min : focus_zone.y_max + 1,
        focus_zone.x_min : focus_zone.x_max + 1,
    ] = mask[
        focus_zone.y_min : focus_zone.y_max + 1,
        focus_zone.x_min : focus_zone.x_max + 1,
    ]
    return focused_mask
