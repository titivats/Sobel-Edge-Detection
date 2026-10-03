"""Acceptance limits shared by production logic and the settings dialog."""

from __future__ import annotations

import math


def validated_limits(inner, outer) -> dict[str, float]:
    limits = {}
    for key, caption, raw in (
        ("inner_max_mm", "Max Inner", inner),
        ("outer_max_mm", "Max Outer", outer),
    ):
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{caption}: enter a non-negative number in mm.") from None
        if isinstance(raw, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{caption}: enter a finite, non-negative number in mm.")
        limits[key] = value
    return limits
