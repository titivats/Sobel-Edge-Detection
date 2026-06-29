from __future__ import annotations

import csv
from pathlib import Path

from .measurement_data_models import IntrusionMeasurement


def write_csv(measurements: list[IntrusionMeasurement], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "image",
                "width_px",
                "height_px",
                "orientation",
                "baseline_left_px",
                "baseline_right_px",
                "baseline_top_px",
                "baseline_bottom_px",
                "min_slot_width_px",
                "min_slot_height_px",
                "min_left_intrusion_px",
                "max_left_intrusion_px",
                "min_right_intrusion_px",
                "max_right_intrusion_px",
                "min_top_intrusion_px",
                "max_top_intrusion_px",
                "min_bottom_intrusion_px",
                "max_bottom_intrusion_px",
                "max_intrusion_px",
                "max_intrusion_mm",
                "max_intrusion_side",
            ],
        )
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(
                {
                    "image": str(measurement.image_path),
                    "width_px": measurement.width_px,
                    "height_px": measurement.height_px,
                    "orientation": measurement.orientation,
                    "baseline_left_px": f"{measurement.baseline_left_px:.3f}",
                    "baseline_right_px": f"{measurement.baseline_right_px:.3f}",
                    "baseline_top_px": f"{measurement.baseline_top_px:.3f}",
                    "baseline_bottom_px": f"{measurement.baseline_bottom_px:.3f}",
                    "min_slot_width_px": f"{measurement.min_slot_width_px:.3f}",
                    "min_slot_height_px": f"{measurement.min_slot_height_px:.3f}",
                    "min_left_intrusion_px": f"{measurement.min_left_intrusion_px:.3f}",
                    "max_left_intrusion_px": f"{measurement.max_left_intrusion_px:.3f}",
                    "min_right_intrusion_px": f"{measurement.min_right_intrusion_px:.3f}",
                    "max_right_intrusion_px": f"{measurement.max_right_intrusion_px:.3f}",
                    "min_top_intrusion_px": f"{measurement.min_top_intrusion_px:.3f}",
                    "max_top_intrusion_px": f"{measurement.max_top_intrusion_px:.3f}",
                    "min_bottom_intrusion_px": f"{measurement.min_bottom_intrusion_px:.3f}",
                    "max_bottom_intrusion_px": f"{measurement.max_bottom_intrusion_px:.3f}",
                    "max_intrusion_px": f"{measurement.max_intrusion_px:.3f}",
                    "max_intrusion_mm": f"{measurement.max_intrusion_mm:.3f}",
                    "max_intrusion_side": measurement.max_intrusion_side,
                }
            )
