from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

from sobel_tuning_ui_main import (
    TuneSettings,
    _center_width_to_bounds,
    _command_text,
    save_recipe,
)


class SobelTuningUiTest(unittest.TestCase):
    def test_center_width_to_bounds_clamps_to_image_ratio(self) -> None:
        self.assertEqual(_center_width_to_bounds(0.05, 0.30), (0.0, 0.30))
        lower, upper = _center_width_to_bounds(0.95, 0.30)
        self.assertAlmostEqual(lower, 0.70)
        self.assertAlmostEqual(upper, 1.0)

    def test_command_text_contains_tuned_inspection_values(self) -> None:
        settings = TuneSettings(
            sobel_threshold_ratio=0.145,
            display_thickness=3,
            edge_close_kernel=5,
            edge_close_iterations=2,
            edge_dilate_iterations=1,
            black_threshold=62,
            x_min_ratio=0.20,
            x_max_ratio=0.85,
            y_min_ratio=0.15,
            y_max_ratio=0.80,
            view_mode=2,
        )

        command = _command_text(settings)

        self.assertIn("--sobel-threshold-ratio 0.145", command)
        self.assertIn("--edge-close-kernel 5", command)
        self.assertIn("--edge-close-iterations 2", command)
        self.assertIn("--edge-dilate-iterations 1", command)
        self.assertIn("--display-edge-thickness 3", command)
        self.assertIn("--edge-view y", command)
        self.assertNotIn("--pixels-per-mm", command)
        self.assertNotIn("--inspection-x-min-ratio", command)

    def test_save_recipe_writes_json_and_index(self) -> None:
        settings = TuneSettings(
            sobel_threshold_ratio=0.145,
            display_thickness=3,
            edge_close_kernel=5,
            edge_close_iterations=2,
            edge_dilate_iterations=1,
            black_threshold=62,
            x_min_ratio=0.20,
            x_max_ratio=0.85,
            y_min_ratio=0.15,
            y_max_ratio=0.80,
            view_mode=2,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recipe_path = save_recipe(
                settings=settings,
                recipe_dir=Path(temp_dir),
                program_name="160914002C01.rcp",
                sample_image=Path("SepData/camera/20260619_082849.bmp"),
            )
            recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
            index_text = (Path(temp_dir) / "recipe_index.csv").read_text(encoding="utf-8")

        self.assertEqual(recipe_path.name, "160914002C01.rcp.json")
        self.assertEqual(recipe["program_name"], "160914002C01.rcp")
        self.assertEqual(recipe["detection"]["black_threshold"], 62)
        self.assertEqual(recipe["detection"]["sobel_threshold_ratio"], 0.145)
        self.assertEqual(recipe["detection"]["edge_close_kernel"], 5)
        self.assertEqual(recipe["detection"]["edge_close_iterations"], 2)
        self.assertEqual(recipe["detection"]["edge_dilate_iterations"], 1)
        self.assertEqual(recipe["detection"]["edge_view"], "y")
        self.assertEqual(recipe["inspection_zone"]["x_min_ratio"], 0.2)
        self.assertNotIn("calibration", recipe)
        self.assertNotIn("classification", recipe)
        self.assertIn("160914002C01.rcp", index_text)
        self.assertIn("0.145", index_text)
        self.assertIn("edge_close_kernel", index_text)


if __name__ == "__main__":
    unittest.main()
