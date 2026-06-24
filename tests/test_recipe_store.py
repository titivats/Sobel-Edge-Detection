from __future__ import annotations

import unittest
import tempfile
import json
from pathlib import Path

from edge_view import edge_view_name
from recipe_store import (
    TuneSettings,
    save_recipe,
)


class SobelTuningUiTest(unittest.TestCase):
    def test_edge_view_name_maps_recipe_modes(self) -> None:
        self.assertEqual(edge_view_name(0), "all")
        self.assertEqual(edge_view_name(1), "x")
        self.assertEqual(edge_view_name(2), "y")
        self.assertEqual(edge_view_name(3), "all")

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

    def test_save_recipe_preserves_created_at_when_updating(self) -> None:
        settings = TuneSettings(
            sobel_threshold_ratio=0.12,
            display_thickness=1,
            edge_close_kernel=1,
            edge_close_iterations=0,
            edge_dilate_iterations=0,
            black_threshold=55,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            recipe_dir = Path(temp_dir)
            recipe_path = save_recipe(
                settings,
                recipe_dir,
                "MANUAL_TUNED",
                Path("first.bmp"),
            )
            original = json.loads(recipe_path.read_text(encoding="utf-8"))
            updated_path = save_recipe(
                settings,
                recipe_dir,
                "MANUAL_TUNED",
                Path("second.bmp"),
            )
            updated = json.loads(updated_path.read_text(encoding="utf-8"))

        self.assertEqual(updated["created_at"], original["created_at"])
        self.assertEqual(updated["sample_image"], "second.bmp")


if __name__ == "__main__":
    unittest.main()
