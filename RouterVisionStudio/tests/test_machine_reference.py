from __future__ import annotations

import copy
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
from router_vision.machine_reference import SCHEMA, load_manifest, reference_for_image, sha256
from router_vision.measurement_trial import main


class MachineReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.picture = self.root / "machine/Picture/board.png"
        self.picture.parent.mkdir(parents=True)
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        image[190:] = 180
        cv2.imwrite(str(self.picture), image)
        self.recipe = self.root / "machine/Recipe/board.rcp"
        self.recipe.parent.mkdir()
        self.recipe.write_bytes(b"recipe fixture")
        self.manifest_path = self.root / "reference.json"
        self.data = {
            "schema": SCHEMA,
            "source_files": [{"path": str(self.recipe), "sha256": sha256(self.recipe)}],
            "scale_mm_per_px_xy": [0.01, 0.01],
            "images": [
                {
                    "name": self.picture.name,
                    "sha256": sha256(self.picture),
                    "size_px": [320, 240],
                    "csv_sn": "csv-1",
                    "database_sn": 91,
                    "cut_point": 1,
                    "acquisition_log_line": 123,
                    "mapping_basis": "Fixture correlation",
                    "projection": {
                        "reconstructed_nc_xy_mm": [[-1.1, 0], [0.9, 0]],
                        "reconstructed_g87_camera_xy_mm": [0, 0],
                        "bit_diameter_mm": 1.3,
                        "upper_A_px": [[50, 55], [250, 55]],
                        "lower_B_px": [[50, 185], [250, 185]],
                    },
                }
            ],
        }

    def load(self):
        self.manifest_path.write_text(json.dumps(self.data), encoding="utf-8")
        return load_manifest(self.manifest_path)

    def test_tangents_use_captured_frame_and_radius_with_correct_material_side(self):
        manifest = self.load()
        a = reference_for_image(manifest, self.picture, "A")
        b = reference_for_image(manifest, self.picture, "B")
        np.testing.assert_allclose(a["line"], [[50, 55], [250, 55]])
        np.testing.assert_allclose(b["line"], [[50, 185], [250, 185]])
        self.assertEqual((a["side"], b["side"]), (-1, 1))
        self.assertFalse(b["provenance"]["production_reference_verified"])
        self.assertFalse(b["provenance"]["edge_identity_verified"])
        self.assertNotEqual(b["provenance"]["csv_sn"], b["provenance"]["database_sn"])
        # Center is x=150, not automatically recentered to the image's x=160.
        self.assertAlmostEqual(np.asarray(b["line"])[:, 0].mean(), 150)

    def test_reversing_cut_direction_keeps_B_on_lower_PCB(self):
        projection = self.data["images"][0]["projection"]
        for key in ("reconstructed_nc_xy_mm", "upper_A_px", "lower_B_px"):
            projection[key].reverse()
        selected = reference_for_image(self.load(), self.picture)
        np.testing.assert_allclose(selected["line"], [[250, 185], [50, 185]])
        self.assertEqual(selected["side"], -1)

    def test_changed_machine_source_is_rejected(self):
        self.load()
        self.recipe.write_bytes(b"new recipe")
        with self.assertRaisesRegex(ValueError, "source changed"):
            load_manifest(self.manifest_path)

    def test_changed_image_is_rejected_even_with_same_filename(self):
        manifest = self.load()
        self.picture.write_bytes(b"replacement image")
        with self.assertRaisesRegex(ValueError, "Image hash"):
            reference_for_image(manifest, self.picture)

    def test_duplicate_record_is_rejected(self):
        self.data["images"].append(copy.deepcopy(self.data["images"][0]))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            reference_for_image(self.load(), self.picture)

    def test_tangent_moved_to_fit_visible_edge_is_rejected(self):
        self.data["images"][0]["projection"]["lower_B_px"][0][1] += 5
        with self.assertRaisesRegex(ValueError, "Stored tangent differs"):
            reference_for_image(self.load(), self.picture)

    def test_invalid_scale_is_rejected(self):
        self.data["scale_mm_per_px_xy"][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "geometry / scale"):
            reference_for_image(self.load(), self.picture)

    def test_cli_measures_without_manual_line_and_preserves_provenance(self):
        self.load()
        output = self.root / "trials"
        original = self.picture.read_bytes()
        with redirect_stdout(StringIO()):
            code = main(
                [
                    "--image",
                    str(self.picture),
                    "--machine-reference",
                    str(self.manifest_path),
                    "--output",
                    str(output),
                    "--radius",
                    "24",
                ]
            )
        self.assertEqual(code, 0)
        self.assertEqual(original, self.picture.read_bytes())
        result = json.loads(next(output.glob("*/measurement.json")).read_text())
        self.assertAlmostEqual(result["inner_line_max_mm"], 0.045, delta=0.01)
        self.assertEqual(result["outer_line_max_mm"], 0)
        self.assertEqual(result["machine_reference"]["selected_edge"], "B")
        self.assertEqual(result["machine_reference"]["manifest_sha256"], sha256(self.manifest_path))
        self.assertFalse(result["calibration_verified"])
        self.assertIsNone(result["production_decision"])

    def test_cli_forbids_silent_manual_override_of_recorded_reference(self):
        self.load()
        output = self.root / "trials"
        with redirect_stdout(StringIO()):
            code = main(
                [
                    "--image",
                    str(self.picture),
                    "--machine-reference",
                    str(self.manifest_path),
                    "--output",
                    str(output),
                    "--line",
                    "50",
                    "190",
                    "250",
                    "190",
                ]
            )
        self.assertEqual(code, 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
