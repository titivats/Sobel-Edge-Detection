from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from router_vision.guard import ProtectedPathError
from router_vision.measurement_trial import pick_reference, run_trial, side_from_point
from router_vision.reference import image_signature


class MeasurementTrialTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.picture = self.root / "machine" / "Picture"
        self.picture.mkdir(parents=True)
        self.path = self.picture / "board.png"
        self.image = np.zeros((240, 320, 3), dtype=np.uint8)
        self.image[120:] = 180
        self.output = self.root / "reports"

    def measure(self, line):
        cv2.imwrite(str(self.path), self.image)
        original = self.path.read_bytes()
        folder = run_trial(
            self.path, line, side=1, scales=(0.01, 0.02), radius=24, output_root=self.output
        )
        self.assertEqual(original, self.path.read_bytes())
        self.assertTrue((folder / "profile.csv").is_file())
        self.assertIsNotNone(cv2.imread(str(folder / "annotated.png")))
        return json.loads((folder / "measurement.json").read_text())

    def test_inward_and_protrusion_have_correct_sign_and_mm(self):
        inward = self.measure([[30, 109.5], [285, 109.5]])
        self.assertEqual(inward["measurement_status"], "COMPLETE_ESTIMATE")
        self.assertAlmostEqual(inward["inward_mm"], 0.2, delta=0.015)
        self.assertAlmostEqual(inward["inner_line_max_mm"], 0.2, delta=0.015)
        self.assertEqual(inward["outer_line_max_mm"], 0)
        self.assertAlmostEqual(inward["protrusion_mm"], 0, delta=0.015)
        outward = self.measure([[30, 129.5], [285, 129.5]])
        self.assertAlmostEqual(outward["protrusion_mm"], 0.2, delta=0.015)
        self.assertAlmostEqual(outward["outer_line_max_mm"], 0.2, delta=0.015)
        self.assertEqual(outward["inner_line_max_mm"], 0)
        self.assertAlmostEqual(outward["max_blue_yellow_mm"], 0.2, delta=0.015)
        self.assertIsNotNone(outward["max_edge_point_px"])
        self.assertEqual(inward["max_blue_yellow_mm"], 0)
        self.assertFalse(outward["calibration_verified"])
        self.assertIsNone(outward["production_decision"])

    def test_blank_image_does_not_report_zero_defect(self):
        self.image[:] = 0
        result = self.measure([[30, 119.5], [285, 119.5]])
        self.assertEqual(result["measurement_status"], "INCOMPLETE")
        self.assertIsNone(result["inward_mm"])
        self.assertIsNone(result["inner_line_max_mm"])
        self.assertIsNone(result["outer_line_max_mm"])
        self.assertIsNone(result["observed_max_blue_yellow_mm"])
        self.assertIsNone(result["accepted_samples_only"]["protrusion_mm"])

    def test_partial_edge_does_not_report_complete_maximum(self):
        self.image[:, 100:150] = 0
        result = self.measure([[30, 119.5], [285, 119.5]])
        self.assertGreater(result["coverage"], 0)
        self.assertLess(result["coverage"], 1)
        self.assertIsNone(result["protrusion_mm"])
        self.assertIsNone(result["max_blue_yellow_mm"])

    def test_output_inside_machine_is_blocked(self):
        cv2.imwrite(str(self.path), self.image)
        with self.assertRaises(ProtectedPathError):
            run_trial(
                self.path,
                [[30, 119.5], [285, 119.5]],
                scales=(0.01, 0.01),
                output_root=self.picture.parent / "reports",
            )
        self.assertFalse((self.picture.parent / "reports").exists())

    def test_invalid_scale_is_rejected(self):
        cv2.imwrite(str(self.path), self.image)
        with self.assertRaises(ValueError):
            run_trial(
                self.path, [[30, 119.5], [285, 119.5]], scales=(0, 0.01), output_root=self.output
            )
        self.assertFalse(self.output.exists())

    def test_pcb_side_click_is_independent_of_line_direction(self):
        self.assertEqual(side_from_point([[30, 100], [280, 100]], [100, 150]), 1)
        self.assertEqual(side_from_point([[280, 100], [30, 100]], [100, 150]), -1)
        with self.assertRaises(ValueError):
            side_from_point([[30, 100], [280, 100]], [100, 100])

    def test_image_changed_while_picking_reference_is_rejected(self):
        cv2.imwrite(str(self.path), self.image)
        expected = image_signature(self.path)
        self.path.write_bytes(b"replaced image")
        with self.assertRaisesRegex(ValueError, "Source image changed"):
            run_trial(
                self.path,
                [[30, 119.5], [285, 119.5]],
                scales=(0.01, 0.01),
                output_root=self.output,
                expected_source=expected,
            )
        self.assertFalse(self.output.exists())

    def test_picker_maps_display_clicks_back_to_original_pixels(self):
        image = np.zeros((1400, 2000, 3), dtype=np.uint8)

        def click(_name, callback):
            for x, y in ((100, 100), (400, 100), (200, 200)):
                callback(cv2.EVENT_LBUTTONDOWN, x, y, 0, None)

        with (
            patch("cv2.namedWindow"),
            patch("cv2.setMouseCallback", side_effect=click),
            patch("cv2.imshow"),
            patch("cv2.waitKey", return_value=13),
            patch("cv2.getWindowProperty", return_value=1),
            patch("cv2.destroyAllWindows"),
        ):
            line, side = pick_reference(image)
        self.assertEqual(line, [[200, 200], [800, 200]])
        self.assertEqual(side, 1)


if __name__ == "__main__":
    unittest.main()
