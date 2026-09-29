from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import production_app as ui
from router_vision.machine_reference import SCHEMA, sha256
from router_vision.model import SobelConfig
from router_vision.training_edge import context_key, load_edge_review, saved_edge
from router_vision.training_edge_ui import TrainingEdgeDialog

from tests import test_settings_workflow as workflow_tests


class TrainingEdgeTests(unittest.TestCase):
    setUpClass = classmethod(workflow_tests.SettingsWorkflowTests.setUpClass.__func__)

    def setUp(self):
        workflow_tests.SettingsWorkflowTests.setUp(self)
        self.recipe = self.source / "Recipe/PRODUCT-A.rcp"
        self.recipe.write_bytes(b"fixture recipe")
        self.manifest = self.root / "reference.json"
        records = []
        for index, path in enumerate(self.window.settings_image_paths):
            meta = self.window.settings_image_metadata[self.window._sobel_record_key(path)]
            dx = 0.05 if index >= 2 else 0
            center = np.array([[170.0, 200.0], [330.0, 200.0]]) + [dx / 0.01, 0]
            context = {
                "product_id": "PRODUCT-A",
                "recipe_name": self.recipe.name,
                "recipe_sha256": sha256(self.recipe),
                "table": "LeftTable",
                "program_key": 1,
                "layer": "SubBoard_01",
                "cut_point": meta.cut_point,
            }
            records.append(
                {
                    "name": Path(path).name,
                    "sha256": sha256(path),
                    "size_px": [500, 400],
                    "csv_sn": meta.panel_sn,
                    "database_sn": 100 + index // 2,
                    "cut_point": meta.cut_point,
                    "acquisition_log_line": 100 + index,
                    "mapping_basis": "Test fixture",
                    "reference_context": context,
                    "projection": {
                        "source_recipe_sha256": sha256(self.recipe),
                        "reconstructed_nc_xy_mm": [[-0.8 + dx, 0], [0.8 + dx, 0]],
                        "reconstructed_g87_camera_xy_mm": [0, 0],
                        "bit_diameter_mm": 1.3,
                        "upper_A_px": (center - [0, 65]).tolist(),
                        "lower_B_px": (center + [0, 65]).tolist(),
                    },
                }
            )
        self.data = {
            "schema": SCHEMA,
            "source_files": [{"path": str(self.recipe), "sha256": sha256(self.recipe)}],
            "scale_mm_per_px_xy": [0.01, 0.01],
            "images": records,
        }
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        self.path = self.window.settings_image_paths[0]

    def review(self, path=None):
        path = path or self.path
        return load_edge_review(self.manifest, path, **self.window._training_edge_inputs(path))

    def dialog(self, callback=None):
        dialog = TrainingEdgeDialog(
            self.path,
            self.window._training_edge_inputs(self.path),
            self.window.edge_review.get("choices", {}),
            SobelConfig(),
            self.manifest,
            save_callback=callback,
        )
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_save_reload_reuses_side_for_next_panel_but_uses_its_own_coordinates(self):
        first = self.review()
        self.window._save_training_edge(self.path, first, "B")
        state = self.window._load_edge_review_state()
        other = self.review(self.window.settings_image_paths[2])
        self.assertEqual(first["key"], other["key"])
        self.assertEqual(saved_edge(state["choices"], other), "B")
        self.assertNotEqual(first["B"]["line"], other["B"]["line"])
        self.assertIsNone(
            saved_edge(state["choices"], self.review(self.window.settings_image_paths[1]))
        )
        self.window._save_sobel_records()
        self.assertEqual(self.window._load_edge_review_state(), state)

    def test_uncertain_replaces_side_without_changing_good_ng_or_model_contract(self):
        self.window._label_sobel_image("GOOD")
        labels = copy.deepcopy(self.window.sobel_records)
        self.window.settings_trial_approved = True
        self.window._save_training_edge(self.path, self.review(), "A")
        self.window._save_training_edge(self.path, self.review(), "UNSURE")
        self.assertEqual(saved_edge(self.window.edge_review["choices"], self.review()), "UNSURE")
        self.assertEqual(self.window.sobel_records, labels)
        self.assertTrue(self.window.settings_trial_approved)
        self.window._save_current_sobel(image_path=self.path)
        self.assertEqual(
            saved_edge(self.window._load_edge_review_state()["choices"], self.review()), "UNSURE"
        )

    def test_write_failure_keeps_prior_choice_in_memory_and_on_disk(self):
        self.window._save_training_edge(self.path, self.review(), "B")
        old = copy.deepcopy(self.window.edge_review)
        payload = ui.SOBEL_WORKFLOW_PATH.read_bytes()
        with patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.window._save_training_edge(self.path, self.review(), "A")
        self.assertEqual(self.window.edge_review, old)
        self.assertEqual(ui.SOBEL_WORKFLOW_PATH.read_bytes(), payload)

    def test_recipe_product_table_program_and_point_are_distinct_contexts(self):
        context = self.review()["context"]
        for key, value in (
            ("recipe_sha256", "new hash"),
            ("product_id", "OTHER"),
            ("table", "RightTable"),
            ("program_key", 2),
            ("cut_point", 2),
            ("layer", "SubBoard_02"),
        ):
            self.assertNotEqual(context_key(context), context_key({**context, key: value}))

    def test_stale_reference_and_ambiguous_recipe_are_rejected(self):
        self.recipe.write_bytes(b"replacement recipe")
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.review()
        other = self.source / "Recipe/duplicate/PRODUCT-A.rcp"
        other.parent.mkdir()
        other.write_bytes(self.recipe.read_bytes())
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            self.window._training_edge_inputs(self.path)

    def test_product_or_csv_sn_mismatch_is_rejected(self):
        inputs = self.window._training_edge_inputs(self.path)
        for change in ({"product_id": "WRONG"}, {"csv_sn": "WRONG"}, {"cut_point": 2}):
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_edge_review(self.manifest, self.path, **{**inputs, **change})

    def test_malformed_context_reports_missing_identity(self):
        self.data["images"][0]["reference_context"] = None
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "identity"):
            self.review()

    def test_malformed_saved_choice_is_not_treated_as_reviewed(self):
        review = self.review()
        self.assertIsNone(saved_edge({review["key"]: None}, review))

    def test_dialog_overlay_does_not_modify_original_or_sobel(self):
        callback = Mock()
        original = Path(self.path).read_bytes()
        dialog = self.dialog(callback)
        edge = dialog.sobel.copy()
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.render()
        np.testing.assert_array_equal(edge, dialog.sobel)
        self.assertEqual(original, Path(self.path).read_bytes())
        dialog.confirm()
        self.assertEqual(callback.call_args.args[1], "B")

    def test_detail_view_uses_one_roi_and_keeps_reference_coordinates(self):
        dialog = self.dialog()
        original_lines = copy.deepcopy(dialog.review)
        x0, y0, x1, y1 = dialog.display_roi
        expected = dialog.image[y0:y1, x0:x1]
        np.testing.assert_array_equal(dialog.preview_array[:, : x1 - x0], expected)
        for name in ("A", "B"):
            for x, y in dialog.review[name]["line"]:
                self.assertTrue(x0 <= x < x1 and y0 <= y < y1)
        dialog.full_button.click()
        self.assertEqual(dialog.display_roi, (0, 0, 500, 400))
        np.testing.assert_array_equal(dialog.preview_array[:, :500], dialog.image)
        self.assertEqual(dialog.review, original_lines)
        dialog.detail_button.click()
        self.assertEqual(dialog.display_roi, (x0, y0, x1, y1))

    def test_changing_saved_choice_is_shown_as_unsaved(self):
        self.window._save_training_edge(self.path, self.review(), "B")
        dialog = self.dialog()
        self.assertEqual(dialog.status.property("state"), "saved")
        dialog.choice.setCurrentIndex(dialog.choice.findData("A"))
        self.assertEqual(dialog.status.property("state"), "changed")
        self.assertIn("Not saved", dialog.status.text())
        self.assertEqual(saved_edge(self.window.edge_review["choices"], self.review()), "B")

    def test_dialog_change_during_review_blocks_save_and_removes_lines(self):
        callback = Mock()
        dialog = self.dialog(callback)
        Path(self.path).write_bytes(b"changed image")
        dialog.confirm()
        callback.assert_not_called()
        self.assertIsNone(dialog.review)
        self.assertFalse(dialog.save_button.isEnabled())

    def test_dialog_save_error_stays_open_and_does_not_claim_saved(self):
        dialog = self.dialog(Mock(side_effect=OSError("disk full")))
        dialog.confirm()
        self.assertNotEqual(dialog.result(), ui.QDialog.Accepted)
        self.assertIn("not saved", dialog.status.text())

    def test_training_button_reviews_the_correct_current_image(self):
        self.window.cmb_settings_image.setCurrentIndex(0)
        self.window.cmb_training_image.setCurrentIndex(2)
        with patch.object(ui, "TrainingEdgeDialog") as factory:
            self.window.btn_training_edge.click()
            self.assertEqual(factory.call_args.args[0], self.window.settings_image_paths[2])


if __name__ == "__main__":
    unittest.main()
