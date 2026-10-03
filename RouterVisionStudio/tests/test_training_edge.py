from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import production_app as ui
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from router_vision.machine_reference import SCHEMA, sha256
from router_vision.model import SobelConfig
from router_vision.training_edge import (
    confirmation_key,
    context_key,
    load_edge_review,
    measure_image_edge,
    measure_selected_edge,
    saved_edge,
)
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
            confirmations=self.window.edge_review.get("confirmations", {}),
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
        self.measurement_image()
        callback = Mock()
        original = Path(self.path).read_bytes()
        dialog = self.dialog(callback)
        edge = dialog.sobel.copy()
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.render()
        np.testing.assert_array_equal(edge, dialog.sobel)
        self.assertEqual(original, Path(self.path).read_bytes())
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.save_button.click()
        self.assertEqual(callback.call_args.args[1], "B")
        self.assertTrue(callback.call_args.args[2]["human_confirmed"])

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
        self.assertEqual(dialog.status.property("state"), "pending")
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

    def test_sobel_button_reviews_the_correct_current_image(self):
        self.window.cmb_settings_image.setCurrentIndex(0)
        self.window.cmb_training_image.setCurrentIndex(2)
        self.assertTrue(
            self.window.settings_workflow.widget(1).isAncestorOf(self.window.btn_sobel_edge)
        )
        self.assertFalse(
            self.window.settings_workflow.widget(2).isAncestorOf(self.window.btn_sobel_edge)
        )
        with patch.object(ui, "TrainingEdgeDialog") as factory:
            self.window.btn_sobel_edge.click()
            self.assertEqual(factory.call_args.args[0], self.window.settings_image_paths[0])

    def measurement_image(self):
        image = np.zeros((400, 500, 3), dtype=np.uint8)
        image[:125] = 180
        image[275:] = 180
        image[275:280, 200:230] = 0
        image[255:275, 260:290] = 180
        cv2.imwrite(self.path, image)
        self.data["images"][0]["sha256"] = sha256(self.path)
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        return image

    def test_selected_side_measures_known_inner_and_outer_deviations(self):
        image = self.measurement_image()
        review = self.review()
        upper, _ = measure_selected_edge(image, review, "A")
        lower, _ = measure_selected_edge(image, review, "B")
        self.assertAlmostEqual(upper["inner_line_max_mm"], 0.105, delta=0.005)
        self.assertEqual(upper["outer_line_max_mm"], 0)
        self.assertAlmostEqual(lower["inner_line_max_mm"], 0.145, delta=0.005)
        self.assertAlmostEqual(lower["outer_line_max_mm"], 0.105, delta=0.005)
        self.assertEqual(lower["reference_line_px"], review["B"]["line"])
        self.assertFalse(lower["calibration_verified"])
        self.assertIsNone(lower["production_decision"])

    def test_partial_and_missing_edges_do_not_claim_complete_or_zero_measurement(self):
        image = self.measurement_image()
        image[:, :245] = 0
        partial, _ = measure_selected_edge(image, self.review(), "B")
        self.assertEqual(partial["measurement_status"], "PARTIAL")
        self.assertGreater(partial["coverage"], 0)
        self.assertLess(partial["coverage"], 1)
        self.assertIsNotNone(partial["outer_line_max_mm"])
        self.assertIsNone(partial["complete_inner_mm"])
        self.assertIsNone(partial["complete_outer_mm"])
        blank, _ = measure_selected_edge(np.zeros_like(image), self.review(), "B")
        self.assertEqual(blank["measurement_status"], "NO_EDGE")
        self.assertIsNone(blank["inner_line_max_mm"])
        self.assertIsNone(blank["outer_line_max_mm"])
        with self.assertRaises(ValueError):
            measure_selected_edge(image, self.review(), "UNSURE")

    def test_dialog_recomputes_saved_choice_and_clears_measurement_for_unsure(self):
        self.measurement_image()
        self.window._save_training_edge(self.path, self.review(), "B")
        dialog = self.dialog()
        self.assertEqual(dialog.measurement["edge"], "B")
        self.assertIn("ESTIMATED", dialog.measurement_label.text())
        self.assertIn("Blue: visual guide only", dialog.measurement_label.text())
        dialog.choice.setCurrentIndex(dialog.choice.findData("A"))
        self.assertEqual(dialog.measurement["edge"], "A")
        self.assertEqual(dialog.measurement["outer_line_max_mm"], 0)
        dialog.choice.setCurrentIndex(dialog.choice.findData("UNSURE"))
        self.assertIsNone(dialog.measurement)
        self.assertIsNone(dialog.measurement_profile)
        self.assertIn("Select A or B", dialog.measurement_label.text())

    def test_changed_manifest_invalidates_measurement_and_lines(self):
        self.measurement_image()
        dialog = self.dialog()
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        self.assertIsNotNone(dialog.measurement)
        self.data["note"] = "changed while reviewing"
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        dialog.choice.setCurrentIndex(dialog.choice.findData("A"))
        self.assertIsNone(dialog.measurement)
        self.assertIsNone(dialog.review)
        self.assertFalse(dialog.save_button.isEnabled())
        self.assertIn("UNAVAILABLE", dialog.measurement_label.text())

    def test_reused_choice_recomputes_image_baseline_without_reusing_confirmation(self):
        image = self.measurement_image()
        other_path = self.window.settings_image_paths[2]
        cv2.imwrite(other_path, np.roll(image, 10, axis=0))
        record = self.data["images"][2]
        record["sha256"] = sha256(other_path)
        projection = record["projection"]
        for point in projection["reconstructed_nc_xy_mm"]:
            point[1] += 0.1
        for key in ("upper_A_px", "lower_B_px"):
            for point in projection[key]:
                point[1] += 10
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        self.window._save_training_edge(self.path, self.review(), "B")
        dialog = TrainingEdgeDialog(
            other_path,
            self.window._training_edge_inputs(other_path),
            self.window.edge_review["choices"],
            SobelConfig(),
            self.manifest,
        )
        self.addCleanup(dialog.deleteLater)
        self.assertEqual(dialog.measurement["edge"], "B")
        self.assertNotEqual(dialog.measurement["reference_line_px"], projection["lower_B_px"])
        self.assertAlmostEqual(dialog.measurement["reference_line_px"][0][1], 284.5, delta=0.1)
        self.assertAlmostEqual(dialog.measurement["inner_line_max_mm"], 0.05, delta=0.005)
        self.assertAlmostEqual(dialog.measurement["outer_line_max_mm"], 0.20, delta=0.005)
        self.assertNotEqual(dialog.status.property("state"), "saved")
        self.assertTrue(dialog.save_button.isEnabled())

    def test_save_button_confirms_without_checkbox_and_keeps_labels_separate(self):
        self.measurement_image()
        labels = copy.deepcopy(self.window.sobel_records)
        callback = Mock(
            side_effect=lambda r, e, c: self.window._save_training_edge(self.path, r, e, c)
        )
        dialog = self.dialog(callback)
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        callback.assert_not_called()
        self.assertFalse(hasattr(dialog, "confirm_check"))
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.save_button.click()
        state = self.window._load_edge_review_state()
        record = state["confirmations"][confirmation_key(self.review())]
        self.assertTrue(record["human_confirmed"])
        self.assertFalse(record["used_for_classifier_training"])
        self.assertEqual(record["snapshot"]["edge"], "B")
        self.assertGreater(len(record["snapshot"]["accepted_edge_points_px"]), 0)
        self.assertEqual(self.window.sobel_records, labels)
        reopened = self.dialog()
        self.assertIn("Confirmed", reopened.status.text())
        reopened.choice.setCurrentIndex(reopened.choice.findData("A"))
        self.assertNotEqual(reopened.status.property("state"), "saved")

    def test_rejected_detector_cannot_be_confirmed_but_not_sure_can_be_saved(self):
        callback = Mock()
        dialog = self.dialog(callback)  # random-noise fixture has no supported baseline
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        self.assertIsNone(dialog.snapshot)
        self.assertFalse(hasattr(dialog, "confirm_check"))
        self.assertFalse(dialog.save_button.isEnabled())
        dialog.confirm()
        callback.assert_not_called()
        dialog.choice.setCurrentIndex(dialog.choice.findData("UNSURE"))
        dialog.confirm()
        self.assertIsNone(callback.call_args.args[2])

    def test_confirmation_is_not_inherited_by_another_panel_even_with_same_pixels(self):
        image = self.measurement_image()
        other = self.window.settings_image_paths[2]
        cv2.imwrite(other, image)
        self.data["images"][2]["sha256"] = sha256(other)
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        dialog = self.dialog(lambda r, e, c: self.window._save_training_edge(self.path, r, e, c))
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.confirm()
        other_dialog = TrainingEdgeDialog(
            other,
            self.window._training_edge_inputs(other),
            self.window.edge_review["choices"],
            SobelConfig(),
            self.manifest,
            confirmations=self.window.edge_review["confirmations"],
        )
        self.addCleanup(other_dialog.deleteLater)
        self.assertEqual(other_dialog.choice.currentData(), "B")
        self.assertNotEqual(other_dialog.status.property("state"), "saved")
        self.assertTrue(other_dialog.save_button.isEnabled())

    def test_tampered_confirmation_is_rejected_and_write_failure_preserves_state(self):
        self.measurement_image()
        callback = Mock()
        dialog = self.dialog(callback)
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.confirm()
        review, edge, confirmation = callback.call_args.args
        changed = copy.deepcopy(confirmation)
        changed["snapshot"]["accepted_edge_points_px"][0][1] += 10
        with self.assertRaisesRegex(ValueError, "confirmed edge changed"):
            self.window._save_training_edge(self.path, review, edge, changed)
        self.window._save_training_edge(self.path, review, edge, confirmation)
        old = copy.deepcopy(self.window.edge_review)
        disk = ui.SOBEL_WORKFLOW_PATH.read_bytes()
        with patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.window._save_training_edge(self.path, review, "UNSURE")
        self.assertEqual(self.window.edge_review, old)
        self.assertEqual(ui.SOBEL_WORKFLOW_PATH.read_bytes(), disk)

    def test_unsure_revokes_confirmation_for_this_image(self):
        self.measurement_image()
        dialog = self.dialog(lambda r, e, c: self.window._save_training_edge(self.path, r, e, c))
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.confirm()
        self.window._save_training_edge(self.path, self.review(), "UNSURE")
        self.assertNotIn(confirmation_key(self.review()), self.window.edge_review["confirmations"])

    def test_preview_settings_change_requires_a_new_visual_confirmation(self):
        self.measurement_image()
        dialog = self.dialog(lambda r, e, c: self.window._save_training_edge(self.path, r, e, c))
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.confirm()
        changed = TrainingEdgeDialog(
            self.path,
            self.window._training_edge_inputs(self.path),
            self.window.edge_review["choices"],
            SobelConfig(edge_gain=2),
            self.manifest,
            confirmations=self.window.edge_review["confirmations"],
        )
        self.addCleanup(changed.deleteLater)
        self.assertNotEqual(changed.status.property("state"), "saved")
        self.assertTrue(changed.save_button.isEnabled())

    def test_partial_confirmation_keeps_missing_columns_unconfirmed(self):
        image = self.measurement_image()
        image[240:350, 240:250] = np.linspace(0, 180, 110).astype(np.uint8)[:, None, None]
        cv2.imwrite(self.path, image)
        self.data["images"][0]["sha256"] = sha256(self.path)
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        dialog = self.dialog(lambda r, e, c: self.window._save_training_edge(self.path, r, e, c))
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        self.assertEqual(dialog.measurement["measurement_status"], "PARTIAL")
        self.assertIn("accepted samples only", dialog.measurement_label.text())
        dialog.confirm()
        record = self.window.edge_review["confirmations"][confirmation_key(self.review())]
        self.assertIsNone(record["snapshot"]["measurement"]["complete_outer_mm"])
        self.assertGreater(len(record["snapshot"]["measurement"]["rejected_x_px"]), 0)

    def test_selected_side_is_the_only_blue_contour(self):
        self.measurement_image()
        dialog = self.dialog()
        for side in ("A", "B"):
            dialog.choice.setCurrentIndex(dialog.choice.findData(side))
            x0, y0, x1, _ = dialog.display_roi
            overlay = dialog.preview_array[:, x1 - x0 :]
            blue = np.all(overlay == [255, 190, 0], axis=2)
            ys = np.nonzero(blue)[0] + y0
            self.assertGreater(len(ys), 0)
            self.assertTrue(np.all(ys < 200) if side == "A" else np.all(ys > 200))
            guide = dialog.measurement["preview_reference"]
            self.assertEqual(guide["line_px"], dialog.measurement["reference_line_px"])
            blue_reference = np.all(overlay == [255, 100, 0], axis=2)
            line_ys = np.nonzero(blue_reference)[0] + y0
            self.assertGreater(len(line_ys), 0)
            self.assertTrue(np.all(line_ys < 200) if side == "A" else np.all(line_ys > 200))

    def test_measurement_button_shows_max_mm_and_follows_selected_side(self):
        self.measurement_image()
        callback = Mock()
        dialog = self.dialog(callback)
        self.assertFalse(dialog.measure_button.isEnabled())
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        self.assertTrue(dialog.cut_measurement_label.isHidden())
        dialog.measure_button.click()
        self.assertIn("Inner Cut (MAX): 0.0500 mm", dialog.cut_measurement_label.text())
        self.assertIn("Outer Cut (MAX): 0.2000 mm", dialog.cut_measurement_label.text())
        self.assertFalse(dialog.cut_measurement_label.isHidden())
        callback.assert_not_called()
        dialog.choice.setCurrentIndex(dialog.choice.findData("A"))
        self.assertIn("EDGE A", dialog.cut_measurement_label.text())
        self.assertIn("Outer Cut (MAX): 0.0000 mm", dialog.cut_measurement_label.text())
        dialog.choice.setCurrentIndex(dialog.choice.findData("UNSURE"))
        self.assertFalse(dialog.measure_button.isEnabled())
        self.assertIn("N/A mm", dialog.cut_measurement_label.text())

    def test_click_measure_and_focus_reuse_analysis_and_render_once_per_click(self):
        self.measurement_image()
        dialog = self.dialog()
        with (
            patch(
                "router_vision.training_edge_ui.measure_image_edge", wraps=measure_image_edge
            ) as analyse,
            patch.object(dialog, "render", wraps=dialog.render) as render,
        ):
            dialog.select_edge_at(np.array([270, 255]))
            self.assertEqual(dialog.choice.currentData(), "B")
            self.assertEqual(analyse.call_count, 2)  # one analysis per candidate side
            render.assert_called_once()
            before = copy.deepcopy(dialog.snapshot)
            dialog.measure_button.click()
            dialog.outer_focus_button.click()
            dialog.select_edge_at(np.array([350, 125]))
            dialog.select_edge_at(np.array([270, 255]))
            self.assertEqual(analyse.call_count, 2)
            self.assertEqual(dialog.snapshot, before)
            self.assertNotIn("0.2000", dialog.measurement_label.text())
            self.assertIn("0.2000", dialog.cut_measurement_label.text())

    def test_reloading_changed_scale_replaces_cached_analysis(self):
        self.measurement_image()
        dialog = self.dialog()
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        self.assertAlmostEqual(dialog.measurement["outer_line_max_mm"], 0.2, places=4)
        self.data["scale_mm_per_px_xy"] = [0.02, 0.02]
        # Keep the fixture's stored machine tangents consistent with the new scale.
        for record in self.data["images"]:
            for name in ("upper_A_px", "lower_B_px"):
                points = np.asarray(record["projection"][name])
                record["projection"][name] = ([250, 200] + (points - [250, 200]) / 2).tolist()
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        dialog.measure_button.click()
        self.assertIsNone(dialog.measurement)
        self.assertFalse(dialog.measure_button.isEnabled())
        dialog.load_reference(str(self.manifest))
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        self.assertAlmostEqual(dialog.measurement["outer_line_max_mm"], 0.4, places=4)

    def test_vertical_edges_and_reversed_travel_keep_same_sides_and_mm(self):
        image = self.measurement_image()
        horizontal, _ = measure_image_edge(image, self.review(), "B")
        vertical_image = cv2.transpose(image)
        cv2.imwrite(self.path, vertical_image)
        record = self.data["images"][0]
        record["sha256"] = sha256(self.path)
        record["size_px"] = [400, 500]
        projection = record["projection"]
        for name in ("reconstructed_nc_xy_mm", "upper_A_px", "lower_B_px"):
            projection[name] = [point[::-1] for point in projection[name]]
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        results = []
        for reverse in (False, True):
            if reverse:
                for name in ("reconstructed_nc_xy_mm", "upper_A_px", "lower_B_px"):
                    projection[name].reverse()
                self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
            review = self.review()
            result, profile = measure_image_edge(vertical_image, review, "B")
            self.assertEqual(review["A"]["pcb_side_name"], "left")
            self.assertEqual(review["B"]["pcb_side_name"], "right")
            self.assertEqual(result["cut_path"]["sampling_axis"], "y")
            for name in ("inner_line_max_mm", "outer_line_max_mm"):
                self.assertAlmostEqual(result[name], horizontal[name], places=6)
            self.assertTrue(np.all(profile.points[profile.accepted, 0] > 200))
            results.append(result)
        self.assertEqual(
            results[0]["cut_path"]["axis_extent_px"], results[1]["cut_path"]["axis_extent_px"]
        )
        dialog = self.dialog()
        self.assertIn("LEFT", dialog.choice.itemText(dialog.choice.findData("A")))
        self.assertIn("RIGHT", dialog.choice.itemText(dialog.choice.findData("B")))
        dialog.select_edge_at(np.array([255, 270]))
        self.assertEqual(dialog.choice.currentData(), "B")
        dialog.outer_focus_button.click()
        self.assertIn("0.2000 mm", dialog.cut_measurement_label.text())

    def test_extreme_outside_tool_travel_is_excluded_from_maximum(self):
        image = self.measurement_image()
        image[275:295, 350:365] = (
            0  # deeper defect outside the tool span and its local zero samples
        )
        cv2.imwrite(self.path, image)
        record = self.data["images"][0]
        record["sha256"] = sha256(self.path)
        projection = record["projection"]
        projection.update(
            reconstructed_nc_xy_mm=[[-0.5, 0], [0.5, 0]],
            bit_diameter_mm=0.2,
            upper_A_px=[[200, 190], [300, 190]],
            lower_B_px=[[200, 210], [300, 210]],
        )
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        result, profile = measure_image_edge(image, self.review(), "B")
        self.assertEqual(result["cut_path"]["axis_extent_px"], [190, 310])
        self.assertEqual(result["measured_x_range"], [190, 310])
        self.assertAlmostEqual(result["inner_line_max_mm"], 0.05, places=4)
        self.assertAlmostEqual(result["outer_line_max_mm"], 0.20, places=4)
        self.assertTrue(np.all((profile.points[:, 0] >= 190) & (profile.points[:, 0] <= 310)))
        for name in ("reconstructed_nc_xy_mm", "upper_A_px", "lower_B_px"):
            projection[name].reverse()
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        reverse, _ = measure_image_edge(image, self.review(), "B")
        for name in ("inner_line_max_mm", "outer_line_max_mm", "reference_line_px"):
            self.assertEqual(reverse[name], result[name])

    def test_blue_guide_is_not_used_to_gate_or_compute_measurement(self):
        image = self.measurement_image()
        image[275:281, 375:] = 0  # old global straight-line agreement check rejected this image
        cv2.imwrite(self.path, image)
        self.data["images"][0]["sha256"] = sha256(self.path)
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        review = self.review()
        result, _ = measure_image_edge(image, review, "B")
        self.assertIsNotNone(result["inner_line_max_mm"])
        self.assertFalse(result["preview_reference"]["used_for_mm"])
        self.assertEqual(result["baseline_source"], "normal_pcb_sobel_outside_cut")
        lo, hi = result["cut_path"]["axis_extent_px"]
        self.assertTrue(all(p[0] < lo or p[0] > hi for p in result["normal_pcb_points_px"]))
        with patch("router_vision.training_edge.preview_reference", return_value=None):
            without_guide, _ = measure_image_edge(image, review, "B")
        for name in ("inner_line_max_mm", "outer_line_max_mm", "reference_line_px"):
            self.assertEqual(without_guide[name], result[name])

    def test_transpose_preserves_measurement_with_unequal_xy_pixel_scales(self):
        image = self.measurement_image()
        self.data["scale_mm_per_px_xy"] = [0.01, 0.02]
        record = self.data["images"][0]
        projection = record["projection"]
        for name in ("upper_A_px", "lower_B_px"):
            points = np.asarray(projection[name])
            points[:, 1] = 200 + (points[:, 1] - 200) / 2
            projection[name] = points.tolist()
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        horizontal, _ = measure_image_edge(image, self.review(), "B")
        self.assertAlmostEqual(horizontal["outer_line_max_mm"], 0.4, places=4)
        transposed = cv2.transpose(image)
        cv2.imwrite(self.path, transposed)
        record.update(sha256=sha256(self.path), size_px=[400, 500])
        self.data["scale_mm_per_px_xy"] = [0.02, 0.01]
        for name in ("reconstructed_nc_xy_mm", "upper_A_px", "lower_B_px"):
            projection[name] = [point[::-1] for point in projection[name]]
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        vertical, _ = measure_image_edge(transposed, self.review(), "B")
        self.assertAlmostEqual(
            vertical["outer_line_max_mm"], horizontal["outer_line_max_mm"], places=6
        )
        self.assertAlmostEqual(
            vertical["inner_line_max_mm"], horizontal["inner_line_max_mm"], places=6
        )

    def test_measurement_button_rejects_changed_sources_without_stale_values(self):
        self.measurement_image()
        dialog = self.dialog()
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.measure_button.click()
        self.data["note"] = "source changed before measurement click"
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        dialog.measure_button.click()
        self.assertIsNone(dialog.measurement)
        self.assertFalse(dialog.measure_button.isEnabled())
        self.assertIn("N/A mm", dialog.cut_measurement_label.text())

    def test_wheel_zoom_pan_and_edge_click_keep_source_coordinates_and_mm(self):
        self.measurement_image()
        dialog = self.dialog()
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        dialog.show()
        self.app.processEvents()
        saved_snapshot = copy.deepcopy(dialog.snapshot)
        original_roi = dialog.display_roi
        pos = QPointF(dialog.preview.contentsRect().center())
        point_before = dialog.source_point_from_preview(pos)
        wheel = QWheelEvent(
            pos, pos, QPoint(), QPoint(0, 120), Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False
        )
        self.app.sendEvent(dialog.preview, wheel)
        self.assertLess(
            dialog.display_roi[2] - dialog.display_roi[0], original_roi[2] - original_roi[0]
        )
        np.testing.assert_allclose(dialog.source_point_from_preview(pos), point_before, atol=2)
        self.assertEqual(dialog.snapshot, saved_snapshot)
        dialog.zoom_in_button.click()
        before_pan = dialog.display_roi
        start = dialog.preview.contentsRect().center()
        QTest.mousePress(dialog.preview, Qt.RightButton, pos=start)
        QTest.mouseMove(dialog.preview, start + QPoint(25, 0))
        QTest.mouseRelease(dialog.preview, Qt.RightButton, pos=start + QPoint(25, 0))
        self.assertLess(dialog.display_roi[0], before_pan[0])
        self.assertEqual(dialog.snapshot, saved_snapshot)
        dialog.focus_maximum("outer")
        x0, y0, x1, y1 = dialog.display_roi
        pixmap = dialog.preview.pixmap()
        pw, ph = (
            pixmap.width() / pixmap.devicePixelRatio(),
            pixmap.height() / pixmap.devicePixelRatio(),
        )
        rect = dialog.preview.contentsRect()
        index = dialog.measurement["max_indices"]["outer"]
        x, y = dialog.measurement_profile.points[index]
        click = QPoint(
            round(rect.x() + (rect.width() - pw) / 2 + (x - x0) / (x1 - x0) * pw),
            round(rect.y() + (rect.height() - ph) / 2 + (y - y0) / (y1 - y0) * ph),
        )
        QTest.mouseClick(dialog.preview, Qt.LeftButton, pos=click)
        self.assertEqual(dialog.choice.currentData(), "B")
        self.assertIsNotNone(dialog.clicked_edge_point)
        self.assertEqual(dialog.snapshot, saved_snapshot)
        dialog.zoom_reset_button.click()
        self.assertEqual(dialog.display_roi, dialog.base_roi)

    def test_maximum_buttons_focus_measured_points_without_changing_measurements(self):
        self.measurement_image()
        dialog = self.dialog()
        self.assertFalse(dialog.inner_focus_button.isEnabled())
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        baseline = copy.deepcopy(dialog.measurement)
        for name, button in (
            ("inner", dialog.inner_focus_button),
            ("outer", dialog.outer_focus_button),
        ):
            button.click()
            index = dialog.measurement["max_indices"][name]
            x, y = dialog.measurement_profile.points[index]
            x0, y0, x1, y1 = dialog.display_roi
            self.assertTrue(x0 <= x < x1 and y0 <= y < y1)
            self.assertLess(x1 - x0, dialog.base_roi[2] - dialog.base_roi[0])
            self.assertEqual(dialog.measurement, baseline)
            self.assertFalse(dialog.cut_measurement_label.isHidden())
        dialog.choice.setCurrentIndex(dialog.choice.findData("A"))
        self.assertFalse(dialog.inner_focus_button.isEnabled())
        self.assertFalse(dialog.outer_focus_button.isEnabled())
        dialog.set_detail_view(False)
        self.assertEqual(dialog.display_roi, (0, 0, 500, 400))

    def test_click_selects_an_observed_edge_and_ignores_letterbox_padding(self):
        self.measurement_image()
        callback = Mock()
        dialog = self.dialog(callback)
        dialog.set_detail_view(False)
        dialog.preview.setFixedSize(600, 180)
        dialog.show()
        self.app.processEvents()

        def screen_point(x, y):
            pixmap = dialog.preview.pixmap()
            width = pixmap.width() / pixmap.devicePixelRatio()
            height = pixmap.height() / pixmap.devicePixelRatio()
            rect = dialog.preview.contentsRect()
            return QPoint(
                round(rect.x() + (rect.width() - width) / 2 + x / 500 * width),
                round(rect.y() + (rect.height() - height) / 2 + y / 400 * height),
            )

        QTest.mouseClick(dialog.preview, Qt.LeftButton, pos=screen_point(350, 275))
        self.assertEqual(dialog.choice.currentData(), "B")
        self.assertTrue(dialog.save_button.isEnabled())
        callback.assert_not_called()
        self.assertIsNone(dialog.source_point_from_preview(QPoint(0, 0)))
        QTest.mouseClick(dialog.preview, Qt.LeftButton, pos=QPoint(1, 1))
        self.assertEqual(dialog.choice.currentData(), "B")
        QTest.mouseClick(dialog.preview, Qt.LeftButton, pos=screen_point(350, 125))
        self.assertEqual(dialog.choice.currentData(), "A")
        dialog.save_button.click()
        self.assertEqual(callback.call_args.args[1], "A")

    def test_edge_only_annotation_is_saveable_but_has_no_mm_values(self):
        image = self.measurement_image()
        image[240:350, 400:450] = np.linspace(0, 180, 110).astype(np.uint8)[:, None, None]
        cv2.imwrite(self.path, image)
        self.data["images"][0]["sha256"] = sha256(self.path)
        self.manifest.write_text(json.dumps(self.data), encoding="utf-8")
        dialog = self.dialog(lambda r, e, c: self.window._save_training_edge(self.path, r, e, c))
        dialog.choice.setCurrentIndex(dialog.choice.findData("B"))
        self.assertEqual(dialog.measurement["measurement_status"], "EDGE_ONLY")
        self.assertIsNone(dialog.measurement["reference_line_px"])
        self.assertEqual(dialog.measurement["preview_reference"]["status"], "VISUAL_ESTIMATE_ONLY")
        self.assertIn("visual estimate only", dialog.measurement_label.text())
        dialog.measure_button.click()
        self.assertIn("N/A mm", dialog.cut_measurement_label.text())
        self.assertIn("Normal PCB edge outside the cut", dialog.cut_measurement_label.text())
        self.assertTrue(dialog.save_button.isEnabled())
        dialog.save_button.click()
        snapshot = self.window.edge_review["confirmations"][confirmation_key(self.review())][
            "snapshot"
        ]
        self.assertGreater(len(snapshot["accepted_edge_points_px"]), 0)
        self.assertIsNone(snapshot["measurement"]["inner_mm"])
        self.assertIsNone(snapshot["measurement"]["outer_mm"])
        self.assertFalse(snapshot["measurement"]["preview_reference"]["used_for_mm"])
        self.assertIn("no baseline or mm", snapshot["review_scope"])


if __name__ == "__main__":
    unittest.main()
