from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import production_app as ui
import torch
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QFontDatabase
from PySide6.QtWidgets import QApplication
from router_vision.config import AppConfig
from router_vision.interlock import GateState
from router_vision.model import CropBox, CutClassifier, SobelConfig, TrainReport


class SettingsWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if Path("C:/Windows/Fonts/segoeui.ttf").is_file():
            QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf")
        cls.app.setStyle("Fusion")
        cls.app.setStyleSheet(ui.APP_STYLE)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        for folder in ("Picture", "Result", "Recipe"):
            (self.source / folder).mkdir(parents=True, exist_ok=True)
        for stamp in ("120010", "120015", "120210", "120215"):
            # Include the default crop origin (400, 290) for real Sobel preparation.
            picture = np.random.default_rng(8).integers(0, 256, (400, 500, 3), dtype=np.uint8)
            cv2.imwrite(str(self.source / "Picture" / f"20260903_{stamp}.bmp"), picture)
        header = "SN,Barcode,Recipe_Name,ProductId,Result,CuttingTime\n"
        for stamp, sn in (("120030", "1"), ("120230", "2")):
            (self.source / "Result" / f"_20260903_{stamp}.csv").write_text(
                header + f"{sn},,PRODUCT-A.rcp,PRODUCT-A,True,20\n", encoding="utf-8"
            )
        for name, value in (
            ("CONFIG_PATH", self.root / "config.json"),
            ("SOBEL_WORKFLOW_PATH", self.root / "workflow.json"),
            ("SOBEL_OUTPUT_DIR", self.root / "edges"),
            ("DEMO_MODE", False),
        ):
            p = patch.object(ui, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.window = ui.MainWindow()
        self.addCleanup(self.window.deleteLater)
        self.addCleanup(self.window.close)
        self.window.cfg = AppConfig(
            model_dir=str(self.root / "models"),
            picture_dir=str(self.source / "Picture"),
            result_dir=str(self.source / "Result"),
            recipe_dir=str(self.source / "Recipe"),
            eqp_cfg_path=str(self.source / "Config" / "Eqp.cfg"),
        )
        self.window._set_settings_path(str(self.source))
        self.window.sobel_preview_timer.stop()
        self.window._sync_settings_controls()

    def model(self, *, sobel=None, quality=1.0):
        classifier = CutClassifier(crop=CropBox(0, 0, 100, 80), sobel=sobel, model_key="PRODUCT-A")
        classifier.classes = ["GOOD", "NG"]
        classifier.head = torch.nn.Linear(384, 2).eval()
        classifier.report = TrainReport(
            classes=classifier.classes,
            n_train=8,
            n_val=2,
            val_acc=quality,
            per_class={"GOOD": {"n": 1, "correct": 1}, "NG": {"n": 1, "correct": 1}},
        )
        target = self.window._model_path("PRODUCT-A")
        classifier.save(target)
        self.window.settings_classifier = classifier
        self.window.settings_model_name = target.name
        return classifier

    def save_record(self, path, *, label="GOOD", trained=True):
        target = self.window._model_path("PRODUCT-A")
        output = self.root / (Path(path).stem + ".png")
        output.write_bytes(b"saved")
        record = {
            "source": self.window._source_signature(path),
            "sobel_output": str(output),
            "sobel_parameters": self.window._sobel_parameter_record(),
            "crop": asdict(self.window._settings_crop()),
            "training_label": label,
            "trained_label": label,
            "trained": trained,
            "trained_model": target.name,
            "trained_model_signature": self.window._source_signature(str(target)),
            "trained_sobel_parameters": self.window._sobel_parameter_record(),
        }
        self.window.sobel_records[self.window._sobel_record_key(path)] = record
        return record

    def test_technician_label_prepares_selected_training_image_and_advances(self):
        self.window.cmb_training_image.setCurrentIndex(1)
        path = self.window._current_training_image_path()
        source_before = Path(path).read_bytes()
        self.assertNotEqual(path, self.window.settings_image_path)
        parameters = self.window._sobel_parameter_record()
        self.window._label_and_prepare_image("NG")
        self.assertEqual(self.window._training_label_for(path), "NG")
        self.assertTrue(self.window._record_is_saved(path, current_parameters=True))
        self.assertNotEqual(self.window._current_training_image_path(), path)
        record = self.window.sobel_records[self.window._sobel_record_key(path)]
        self.assertEqual(record["sobel_parameters"], parameters)
        self.assertEqual(Path(path).read_bytes(), source_before)
        self.assertTrue(self.window.training_advanced.isHidden())

    def test_technician_save_failure_does_not_label_or_advance(self):
        path = self.window._current_training_image_path()
        with (
            patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")),
            patch.object(ui.QMessageBox, "warning"),
        ):
            self.window._label_and_prepare_image("GOOD")
        self.assertEqual(self.window._current_training_image_path(), path)
        self.assertEqual(self.window._training_label_for(path), "")
        self.assertFalse(self.window._record_is_saved(path))

    def test_technician_label_failure_retains_saved_image_without_advancing(self):
        path = self.window._current_training_image_path()
        original_save = self.window._save_sobel_records
        calls = []

        def fail_second_save(records):
            calls.append(records)
            if len(calls) == 2:
                raise OSError("label storage unavailable")
            return original_save(records)

        with (
            patch.object(self.window, "_save_sobel_records", side_effect=fail_second_save),
            patch.object(ui.QMessageBox, "warning"),
        ):
            self.window._label_and_prepare_image("GOOD")
        self.assertEqual(self.window._current_training_image_path(), path)
        self.assertEqual(self.window._training_label_for(path), "")
        self.assertTrue(self.window._record_is_saved(path, current_parameters=True))

    def test_technician_skip_does_not_create_a_label(self):
        path = self.window._current_training_image_path()
        self.window._next_training_image_to_review()
        self.assertNotEqual(self.window._current_training_image_path(), path)
        self.assertEqual(self.window._training_label_for(path), "")
        self.assertIn("PREPARE IN 2 SOBEL TUNING", self.window.lbl_training_next_step.text())

    def test_manual_review_saves_label_without_changing_selected_image(self):
        path = self.window._current_training_image_path()
        self.window.chk_training_auto_next.setChecked(False)
        self.window._label_and_prepare_image("GOOD")
        self.assertEqual(self.window._current_training_image_path(), path)
        self.assertEqual(self.window._training_label_for(path), "GOOD")
        self.assertTrue(self.window._record_is_saved(path, current_parameters=True))
        self.assertIn("SAVE LABEL", self.window.btn_label_good.text())
        self.assertEqual(self.window.training_good_progress.value(), 1)

    def test_training_guidance_moves_from_existing_model_to_completed_test(self):
        self.model()
        self.window._refresh_training_workflow_status()
        self.assertIn("4a TEST MODEL", self.window.lbl_training_next_step.text())
        self.window.trial_input_signature = self.window._trial_signature()
        self.window._vision_trial_done(
            [(path, "GOOD", 0.99) for path in self.window.settings_image_paths]
        )
        self.window._refresh_training_workflow_status()
        self.assertIn("4b SAVE TESTED SETTINGS", self.window.lbl_training_next_step.text())

    def test_training_actions_are_disabled_during_inspection(self):
        self.window.auto_running = True
        self.window._refresh_training_workflow_status()
        for button in (
            self.window.btn_label_good,
            self.window.btn_label_ng,
            self.window.btn_clear_label,
            self.window.btn_back_to_sobel,
            self.window.btn_train_model,
        ):
            self.assertFalse(button.isEnabled())

    def test_sobel_labels_flow_to_training_without_selecting_a_folder(self):
        first, second = self.window.settings_image_paths[:2]
        self.window.cmb_training_image.setCurrentIndex(1)
        self.assertEqual(self.window.settings_image_path, first)
        with patch.object(ui.QFileDialog, "getExistingDirectory") as folder_picker:
            self.window.btn_sobel_good.click()
            self.window.cmb_settings_image.setCurrentIndex(1)
            self.window.btn_sobel_ng.click()
            self.window.btn_continue_training.click()
        folder_picker.assert_not_called()
        self.assertEqual(self.window.settings_workflow.currentIndex(), 2)
        self.assertEqual(self.window._current_training_image_path(), second)
        self.assertEqual(dict(self.window._training_items()), {first: "GOOD", second: "NG"})
        self.assertEqual(self.window.training_good_progress.value(), 1)
        self.assertEqual(self.window.training_ng_progress.value(), 1)
        self.assertFalse(hasattr(self.window, "btn_training_source"))
        self.window.sobel_records = self.window._load_sobel_records()
        self.assertEqual(dict(self.window._training_items()), {first: "GOOD", second: "NG"})

    def test_sobel_label_save_failure_never_adds_image_to_training(self):
        with (
            patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")),
            patch.object(ui.QMessageBox, "warning"),
        ):
            self.window.btn_sobel_good.click()
        self.assertEqual(self.window._training_items(), [])
        self.assertIn("UNLABELED", self.window.lbl_sobel_label.text())

    def test_sobel_label_and_save_status_follow_saving_edits_and_image_selection(self):
        window = self.window
        self.assertIn("NOT SAVED", window.lbl_sobel_label.text())
        window.btn_sobel_good.click()
        self.assertEqual(window.btn_sobel_good.text(), "SAVED AS GOOD")
        self.assertIn("LABEL GOOD  |  SAVED", window.lbl_current_image_context.text())
        window.btn_sobel_ng.click()
        self.assertEqual(window.btn_sobel_good.text(), "SAVE AS GOOD")
        self.assertEqual(window.btn_sobel_ng.text(), "SAVED AS NG")
        self.assertIn("LABEL NG  |  SAVED", window.lbl_current_image_context.text())
        self.assertIn("LABEL NG", window.cmb_settings_image.currentText())
        original = window.slider_gradient_x.value()
        window.slider_gradient_x.setValue(original + 1)
        self.assertIn("NOT SAVED", window.lbl_current_image_context.text())
        self.assertIn("NOT SAVED", window.cmb_settings_image.currentText())
        self.assertEqual(window.btn_sobel_ng.text(), "SAVE AS NG")
        with (
            patch.object(window, "_save_sobel_records", side_effect=OSError("disk full")),
            patch.object(ui.QMessageBox, "warning"),
        ):
            window.btn_sobel_good.click()
        self.assertIn("LABEL NG  |  NOT SAVED", window.lbl_current_image_context.text())
        window.slider_gradient_x.setValue(original)
        self.assertEqual(window.btn_sobel_ng.text(), "SAVED AS NG")
        window.cmb_settings_image.setCurrentIndex(1)
        self.assertIn("UNLABELED", window.lbl_sobel_label.text())
        self.assertEqual(window.btn_sobel_ng.text(), "SAVE AS NG")
        window.cmb_settings_image.setCurrentIndex(0)
        self.assertEqual(window.btn_sobel_ng.text(), "SAVED AS NG")

    def test_training_shortcut_does_not_bypass_settings_authentication(self):
        self.window.show()
        self.window.settings_panel.hide()
        with patch.object(self.window, "_toggle_settings") as authenticate:
            self.window._open_image_training()
        authenticate.assert_called_once()
        self.assertFalse(self.window.settings_panel.isVisible())

    def test_model_preprocessing_is_preserved_and_crop_belongs_to_settings(self):
        classifier = self.model(sobel=SobelConfig(blur_ksize=7))
        self.window.classifier = classifier
        self.window._apply_sobel_to_classifier()
        self.assertEqual(classifier.extractor.sobel.blur_ksize, 7)
        self.window.classifier = CutClassifier(crop=CropBox(12, 13, 40, 50))
        self.assertEqual(self.window._settings_crop(), classifier.crop)

    def test_reference_editor_uses_selected_original_without_changing_model_or_gate(self):
        self.model()
        original_config = asdict(self.window.cfg)
        original_gate = self.window.link.state
        classifier = self.window.settings_classifier
        with patch.object(ui, "ReferenceMeasurementDialog") as editor:
            self.window._open_reference_measurement()
            self.assertEqual(editor.call_args.args[0], self.window._current_training_image_path())
            editor.return_value.exec.assert_called_once()
        self.assertEqual(asdict(self.window.cfg), original_config)
        self.assertEqual(self.window.link.state, original_gate)
        self.assertIs(self.window.settings_classifier, classifier)

    def test_reference_editor_is_blocked_during_inspection_or_training(self):
        with patch.object(ui, "ReferenceMeasurementDialog") as editor:
            self.window.auto_running = True
            self.window._open_reference_measurement()
            self.window.auto_running = False
            self.window.training_worker = Mock()
            self.window._open_reference_measurement()
            self.window.training_worker = None
        editor.assert_not_called()

    def test_changed_saved_sobel_or_cleared_label_requires_retraining(self):
        self.model()
        path = self.window.settings_image_paths[0]
        record = self.save_record(path)
        self.assertTrue(self.window._record_is_trained(path))
        self.assertFalse(self.window._settings_model_needs_retraining())
        record["sobel_parameters"] = asdict(SobelConfig(blur_ksize=7))
        self.assertFalse(self.window._record_is_trained(path))
        self.assertTrue(self.window._settings_model_needs_retraining())
        record["sobel_parameters"] = self.window._sobel_parameter_record()
        record["training_label"] = ""
        self.assertTrue(self.window._settings_model_needs_retraining())

    def test_training_excludes_images_saved_with_different_parameters(self):
        self.model()
        path = self.window.settings_image_paths[0]
        record = self.save_record(path)
        self.assertEqual(self.window._training_items(), [(path, "GOOD")])
        record["sobel_parameters"] = asdict(SobelConfig(blur_ksize=7))
        self.assertEqual(self.window._training_items(), [])

    def test_fine_slider_values_are_displayed_and_recorded_without_rounding(self):
        self.window.slider_gradient_x.setValue(1001)
        self.window.slider_gradient_y.setValue(1002)
        self.window.slider_edge_gain.setValue(1003)
        parameters = self.window._sobel_parameter_record()
        self.assertEqual(parameters["gradient_x_weight"], 1.001)
        self.assertEqual(parameters["gradient_y_weight"], 1.002)
        self.assertEqual(parameters["edge_gain"], 1.003)
        self.assertEqual(self.window.lbl_gradient_x_value.text(), "1.001")
        self.assertEqual(self.window.lbl_gradient_y_value.text(), "1.002")
        self.assertTrue(self.window.lbl_edge_gain_value.text().startswith("1.003"))

    def test_fine_parameter_change_excludes_old_saved_image_from_training(self):
        self.model()
        path = self.window.settings_image_paths[0]
        self.save_record(path)
        self.window.slider_gradient_x.setValue(1001)
        self.assertEqual(self.window._training_items(), [])
        self.assertTrue(self.window._settings_model_needs_retraining())

    def test_compact_editors_preserve_precision_and_follow_loaded_values(self):
        editors = dict(self.window.sobel_parameter_editors)
        editors[self.window.slider_gradient_x].setValue(1.001)
        editors[self.window.slider_gradient_y].setValue(1.002)
        editors[self.window.slider_edge_gain].setValue(1.003)
        self.assertEqual(self.window.slider_gradient_x.value(), 1001)
        self.assertEqual(self.window.slider_gradient_y.value(), 1002)
        self.assertEqual(self.window.slider_edge_gain.value(), 1003)
        editors[self.window.slider_sobel_kernel].setCurrentIndex(3)
        self.assertEqual(self.window._sobel_from_controls().sobel_ksize, 7)
        self.window.slider_gradient_x.setValue(1234)
        self.assertEqual(editors[self.window.slider_gradient_x].value(), 1.234)

    def test_sobel_controls_use_one_column_and_actions_follow_requested_order(self):
        window = self.window
        window.settings_panel.show()
        window.image_card.hide()
        window.log_card.hide()
        window.settings_workflow.setCurrentIndex(1)
        window.show()
        for width, height in ((960, 640), (1366, 768)):
            window.resize(width, height)
            for _ in range(8):
                self.app.processEvents()
            panel = window.sobel_parameter_panel
            scroll = window.sobel_parameter_scroll
            self.assertEqual(scroll.horizontalScrollBar().maximum(), 0)
            self.assertEqual(window.sobel_parameter_columns, 1)
            tab = window.settings_workflow.widget(1)
            self.assertTrue(tab.rect().contains(panel.mapTo(tab, panel.rect().bottomRight())))
            for _, editor in window.sobel_parameter_editors:
                scroll.ensureWidgetVisible(editor)
                self.app.processEvents()
                self.assertTrue(editor.isVisible())
                self.assertTrue(
                    panel.rect().contains(editor.mapTo(panel, editor.rect().bottomRight()))
                )
            positions = [
                button.mapTo(tab, button.rect().topLeft()).x()
                for button in (
                    window.btn_previous_image,
                    window.btn_sobel_good,
                    window.btn_sobel_ng,
                    window.btn_next_image,
                )
            ]
            self.assertEqual(positions, sorted(positions))
            footer = window.btn_continue_training.mapTo(
                tab, window.btn_continue_training.rect().topLeft()
            )
            self.assertGreater(footer.y(), panel.mapTo(tab, panel.rect().bottomRight()).y())

    def test_parameter_plus_minus_preserve_step_and_limits(self):
        for slider, increase, decrease in self.window.sobel_parameter_step_buttons:
            slider.setValue(slider.minimum())
            decrease.click()
            self.assertEqual(slider.value(), slider.minimum())
            increase.click()
            self.assertEqual(slider.value(), slider.minimum() + slider.singleStep())
            slider.setValue(slider.maximum())
            increase.click()
            self.assertEqual(slider.value(), slider.maximum())
            decrease.click()
            self.assertEqual(slider.value(), slider.maximum() - slider.singleStep())

    def test_vertical_mouse_wheel_adjusts_numeric_parameter(self):
        from PySide6.QtCore import QPoint, QPointF
        from PySide6.QtGui import QWheelEvent

        slider = self.window.slider_gradient_x
        editor = dict(self.window.sobel_parameter_editors)[slider]
        slider.setValue(1000)
        for delta, expected in ((120, 1001), (-120, 1000)):
            event = QWheelEvent(
                QPointF(10, 10),
                QPointF(10, 10),
                QPoint(),
                QPoint(0, delta),
                Qt.NoButton,
                Qt.NoModifier,
                Qt.NoScrollPhase,
                False,
            )
            self.app.sendEvent(editor, event)
            self.assertEqual(slider.value(), expected)

    def test_sobel_save_reloads_exact_fine_parameters(self):
        self.model()
        self.window.slider_gradient_x.setValue(1001)
        self.window.slider_gradient_y.setValue(1002)
        self.window.slider_edge_gain.setValue(1003)
        self.window._save_current_sobel()
        self.window.slider_gradient_x.setValue(1500)
        self.window.slider_gradient_y.setValue(1500)
        self.window.slider_edge_gain.setValue(1500)
        self.window._load_current_image_parameters()
        self.assertEqual(self.window.slider_gradient_x.value(), 1001)
        self.assertEqual(self.window.slider_gradient_y.value(), 1002)
        self.assertEqual(self.window.slider_edge_gain.value(), 1003)

    def test_label_write_failure_keeps_previous_label_and_training_record(self):
        self.model()
        path = self.window._current_training_image_path()
        previous = deepcopy(self.save_record(path))
        with (
            patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")),
            patch.object(ui.QMessageBox, "warning") as warning,
        ):
            self.window._set_training_label("NG")
        warning.assert_called_once()
        self.assertEqual(self.window.sobel_records[self.window._sobel_record_key(path)], previous)
        self.assertEqual(self.window._training_label_for(path), "GOOD")

    def test_sobel_status_write_failure_preserves_previous_image_and_record(self):
        self.model()
        path = self.window.settings_image_path
        self.window._save_current_sobel()
        key = self.window._sobel_record_key(path)
        previous = deepcopy(self.window.sobel_records[key])
        original_output = Path(previous["sobel_output"])
        previous_bytes = original_output.read_bytes()
        self.window.slider_noise_floor.setValue(200)
        with (
            patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")),
            patch.object(ui.QMessageBox, "warning") as warning,
        ):
            self.window._save_current_sobel()
        warning.assert_called_once()
        self.assertEqual(self.window.sobel_records[key], previous)
        self.assertEqual(original_output.read_bytes(), previous_bytes)
        self.assertFalse(self.window._record_is_saved(path, current_parameters=True))

    def test_sobel_output_supports_unicode_directory(self):
        self.model()
        with (
            patch.object(ui, "SOBEL_OUTPUT_DIR", self.root / "\u0e20\u0e32\u0e1e Sobel"),
            patch.object(ui.QMessageBox, "warning") as warning,
        ):
            self.window._save_current_sobel()
        warning.assert_not_called()
        record = self.window.sobel_records[
            self.window._sobel_record_key(self.window.settings_image_path)
        ]
        decoded = cv2.imdecode(
            np.fromfile(record["sobel_output"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        self.assertIsNotNone(decoded)

    def test_modified_sobel_output_is_no_longer_saved(self):
        self.model()
        path = self.window.settings_image_path
        self.window._save_current_sobel()
        self.assertTrue(self.window._record_is_saved(path))
        record = self.window.sobel_records[self.window._sobel_record_key(path)]
        Path(record["sobel_output"]).write_bytes(b"damaged output")
        self.assertFalse(self.window._record_is_saved(path))

    def test_failed_first_sobel_save_does_not_leave_a_saved_record_or_output(self):
        self.model()
        with (
            patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")),
            patch.object(ui.QMessageBox, "warning"),
        ):
            self.window._save_current_sobel()
        self.assertFalse(self.window._record_is_saved(self.window.settings_image_path))
        self.assertEqual(list(ui.SOBEL_OUTPUT_DIR.iterdir()), [])

    def test_workflow_replace_failure_preserves_file_and_cleans_temporary_file(self):
        self.window._save_sobel_records()
        previous = ui.SOBEL_WORKFLOW_PATH.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("file locked")):
            with self.assertRaises(OSError):
                self.window._save_sobel_records({"new": {"training_label": "NG"}})
        self.assertEqual(ui.SOBEL_WORKFLOW_PATH.read_bytes(), previous)
        self.assertEqual(list(self.root.glob(".workflow.json.*.tmp")), [])

    def test_training_status_write_failure_does_not_mark_images_trained_in_memory(self):
        classifier = self.model()
        path = self.window.settings_image_paths[0]
        self.save_record(path, trained=False)
        previous = deepcopy(self.window.sobel_records)
        self.window.training_model_target = self.window._model_path("PRODUCT-A")
        self.window.training_paths_in_progress = [path]
        self.window.training_input_signatures = {path: self.window._source_signature(path)}
        with patch.object(self.window, "_save_sobel_records", side_effect=OSError("disk full")):
            self.window._settings_training_done(classifier, classifier.report)
        self.assertEqual(self.window.sobel_records, previous)
        self.assertFalse(self.window.training_last_success)
        self.assertIn("training status could not be saved", self.window.training_last_message)

    def test_workflow_cannot_be_written_inside_machine_source(self):
        target = self.source / "workflow.json"
        with patch.object(ui, "SOBEL_WORKFLOW_PATH", target):
            with self.assertRaises(ui.ProtectedPathError):
                self.window._save_sobel_records()
        self.assertFalse(target.exists())

    def test_failed_test_never_enables_save_and_unapproved_settings_do_not_mutate(self):
        old_cfg = asdict(self.window.cfg)
        self.window.settings_trial_approved = True
        self.window._vision_trial_failed("broken input")
        self.window._vision_trial_finished()
        self.assertFalse(self.window.btn_apply_settings.isEnabled())
        self.assertFalse(self.window._apply_settings())
        self.assertEqual(old_cfg, asdict(self.window.cfg))

    def test_test_results_preserve_low_confidence_good_and_render_only_one_image(self):
        self.model()
        paths = self.window.settings_image_paths
        self.window.trial_input_signature = self.window._trial_signature()
        results = [(path, "GOOD", 0.72) for path in paths]
        with patch.object(
            ui, "render_prediction_overview", wraps=ui.render_prediction_overview
        ) as renderer:
            self.window._vision_trial_done(results)
        self.assertEqual(len(renderer.call_args.args[1].details), 1)
        self.assertEqual(self.window.trial_result.details[0].label, "GOOD")
        self.assertIn("BELOW THRESHOLD", self.window.lbl_training_image_context.text())
        self.assertLess(
            self.window.trial_preview_pixmap.width() * self.window.trial_preview_pixmap.height(),
            1_000_000,
        )
        self.window._next_training_image()
        self.assertIn(Path(paths[1]).name, self.window.lbl_training_image_context.text())
        self.window.spin_confidence.setValue(96)
        self.assertFalse(self.window.settings_trial_approved)
        self.assertEqual(self.window.trial_predictions, {})

    def test_quality_failure_keeps_existing_model_untouched(self):
        classifier = self.model()
        target = self.window._model_path("PRODUCT-A")
        previous = target.read_bytes()
        self.window.training_model_target = target
        classifier.report.val_acc = 0.375
        self.window._settings_training_done(classifier, classifier.report)
        self.assertEqual(target.read_bytes(), previous)
        self.assertFalse(self.window.training_last_success)
        self.assertIn("MODEL NOT ACCEPTED", self.window.training_last_message)

    def test_successful_save_connects_auo_source_and_product_model_to_operator_queue(self):
        self.model()
        self.window.trial_input_signature = self.window._trial_signature()
        self.window._vision_trial_done(
            [(p, "GOOD", 0.99) for p in self.window.settings_image_paths]
        )
        self.assertTrue(self.window._apply_settings())
        self.assertEqual(self.window.cmb_recipe.currentText(), "PRODUCT-A")
        self.assertEqual([len(run.pictures) for run in self.window.queue], [2, 2])
        self.assertEqual(self.window.classifier.model_key, "PRODUCT-A")
        self.assertEqual(self.window.recipe_specs["PRODUCT-A"]["expected"], 2)
        self.assertEqual(self.window.link.state, GateState.FAULT)
        self.window._rebuild_queue()
        self.assertEqual(self.window.link.state, GateState.FAULT)
        self.window._acknowledge()
        self.assertEqual(self.window.link.state, GateState.READY)

    def test_source_change_during_job_is_blocked_and_shutdown_defers(self):
        worker = Mock()
        worker.isRunning.return_value = True
        self.window.settings_worker = worker
        old_source = self.window.settings_image_dir
        self.window._set_settings_path(str(self.root))
        self.assertEqual(self.window.settings_image_dir, old_source)
        event = QCloseEvent()
        with patch.object(ui.QTimer, "singleShot"):
            self.window.closeEvent(event)
        self.assertFalse(event.isAccepted())
        worker.requestInterruption.assert_called_once()
        self.window.settings_worker = None

    def test_small_screen_keeps_vision_controls_accessible(self):
        self.model()
        self.window.trial_input_signature = self.window._trial_signature()
        self.window._vision_trial_done(
            [(p, "GOOD", 0.99) for p in self.window.settings_image_paths]
        )
        self.window.settings_panel.show()
        self.window.image_card.hide()
        self.window.log_card.hide()
        self.window.settings_workflow.setCurrentIndex(2)
        self.window.resize(960, 640)
        self.window.show()
        for _ in range(5):
            self.app.processEvents()
        self.assertEqual(self.window.vision_workspace.orientation(), Qt.Vertical)
        self.assertGreaterEqual(self.window.vision_workflow_scroll.viewport().width(), 430)
        self.assertEqual(self.window.vision_workflow_scroll.horizontalScrollBar().maximum(), 0)
        self.window.resize(1366, 768)
        for _ in range(5):
            self.app.processEvents()
        self.assertEqual(self.window.vision_workflow_scroll.horizontalScrollBar().maximum(), 0)


if __name__ == "__main__":
    unittest.main()
