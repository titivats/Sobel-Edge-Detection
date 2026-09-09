from __future__ import annotations

import json
import math
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMessageBox
from router_vision.guard import ProtectedPathError
from router_vision.reference import (
    ReferenceStore,
    image_signature,
    measure_reference,
    normal_scale,
)
from router_vision.reference_ui import ReferenceMeasurementDialog


def board_image():
    image = np.zeros((240, 320, 3), np.uint8)
    image[120:, :] = 180
    return image


class ReferenceMeasureTests(unittest.TestCase):
    def test_straight_nominal_edge_is_zero(self):
        result = measure_reference(board_image(), [[30, 119.5], [285, 119.5]])
        self.assertTrue(result.valid, result.reason)
        self.assertAlmostEqual(result.inward_px, 0, delta=0.05)
        self.assertAlmostEqual(result.protrusion_px, 0, delta=0.05)

    def test_inward_and_protrusion_extrema_are_not_averaged_away(self):
        image = board_image()
        image[120:125, 100:150] = 0
        image[113:120, 190:240] = 180
        result = measure_reference(image, [[30, 119.5], [285, 119.5]])
        self.assertAlmostEqual(result.inward_px, 5, delta=0.1)
        self.assertAlmostEqual(result.protrusion_px, 7, delta=0.1)
        self.assertGreater(result.coverage, 0.95)

    def test_reversing_line_and_side_preserves_material_sign(self):
        first = measure_reference(board_image(), [[30, 114.5], [285, 114.5]])
        reverse = measure_reference(board_image(), [[285, 114.5], [30, 114.5]], side=-1)
        self.assertTrue(first.valid)
        self.assertTrue(reverse.valid)
        self.assertAlmostEqual(first.inward_px, 5, delta=0.1)
        self.assertAlmostEqual(first.inward_px, reverse.inward_px, delta=0.01)

    def test_wrong_material_side_does_not_pass(self):
        result = measure_reference(board_image(), [[30, 119.5], [285, 119.5]], side=-1)
        self.assertFalse(result.valid)
        self.assertEqual(result.coverage, 0)

    def test_dark_material_polarity(self):
        result = measure_reference(255 - board_image(), [[30, 119.5], [285, 119.5]], polarity=-1)
        self.assertTrue(result.valid)

    def test_blank_clipped_and_ambiguous_edges_are_invalid(self):
        self.assertFalse(
            measure_reference(np.zeros((240, 320, 3), np.uint8), [[30, 119.5], [285, 119.5]]).valid
        )
        self.assertFalse(measure_reference(board_image(), [[30, 5], [285, 5]]).valid)
        self.assertFalse(measure_reference(board_image(), [[30, 97], [285, 97]], radius=24).valid)
        image = board_image()
        image[112:120] = 180
        image[118:124] = 0
        self.assertFalse(measure_reference(image, [[30, 119.5], [285, 119.5]]).valid)

    def test_invalid_geometry_is_refused(self):
        for line in ([[1, 1], [2, 2]], [[-1, 120], [100, 120]], [[30, float("nan")], [100, 120]]):
            self.assertFalse(measure_reference(board_image(), line).valid)

    def test_diagonal_edge_is_measured_along_its_normal(self):
        yy, xx = np.indices((240, 320))
        image = np.where(yy >= 0.25 * xx + 70, 180, 0).astype(np.uint8)
        result = measure_reference(image, [[40, 79.5], [260, 134.5]])
        self.assertTrue(result.valid)
        self.assertLess(max(result.inward_px, result.protrusion_px), 0.7)

    def test_anisotropic_mm_conversion_is_perpendicular_to_physical_line(self):
        self.assertAlmostEqual(normal_scale([0, 1], 0.01, 0.02), 0.02)
        n = [1 / math.sqrt(2), 1 / math.sqrt(2)]
        self.assertAlmostEqual(normal_scale(n, 0.01, 0.02), 1 / math.sqrt(5000 + 1250))
        for scale in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                normal_scale([0, 1], scale, 0.02)


class ReferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = self.root / "image.bmp"
        self.image.write_bytes(b"original")
        self.store = ReferenceStore(self.root / "references.json")

    def test_reference_round_trip_is_per_image_and_source_bound(self):
        settings = {"line": [[30, 119.5], [285, 119.5]]}
        self.store.save(self.image, settings, image_signature(self.image))
        self.assertEqual(self.store.load(self.image), settings)
        other = self.root / "other.bmp"
        other.write_bytes(b"other")
        self.assertIsNone(self.store.load(other))
        self.image.write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.store.load(self.image)

    def test_replace_failure_does_not_overwrite_or_leave_temporary_file(self):
        self.store.save(self.image, {"old": True}, image_signature(self.image))
        old = self.store.path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("locked")):
            with self.assertRaises(OSError):
                self.store.save(self.image, {"new": True}, image_signature(self.image))
        self.assertEqual(self.store.path.read_bytes(), old)
        self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_corrupt_store_and_machine_paths_are_not_overwritten(self):
        self.store.path.write_text("broken", encoding="utf-8")
        with self.assertRaises(ValueError):
            self.store.save(self.image, {}, image_signature(self.image))
        self.assertEqual(self.store.path.read_text(), "broken")
        self.store.protected = [self.root]
        with self.assertRaises(ProtectedPathError):
            self.store.save(self.image, {}, image_signature(self.image))

    def test_source_change_during_edit_refuses_save(self):
        signature = image_signature(self.image)
        self.image.write_bytes(b"replacement image")
        with self.assertRaises(ValueError):
            self.store.save(self.image, {}, signature)
        self.assertFalse(self.store.path.exists())


class ReferenceUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        font = Path("C:/Windows/Fonts/segoeui.ttf")
        if font.exists():
            QFontDatabase.addApplicationFont(str(font))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.path = root / "image.bmp"
        cv2.imencode(".bmp", board_image())[1].tofile(self.path)
        self.dialog = ReferenceMeasurementDialog(self.path, root / "reference.json")
        self.addCleanup(self.dialog.deleteLater)
        self.addCleanup(self.close_dialog)

    def close_dialog(self):
        self.dialog.dirty = False
        self.dialog.reject()

    def set_line(self):
        self.dialog.canvas.line = [[30, 119.5], [285, 119.5]]
        self.dialog.line_edited()
        self.dialog.measure()

    def test_pixels_are_default_and_scale_requires_explicit_verification(self):
        self.set_line()
        self.assertIn("ALIGNMENT NOT CONFIRMED", self.dialog.result.text())
        self.dialog.aligned.setChecked(True)
        self.assertIn("WITHIN TRIAL LIMITS", self.dialog.result.text())
        self.assertEqual(self.dialog.inward.suffix(), " px")
        self.dialog.scale_x.setValue(0.01)
        self.dialog.scale_y.setValue(0.02)
        self.assertFalse(self.dialog.calibrated.isChecked())
        self.dialog.calibrated.setChecked(True)
        self.assertEqual(self.dialog.inward.suffix(), " mm")
        self.assertAlmostEqual(self.dialog.inward.value(), 0.04)
        self.dialog.scale_x.setValue(0.02)
        self.assertFalse(self.dialog.calibrated.isChecked())
        self.assertEqual(self.dialog.inward.suffix(), " px")

    def test_drawing_mouse_gesture_creates_original_pixel_line(self):
        self.dialog.show()
        self.app.processEvents()
        self.dialog.canvas.fit_image()
        self.dialog.draw_line()
        a = self.dialog.canvas.mapFromScene(30, 119)
        b = self.dialog.canvas.mapFromScene(285, 119)
        viewport = self.dialog.canvas.viewport()
        QTest.mousePress(viewport, Qt.LeftButton, pos=a)
        QTest.mouseMove(viewport, b)
        QTest.mouseRelease(viewport, Qt.LeftButton, pos=b)
        self.dialog.measure()
        self.assertAlmostEqual(self.dialog.canvas.line[0][0], 30, delta=1)
        self.assertAlmostEqual(self.dialog.canvas.line[1][0], 285, delta=1)
        self.assertTrue(self.dialog.dirty)

    def test_save_does_not_persist_alignment_or_verified_scale_claims(self):
        self.set_line()
        self.dialog.aligned.setChecked(True)
        self.dialog.save_reference()
        raw = json.loads(self.dialog.store.path.read_text())
        record = next(iter(raw["images"].values()))["settings"]
        self.assertNotIn("aligned", record)
        self.assertNotIn("calibrated", record)
        self.assertFalse(self.dialog.dirty)

    def test_escape_and_close_respect_unsaved_changes(self):
        self.dialog.show()
        self.set_line()
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Cancel):
            self.dialog.reject()
            self.assertTrue(self.dialog.isVisible())
            self.dialog.close()
            self.assertTrue(self.dialog.isVisible())
        with patch.object(QMessageBox, "question", return_value=QMessageBox.Discard):
            self.dialog.close()
        self.assertFalse(self.dialog.isVisible())

    def test_measurement_and_actions_remain_visible_on_small_screen(self):
        self.dialog.resize(960, 640)
        self.set_line()
        self.dialog.show()
        for _ in range(3):
            self.app.processEvents()
        self.assertTrue(self.dialog.result.isVisible())
        self.assertTrue(self.dialog.save_button.isVisible())
        self.assertGreaterEqual(self.dialog.result.height(), 90)
        self.assertLessEqual(
            self.dialog.save_button.mapTo(
                self.dialog, QPoint(0, self.dialog.save_button.height())
            ).y(),
            self.dialog.height(),
        )


if __name__ == "__main__":
    unittest.main()
