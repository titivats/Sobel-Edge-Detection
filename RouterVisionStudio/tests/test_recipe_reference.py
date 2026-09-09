from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import struct
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QDialog
from router_vision.recipe_reference import (
    project_segment,
    recipe_segments,
    recipe_signature,
    validate_recipe_reference,
)
from router_vision.recipe_reference_ui import RecipeLineDialog
from router_vision.reference_ui import ReferenceMeasurementDialog


def mapping(**changes):
    result = dict(
        center_x_mm=0, center_y_mm=0, scale_x=0.1, scale_y=0.1, image_x_axis_deg=0, y_up=True
    )
    result.update(changes)
    return result


class RecipeProjectionTests(unittest.TestCase):
    def test_camera_translation_anisotropic_scale_and_y_direction(self):
        line = project_segment(
            [[10, 21], [12, 21]], mapping(center_x_mm=10, center_y_mm=20, scale_y=0.2), (241, 321)
        )
        np.testing.assert_allclose(line, [[160, 115], [180, 115]])
        line = project_segment(
            [[10, 21], [12, 21]],
            mapping(center_x_mm=10, center_y_mm=20, scale_y=0.2, y_up=False),
            (241, 321),
        )
        np.testing.assert_allclose(line, [[160, 125], [180, 125]])

    def test_rotated_camera_and_signed_tool_offset(self):
        line = project_segment([[0, -5], [0, 5]], mapping(image_x_axis_deg=90), (241, 321))
        np.testing.assert_allclose(line, [[110, 120], [210, 120]])
        line = project_segment([[-5, 0], [5, 0]], mapping(), (241, 321), offset_mm=0.5)
        np.testing.assert_allclose(line, [[110, 115], [210, 115]])

    def test_clipping_keeps_pose_and_does_not_scale_to_fit(self):
        line = project_segment([[-50, 1], [50, 1]], mapping(), (241, 321))
        np.testing.assert_allclose(line, [[1, 110], [319, 110]])
        with self.assertRaisesRegex(ValueError, "outside"):
            project_segment([[-50, 50], [50, 50]], mapping(), (241, 321))

    def test_invalid_or_missing_scale_and_too_short_segment_rejected(self):
        for value in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                project_segment([[-5, 0], [5, 0]], mapping(scale_x=value), (241, 321))
        with self.assertRaises(ValueError):
            project_segment([[0, 0], [0.1, 0]], mapping(), (241, 321))


class RecipeReferenceUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.recipe = self.root / "example.rcp"
        self.recipe.write_bytes(
            b"recipe header"
            + b"".join(b"\x01" + struct.pack("<fff", x, 0, 0) for x in (-5, 5))
            + b"tail"
        )
        self.image = self.root / "image.png"
        picture = np.zeros((241, 321, 3), np.uint8)
        picture[120:] = 180
        cv2.imencode(".png", picture)[1].tofile(self.image)
        self.record = dict(
            path=str(self.recipe),
            sha256=recipe_signature(self.recipe),
            segment=recipe_segments(self.recipe)[0],
            mapping=mapping(),
            offset_mm=0,
        )

    def test_recipe_identity_rechecked_even_when_file_size_unchanged(self):
        validate_recipe_reference(self.record, (241, 321))
        data = self.recipe.read_bytes()
        self.recipe.write_bytes(data.replace(b"tail", b"edit"))
        with self.assertRaisesRegex(ValueError, "Recipe changed"):
            validate_recipe_reference(self.record, (241, 321))

    def test_dialog_requires_mapping_and_reverification_after_edit(self):
        dialog = RecipeLineDialog((241, 321), self.root)
        self.addCleanup(dialog.deleteLater)
        dialog.load_recipe(self.recipe)
        self.assertTrue(dialog.setup_panel.isHidden())
        self.assertFalse(dialog.use_button.isEnabled())
        self.assertIn("CAMERA SETUP NEEDED", dialog.status.text())
        dialog.use_reference()
        self.assertEqual(dialog.result(), QDialog.Rejected)
        dialog.verified.setChecked(True)
        dialog.use_reference()
        self.assertIn("positive", dialog.status.text())
        for key, value in mapping().items():
            if key != "y_up":
                dialog.fields[key].setValue(value)
        self.assertFalse(dialog.verified.isChecked())
        dialog.verified.setChecked(True)
        self.assertTrue(dialog.use_button.isEnabled())
        dialog.use_reference()
        self.assertEqual(dialog.result(), QDialog.Accepted)
        np.testing.assert_allclose(dialog.line, [[110, 120], [210, 120]])

    def test_recipe_line_is_locked_and_provenance_survives_save_restore(self):
        dialog = ReferenceMeasurementDialog(self.image, self.root / "references.json")
        self.addCleanup(dialog.deleteLater)
        dialog.apply_recipe_reference(self.record)
        line = np.array(dialog.canvas.line)
        dialog.nudge(1)
        np.testing.assert_allclose(dialog.canvas.line, line)
        self.assertTrue(dialog.canvas.line_locked)
        self.assertFalse(dialog.scale_x.isEnabled())
        dialog.save_reference()
        saved = dialog.store.load(self.image)
        self.assertEqual(saved["recipe_reference"], self.record)
        other = ReferenceMeasurementDialog(self.image, self.root / "references.json")
        self.addCleanup(other.deleteLater)
        self.assertTrue(other.canvas.line_locked)
        self.assertFalse(other.aligned.isChecked())
        self.assertFalse(other.calibrated.isChecked())
        other.draw_line()
        self.assertIsNone(other.recipe_reference)
        self.assertFalse(other.canvas.line_locked)

    def test_tampered_saved_projection_is_rejected(self):
        dialog = ReferenceMeasurementDialog(self.image, self.root / "references.json")
        self.addCleanup(dialog.deleteLater)
        dialog.apply_recipe_reference(self.record)
        record = dialog.settings()
        record["line"] = [[110, 121], [210, 121]]
        with self.assertRaisesRegex(ValueError, "no longer matches"):
            dialog.restore(record)
