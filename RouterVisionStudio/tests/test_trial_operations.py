"""Operator actions for commissioning a bound program and recovering a held board."""

import json
import unittest
from unittest.mock import patch

from tests import test_settings_workflow as workflow


class TrialOperationsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow.SettingsWorkflowTests.setUpClass()

    def setUp(self):
        self.fx = workflow.SettingsWorkflowTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.window = self.fx.window

    def test_source_image_count_never_authorizes_cut_count(self):
        self.fx.approve_trial()
        self.assertTrue(self.window._apply_settings())
        program = self.window.cfg.active_program_key
        for stamp in ("120015", "120215"):
            (self.fx.source / "Picture" / f"20260903_{stamp}.bmp").unlink()
        self.window._set_settings_path(str(self.fx.source))
        self.assertEqual(self.window.spin_expected_cuts.value(), 2)
        self.assertEqual(self.window.auo_dataset.expected_cut_points, 1)
        self.fx.approve_trial()
        self.assertFalse(self.window._apply_settings())
        self.assertIn("CUT count", self.window.lbl_trial_result.text())
        config = json.loads(workflow.ui.CONFIG_PATH.read_text())
        self.assertEqual(config["expected_images_by_route"][program], 2)

    def test_clean_result_folder_can_arm_only_with_configured_captures(self):
        self.fx.approve_trial()
        self.assertTrue(self.window._apply_settings())
        for path in (self.fx.source / "Result").glob("*.csv"):
            path.unlink()
        self.assertTrue(self.window._load_dataset())
        self.assertEqual(self.window.cmb_recipe.currentText(), self.window.cfg.active_recipe)
        self.assertIsNotNone(self.window.classifier)
        self.window._acknowledge()
        self.window._toggle_auto()
        self.assertFalse(self.window.auto_running)
        self.assertIn("CAPTURE JSON FOLDER", self.window.link.reason)
        capture_dir = self.fx.root / "captures"
        capture_dir.mkdir()
        with patch.object(
            workflow.ui.QFileDialog, "getExistingDirectory", return_value=str(capture_dir)
        ):
            self.window.btn_capture_source.click()
        self.assertEqual(self.window.capture_path_label.text(), str(capture_dir))
        self.window._acknowledge()
        with patch.object(workflow.ui.LiveScanWorker, "start") as start:
            self.window._toggle_auto()
        start.assert_called_once()
        self.assertTrue(self.window.auto_running)
        self.assertEqual(self.window.live_scan_worker.expected, 2)
        self.window.live_scan_worker = None

    def test_authenticated_settings_retry_preserves_interrupted_attempt(self):
        self.fx.approve_trial()
        self.assertTrue(self.window._apply_settings())
        route = self.window.cfg.active_recipe
        store = self.window.production_store
        store.claim("interrupted", route, "fixture")
        self.window.show()
        with (
            patch.object(workflow.ui, "SETTINGS_PASSWORD", "fixture"),
            patch.object(workflow.ui.QInputDialog, "getText", return_value=("fixture", True)),
        ):
            self.window.btn_settings.click()
        self.assertTrue(self.window.btn_retry_board.isVisible())
        self.window.btn_retry_board.click()
        self.assertEqual(store.state("interrupted")[0], "RETRY_APPROVED")
        self.assertTrue(self.window.btn_retry_board.isHidden())
        with store.connect() as db:
            self.assertEqual(db.execute("SELECT status FROM attempts").fetchall(), [("FAULT",)])
        self.assertFalse(self.window.auto_running)
        self.assertFalse(self.window.link.signals.release)
