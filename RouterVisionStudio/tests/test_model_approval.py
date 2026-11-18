from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from router_vision.config import AppConfig
from router_vision.machine_reference import sha256

from tests import test_settings_workflow as workflow


class ModelApprovalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow.SettingsWorkflowTests.setUpClass()

    def setUp(self):
        self.fixture = workflow.SettingsWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window = self.fixture.window
        self.fixture.approve_trial()
        self.assertTrue(self.window._apply_settings())
        self.approved = self.window.active_model_path
        self.original = sha256(self.approved)
        self.candidate = self.window._model_path(self.window.cfg.active_program_key)
        self.config = workflow.ui.CONFIG_PATH

    def reload_production(self):
        self.window.cfg = AppConfig.load(self.config)
        self.window._load_active_model(self.window.cfg.active_recipe)

    def test_retraining_and_restart_keep_last_approved_checkpoint_until_save(self):
        self.assertNotEqual(self.approved, self.candidate)
        before = self.config.read_bytes()
        self.window._invalidate_settings_trial()
        candidate = self.fixture.model()
        self.window.training_model_target = self.candidate
        self.window._settings_training_done(candidate, candidate.report)
        self.assertTrue(self.window.training_last_success)
        self.assertFalse(self.window.settings_trial_approved)
        new_digest = sha256(self.candidate)
        self.assertNotEqual(new_digest, self.original)
        self.assertEqual(self.config.read_bytes(), before)
        self.reload_production()
        self.assertEqual(self.window.classifier.checkpoint_sha256, self.original)

        self.window.trial_input_signature = self.window._trial_signature()
        self.window._vision_trial_done(
            [(path, "GOOD", 0.99) for path in self.window.settings_image_paths]
        )
        self.assertTrue(self.window._apply_settings())
        self.reload_production()
        self.assertEqual(self.window.classifier.checkpoint_sha256, new_digest)
        self.assertEqual(sha256(self.approved), self.original)

    def test_same_size_and_mtime_candidate_replacement_cannot_be_promoted(self):
        self.fixture.approve_trial()
        before = self.config.read_bytes()
        stat = self.candidate.stat()
        payload = bytearray(self.candidate.read_bytes())
        payload[-1] ^= 1
        self.candidate.write_bytes(payload)
        os.utime(self.candidate, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertFalse(self.window._apply_settings())
        self.assertEqual(self.config.read_bytes(), before)
        self.reload_production()
        self.assertEqual(self.window.classifier.checkpoint_sha256, self.original)

    def test_config_write_failure_keeps_original_production_model(self):
        self.fixture.approve_trial()
        with (
            patch.object(AppConfig, "save", side_effect=OSError("config locked")),
            patch.object(workflow.ui.QMessageBox, "critical"),
        ):
            self.assertFalse(self.window._apply_settings())
        self.reload_production()
        self.assertEqual(self.window.classifier.checkpoint_sha256, self.original)

    def test_changed_approved_bytes_and_unpinned_legacy_config_hold(self):
        self.approved.write_bytes(b"changed")
        self.reload_production()
        self.assertIsNone(self.window.classifier)
        self.assertIn("checkpoint changed", self.window.model_error)
        self.window.cfg.active_model_sha256 = ""
        self.window.cfg.save(self.config)
        self.reload_production()
        self.assertIsNone(self.window.classifier)
        self.assertIn("run its model test", self.window.model_error)
        self.window._load_settings_model()
        self.assertEqual(self.window.settings_classifier.checkpoint_sha256, self.original)


if __name__ == "__main__":
    unittest.main()
