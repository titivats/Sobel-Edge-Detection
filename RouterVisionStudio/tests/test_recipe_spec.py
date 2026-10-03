from __future__ import annotations

import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from router_vision.config import AppConfig
from router_vision.recipe_spec_ui import RecipeSpecDialog, validated_limits

from tests import test_settings_workflow as workflow


class RecipeSpecTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow.SettingsWorkflowTests.setUpClass()

    def setUp(self):
        self.fixture = workflow.SettingsWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window = self.fixture.window
        self.window.recipe_specs = {"A": {"recipe": "PROGRAM-A"}, "B": {"recipe": "PROGRAM-B"}}

    def dialog(self, callback=None, recipes=None):
        dialog = RecipeSpecDialog(
            {"A": "PROGRAM-A", "B": "PROGRAM-B"} if recipes is None else recipes,
            self.window.cfg.edge_limits_by_route,
            "A",
            callback or self.window._save_recipe_specs,
            self.window,
        )
        self.addCleanup(dialog.deleteLater)
        return dialog

    def test_save_two_recipes_and_reload_without_changing_active_recipe(self):
        active = self.window.cfg.active_recipe
        dialog = self.dialog()
        dialog.inner.setText("0.05")
        dialog.outer.setText("0.20")
        dialog.recipe.setCurrentIndex(1)
        dialog.inner.setText("0")
        dialog.outer.setText("0.125")
        dialog.recipe.setCurrentIndex(0)
        self.assertEqual(dialog.inner.text(), "0.05")
        dialog.save_button.click()
        self.assertEqual(dialog.result(), workflow.ui.QDialog.DialogCode.Accepted)
        saved = AppConfig.load(workflow.ui.CONFIG_PATH)
        self.assertEqual(
            saved.edge_limits_by_route,
            {
                "A": {"inner_max_mm": 0.05, "outer_max_mm": 0.20},
                "B": {"inner_max_mm": 0.0, "outer_max_mm": 0.125},
            },
        )
        self.assertEqual(saved.active_recipe, active)
        self.assertEqual(saved.last_edge_spec_route, "A")
        reopened = self.dialog()
        self.assertEqual(float(reopened.outer.text()), 0.20)
        reopened.recipe.setCurrentIndex(1)
        self.assertEqual(float(reopened.outer.text()), 0.125)

    def test_cancel_does_not_write(self):
        before = deepcopy(self.window.cfg.edge_limits_by_route)
        dialog = self.dialog()
        dialog.inner.setText("0.2")
        dialog.reject()
        self.assertEqual(self.window.cfg.edge_limits_by_route, before)
        self.assertFalse(workflow.ui.CONFIG_PATH.exists())

    def test_invalid_input_on_other_recipe_blocks_all_saves(self):
        callback = Mock()
        dialog = self.dialog(callback)
        dialog.inner.setText("-0.1")
        dialog.outer.setText("0.2")
        dialog.recipe.setCurrentIndex(1)
        dialog.inner.setText("0.1")
        dialog.outer.setText("0.2")
        dialog.save()
        callback.assert_not_called()
        self.assertIn("SPEC NOT SAVED", dialog.status.text())
        self.assertEqual(dialog.recipe.currentData(), "A")

    def test_failed_disk_write_preserves_memory_and_dialog(self):
        self.window.cfg.edge_limits_by_route = {"B": {"inner_max_mm": 0.3, "outer_max_mm": 0.4}}
        before = deepcopy(self.window.cfg)
        dialog = self.dialog()
        dialog.inner.setText("0.1")
        dialog.outer.setText("0.2")
        with patch.object(AppConfig, "save", side_effect=OSError("disk full")):
            dialog.save()
        self.assertEqual(self.window.cfg, before)
        self.assertNotEqual(dialog.result(), workflow.ui.QDialog.DialogCode.Accepted)
        self.assertIn("disk full", dialog.status.text())

    def test_spec_button_and_save_are_blocked_during_inspection(self):
        self.window.auto_running = True
        self.window._set_busy(False)
        self.assertFalse(self.window.btn_spec.isEnabled())
        with patch.object(workflow.ui, "RecipeSpecDialog") as factory:
            self.window._open_recipe_spec()
        factory.assert_not_called()
        with self.assertRaises(ValueError):
            self.window._save_recipe_specs({"A": {"inner_max_mm": 0.1, "outer_max_mm": 0.2}})
        self.window.auto_running = False

    def test_empty_recipe_list_disables_save(self):
        dialog = self.dialog(recipes={})
        self.assertFalse(dialog.save_button.isEnabled())

    def test_nonfinite_negative_and_missing_values_rejected(self):
        for raw in ("", None, "nan", "inf", "-0.01", True, "text"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                validated_limits(raw, 0.2)

    def test_save_preserves_other_settings_and_unedited_recipe(self):
        self.window.cfg.edge_limits_by_route = {"B": {"inner_max_mm": 0.3, "outer_max_mm": 0.4}}
        before = deepcopy(self.window.cfg)
        self.window._save_recipe_specs({"A": {"inner_max_mm": 0.1, "outer_max_mm": 0.2}})
        expected = deepcopy(before)
        expected.edge_limits_by_route["A"] = {"inner_max_mm": 0.1, "outer_max_mm": 0.2}
        expected.last_edge_spec_route = "A"
        self.assertEqual(self.window.cfg, expected)

    def test_unknown_recipe_rejected(self):
        with self.assertRaises(ValueError):
            self.window._save_recipe_specs({"missing": {"inner_max_mm": 0.1, "outer_max_mm": 0.2}})

    def test_spec_button_shows_latest_saved_recipe_after_reload(self):
        self.window._save_recipe_specs({"A": {"inner_max_mm": 0.05, "outer_max_mm": 0.2}})
        self.window._save_recipe_specs({"B": {"inner_max_mm": 0.1, "outer_max_mm": 0.3}})
        self.window.cfg = AppConfig.load(workflow.ui.CONFIG_PATH)
        self.window._refresh_spec_button()
        self.assertEqual(self.window._latest_spec_route(), "B")
        self.assertIn("SPEC | B | INNER ≤ 0.1 / OUTER ≤ 0.3 mm", self.window.btn_spec.text())
        with patch.object(workflow.ui, "RecipeSpecDialog") as factory:
            self.window._open_recipe_spec()
        self.assertEqual(factory.call_args.args[2], "B")

    def test_existing_single_spec_is_displayed_without_fabricating_latest(self):
        self.window.cfg.edge_limits_by_route = {"B": {"inner_max_mm": 1.0, "outer_max_mm": 1.0}}
        self.window._refresh_spec_button()
        self.assertIn("SPEC | B | INNER ≤ 1 / OUTER ≤ 1 mm", self.window.btn_spec.text())
        self.assertFalse(hasattr(self.window, "btn_training_entry"))


if __name__ == "__main__":
    unittest.main()
