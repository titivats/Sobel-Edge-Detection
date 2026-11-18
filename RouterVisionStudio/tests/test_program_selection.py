"""Recipe selection must isolate images, models, SPEC and the production queue."""

from __future__ import annotations

import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock, patch

from router_vision.config import AppConfig
from router_vision.interlock import GateState
from router_vision.machine import read_run

from tests import test_settings_workflow as workflow


class ProgramSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow.SettingsWorkflowTests.setUpClass()

    def setUp(self):
        self.fixture = workflow.SettingsWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.window = self.fixture.window
        self.source = self.fixture.source
        self.second = self.source / "Result/_20260903_120230.csv"
        self.second.write_text(
            self.second.read_text().replace("PRODUCT-A.rcp", "PROGRAM-B.rcp"), encoding="utf-8"
        )
        for name in ("PRODUCT-A.rcp", "PROGRAM-B.rcp", "NO-RESULT.rcp"):
            (self.source / "Recipe" / name).write_bytes(b"synthetic recipe")
        self.window._set_settings_path(str(self.source))
        self.window.sobel_preview_timer.stop()
        self.a = next(
            p for p in self.window.settings_programs.values() if p.recipe == "PRODUCT-A.rcp"
        )
        self.b = next(
            p for p in self.window.settings_programs.values() if p.recipe == "PROGRAM-B.rcp"
        )

    def select(self, program):
        self.window.cmb_settings_recipe.setCurrentIndex(
            self.window.cmb_settings_recipe.findData(program.key)
        )
        self.window.sobel_preview_timer.stop()

    def test_same_product_different_recipes_have_disjoint_images_and_identities(self):
        self.select(self.a)
        images_a = set(self.window.settings_image_paths)
        self.assertEqual([p.sn for p in self.window.auo_dataset.panels], ["1"])
        self.assertEqual(self.window.lbl_auo_panels.text(), "1")
        self.select(self.b)
        images_b = set(self.window.settings_image_paths)
        self.assertEqual([p.sn for p in self.window.auo_dataset.panels], ["2"])
        self.assertEqual(self.window.lbl_auo_recipe.text(), "PROGRAM-B.rcp")
        self.assertEqual(len(images_a), 2)
        self.assertEqual(len(images_b), 2)
        self.assertFalse(images_a & images_b)
        self.assertNotEqual(self.a.key, self.b.key)
        self.assertEqual(self.window._settings_product_id(), "PRODUCT-A")
        self.assertEqual(self.window._settings_model_key(), self.b.key)
        self.assertEqual([i.cut_point for i in self.window.auo_dataset.images], [1, 2])

    def test_switch_invalidates_trial_and_rejects_another_program_model(self):
        self.select(self.a)
        self.fixture.model()
        self.window.settings_trial_approved = True
        self.window.trial_predictions = {next(iter(self.window.settings_image_paths)): ("GOOD", 1)}
        wrong = self.window._model_path(self.b.key)
        wrong.write_bytes(self.window._model_path(self.a.key).read_bytes())
        self.select(self.b)
        self.assertFalse(self.window.settings_trial_approved)
        self.assertEqual(self.window.trial_predictions, {})
        self.assertIsNone(self.window.settings_classifier)
        self.assertIn("identity", self.window.settings_model_error)
        self.assertFalse(self.window._apply_settings())
        self.select(self.a)
        self.assertEqual(self.window.settings_classifier.model_key, self.a.key)

    def test_spec_follows_selected_recipe_and_dialog_is_locked_to_it(self):
        self.window._save_recipe_specs({self.a.key: {"inner_max_mm": 0.1, "outer_max_mm": 0.2}})
        self.window._save_recipe_specs({self.b.key: {"inner_max_mm": 0.3, "outer_max_mm": 0.4}})
        self.select(self.a)
        self.assertIn("INNER ≤ 0.1 / OUTER ≤ 0.2", self.window.btn_spec.text())
        with patch.object(workflow.ui, "RecipeSpecDialog") as factory:
            self.window._open_recipe_spec()
        self.assertEqual(list(factory.call_args.args[0]), [self.a.key])
        factory.return_value.recipe.setEnabled.assert_called_once_with(False)
        self.select(self.b)
        self.assertIn("INNER ≤ 0.3 / OUTER ≤ 0.4", self.window.btn_spec.text())

    def test_missing_program_spec_never_inherits_latest_or_single_other_spec(self):
        self.window._save_recipe_specs({self.a.key: {"inner_max_mm": 0.1, "outer_max_mm": 0.2}})
        self.select(self.b)
        self.assertEqual(self.window.btn_spec.text(), "SPEC | NOT SET")
        self.assertNotIn(self.b.key, self.window.cfg.edge_limits_by_route)

    def test_apply_saves_exact_program_and_keeps_foreign_expected_counts(self):
        self.select(self.b)
        self.fixture.model()
        self.window.cfg.expected_images_by_route["OTHER"] = 99
        self.window.trial_input_signature = self.window._trial_signature()
        self.window._vision_trial_done(
            [(p, "GOOD", 0.99) for p in self.window.settings_image_paths]
        )
        self.assertTrue(self.window._apply_settings())
        saved = AppConfig.load(workflow.ui.CONFIG_PATH)
        self.assertEqual(saved.active_program_key, self.b.key)
        self.assertEqual(saved.active_program_recipe, "PROGRAM-B.rcp")
        self.assertEqual(saved.expected_images_by_route["OTHER"], 99)
        self.assertEqual([run.sn for run in self.window.queue], ["2"])
        self.assertEqual(self.window.classifier.model_key, self.b.key)

    def test_recipe_without_result_clears_old_preview_and_cannot_use_other_images(self):
        unknown = next(p for p in self.window.settings_programs.values() if not p.product)
        self.fixture.model()
        self.select(unknown)
        self.assertEqual(self.window.settings_image_paths, [])
        self.assertEqual(self.window.auo_dataset.panels, [])
        self.assertIsNone(self.window.settings_classifier)
        self.assertIsNone(self.window.settings_preview_pixmap)
        self.assertFalse(self.window.btn_continue_sobel.isEnabled())
        self.assertFalse(self.window.btn_spec.isEnabled())

    def test_table_is_part_of_program_identity(self):
        first = self.source / "Result/_20260903_120030.csv"
        for path, table in ((first, "Left"), (self.second, "Right")):
            lines = path.read_text().replace("PROGRAM-B.rcp", "PRODUCT-A.rcp").splitlines()
            path.write_text(f"{lines[0]},Table\n{lines[1]},{table}\n", encoding="utf-8")
        self.window._set_settings_path(str(self.source))
        known = [p for p in self.window.settings_programs.values() if p.product]
        self.assertEqual(len(known), 2)
        self.assertNotEqual(known[0].key, known[1].key)
        self.select(next(p for p in known if p.table == "Right"))
        self.assertEqual([p.sn for p in self.window.auo_dataset.panels], ["2"])

    def test_reload_restores_selected_program_without_overwriting_labels_or_specs(self):
        self.select(self.b)
        records = deepcopy(self.window.sobel_records)
        self.window.cmb_settings_recipe.blockSignals(True)
        self.window.cmb_settings_recipe.clear()
        self.window.cmb_settings_recipe.blockSignals(False)
        self.window._set_settings_path(str(self.source))
        self.assertEqual(self.window.cmb_settings_recipe.currentData(), self.b.key)
        self.assertEqual(self.window.sobel_records, records)

    def test_changing_foreign_result_invalidates_old_assignment_and_test(self):
        self.select(self.a)
        self.fixture.model()
        self.window.settings_trial_approved = True
        self.window.trial_input_signature = self.window._trial_signature()
        self.second.write_text(self.second.read_text().replace("True", "False"))
        self.assertFalse(self.window._apply_settings())
        self.assertFalse(self.window.settings_trial_approved)

    def test_program_selection_cannot_change_during_training(self):
        self.select(self.a)
        self.window.training_worker = Mock()
        self.select(self.b)
        self.assertEqual(self.window.cmb_settings_recipe.currentData(), self.a.key)
        self.assertEqual(self.window._settings_model_key(), self.a.key)
        self.window.training_worker = None

    def test_program_overview_cannot_train_merged_recipes(self):
        self.window.cmb_settings_recipe.setCurrentIndex(0)
        self.assertEqual(self.window.lbl_auo_recipe.text(), "NOT SELECTED")
        self.assertEqual(self.window.lbl_auo_cut_points.text(), "UNASSIGNED")
        self.assertEqual(self.window._settings_product_id(), "")
        self.assertIsNone(self.window.settings_classifier)
        self.assertFalse(self.window.btn_spec.isEnabled())
        self.assertFalse(self.window.btn_train_model.isEnabled())
        self.assertFalse(self.window.spin_expected_cuts.isEnabled())
        self.assertTrue(self.window.btn_retry_board.isHidden())
        self.assertEqual(self.window.btn_spec.parent(), self.window.cmb_settings_recipe.parent())

    def test_corrupt_program_binding_is_rejected_before_config_save(self):
        cfg = deepcopy(self.window.cfg)
        cfg.active_recipe = self.b.route
        cfg.active_program_recipe = self.b.recipe
        cfg.active_program_key = self.a.key
        with self.assertRaisesRegex(ValueError, "program identity"):
            cfg.save(self.fixture.root / "invalid.json")

    def test_production_cannot_use_legacy_or_wrong_recipe_binding(self):
        self.window.cmb_recipe.addItem(self.a.route)
        self.window.recipe_specs[self.a.route] = {"expected": 2}
        self.select(self.a)
        self.window.classifier = self.fixture.model()
        run = read_run(self.source / "Result/_20260903_120030.csv", strict=True)
        self.window.link.state = GateState.READY
        with patch.object(workflow.ui, "InspectionWorker") as factory:
            self.window._inspect_next(SimpleNamespace(run=run))
        factory.assert_not_called()
        self.assertIs(self.window.link.state, GateState.FAULT)
        self.assertIn("Choose Recipe", self.window.link.reason)

        self.window.cfg.active_recipe = self.b.route
        self.window.cfg.active_program_key = self.b.key
        self.window.cfg.active_program_recipe = self.b.recipe
        self.window.link.state = GateState.READY
        with patch.object(workflow.ui, "InspectionWorker") as factory:
            self.window._inspect_next(SimpleNamespace(run=run))
        factory.assert_not_called()
        self.assertIs(self.window.link.state, GateState.FAULT)
        self.assertIn("Panel program differs", self.window.link.reason)


if __name__ == "__main__":
    unittest.main()
