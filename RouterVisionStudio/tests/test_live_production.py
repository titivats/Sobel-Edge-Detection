from __future__ import annotations

import json
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from router_vision.capture_records import CAPTURE_SCHEMA
from router_vision.config import AppConfig
from router_vision.live_source import LiveSource
from router_vision.machine import Run
from router_vision.machine_reference import SCHEMA, sha256
from router_vision.production import PanelDecision
from router_vision.production_measurement import CaptureReferences, inspect_with_spec
from router_vision.production_store import ProductionStore, panel_key
from router_vision.training_edge import (
    edge_snapshot,
    make_confirmation,
    measure_image_edge,
    selection_record,
)


class Classifier:
    classes = ["GOOD", "NG"]

    def __init__(self, predictions=None):
        self.predictions = predictions
        self.calls = 0

    def predict(self, paths, cancelled=None):
        self.calls += 1
        return self.predictions or [("GOOD", 0.99)] * len(paths)


class LiveProductionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.machine = self.root / "machine"
        for folder in ("Picture", "Result", "Recipe"):
            (self.machine / folder).mkdir(parents=True)
        self.recipe = self.machine / "Recipe/P.rcp"
        self.recipe.write_bytes(b"version 1")
        self.result_path = self.machine / "Result/_20260903_120030.csv"
        self.result_path.write_text(
            "SN,Recipe_Name,ProductId,Result,CuttingTime\n1,P.rcp,P,True,20\n"
        )
        self.run = Run(
            self.result_path.name,
            "1",
            "run-1",
            "P",
            "",
            "P.rcp",
            True,
            0,
            0,
            0,
            datetime(2026, 9, 3, 12),
            datetime(2026, 9, 3, 12, 0, 30),
            "20",
            "",
            "",
        )
        self.cfg = AppConfig(
            picture_dir=str(self.machine / "Picture"),
            result_dir=str(self.machine / "Result"),
            recipe_dir=str(self.machine / "Recipe"),
            eqp_cfg_path=str(self.machine / "Config/Eqp.cfg"),
        )
        self.store = ProductionStore(self.root / "history.sqlite", self.cfg)
        self.manifest_path = self.root / "captures.json"
        records = []
        self.state = {"choices": {}, "confirmations": {}}
        for index, stamp in enumerate(("120010", "120015"), 1):
            path = self.machine / "Picture" / f"20260903_{stamp}.bmp"
            image = np.zeros((400, 500, 3), dtype=np.uint8)
            image[:125] = 180
            image[275:] = 180
            image[275:280, 200:230] = 0
            image[255:275, 260:290] = 180
            cv2.imwrite(str(path), image)
            self.run.pictures.append(str(path))
            context = {
                "product_id": "P",
                "recipe_name": "P.rcp",
                "recipe_sha256": sha256(self.recipe),
                "table": "LeftTable",
                "program_key": 1,
                "layer": "Cut",
                "cut_point": index,
            }
            records.append(
                {
                    "name": path.name,
                    "sha256": sha256(path),
                    "size_px": [500, 400],
                    "csv_sn": "1",
                    "database_sn": None,
                    "cut_point": index,
                    "acquisition_log_line": None,
                    "mapping_basis": "fixture",
                    "reference_context": context,
                    "projection": {
                        "source_recipe_sha256": sha256(self.recipe),
                        "reconstructed_nc_xy_mm": [[-0.8, 0], [0.8, 0]],
                        "reconstructed_g87_camera_xy_mm": [0, 0],
                        "bit_diameter_mm": 1.3,
                        "upper_A_px": [[170, 135], [330, 135]],
                        "lower_B_px": [[170, 265], [330, 265]],
                    },
                }
            )
        self.data = {
            "schema": SCHEMA,
            "source_files": [{"path": str(self.recipe), "sha256": sha256(self.recipe)}],
            "scale_mm_per_px_xy": [0.01, 0.01],
            "images": records,
        }
        self.save_manifest()
        self.refs = CaptureReferences(self.manifest_path)
        for index, path in enumerate(self.run.pictures, 1):
            review = self.refs.review(path, self.run, self.recipe, index)
            summary, profile = measure_image_edge(cv2.imread(path), review, "B")
            snapshot = edge_snapshot(review, summary, profile, {})
            self.state["choices"][review["key"]] = selection_record(review, path, "B")
            self.state["confirmations"][str(index)] = make_confirmation(snapshot)
        self.limits = {"inner_max_mm": 1.0, "outer_max_mm": 1.0}

    def save_manifest(self):
        self.manifest_path.write_text(json.dumps(self.data), encoding="utf-8")

    def inspect(self, classifier=None, limits=None, state=None, refs=None):
        return inspect_with_spec(
            classifier or Classifier(),
            self.run,
            2,
            0.95,
            self.limits if limits is None else limits,
            self.refs if refs is None else refs,
            self.recipe,
            self.state if state is None else state,
        )

    def test_model_then_actual_sobel_then_spec(self):
        result, evidence = self.inspect()
        self.assertEqual(result.status, "GOOD")
        self.assertEqual(evidence["model_stage"], "GOOD")
        self.assertEqual(len(evidence["measurements"]), 2)
        self.assertGreater(evidence["measurements"][0]["measurement"]["outer_line_max_mm"], 0.15)

    def test_spec_exceeded_is_ng(self):
        result, _ = self.inspect(limits={"inner_max_mm": 1.0, "outer_max_mm": 0.1})
        self.assertEqual(result.status, "NG")

    def test_values_equal_to_spec_are_accepted(self):
        _, evidence = self.inspect()
        m = evidence["measurements"][0]["measurement"]
        result, _ = self.inspect(
            limits={"inner_max_mm": m["inner_line_max_mm"], "outer_max_mm": m["outer_line_max_mm"]}
        )
        self.assertEqual(result.status, "GOOD")

    def test_ng_and_uncertain_skip_measurement(self):
        for predictions, status in (([("NG", 0.9)] * 2, "NG"), ([("GOOD", 0.7)] * 2, "FAULT")):
            refs = Mock()
            result, evidence = self.inspect(classifier=Classifier(predictions), refs=refs)
            self.assertEqual(result.status, status)
            self.assertEqual(evidence["measurement_stage"], "NOT_RUN")
            refs.review.assert_not_called()

    def test_missing_spec_fails_closed(self):
        result, _ = self.inspect(limits={})
        self.assertEqual(result.status, "FAULT")

    def test_unsaved_side_cannot_release(self):
        result, _ = self.inspect(state={"choices": {}, "confirmations": {}})
        self.assertEqual(result.status, "FAULT")

    def test_partial_measurement_cannot_release(self):
        for record in self.data["images"]:
            p = record["projection"]
            p["reconstructed_nc_xy_mm"] = [[-1.9, 0], [1.9, 0]]
            p["upper_A_px"] = [[60, 135], [440, 135]]
            p["lower_B_px"] = [[60, 265], [440, 265]]
        self.save_manifest()
        result, evidence = self.inspect()
        self.assertEqual(result.status, "FAULT")
        self.assertTrue(evidence["measurements"])

    def test_changed_image_during_model_cannot_release(self):
        classifier = Classifier()
        original = classifier.predict

        def changed(paths, cancelled=None):
            predictions = original(paths, cancelled)
            Path(paths[0]).write_bytes(b"changed")
            return predictions

        classifier.predict = changed
        result, _ = self.inspect(classifier=classifier)
        self.assertEqual(result.status, "FAULT")

    def test_capture_replaced_while_loading_cannot_release(self):
        from router_vision import capture_index as measurement

        real_load = measurement.load_edge_review

        def replace_after_load(*args, **kwargs):
            review = real_load(*args, **kwargs)
            self.data["images"][0]["mapping_basis"] = "changed during loading"
            self.save_manifest()
            return review

        with patch.object(measurement, "load_edge_review", side_effect=replace_after_load):
            result, evidence = self.inspect()
        self.assertEqual(result.status, "FAULT")
        self.assertIn("changed", evidence["measurements"][0]["reason"])

    def test_history_write_failure_cannot_publish_pass(self):
        from tests import test_settings_workflow as workflow

        workflow.SettingsWorkflowTests.setUpClass()
        window = SimpleNamespace(
            current_result=None,
            active_claim="pending-board",
            worker=SimpleNamespace(evidence={}),
            cfg=self.cfg,
            production_store=Mock(),
            link=Mock(),
            _software_hold=Mock(),
            _show_result_image=Mock(),
            _append_log=Mock(),
        )
        window.link.state = workflow.ui.GateState.INSPECTING
        window.link.active_panel = workflow.ui.panel_id(self.run)
        window.production_store.finish.side_effect = OSError("disk full")
        result = PanelDecision(run=self.run, status="GOOD")
        with patch.object(workflow.ui, "CONFIG_PATH", self.root / "config.json"):
            workflow.ui.MainWindow._inspection_done(
                window, result, np.zeros((40, 40, 3), dtype=np.uint8), 5.0
            )
        self.assertEqual(result.status, "FAULT")
        self.assertIn("disk full", result.note)
        window._software_hold.assert_called_once()
        window.link.publish.assert_not_called()
        self.assertEqual(window.active_claim, "pending-board")

    def test_persistence_claim_prevents_duplicate_after_restart(self):
        key = panel_key(self.cfg.result_dir, self.run)
        self.store.claim(key, self.run.key, "sig")
        result, evidence = self.inspect()
        self.store.finish(key, result, evidence)
        reopened = ProductionStore(self.store.path, self.cfg)
        self.assertEqual(reopened.state(key), ("DONE", "sig"))
        with self.assertRaises(sqlite3.IntegrityError):
            reopened.claim(key, self.run.key, "sig")
        self.assertEqual(reopened.recent()[0][0]["result"]["status"], "GOOD")
        self.assertEqual(reopened.recent()[0][0]["evidence"]["spec"], self.limits)

    def test_live_waits_for_stable_files_then_picks_new_board(self):
        clock = [0.0]
        source = LiveSource(self.cfg, self.store, clock=lambda: clock[0])
        self.assertEqual(source.poll("P", 2), [])
        clock[0] = 3.0
        panels = source.poll("P", 2)
        self.assertEqual(len(panels), 1)
        self.assertEqual(panels[0].run.sn, "1")
        self.store.claim(panels[0].key, "P", panels[0].signature)
        self.store.finish(panels[0].key, PanelDecision(run=panels[0].run), {})
        self.assertEqual(source.poll("P", 2), [])
        new_result = self.machine / "Result/_20260903_120230.csv"
        new_result.write_text("SN,Recipe_Name,ProductId,Result,CuttingTime\n2,P.rcp,P,True,20\n")
        for stamp in ("120210", "120215"):
            cv2.imwrite(
                str(self.machine / "Picture" / f"20260903_{stamp}.bmp"),
                np.zeros((400, 500, 3), np.uint8),
            )
        self.assertEqual(source.poll("P", 2), [])
        clock[0] = 6.0
        panels = source.poll("P", 2)
        self.assertEqual([p.run.sn for p in panels], ["2"])

    def test_changed_previously_inspected_panel_requires_review(self):
        clock = [0.0]
        source = LiveSource(self.cfg, self.store, clock=lambda: clock[0])
        source.poll("P", 2)
        clock[0] = 3.0
        panel = source.poll("P", 2)[0]
        self.store.claim(panel.key, "P", panel.signature)
        self.store.finish(panel.key, PanelDecision(run=panel.run), {})
        self.result_path.write_text(self.result_path.read_text().replace("True", "False"))
        source.poll("P", 2)
        clock[0] = 6.0
        with self.assertRaisesRegex(ValueError, "source files changed"):
            source.poll("P", 2)

    def test_incomplete_board_waits_then_reports_fault_without_guessing(self):
        Path(self.run.pictures[1]).unlink()
        clock = [0.0]
        source = LiveSource(self.cfg, self.store, clock=lambda: clock[0])
        source.poll("P", 2)
        clock[0] = 3.0
        self.assertEqual(source.poll("P", 2), [])
        clock[0] = 34.0
        panels = source.poll("P", 2)
        self.assertEqual(len(panels), 1)
        self.assertIn("Expected 2", panels[0].readiness_error)

    def test_pending_claim_is_not_replayed_automatically(self):
        clock = [0.0]
        source = LiveSource(self.cfg, self.store, clock=lambda: clock[0])
        source.poll("P", 2)
        clock[0] = 3.0
        panel = source.poll("P", 2)[0]
        self.store.claim(panel.key, "P", panel.signature)
        with self.assertRaisesRegex(ValueError, "unfinished inspection"):
            source.poll("P", 2)

    def test_retry_keeps_previous_result_and_rejects_previous_owner(self):
        key = panel_key(self.cfg.result_dir, self.run)
        self.store.claim(key, "P", "sig")
        other = ProductionStore(self.store.path, self.cfg)
        other.retry_held("P")
        other.claim(key, "P", "sig")
        with self.assertRaises(ValueError):
            self.store.finish(key, PanelDecision(run=self.run), {})
        result, evidence = self.inspect()
        other.finish(key, result, evidence)
        self.assertEqual(other.state(key)[0], "DONE")
        with other.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM attempts").fetchone()[0], 2)

    def recorded_captures(self):
        self.result_path.write_text(
            "SN,Recipe_Name,ProductId,Result,CuttingTime,OffsetX,OffsetY,BitDiameter\n1,P.rcp,P,True,20,0,0,1.3\n"
        )
        self.data["schema"] = CAPTURE_SCHEMA
        self.data["source_files"].append(
            {"path": str(self.result_path), "sha256": sha256(self.result_path)}
        )
        for record in self.data["images"]:
            p = record.pop("projection")
            record.update(
                result_path=str(self.result_path),
                result_file=self.result_path.name,
                motion={
                    "start_end_xy_mm": p["reconstructed_nc_xy_mm"],
                    "camera_xy_mm": p["reconstructed_g87_camera_xy_mm"],
                    "diameter_mm": p["bit_diameter_mm"],
                    "offset_xy_mm": [0, 0],
                    "rotation_rad": 0.0,
                    "alignment_applied": True,
                },
            )

    def test_recorded_geometry_supports_all_four_directions(self):
        self.recorded_captures()
        for vertical in (False, True):
            if vertical:
                for path, record in zip(self.run.pictures, self.data["images"]):
                    cv2.imwrite(path, cv2.transpose(cv2.imread(path)))
                    record["size_px"] = [400, 500]
                    record["sha256"] = sha256(path)
            for reverse in (False, True):
                points = [[0, -0.8], [0, 0.8]] if vertical else [[-0.8, 0], [0.8, 0]]
                if reverse:
                    points.reverse()
                for record in self.data["images"]:
                    record["motion"]["start_end_xy_mm"] = points
                self.save_manifest()
                result, evidence = self.inspect()
                self.assertEqual(result.status, "GOOD", result.note)
                self.assertEqual(
                    evidence["measurements"][0]["measurement"]["orientation"],
                    "vertical" if vertical else "horizontal",
                )

    def test_recorded_capture_rejects_missing_and_mismatching_offset(self):
        self.recorded_captures()
        self.data["images"][0]["motion"]["offset_xy_mm"] = [0.1, 0]
        self.save_manifest()
        result, evidence = self.inspect()
        self.assertEqual(result.status, "FAULT")
        self.assertIn("OffsetX", evidence["measurements"][0]["reason"])
        del self.data["images"][0]["motion"]["offset_xy_mm"]
        self.save_manifest()
        result, _ = self.inspect()
        self.assertEqual(result.status, "FAULT")

    def test_live_waits_for_capture_data_before_starting_board(self):
        clock = [0.0]
        source = LiveSource(
            self.cfg, self.store, clock=lambda: clock[0], capture_location=self.root / "incoming"
        )
        source.poll("P", 2)
        clock[0] = 3.0
        self.assertEqual(source.poll("P", 2), [])
        (self.root / "incoming").mkdir()
        (self.root / "incoming/capture.json").write_text(self.manifest_path.read_text())
        self.assertEqual(source.poll("P", 2), [])
        clock[0] = 6.0
        panels = source.poll("P", 2)
        self.assertEqual(len(panels), 1)
        self.assertFalse(panels[0].readiness_error)

    def test_new_board_reuses_only_side_and_measures_new_pixels(self):
        _, before = self.inspect()
        for record, path in zip(self.data["images"], self.run.pictures):
            image = cv2.imread(path)
            image[245:255, 260:290] = 180
            cv2.imwrite(path, image)
            record["sha256"] = sha256(path)
            record["csv_sn"] = "2"
            record["projection"]["reconstructed_nc_xy_mm"] = [[-0.8, 0.02], [0.8, 0.02]]
            record["projection"]["upper_A_px"] = [[170, 137], [330, 137]]
            record["projection"]["lower_B_px"] = [[170, 267], [330, 267]]
        self.run.sn = "2"
        self.save_manifest()
        result, after = self.inspect()
        self.assertEqual(result.status, "GOOD")
        self.assertGreater(
            after["measurements"][0]["measurement"]["outer_line_max_mm"],
            before["measurements"][0]["measurement"]["outer_line_max_mm"],
        )

    def test_ui_live_auto_receives_new_boards_and_restores_history(self):
        from PySide6.QtTest import QTest
        from router_vision.model import CropBox, SobelConfig

        from tests import test_settings_workflow as workflow

        workflow.SettingsWorkflowTests.setUpClass()
        self.cfg.expected_images_by_route = {"P": 2}
        self.cfg.edge_limits_by_route = {"P": self.limits}
        self.cfg.capture_manifest_dir = str(self.manifest_path)
        self.cfg.live_file_stable_seconds = 0
        self.cfg.live_poll_seconds = 0.5
        self.cfg.save(self.root / "config.json")
        patches = [
            patch.object(workflow.ui, "CONFIG_PATH", self.root / "config.json"),
            patch.object(workflow.ui, "SOBEL_WORKFLOW_PATH", self.root / "workflow.json"),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        window = workflow.ui.MainWindow()
        self.addCleanup(window.deleteLater)
        self.addCleanup(window.close)
        classifier = Classifier()
        classifier.crop = CropBox(0, 0, 500, 400)
        classifier.extractor = SimpleNamespace(
            sobel=SobelConfig(), edge_map=lambda p, c: cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        )
        window.classifier = classifier
        model = self.root / "model.pt"
        model.write_bytes(b"test model identity")
        window.active_model_path = model
        window.active_model_sha256 = sha256(model)
        window.edge_review = self.state
        window._acknowledge()
        window._toggle_auto()

        def wait_done(count):
            for _ in range(150):
                QTest.qWait(50)
                if len(window.production_store.recent()) == count and window.worker is None:
                    return
            self.fail(f"Live UI did not persist {count} boards: {window.link.reason}")

        wait_done(1)
        self.assertTrue(window.auto_running)
        for _ in range(80):
            QTest.qWait(50)
            time.sleep(0.005)
            if "WAITING" in window.lbl_state_big.text():
                break
        self.assertEqual(classifier.calls, 1)
        self.assertIn("WAITING", window.lbl_state_big.text())
        new_result = self.machine / "Result/_20260903_120230.csv"
        new_result.write_text("SN,Recipe_Name,ProductId,Result,CuttingTime\n2,P.rcp,P,True,20\n")
        for old, stamp in zip(list(self.data["images"]), ("120210", "120215")):
            record = json.loads(json.dumps(old))
            picture = self.machine / "Picture" / f"20260903_{stamp}.bmp"
            cv2.imwrite(str(picture), cv2.imread(str(self.machine / "Picture" / old["name"])))
            record.update(name=picture.name, sha256=sha256(picture), csv_sn="2")
            self.data["images"].append(record)
        self.save_manifest()
        wait_done(2)
        self.assertEqual(classifier.calls, 2)
        self.assertEqual(
            [r[0]["result"]["status"] for r in window.production_store.recent()], ["GOOD", "GOOD"]
        )
        window._software_hold()
        for _ in range(40):
            QTest.qWait(25)
            if window.live_scan_worker is None:
                break
        window._save_recipe_specs({"P": {"inner_max_mm": 1.0, "outer_max_mm": 0.1}})
        new_result = self.machine / "Result/_20260903_120430.csv"
        new_result.write_text("SN,Recipe_Name,ProductId,Result,CuttingTime\n3,P.rcp,P,True,20\n")
        for old, stamp in zip(list(self.data["images"][:2]), ("120410", "120415")):
            record = json.loads(json.dumps(old))
            picture = self.machine / "Picture" / f"20260903_{stamp}.bmp"
            cv2.imwrite(str(picture), cv2.imread(str(self.machine / "Picture" / old["name"])))
            record.update(name=picture.name, sha256=sha256(picture), csv_sn="3")
            self.data["images"].append(record)
        self.save_manifest()
        window._acknowledge()
        window._toggle_auto()
        wait_done(3)
        self.assertFalse(window.auto_running)
        latest = window.production_store.recent()[0][0]
        self.assertEqual(latest["result"]["status"], "NG")
        self.assertEqual(latest["evidence"]["model_stage"], "GOOD")
        self.assertEqual(latest["evidence"]["measurement_stage"], "NG")
        window.close()
        reopened = workflow.ui.MainWindow()
        self.addCleanup(reopened.deleteLater)
        self.addCleanup(reopened.close)
        self.assertEqual(reopened.table.rowCount(), 3)


if __name__ == "__main__":
    unittest.main()
