from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
from router_vision.capture_index import CaptureReferences
from router_vision.capture_records import CAPTURE_SCHEMA
from router_vision.config import AppConfig, read_pixel_size
from router_vision.guard import ProtectedPathError, check_write_target, protected_roots
from router_vision.images import read_image
from router_vision.interlock import GateState
from router_vision.live_source import LiveSource
from router_vision.machine import read_run
from router_vision.machine_reference import sha256
from router_vision.production import PanelDecision
from router_vision.production_store import panel_key

from tests import test_live_production as live


class ProductionResilienceTests(unittest.TestCase):
    def setUp(self):
        self.fx = live.LiveProductionTests()
        self.fx.setUp()
        self.addCleanup(self.fx.doCleanups)
        self.clock = [0.0]

    def source(self, **kwargs):
        return LiveSource(
            self.fx.cfg,
            self.fx.store,
            clock=lambda: self.clock[0],
            capture_location=self.fx.manifest_path,
            **kwargs,
        )

    def stable(self, source):
        source.poll("P", 2)
        self.clock[0] = 3.0
        return source.poll("P", 2)

    def window(self):
        from tests import test_settings_workflow as workflow

        fx = workflow.SettingsWorkflowTests()
        workflow.SettingsWorkflowTests.setUpClass()
        fx.setUp()
        self.addCleanup(fx.doCleanups)
        window = fx.window
        window.cfg, window.production_store = self.fx.cfg, self.fx.store
        window.worker = Mock()
        window.worker.evidence = {}
        window.worker.isRunning.return_value = False
        window.current_run = self.fx.run
        window.link.begin_inspection(workflow.ui.panel_id(self.fx.run))
        key = panel_key(self.fx.cfg.result_dir, self.fx.run)
        self.fx.store.claim(key, "P", "signature")
        window.active_claim = key
        return window

    def test_continuous_capture_stream_uses_sn_and_cut_not_time_gaps(self):
        result = self.fx.machine / "Result/_20260903_120031.csv"
        result.write_text("SN,Recipe_Name,ProductId,Result,CuttingTime\n2,P.rcp,P,True,20\n")
        for original, stamp in zip(list(self.fx.data["images"]), ("120020", "120025")):
            record = json.loads(json.dumps(original))
            picture = self.fx.machine / "Picture" / f"20260903_{stamp}.bmp"
            cv2.imwrite(str(picture), cv2.imread(self.fx.run.pictures[0]))
            record.update(name=picture.name, sha256=sha256(picture), csv_sn="2")
            self.fx.data["images"].append(record)
        self.fx.save_manifest()
        with patch("router_vision.live_source.scan_auo6000_dataset") as heuristic:
            panels = self.stable(self.source())
        heuristic.assert_not_called()
        self.assertEqual([panel.run.sn for panel in panels], ["1", "2"])
        self.assertEqual(
            [Path(p).stem for p in panels[1].run.pictures],
            ["20260903_120020", "20260903_120025"],
        )

    def test_unchanged_board_does_not_reparse_results_or_json(self):
        source = self.source()
        with patch("router_vision.live_source.read_run", wraps=read_run) as reader:
            panel = self.stable(source)[0]
            self.fx.store.claim(panel.key, "P", panel.signature)
            self.fx.store.finish(panel.key, PanelDecision(run=panel.run), {})
            with patch.object(Path, "read_text", side_effect=AssertionError("old JSON reread")):
                for now in (6.0, 9.0, 12.0):
                    self.clock[0] = now
                    self.assertEqual(source.poll("P", 2), [])
        self.assertEqual(reader.call_count, 1)

    def test_duplicate_capture_records_cannot_release(self):
        directory = self.fx.root / "captures"
        directory.mkdir()
        for name in ("one.json", "two.json"):
            (directory / name).write_text(json.dumps(self.fx.data))
        self.fx.manifest_path = directory
        source = self.source()
        self.assertEqual(self.stable(source), [])
        self.clock[0] = 34.0
        panel = source.poll("P", 2)[0]
        self.assertIn("unique captures", panel.readiness_error)

    def test_duplicate_image_basenames_cannot_release(self):
        duplicate = self.fx.machine / "Picture/other"
        duplicate.mkdir()
        (duplicate / Path(self.fx.run.pictures[0]).name).write_bytes(
            Path(self.fx.run.pictures[0]).read_bytes()
        )
        source = self.source()
        self.assertEqual(self.stable(source), [])
        self.clock[0] = 34.0
        self.assertIn("ambiguous", source.poll("P", 2)[0].readiness_error)

    def test_never_stable_image_times_out_instead_of_waiting_forever(self):
        source = self.source()
        source.poll("P", 2)
        path = Path(self.fx.run.pictures[0])
        for now in (3.0, 10.0, 20.0, 40.0):
            self.clock[0] = now
            path.write_bytes(path.read_bytes())
            panels = source.poll("P", 2)
        self.assertEqual(len(panels), 1)
        self.assertIn("stable", panels[0].readiness_error)

    def test_pending_claim_blocks_even_when_its_result_has_disappeared(self):
        source = self.source()
        panel = self.stable(source)[0]
        self.fx.store.claim(panel.key, "P", panel.signature)
        self.fx.result_path.unlink()
        with self.assertRaisesRegex(ValueError, "unfinished inspection"):
            source.poll("P", 2)

    def test_source_quarantine_requires_retry_even_after_restoration(self):
        source = self.source()
        panel = self.stable(source)[0]
        self.fx.store.claim(panel.key, "P", panel.signature)
        self.fx.store.finish(panel.key, PanelDecision(run=panel.run), {})
        self.fx.store.quarantine(panel.key, "operator must review")
        with self.assertRaisesRegex(ValueError, "operator review"):
            source.poll("P", 2)

    def test_inflight_source_configuration_is_an_immutable_snapshot(self):
        source = self.source()
        self.fx.cfg.picture_dir = "missing"
        self.assertEqual(len(self.stable(source)), 1)

    def test_malformed_stable_csv_waits_then_stops_without_silent_skipping(self):
        self.fx.result_path.write_text("wrong,columns\n1,2\n")
        source = self.source()
        self.assertEqual(self.stable(source), [])
        self.clock[0] = 34.0
        with self.assertRaisesRegex(ValueError, "Invalid stable Result"):
            source.poll("P", 2)

    def test_result_metadata_and_hashed_bytes_must_be_the_same_snapshot(self):
        def changed_after_read(path, **kwargs):
            run = read_run(path, **kwargs)
            Path(path).write_text(Path(path).read_text().replace("True", "False"))
            return run

        source = self.source()
        with patch("router_vision.live_source.read_run", side_effect=changed_after_read):
            self.assertEqual(self.stable(source), [])

    def test_recorded_capture_cannot_point_at_another_result_directory(self):
        self.fx.recorded_captures()
        copied = self.fx.root / self.fx.result_path.name
        copied.write_bytes(self.fx.result_path.read_bytes())
        for record in self.fx.data["images"]:
            record["result_path"] = str(copied)
        self.fx.data["source_files"].append({"path": str(copied), "sha256": sha256(copied)})
        self.fx.save_manifest()
        source = self.source()
        self.assertEqual(self.stable(source), [])
        self.clock[0] = 34.0
        self.assertIn("outside", source.poll("P", 2)[0].readiness_error)

    def test_boolean_capture_cut_is_not_an_integer_identity(self):
        self.fx.data["images"][0]["cut_point"] = True
        self.fx.save_manifest()
        source = self.source()
        self.assertEqual(self.stable(source), [])
        self.clock[0] = 34.0
        self.assertTrue(source.poll("P", 2)[0].readiness_error)

    def test_route_mismatch_cannot_finish_another_claim(self):
        key = panel_key(self.fx.cfg.result_dir, self.fx.run)
        self.fx.store.claim(key, "other-route", "signature")
        with self.assertRaisesRegex(ValueError, "route differs"):
            self.fx.store.finish(key, PanelDecision(run=self.fx.run), {})
        self.assertEqual(self.fx.store.state(key)[0], "PENDING")

    def test_history_orders_by_attempt_even_when_clock_moves_backwards(self):
        for key in ("earlier", "later"):
            self.fx.store.claim(key, "P", "sig")
            self.fx.store.finish(key, PanelDecision(run=self.fx.run, note=key), {})
        with self.fx.store.connect() as db:
            db.execute("UPDATE attempts SET completed_at='2099-01-01' WHERE id=1")
        self.assertEqual(self.fx.store.recent()[0][0]["result"]["note"], "later")

    def test_late_good_after_hold_is_persisted_as_fault(self):
        window = self.window()
        window.link.fault("operator HOLD")
        window._inspection_done(
            PanelDecision(run=self.fx.run, status="GOOD"), np.zeros((50, 70, 3), np.uint8), 1
        )
        self.assertIs(window.link.state, GateState.FAULT)
        self.assertFalse(window.link.signals.release)
        self.assertEqual(self.fx.store.recent()[0][0]["result"]["status"], "FAULT")

    def test_source_changed_after_worker_before_recording_cannot_pass(self):
        window = self.window()
        image = Path(self.fx.run.pictures[0])
        window.worker.evidence = {"source_hashes": {str(image): sha256(image)}}
        image.write_bytes(b"changed after worker")
        window._inspection_done(
            PanelDecision(run=self.fx.run, status="GOOD"), np.zeros((50, 70, 3), np.uint8), 1
        )
        self.assertIs(window.link.state, GateState.FAULT)
        saved = self.fx.store.recent()[0][0]
        self.assertEqual(saved["result"]["status"], "FAULT")
        self.assertIn("source_conflict", saved["evidence"])

    def test_source_changed_while_writing_evidence_is_recorded_as_fault(self):
        from router_vision.production_evidence import save_image

        from tests import test_settings_workflow as workflow

        window = self.window()
        image = Path(self.fx.run.pictures[0])
        window.worker.evidence = {"source_hashes": {str(image): sha256(image)}}

        def changed_after_save(*args, **kwargs):
            save_image(*args, **kwargs)
            image.write_bytes(b"changed while recording evidence")

        with patch.object(workflow.ui, "save_image", side_effect=changed_after_save):
            window._inspection_done(
                PanelDecision(run=self.fx.run, status="GOOD"), np.zeros((50, 70, 3), np.uint8), 1
            )
        self.assertFalse(window.link.signals.release)
        self.assertEqual(self.fx.store.recent()[0][0]["result"]["status"], "FAULT")

    def test_second_app_cannot_share_the_same_runtime_configuration(self):
        from tests import test_settings_workflow as workflow

        self.window()
        with self.assertRaisesRegex(ValueError, "Another AVTR instance"):
            workflow.ui.MainWindow()

    def test_hundred_boards_measure_fresh_pixels_and_keep_exactly_one_attempt_each(self):
        captures = self.fx.root / "batch-captures"
        captures.mkdir()
        self.fx.result_path.unlink()
        base_records = json.loads(json.dumps(self.fx.data["images"]))
        source = LiveSource(
            self.fx.cfg,
            self.fx.store,
            stable_seconds=0,
            clock=lambda: self.clock[0],
            capture_location=captures,
        )
        expected_statuses = []
        original = read_image(self.fx.run.pictures[0])
        for number in range(1, 101):
            stamp = (datetime(2026, 9, 4) + timedelta(minutes=number)).strftime("%Y%m%d_%H%M%S")
            result_path = self.fx.machine / "Result" / f"_{stamp}.csv"
            result_path.write_text(
                f"SN,Recipe_Name,ProductId,Result,CuttingTime,OffsetX,OffsetY,BitDiameter\n{number},P.rcp,P,True,20,0,0,1.3\n"
            )
            records = json.loads(json.dumps(base_records))
            for cut, record in enumerate(records, 1):
                picture = self.fx.machine / "Picture" / f"board-{number:03d}-cut-{cut:02d}.bmp"
                image = original.copy()
                image[0, 0] = [number, cut, 200]
                okay, encoded = cv2.imencode(".png", image)
                self.assertTrue(okay)
                encoded.tofile(picture)
                projection = record.pop("projection")
                record.update(
                    name=picture.name,
                    sha256=sha256(picture),
                    csv_sn=str(number),
                    result_file=result_path.name,
                    result_path=str(result_path),
                    motion={
                        "alignment_applied": True,
                        "start_end_xy_mm": projection["reconstructed_nc_xy_mm"],
                        "camera_xy_mm": projection["reconstructed_g87_camera_xy_mm"],
                        "diameter_mm": 1.3,
                        "offset_xy_mm": [0, 0],
                        "rotation_rad": 0.0,
                    },
                )
            manifest = captures / f"board-{number:03d}.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": CAPTURE_SCHEMA,
                        "scale_mm_per_px_xy": [0.01, 0.01],
                        "source_files": [
                            {"path": str(self.fx.recipe), "sha256": sha256(self.fx.recipe)},
                            {"path": str(result_path), "sha256": sha256(result_path)},
                        ],
                        "images": records,
                    }
                )
            )
            source.poll("P", 2)
            self.clock[0] += 0.1
            panels = source.poll("P", 2)
            self.assertEqual(len(panels), 1)
            panel = panels[0]
            self.assertFalse(panel.readiness_error)
            self.fx.run = panel.run
            limits = {"inner_max_mm": 1.0, "outer_max_mm": 0.1 if number % 10 == 0 else 1.0}
            decision, evidence = self.fx.inspect(limits=limits, refs=CaptureReferences(manifest))
            expected = "NG" if number % 10 == 0 else "GOOD"
            self.assertEqual(decision.status, expected, decision.note)
            self.fx.store.claim(panel.key, "P", panel.signature)
            self.fx.store.finish(panel.key, decision, evidence)
            expected_statuses.append(expected)
            self.assertEqual(source.poll("P", 2), [])
        history = self.fx.store.recent(1000)
        self.assertEqual(len(history), 100)
        self.assertEqual([p[0]["result"]["status"] for p in reversed(history)], expected_statuses)


class ConfigurationResilienceTests(unittest.TestCase):
    def test_legacy_null_optional_paths_mean_not_configured(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"capture_manifest_dir": None, "model_dir": None}))
            cfg = AppConfig.load(path)
            self.assertFalse(getattr(cfg, "load_error", ""))
            self.assertEqual(cfg.capture_manifest_dir, "")
            self.assertEqual(cfg.model_dir, "")

    def test_invalid_config_and_missing_source_start_in_fault_with_a_working_timer(self):
        from tests import test_settings_workflow as workflow

        workflow.SettingsWorkflowTests.setUpClass()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "config.json"
            for invalid in (True, False):
                path.write_text("[]") if invalid else AppConfig(
                    picture_dir=str(root / "missing"), result_dir=str(root / "missing-result")
                ).save(path)
                with (
                    patch.object(workflow.ui, "CONFIG_PATH", path),
                    patch.object(workflow.ui, "SOBEL_WORKFLOW_PATH", root / "workflow.json"),
                    patch.object(workflow.ui.QMessageBox, "critical"),
                ):
                    window = workflow.ui.MainWindow()
                    try:
                        self.assertIs(window.link.state, GateState.FAULT)
                        self.assertIsNotNone(window.auto_timer)
                    finally:
                        window.close()
                        window.deleteLater()

    def test_unicode_image_paths_and_unreadable_files(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "\u0e20\u0e32\u0e1e-\u0e1a\u0e2d\u0e23\u0e4c\u0e14.bmp"
            image = np.random.default_rng(7).integers(0, 255, (80, 100, 3), dtype=np.uint8)
            okay, encoded = cv2.imencode(".png", image)
            self.assertTrue(okay)
            encoded.tofile(path)
            np.testing.assert_array_equal(read_image(path), image)
            path.write_bytes(b"not an image")
            self.assertIsNone(read_image(path))
            path.unlink()
            self.assertIsNone(read_image(path))

    def test_invalid_config_types_are_reported_without_crashing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for data in (
                [],
                None,
                {"live_poll_seconds": "bad"},
                {"expected_images_by_route": []},
                {"edge_limits_by_route": {"P": None}},
                {"good_confidence_min": 0.1},
            ):
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    cfg = AppConfig.load(path)
                    self.assertTrue(cfg.load_error)

    def test_invalid_poll_intervals_cannot_be_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            for value in (float("nan"), float("inf"), -1, 0, True):
                with self.subTest(value=value), self.assertRaises(ValueError):
                    AppConfig(live_poll_seconds=value).save(path)
            self.assertFalse(path.exists())

    def test_invalid_pixel_size_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "Eqp.cfg"
            path.write_text('<PixelSize X="0" Y="-0.01"/>')
            with self.assertRaises(ValueError):
                read_pixel_size(path)

    def test_guard_protects_other_sepdata_subfolders_and_capture_inputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "SepData"
            for name in ("Picture", "Result", "captures"):
                (root / name).mkdir(parents=True)
            cfg = AppConfig(
                picture_dir=str(root / "Picture"),
                result_dir=str(root / "Result"),
                capture_manifest_dir=str(root / "captures"),
            )
            for name in (
                "Database/ProcessReport.db",
                "Log/output.json",
                "Temp/snapshot.cfg",
                "captures/output.json",
            ):
                with self.subTest(name=name), self.assertRaises(ProtectedPathError):
                    check_write_target(root / name, protected_roots(cfg))
            with self.assertRaises(ProtectedPathError):
                check_write_target("", [])


if __name__ == "__main__":
    unittest.main()
