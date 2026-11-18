from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from router_vision.model import CropBox, FeatureExtractor, SobelConfig
from router_vision.production import classify_panel
from router_vision.production_archive import archive_panel
from router_vision.production_evidence import save_image, save_json
from router_vision.runtime_log import close_runtime_log, log_event, open_runtime_log
from router_vision.workers import InspectionWorker

from tests import test_live_production as live


class ProductionArchiveTests(unittest.TestCase):
    def setUp(self):
        self.fixture = live.LiveProductionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.cfg, self.run = (
            self.fixture.root,
            self.fixture.cfg,
            self.fixture.run,
        )
        self.classifier = live.Classifier()
        self.classifier.crop = CropBox(20, 20, 480, 380)
        self.classifier.extractor = FeatureExtractor(device="cpu", sobel=SobelConfig())
        self.result = classify_panel(self.classifier, self.run, 2)

    def archive(self, **kwargs):
        return archive_panel(
            self.result, self.classifier, self.root / "archive", self.cfg, "claim", **kwargs
        )

    def test_native_original_and_exact_model_sobel_are_real_jpeg(self):
        manifest, prepared = self.archive()
        self.assertTrue(manifest["complete"])
        self.assertEqual(len(manifest["cuts"]), 2)
        for cut in manifest["cuts"]:
            original = cv2.imread(cut["source"])
            for item in cut["files"]:
                path = Path(item["path"])
                self.assertEqual(path.read_bytes()[:2], b"\xff\xd8")
                decoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
                self.assertEqual(list(decoded.shape[:2][::-1]), item["size_px"])
                expected = (
                    original
                    if item["kind"] == "original"
                    else self.classifier.extractor.edge_map(cut["source"], self.classifier.crop)
                )
                self.assertEqual(decoded.shape, expected.shape)
                self.assertGreater(cv2.PSNR(expected, decoded), 40)
            crop, sobel = prepared[cut["source"]]
            self.assertTrue(crop.flags.owndata)
            np.testing.assert_array_equal(
                sobel, self.classifier.extractor.edge_map(cut["source"], self.classifier.crop)
            )

    def test_retry_has_unique_directory_and_sn_cannot_escape_root(self):
        self.run.sn = "../../unsafe\\name"
        first, _ = self.archive()
        second, _ = self.archive()
        self.assertNotEqual(first["directory"], second["directory"])
        self.assertTrue(Path(first["directory"]).is_relative_to(self.root / "archive"))
        self.assertTrue(Path(first["directory"]).is_dir())

    def test_low_space_rejects_before_writing_any_jpeg(self):
        with patch(
            "router_vision.production_archive.shutil.disk_usage",
            return_value=SimpleNamespace(free=0),
        ):
            with self.assertRaisesRegex(OSError, "disk space"):
                self.archive()
        self.assertEqual(list(self.root.rglob("*.jpeg")), [])

    def test_unreadable_cut_marks_incomplete_and_keeps_other_cut(self):
        Path(self.run.pictures[0]).write_bytes(b"corrupt camera file")
        manifest, _ = self.archive()
        self.assertFalse(manifest["complete"])
        self.assertIn("unreadable", manifest["cuts"][0]["error"])
        self.assertEqual(len(manifest["cuts"][1]["files"]), 2)

    def test_cancellation_rejects_export(self):
        with self.assertRaises(InterruptedError):
            self.archive(cancelled=lambda: True)
        self.assertEqual(list(self.root.rglob("*.jpeg")), [])

    def test_machine_output_is_protected(self):
        with self.assertRaises(Exception):
            archive_panel(
                self.result, self.classifier, self.fixture.machine / "Picture", self.cfg, "claim"
            )
        self.assertEqual(list(self.fixture.machine.rglob("*.jpeg")), [])

    def test_source_change_during_archive_rejects_export(self):
        from router_vision.production_evidence import save_image as actual_save

        def changing_save(path, pixels, cfg):
            actual_save(path, pixels, cfg)
            source = Path(self.run.pictures[0])
            source.write_bytes(source.read_bytes() + b"changed")

        with patch("router_vision.production_archive.save_image", side_effect=changing_save):
            with self.assertRaisesRegex(OSError, "changed during archiving"):
                self.archive()

    def test_replace_failure_preserves_old_file_and_removes_temp(self):
        path = self.root / "evidence.json"
        path.write_text("old")
        with patch.object(Path, "replace", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                save_json(path, {"new": True}, self.cfg)
        self.assertEqual(path.read_text(), "old")
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_failed_encoder_cannot_create_evidence_file(self):
        path = self.root / "cut.jpeg"
        with patch("router_vision.production_evidence.cv2.imencode", return_value=(False, None)):
            with self.assertRaises(OSError):
                save_image(path, np.zeros((10, 10), np.uint8), self.cfg)
        self.assertFalse(path.exists())

    def test_runtime_log_rotates_valid_unicode_json_and_releases_handles(self):
        logger = open_runtime_log(self.root / "logs", self.cfg, max_bytes=300, backups=2)
        self.addCleanup(close_runtime_log, logger)
        for index in range(40):
            log_event(logger, "board.result", sn=index, reason="ทดสอบ", value=float("nan"))
        close_runtime_log(logger)
        files = list((self.root / "logs").glob("avtr.jsonl*"))
        self.assertEqual(len(files), 3)
        for path in files:
            for line in path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                self.assertEqual(event["reason"], "ทดสอบ")
                self.assertIsNone(event["value"])
        self.assertEqual(logger.handlers, [])

    def worker(self):
        return InspectionWorker(
            self.classifier,
            self.run,
            2,
            0.95,
            {
                "archive_root": str(self.root / "archive"),
                "archive_config": self.cfg,
                "claim": "claim",
                "readiness_error": "Setup incomplete",
                "model": {},
                "input_hashes": {},
            },
        )

    def test_worker_archives_even_fault_and_does_not_reread_for_overview(self):
        worker = self.worker()
        completed, failed = [], []
        worker.completed.connect(lambda *args: completed.append(args))
        worker.failed.connect(failed.append)
        with patch(
            "router_vision.overview.read_image", side_effect=AssertionError("duplicate decode")
        ):
            worker.run()
        self.assertEqual(failed, [])
        self.assertEqual(completed[0][0].status, "FAULT")
        self.assertTrue(worker.evidence["archive"]["complete"])
        self.assertTrue(Path(worker.evidence["archive"]["overview"]).is_file())

    def test_worker_archive_error_never_emits_completed(self):
        worker = self.worker()
        completed, failed = [], []
        worker.completed.connect(lambda *args: completed.append(args))
        worker.failed.connect(failed.append)
        with patch("router_vision.workers.archive_panel", side_effect=OSError("disk full")):
            worker.run()
        self.assertEqual(completed, [])
        self.assertIn("disk full", failed[0])

    def test_model_exception_still_archives_fault_images(self):
        worker = self.worker()
        worker.production_context.pop("readiness_error")
        worker.production_context["limits"] = {}
        worker.production_context["recipe"] = self.fixture.recipe
        worker.production_context["edge_state"] = {}
        worker.production_context["references"] = object()
        completed = []
        worker.completed.connect(lambda *args: completed.append(args))
        with patch(
            "router_vision.workers.inspect_with_spec", side_effect=RuntimeError("model failed")
        ):
            worker.run()
        self.assertEqual(completed[0][0].status, "FAULT")
        self.assertIn("model failed", worker.evidence["processing_error"])
        self.assertTrue(worker.evidence["archive"]["complete"])
