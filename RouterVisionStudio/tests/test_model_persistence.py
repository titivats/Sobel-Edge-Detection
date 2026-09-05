from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import torch
import torch.nn as nn

from router_vision.guard import ProtectedPathError
from router_vision.model import (
    BACKBONES,
    MIN_MODEL_CLASS_RECALL,
    MIN_MODEL_VALIDATION_ACCURACY,
    CutClassifier,
    TrainReport,
    model_quality_error,
)


def _classifier(*, model_key: str = "", report: TrainReport | None = None) -> CutClassifier:
    classifier = CutClassifier(model_key=model_key)
    classifier.classes = ["GOOD", "NG"]
    classifier.head = nn.Linear(BACKBONES[classifier.backbone], 2)
    classifier.report = report
    return classifier


def _passing_report() -> TrainReport:
    return TrainReport(
        classes=["GOOD", "NG"],
        n_train=15,
        n_val=5,
        epochs=20,
        train_acc=1.0,
        val_acc=0.8,
        per_class={
            "GOOD": {"n": 3, "correct": 3},
            "NG": {"n": 2, "correct": 1},
        },
        seconds=1.0,
        device="cpu",
        backbone="dinov2_vits14",
    )


class ModelQualityTests(unittest.TestCase):
    def test_quality_threshold_boundaries_are_accepted(self):
        self.assertEqual(MIN_MODEL_VALIDATION_ACCURACY, 0.80)
        self.assertEqual(MIN_MODEL_CLASS_RECALL, 0.50)
        self.assertIsNone(model_quality_error(_passing_report()))

    def test_quality_rejects_low_overall_validation_accuracy(self):
        report = _passing_report()
        report.val_acc = 0.79
        self.assertIn("below the required 80%", model_quality_error(report))

    def test_quality_rejects_missing_class_validation(self):
        report = _passing_report()
        del report.per_class["NG"]
        self.assertIn("NG", model_quality_error(report))

    def test_quality_rejects_low_ng_recall_even_with_high_overall_accuracy(self):
        report = _passing_report()
        report.n_val = 10
        report.val_acc = 0.9
        report.per_class = {
            "GOOD": {"n": 9, "correct": 9},
            "NG": {"n": 1, "correct": 0},
        }
        error = model_quality_error(report)
        self.assertIn("recall for NG", error)
        self.assertIn("below the required 50%", error)

    def test_quality_rejects_non_finite_or_inconsistent_report(self):
        report = _passing_report()
        report.val_acc = float("nan")
        self.assertIn("invalid", model_quality_error(report))

        report = _passing_report()
        report.n_val = 99
        self.assertIn("do not match", model_quality_error(report))


class ModelPersistenceTests(unittest.TestCase):
    def test_identity_round_trip_and_expected_identity_check(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            _classifier(model_key="PRODUCT-A", report=_passing_report()).save(path)

            loaded = CutClassifier.load(path, expected_model_key="PRODUCT-A")
            self.assertEqual(loaded.model_key, "PRODUCT-A")
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["model_key"], "PRODUCT-A")
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                CutClassifier.load(path, expected_model_key="PRODUCT-B")

    def test_legacy_checkpoint_without_identity_loads_only_without_expectation(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            _classifier().save(path)

            self.assertEqual(CutClassifier.load(path).model_key, "")
            with self.assertRaisesRegex(ValueError, "no identity"):
                CutClassifier.load(path, expected_model_key="PRODUCT-A")

    def test_load_rejects_malformed_checkpoint_schema_and_head(self):
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            valid_path = folder_path / "valid.pt"
            _classifier(model_key="PRODUCT-A", report=_passing_report()).save(valid_path)
            valid = torch.load(valid_path, map_location="cpu", weights_only=True)

            cases = []
            missing = copy.deepcopy(valid)
            del missing["sobel"]
            cases.append(("missing", missing, "missing: sobel"))

            wrong_classes = copy.deepcopy(valid)
            wrong_classes["classes"] = ["GOOD", "BAD"]
            cases.append(("classes", wrong_classes, "exactly GOOD and NG"))

            wrong_dimensions = copy.deepcopy(valid)
            wrong_dimensions["head_state"]["weight"] = torch.zeros((2, 10))
            cases.append(("dimensions", wrong_dimensions, "dimensions"))

            non_finite = copy.deepcopy(valid)
            non_finite["head_state"]["weight"][0, 0] = float("nan")
            cases.append(("non_finite", non_finite, "non-finite"))

            for name, payload, message in cases:
                with self.subTest(name=name):
                    path = folder_path / f"{name}.pt"
                    torch.save(payload, path)
                    with self.assertRaisesRegex(ValueError, message):
                        CutClassifier.load(path)

    def test_save_is_atomic_when_serialization_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            sidecar = path.with_suffix(".json")
            path.write_bytes(b"old model")
            sidecar.write_text("old metadata", encoding="utf-8")

            with patch("router_vision.model.torch.save", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    _classifier().save(path)

            self.assertEqual(path.read_bytes(), b"old model")
            self.assertEqual(sidecar.read_text(encoding="utf-8"), "old metadata")
            self.assertEqual(list(Path(folder).glob(".*.tmp")), [])

    def test_save_refuses_a_protected_machine_data_path(self):
        with tempfile.TemporaryDirectory() as folder:
            protected = Path(folder) / "Picture"
            protected.mkdir()
            path = protected / "model.pt"
            with self.assertRaises(ProtectedPathError):
                _classifier().save(path, protected=[protected])
            self.assertFalse(path.exists())
            self.assertFalse(path.with_suffix(".json").exists())


class ModelTrainingInputTests(unittest.TestCase):
    def test_training_requires_two_readable_images_from_each_class(self):
        classifier = _classifier()
        classifier.extractor.embed = Mock()
        with self.assertRaisesRegex(ValueError, "at least 2 labelled images per class"):
            classifier.train(
                [("good-a", "GOOD"), ("ng-a", "NG"), ("ng-b", "NG")],
                epochs=1,
            )
        classifier.extractor.embed.assert_not_called()

    def test_training_rejects_unexpected_or_missing_class(self):
        classifier = _classifier()
        with self.assertRaisesRegex(ValueError, "exactly GOOD and NG"):
            classifier.train(
                [
                    ("good-a", "GOOD"),
                    ("good-b", "GOOD"),
                    ("bad-a", "BAD"),
                    ("bad-b", "BAD"),
                ],
                epochs=1,
            )

        with self.assertRaisesRegex(ValueError, "NG=0"):
            classifier.train(
                [("good-a", "GOOD"), ("good-b", "GOOD")],
                epochs=1,
            )

    def test_training_aborts_if_any_labelled_image_is_dropped(self):
        classifier = _classifier()
        classifier.extractor.embed = Mock(
            return_value=(
                np.zeros((3, BACKBONES[classifier.backbone]), dtype=np.float32),
                [0, 1, 3],
            )
        )
        items = [
            ("good-a", "GOOD"),
            ("good-b", "GOOD"),
            ("ng-a", "NG"),
            ("ng-b", "NG"),
        ]
        with self.assertRaisesRegex(ValueError, "could not be read; training aborted"):
            classifier.train(items, epochs=1)


if __name__ == "__main__":
    unittest.main()
