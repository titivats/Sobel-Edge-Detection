from __future__ import annotations

import unittest
from types import SimpleNamespace

from router_vision.production import STATUS_FAULT, STATUS_GOOD, STATUS_NG, classify_panel


class FakeClassifier:
    classes = ["GOOD", "NG"]

    def __init__(self, predictions):
        self.predictions = predictions

    def predict(self, paths, cancelled=None):
        return list(self.predictions)


def run(*, pictures=3, passed=True):
    return SimpleNamespace(
        pictures=[f"image-{n}.bmp" for n in range(pictures)],
        passed=passed,
        message="router failure" if not passed else "",
    )


class ProductionClassifierTests(unittest.TestCase):
    def test_invalid_configured_threshold_fails_closed(self):
        for threshold in (float("nan"), float("inf"), -.1, 1.2, True, "invalid"):
            with self.subTest(threshold=threshold):
                classifier = FakeClassifier([("GOOD", .99)] * 3)
                result = classify_panel(classifier, run(), 3, threshold)
                self.assertEqual(result.status, STATUS_FAULT)
                self.assertIn("invalid GOOD confidence threshold", result.note)

    def test_all_high_confidence_good_releases(self):
        classifier = FakeClassifier([("GOOD", 0.99), ("GOOD", 0.98), ("GOOD", 0.97)])
        result = classify_panel(classifier, run(), 3, 0.95)
        self.assertEqual(result.status, STATUS_GOOD)

    def test_any_ng_stops_panel(self):
        classifier = FakeClassifier([("GOOD", 0.99), ("NG", 0.55), ("GOOD", 0.99)])
        result = classify_panel(classifier, run(), 3, 0.95)
        self.assertEqual(result.status, STATUS_NG)

    def test_low_confidence_good_fails_closed(self):
        classifier = FakeClassifier([("GOOD", 0.99), ("GOOD", 0.70), ("GOOD", 0.99)])
        result = classify_panel(classifier, run(), 3, 0.95)
        self.assertEqual(result.status, STATUS_FAULT)
        self.assertIn("below", result.note)

    def test_non_finite_confidence_fails_closed(self):
        for confidence in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(confidence=confidence):
                classifier = FakeClassifier([("GOOD", confidence)])
                result = classify_panel(classifier, run(pictures=1), 1, 0.95)

                self.assertEqual(result.status, STATUS_FAULT)
                self.assertIn("invalid confidence", result.note)
                self.assertIn("cut point 1", result.note)

    def test_out_of_range_confidence_fails_closed(self):
        for confidence in (-0.01, 1.01):
            with self.subTest(confidence=confidence):
                classifier = FakeClassifier([("GOOD", confidence)])
                result = classify_panel(classifier, run(pictures=1), 1, 0.95)

                self.assertEqual(result.status, STATUS_FAULT)
                self.assertIn("invalid confidence", result.note)

    def test_invalid_confidence_overrides_ng_status_with_fault(self):
        classifier = FakeClassifier([("NG", float("nan"))])
        result = classify_panel(classifier, run(pictures=1), 1, 0.95)

        self.assertEqual(result.status, STATUS_FAULT)
        self.assertIn("invalid confidence", result.note)
        self.assertIn("NG predicted at cut point 1", result.note)

    def test_unknown_prediction_label_fails_closed(self):
        classifier = FakeClassifier([("MAYBE", 0.99)])
        result = classify_panel(classifier, run(pictures=1), 1, 0.95)

        self.assertEqual(result.status, STATUS_FAULT)
        self.assertIn("invalid/unknown model label at cut point 1", result.note)

    def test_incomplete_panel_fails_closed_without_inference(self):
        classifier = FakeClassifier([])
        result = classify_panel(classifier, run(pictures=2), 3, 0.95)
        self.assertEqual(result.status, STATUS_FAULT)
        self.assertIn("expected 3", result.note)

    def test_incomplete_prediction_set_fails_closed(self):
        classifier = FakeClassifier([("GOOD", 0.99), ("GOOD", 0.99)])
        result = classify_panel(classifier, run(), 3, 0.95)

        self.assertEqual(result.status, STATUS_FAULT)
        self.assertIn("incomplete prediction set", result.note)
        self.assertIn("expected 3, found 2", result.note)

    def test_malformed_prediction_fails_closed(self):
        classifier = FakeClassifier([("GOOD", 0.99), ("NG",), ("GOOD", 0.99)])
        result = classify_panel(classifier, run(), 3, 0.95)

        self.assertEqual(result.status, STATUS_FAULT)
        self.assertIn("malformed model prediction at cut point 2", result.note)

    def test_unreadable_image_fails_closed(self):
        classifier = FakeClassifier([("GOOD", 0.99), ("", 0.0), ("GOOD", 0.99)])
        result = classify_panel(classifier, run(), 3, 0.95)
        self.assertEqual(result.status, STATUS_FAULT)

    def test_reason_lists_every_problem_found_on_the_panel(self):
        classifier = FakeClassifier([("", 0.0), ("NG", 0.91), ("GOOD", 0.70)])
        result = classify_panel(classifier, run(), 3, 0.95)

        self.assertEqual(result.status, STATUS_FAULT)
        self.assertIn("unreadable/blank Sobel input at cut point 1", result.note)
        self.assertIn("NG predicted at cut point 2", result.note)
        self.assertIn("GOOD confidence below 95.0% at cut point 3", result.note)

    def test_router_failure_is_ng_without_inference(self):
        classifier = FakeClassifier([])
        result = classify_panel(classifier, run(passed=False), 3, 0.95)
        self.assertEqual(result.status, STATUS_NG)
        self.assertEqual(result.note, "router failure")
