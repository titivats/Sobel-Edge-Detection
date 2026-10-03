"""Regression checks for image-derived edges used in trainer review."""

import unittest

import cv2
import numpy as np
from router_vision.image_edge import analyse, checked_scale, measurement_summary, preview_reference


def board(inner=0, outer=0):
    image = np.full((1080, 1440, 3), 120, np.uint8)
    edge = np.full(1440, 650)
    edge[550:700] -= outer
    edge[800:900] += inner
    for x in range(1440):
        image[480 : edge[x], x] = 10
    return image


class ImageEdgeTests(unittest.TestCase):
    def test_visual_reference_uses_side_edges_and_preserves_measurement_rejection(self):
        image = board(outer=30)
        image[650:656, 1000:] = 10
        b = analyse(image, allow_edge_only=True)
        a = analyse(cv2.flip(image, 0), side="A", allow_edge_only=True)
        lower, upper = preview_reference(b, 1440), preview_reference(a, 1440)
        for result, guide in ((b, lower), (a, upper)):
            self.assertEqual(guide["status"], "VISUAL_ESTIMATE_ONLY")
            self.assertFalse(guide["used_for_mm"])
            self.assertIsNone(measurement_summary(result, [0.01, 0.01])["inner_mm"])
            self.assertTrue(all(hi < 360 or lo >= 1080 for lo, hi in guide["sample_x_ranges"]))
        np.testing.assert_allclose(
            np.array(upper["line_px"])[:, 1], 1079 - np.array(lower["line_px"])[:, 1], atol=0.01
        )
        # The central protrusion must not pull the visual reference away from the side edges.
        clean = image.copy()
        clean[620:650, 550:700] = 10
        np.testing.assert_allclose(
            lower["line_px"],
            preview_reference(analyse(clean, allow_edge_only=True), 1440)["line_px"],
        )

    def test_visual_reference_does_not_invent_missing_side_support(self):
        result = analyse(board())
        guide = preview_reference(result, 1440)
        self.assertTrue(guide["used_for_mm"])
        result["slope"] = None
        result["accepted"][:] = False
        result["accepted"][400:500] = True  # central tab samples alone are insufficient
        self.assertIsNone(preview_reference(result, 1440))

    def test_edge_review_can_return_a_contour_without_inventing_a_baseline(self):
        image = board()
        image[650:656, 1000:] = 10
        with self.assertRaises(ValueError):
            analyse(image)
        for side, pixels in (("B", image), ("A", cv2.flip(image, 0))):
            result = analyse(pixels, side=side, allow_edge_only=True)
            summary = measurement_summary(result, [0.01, 0.01])
            self.assertEqual(summary["measurement_status"], "EDGE_ONLY")
            self.assertGreater(summary["coverage"], 0)
            self.assertIsNone(summary["inner_mm"])
            self.assertIsNone(summary["outer_mm"])
            self.assertIsNone(result["slope"])

    def test_detached_speck_is_not_a_protrusion_but_attached_tab_is(self):
        image = board()
        image[610:620, 700:712] = 120
        result = analyse(image)
        self.assertAlmostEqual(float(-np.nanmin(result["deviation"])), 0, places=4)
        self.assertGreater(result["detached_bright_columns"], 0)
        image[610:650, 700:712] = 120
        result = analyse(image)
        self.assertAlmostEqual(float(-np.nanmin(result["deviation"])), 40, delta=0.5)

    def test_known_distances_and_mirrored_side(self):
        image = board(inner=8, outer=20)
        b = analyse(image)
        a = analyse(cv2.flip(image, 0), side="A")
        for result in (a, b):
            s = measurement_summary(result, [0.01, 0.02])
            self.assertAlmostEqual(s["inner_px"], 8, delta=0.5)
            self.assertAlmostEqual(s["outer_px"], 20, delta=0.5)
            self.assertAlmostEqual(s["inner_mm"], 0.16, delta=0.01)
            self.assertAlmostEqual(s["outer_mm"], 0.4, delta=0.01)
        np.testing.assert_allclose(a["y"], 1079 - b["y"], equal_nan=True)
        np.testing.assert_allclose(a["normal"], b["normal"] * [1, -1])

    def test_narrow_damage_in_reference_strip_is_rejected(self):
        image = board()
        image[635:650, 200:206] = 120
        with self.assertRaisesRegex(ValueError, "Side segments|disagree"):
            analyse(image)

    def test_two_straight_but_displaced_reference_strips_are_rejected(self):
        image = board()
        image[650:656, 1000:] = 10
        with self.assertRaisesRegex(ValueError, "disagree"):
            analyse(image)

    def test_partial_is_not_reported_as_complete(self):
        image = board(inner=8, outer=20)
        # A blurred, weak boundary leaves material connected but gives no reliable peak.
        image[620:700, 740:750] = np.linspace(10, 120, 80).astype(np.uint8)[:, None, None]
        result = analyse(image)
        summary = measurement_summary(result, [0.01, 0.01])
        self.assertEqual(summary["measurement_status"], "PARTIAL")
        self.assertIsNone(summary["complete_inner_mm"])
        self.assertIsNone(summary["complete_outer_mm"])
        self.assertGreater(len(summary["rejected_x_px"]), 0)

    def test_zero_max_does_not_mark_an_unrelated_point(self):
        summary = measurement_summary(analyse(board()), [0.01, 0.01])
        for name in ("inner", "outer"):
            self.assertEqual(summary[name + "_mm"], 0)
            self.assertIsNone(summary[name + "_index"])

    def test_invalid_scale_and_unit_normal(self):
        for scales in ([0, 0.01], [-1, 1], [np.nan, 0.01], [np.inf, 0.01], [0.01]):
            with self.subTest(scales=scales), self.assertRaises(ValueError):
                checked_scale([0, 1], scales)
        with self.assertRaises(ValueError):
            checked_scale([0, 2], [0.01, 0.01])

    def test_anisotropic_scale_is_perpendicular_distance(self):
        self.assertAlmostEqual(checked_scale([0.6, 0.8], [0.01, 0.02]), 1 / np.hypot(60, 40))

    def test_unreadable_blank_and_invalid_inputs_rejected(self):
        for image in (None, np.zeros((1080, 1440, 3), np.uint8), np.zeros((40, 40, 3), np.uint8)):
            with self.subTest(shape=getattr(image, "shape", None)), self.assertRaises(ValueError):
                analyse(image)
        for kwargs in ({"side": "C"}, {"blur": 2}, {"fraction": np.nan}, {"wings": (0.5, 0.7)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                analyse(board(), **kwargs)
