from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from router_vision.model import (
    BACKBONES,
    PREPROCESS_VERSION,
    CropBox,
    CutClassifier,
    FeatureExtractor,
    SobelConfig,
)


class SobelPreprocessTests(unittest.TestCase):
    def test_edge_map_and_tensor_are_deterministic_shapes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "edge.bmp"
            image = np.zeros((160, 160, 3), dtype=np.uint8)
            cv2.rectangle(image, (35, 20), (125, 140), (255, 255, 255), 4)
            self.assertTrue(cv2.imwrite(str(path), image))

            extractor = FeatureExtractor(device="cpu")
            crop = CropBox(0, 0, 160, 160)
            edge = extractor.edge_map(path, crop)
            tensor = extractor.preprocess(path, crop)

            self.assertEqual(PREPROCESS_VERSION, "sobel-magnitude-v1")
            self.assertEqual(edge.shape, (160, 160))
            self.assertGreater(int(edge.max()), 0)
            self.assertEqual(tensor.shape, (3, 224, 224))

    def test_blank_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "blank.bmp"
            self.assertTrue(cv2.imwrite(str(path), np.zeros((80, 80, 3), dtype=np.uint8)))
            extractor = FeatureExtractor(device="cpu")
            self.assertIsNone(extractor.preprocess(path, CropBox(0, 0, 80, 80)))

    def test_saved_model_round_trip_keeps_preprocess_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            classifier = CutClassifier(
                sobel=SobelConfig(
                    blur_sigma=1.2,
                    gradient_x_weight=0.75,
                    gradient_y_weight=1.25,
                    edge_gain=1.4,
                    noise_floor=18,
                )
            )
            classifier.classes = ["GOOD", "NG"]
            classifier.head = nn.Linear(BACKBONES[classifier.backbone], 2)
            classifier.save(path)

            loaded = CutClassifier.load(path)
            self.assertEqual(loaded.classes, ["GOOD", "NG"])
            self.assertEqual(loaded.sobel.sobel_ksize, 3)
            self.assertEqual(loaded.sobel.blur_sigma, 1.2)
            self.assertEqual(loaded.sobel.gradient_x_weight, 0.75)
            self.assertEqual(loaded.sobel.gradient_y_weight, 1.25)
            self.assertEqual(loaded.sobel.edge_gain, 1.4)
            self.assertEqual(loaded.sobel.noise_floor, 18)

    def test_extended_sobel_controls_are_validated_and_change_output(self):
        config = SobelConfig(
            blur_sigma=99,
            gradient_x_weight=-1,
            gradient_y_weight=0,
            edge_gain=9,
            noise_floor=999,
        ).validated()
        self.assertEqual(config.blur_sigma, 10.0)
        self.assertEqual(config.gradient_x_weight, 1.0)
        self.assertEqual(config.gradient_y_weight, 1.0)
        self.assertEqual(config.edge_gain, 3.0)
        self.assertEqual(config.noise_floor, 254)

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "directional.bmp"
            image = np.zeros((160, 160, 3), dtype=np.uint8)
            cv2.rectangle(image, (25, 45), (135, 115), (255, 255, 255), 5)
            self.assertTrue(cv2.imwrite(str(path), image))
            crop = CropBox(0, 0, 160, 160)
            baseline = FeatureExtractor(device="cpu").edge_map(path, crop)
            tuned = FeatureExtractor(
                device="cpu",
                sobel=SobelConfig(
                    gradient_x_weight=0.0,
                    gradient_y_weight=1.5,
                    edge_gain=0.7,
                    noise_floor=25,
                ),
            ).edge_map(path, crop)
            self.assertIsNotNone(baseline)
            self.assertIsNotNone(tuned)
            self.assertFalse(np.array_equal(baseline, tuned))

    def test_legacy_colour_model_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "legacy.pt"
            torch.save(
                {
                    "backbone": "dinov2_vits14",
                    "classes": ["PASS", "NG"],
                    "crop": {},
                    "head_state": nn.Linear(384, 2).state_dict(),
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "legacy colour input"):
                CutClassifier.load(path)


if __name__ == "__main__":
    unittest.main()
