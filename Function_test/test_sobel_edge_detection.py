from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import cv2
import numpy as np

from command_line_interface import create_sobel_edge_output
from sobel_edge_detection import repair_edge_mask


class SobelEdgeDetectionTest(unittest.TestCase):
    def test_repair_edge_mask_closes_small_gap(self) -> None:
        edge_mask = np.zeros((7, 7), dtype=np.uint8)
        edge_mask[3, 1] = 255
        edge_mask[3, 3] = 255

        repaired = repair_edge_mask(
            edge_mask,
            close_kernel_size=3,
            close_iterations=1,
            dilate_iterations=0,
        )

        self.assertEqual(repaired[3, 2], 255)

    def test_repair_edge_mask_can_dilate_edge(self) -> None:
        edge_mask = np.zeros((7, 7), dtype=np.uint8)
        edge_mask[3, 3] = 255

        repaired = repair_edge_mask(
            edge_mask,
            close_kernel_size=1,
            close_iterations=0,
            dilate_iterations=1,
        )

        self.assertGreater(np.count_nonzero(repaired), np.count_nonzero(edge_mask))

    def test_create_sobel_edge_output_saves_combined_edge_image(self) -> None:
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        image[:, 10:] = 255

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_path = temp_path / "input.bmp"
            cv2.imwrite(str(image_path), image)

            output = create_sobel_edge_output(
                image_path=image_path,
                output_dir=temp_path / "output",
                sobel_threshold_ratio=0.12,
                edge_close_kernel=1,
                edge_close_iterations=0,
                edge_dilate_iterations=0,
                display_edge_thickness=1,
                edge_view="all",
            )

            saved = cv2.imread(str(output.output_path), cv2.IMREAD_GRAYSCALE)

        self.assertTrue(output.output_path.name.endswith("_sobel_edge.png"))
        self.assertIsNotNone(saved)
        self.assertGreater(np.count_nonzero(saved), 0)


if __name__ == "__main__":
    unittest.main()
