from __future__ import annotations

import unittest

import numpy as np

from sobel_fine_tune_gui import GuiTuneSettings, render_sobel_preview


class SobelFineTuneGuiTest(unittest.TestCase):
    def test_render_sobel_preview_returns_color_image(self) -> None:
        image = np.zeros((80, 120, 3), dtype=np.uint8)
        image[:, 50:70] = (255, 255, 255)
        settings = GuiTuneSettings(
            sobel_threshold_ratio=0.10,
            edge_close_kernel=3,
            edge_close_iterations=1,
            edge_dilate_iterations=0,
        )

        preview = render_sobel_preview(image, settings)

        self.assertEqual(preview.shape, image.shape)
        self.assertGreater(np.count_nonzero(preview), 0)
        yellow_pixels = np.all(preview == np.array([0, 255, 255], dtype=np.uint8), axis=2)
        self.assertFalse(np.any(yellow_pixels))
        self.assertTrue(np.array_equal(preview[:, :, 0], preview[:, :, 1]))
        self.assertTrue(np.array_equal(preview[:, :, 1], preview[:, :, 2]))


if __name__ == "__main__":
    unittest.main()
