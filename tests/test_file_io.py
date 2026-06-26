from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from file_io import atomic_write_image, atomic_write_text, read_image


class AtomicWriteTest(unittest.TestCase):
    def test_atomic_write_replaces_existing_text(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "settings.json"
            path.write_text("old", encoding="utf-8")

            atomic_write_text(path, "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_atomic_write_image_creates_readable_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "edge.png"
            image = np.full((8, 8), 255, dtype=np.uint8)

            atomic_write_image(path, image)

            saved = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            self.assertIsNotNone(saved)
            self.assertEqual(int(saved[0, 0]), 255)

    def test_read_image_supports_bmp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "camera image.bmp"
            expected = np.full((12, 16, 3), 127, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(path), expected))

            loaded = read_image(path)

            self.assertEqual(loaded.shape, expected.shape)
            self.assertTrue(np.array_equal(loaded, expected))


if __name__ == "__main__":
    unittest.main()
