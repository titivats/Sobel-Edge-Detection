from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from yolo_training import (
    _split_images,
    load_settings,
    validate_dataset,
)


class YoloTrainingTest(unittest.TestCase):
    def test_load_settings_rejects_missing_model_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "setting.txt").write_text(
                "[training]\n"
                "images=images\n"
                "labels=labels\n"
                "classes=classes.txt\n"
                "model=../missing.pt\n",
                encoding="utf-8",
            )

            with self.assertRaises(FileNotFoundError):
                load_settings(root / "setting.txt")

    def test_validate_dataset_tracks_multi_class_signatures(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (root / "classes.txt").write_text("NG\nPASS\n", encoding="utf-8")
            (root / "model.pt").write_bytes(b"model")
            for name in ("a", "b", "c", "d"):
                (images / f"{name}.png").write_bytes(b"image")
            (labels / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (labels / "b.txt").write_text("1 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (labels / "c.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
                encoding="utf-8",
            )
            (labels / "d.txt").write_text(
                "0 0.5 0.5 0.2 0.2\n1 0.4 0.4 0.1 0.1\n",
                encoding="utf-8",
            )
            setting_path = root / "setting.txt"
            setting_path.write_text(
                "[training]\n"
                "images=images\n"
                "labels=labels\n"
                "classes=classes.txt\n"
                "model=model.pt\n"
                "validation_split=0.5\n",
                encoding="utf-8",
            )

            settings = load_settings(setting_path)
            image_paths, _classes, counts, signatures = validate_dataset(settings)
            train, validation = _split_images(image_paths, signatures, 0.5, 42)

            self.assertEqual(counts, {0: 3, 1: 3})
            self.assertEqual(signatures[images / "c.png"], (0, 1))
            self.assertEqual(len(train), 3)
            self.assertEqual(len(validation), 1)


if __name__ == "__main__":
    unittest.main()
