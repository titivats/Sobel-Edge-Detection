"""Run with the bundle's _runtime/python.exe and the portable folder as the argument."""

from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def exercise_model(bundle: Path, work: Path, *, public: bool = False) -> None:
    import cv2
    import numpy as np
    import torch
    from router_vision.model import CutClassifier

    torch.set_num_threads(2)
    pictures = []
    for index in range(4):
        rng = np.random.default_rng(index)
        image = rng.integers(0, 256, size=(960, 1080, 3), dtype=np.uint8)
        okay, png = cv2.imencode(".png", image)
        require(okay, "OpenCV could not encode the fixture.")
        picture = work / f"cut_{index}.png"
        picture.write_bytes(png.tobytes())
        pictures.append(str(picture))

    candidate = CutClassifier(model_key="portable-smoke-test")
    if public:
        require(
            not any(path.is_file() for path in (bundle / "models").rglob("*")),
            "Public package contains local model files.",
        )
    else:
        checkpoint = bundle / "models" / "cut_classifier_199944000-Optoput-Test.pt"
        classifier = CutClassifier.load(checkpoint, expected_model_key="199944000-Optoput-Test")
        label, score = classifier.predict([pictures[0]])[0]
        require(
            label in classifier.classes and math.isfinite(score), "Packaged model inference failed."
        )
        candidate.extractor = classifier.extractor
    items = [(p, "GOOD" if i < 2 else "NG") for i, p in enumerate(pictures)]
    report = candidate.train(items, epochs=1)
    require(report.n_train == 2 and report.n_val == 2, "Unexpected training split.")
    target = work / "candidate.pt"
    candidate.save(target)
    restored = CutClassifier.load(target, expected_model_key=candidate.model_key)
    require(restored.checkpoint_sha256 == candidate.checkpoint_sha256, "Candidate reload failed.")
    label, score = restored.predict([pictures[0]])[0]
    require(label in restored.classes and math.isfinite(score), "Trained model inference failed.")


def exercise_window(work: Path) -> None:
    import production_app as ui
    from PySide6.QtWidgets import QApplication

    app = QApplication([])
    with patch.multiple(
        ui,
        CONFIG_PATH=work / "config.json",
        SOBEL_WORKFLOW_PATH=work / "workflow.json",
        SOBEL_OUTPUT_DIR=work / "edges",
        DEMO_MODE=False,
    ):
        window = ui.MainWindow()
        try:
            window.show()
            app.processEvents()
            require(window.isVisible(), "The main window did not open.")
            require(
                not window.link.signals.release, "Empty settings unexpectedly released a board."
            )
        finally:
            window.close()
            app.processEvents()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--public", action="store_true", help="Verify a fresh public distribution.")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    runtime = bundle / "_runtime"
    require(Path(sys.executable).resolve() == runtime / "python.exe", "Use the bundled Python.")
    sys.dont_write_bytecode = True
    os.environ["TORCH_HOME"] = str(runtime / "torch")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    sys.path.insert(0, str(bundle))

    network_attempts = []

    def require_offline(event: str, _args: tuple) -> None:
        if event in {
            "socket.connect",
            "socket.getaddrinfo",
            "socket.gethostbyname",
            "socket.sendto",
        }:
            network_attempts.append(event)
            raise RuntimeError(f"Unexpected network access during offline test ({event})")

    sys.addaudithook(require_offline)
    with tempfile.TemporaryDirectory(prefix="AVTR-portable-test-") as directory:
        work = Path(directory)
        exercise_model(bundle, work, public=args.public)
        exercise_window(work)
    for name in ("cv2", "numpy", "torch", "PySide6", "dinov2"):
        origin = Path(sys.modules[name].__file__).resolve()
        require(origin.is_relative_to(runtime), f"{name} came from outside the bundle: {origin}")
    require(not network_attempts, f"Network attempts: {network_attempts}")
    print("Portable runtime verified: offline DINOv2 inference, training, model save/load and Qt.")


if __name__ == "__main__":
    main()
