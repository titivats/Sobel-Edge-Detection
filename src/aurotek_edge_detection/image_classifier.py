from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ClassificationResult:
    label: str
    confidence: float | None = None


class YoloImageClassifier:
    """Lazy-loaded YOLO image-classification model used by the dashboard."""

    def __init__(self) -> None:
        self._model = None
        self._model_path: Path | None = None
        self.last_error: str | None = None

    def reset(self) -> None:
        self._model = None
        self._model_path = None
        self.last_error = None

    def predict(self, image_path: Path, model_path: Path) -> ClassificationResult:
        if not model_path.exists():
            return ClassificationResult("UNKNOWN - model not found")
        if not image_path.exists():
            return ClassificationResult("UNKNOWN - image not found")

        try:
            model = self._load_model(model_path)
            results = model.predict(str(image_path), imgsz=640, verbose=False)
            result = results[0]
            if result.probs is None:
                return ClassificationResult("UNKNOWN - not classification model")

            class_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = str(result.names[class_id]).strip().upper()
            if class_name not in {"PASS", "NG"}:
                return ClassificationResult(f"UNKNOWN - {class_name}", confidence)
            return ClassificationResult(class_name, confidence)
        except Exception as exc:  # noqa: BLE001 - display safe error state in the dashboard.
            self.last_error = str(exc)
            return ClassificationResult("UNKNOWN - classifier error")

    def _load_model(self, model_path: Path):
        if self._model is not None and self._model_path == model_path:
            return self._model

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed. Install image-classification requirements first."
            ) from exc

        self._model = YOLO(str(model_path))
        self._model_path = model_path
        self.last_error = None
        return self._model
