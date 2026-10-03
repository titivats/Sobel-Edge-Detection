"""Qt workers for board intake, production inspection and model training."""

from __future__ import annotations

import time
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from .machine import Run
from .machine_reference import sha256
from .model import CropBox, CutClassifier, SobelConfig
from .overview import render_prediction_overview
from .production import PanelDecision, classify_panel
from .production_measurement import CaptureReferences, inspect_with_spec


class InspectionWorker(QThread):
    completed = Signal(object, object, float)
    failed = Signal(str)

    def __init__(
        self,
        classifier: CutClassifier,
        run: Run,
        expected_images: int,
        good_confidence_min: float,
        production_context=None,
    ):
        super().__init__()
        self.classifier = classifier
        self.target_run = run
        self.expected_images = expected_images
        self.good_confidence_min = good_confidence_min
        self.production_context = production_context
        self.evidence = {}

    def run(self) -> None:
        started = time.perf_counter()
        try:
            if self.production_context is not None:
                context = self.production_context
                if context.get("readiness_error"):
                    result = PanelDecision(run=self.target_run, note=context["readiness_error"])
                    self.evidence = {"model_stage": "NOT_RUN", "measurement_stage": "NOT_RUN"}
                else:
                    result, self.evidence = inspect_with_spec(
                        self.classifier,
                        self.target_run,
                        self.expected_images,
                        self.good_confidence_min,
                        context["limits"],
                        context.get("references")
                        or CaptureReferences(context["manifest_location"]),
                        context["recipe"],
                        context["edge_state"],
                        cancelled=self.isInterruptionRequested,
                    )
                self.evidence.update(model=context["model"], input_files=context["input_hashes"])
                if any(sha256(p) != h for p, h in context["input_hashes"].items()):
                    result.status, result.note = (
                        "FAULT",
                        "Panel input files changed during inspection.",
                    )
            else:
                result = classify_panel(
                    self.classifier,
                    self.target_run,
                    self.expected_images,
                    self.good_confidence_min,
                    cancelled=self.isInterruptionRequested,
                )
            measurement_by_path = {m["path"]: m for m in self.evidence.get("measurements", [])}
            captions = {}
            for detail in result.details:
                measured = measurement_by_path.get(detail.path)
                if measured:
                    values = measured.get("measurement", {})
                    inner, outer = values.get("inner_line_max_mm"), values.get("outer_line_max_mm")
                    mm_text = (
                        f"IN {inner:.4f} / OUT {outer:.4f} mm"
                        if inner is not None and outer is not None
                        else "MM N/A"
                    )
                    captions[detail.path] = (
                        f"CUT {detail.index + 1:02d} | {mm_text} | SPEC {measured['status']} | "
                    )
            overview = render_prediction_overview(
                self.classifier,
                result,
                self.good_confidence_min,
                tile_context=captions or None,
                tile_verdicts={path: entry["status"] for path, entry in measurement_by_path.items()}
                or None,
            )
        except Exception:
            self.failed.emit(traceback.format_exc(limit=6))
            return
        self.completed.emit(result, overview, (time.perf_counter() - started) * 1000.0)


class LiveScanWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, source, route, expected):
        super().__init__()
        self.source, self.route, self.expected = source, route, expected

    def run(self):
        try:
            panels = self.source.poll(self.route, self.expected)
        except Exception as exc:
            self.failed.emit(str(exc))
        else:
            self.completed.emit(panels)


class VisionTrialWorker(QThread):
    completed = Signal(object)
    failed = Signal(str)

    def __init__(self, classifier, image_paths: list[str]):
        super().__init__()
        self.classifier = classifier
        self.image_paths = list(image_paths)

    def run(self) -> None:
        try:
            predictions = self.classifier.predict(
                self.image_paths, cancelled=self.isInterruptionRequested
            )
            if len(predictions) != len(self.image_paths):
                raise RuntimeError("Vision Transformer returned an incomplete batch result.")
            results = []
            for image_path, prediction in zip(self.image_paths, predictions):
                label, confidence = prediction
                if not label:
                    raise RuntimeError(
                        f"Vision Transformer could not read {Path(image_path).name}."
                    )
                results.append((image_path, str(label), float(confidence)))
            self.completed.emit(results)
        except Exception as exc:
            self.failed.emit(str(exc))


class ModelTrainingWorker(QThread):
    """Train a product-only Sobel ViT model without blocking the settings UI."""

    completed = Signal(object, object)
    progress_changed = Signal(int, int, str)
    failed = Signal(str)

    def __init__(
        self,
        items: list[tuple[str, str]],
        sobel: SobelConfig,
        resume_from: CutClassifier | None = None,
        epochs: int = 300,
    ):
        super().__init__()
        self.items = list(items)
        self.sobel = sobel
        self.resume_from = resume_from
        self.epochs = epochs

    def run(self) -> None:
        try:
            previous = self.resume_from
            classifier = CutClassifier(
                backbone=previous.backbone if previous is not None else "dinov2_vits14",
                crop=previous.crop if previous is not None else CropBox(),
                sobel=self.sobel,
            )
            report = classifier.train(
                self.items,
                epochs=self.epochs,
                val_fraction=0.25,
                progress=self.progress_changed.emit,
                cancelled=self.isInterruptionRequested,
                # Always validate a fresh head. Reusing a prior head can leak
                # the new validation images through an older training run.
                resume_from=None,
            )
            if set(classifier.classes) != {"GOOD", "NG"}:
                raise RuntimeError("Training must contain both GOOD and NG classes.")
            self.completed.emit(classifier, report)
        except Exception as exc:
            self.failed.emit(str(exc))
