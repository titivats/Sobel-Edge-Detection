"""AVTR production-focused Sobel-DINOv2 vision gate.

The conveyor link is intentionally simulation-only until a controls engineer
provides the real PLC protocol and I/O map.  The simulated adapter follows the
same fail-safe contract intended for hardware: no valid matching PASS means
HOLD. Metrology and baseline comparison are deliberately outside this gate.
"""

from __future__ import annotations

import hmac
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QLocale, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from router_vision.auo6000 import (
    AUO6000Dataset,
    AUO6000Image,
    scan_auo6000_dataset,
)
from router_vision.config import AppConfig, DEFAULT_CONFIG_NAME
from router_vision.guard import ProtectedPathError, check_write_target, protected_roots
from router_vision.interlock import GateState, SimulatedConveyorLink, VisionVerdict
from router_vision.machine import Run, attach_pictures, available_days, load_runs
from router_vision.model import (
    CropBox, CutClassifier, FeatureExtractor, SobelConfig, model_quality_error,
)
from router_vision.production import (
    STATUS_GOOD,
    STATUS_NG,
    ImageDecision,
    PanelDecision,
    classify_panel,
)


APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / DEFAULT_CONFIG_NAME
MODEL_PREFIX = "cut_classifier_"
APP_NAME = "AVTR"
APP_FULL_NAME = "Automatic Vision Tab Router"
RECENT_RESULT_LIMIT = 5
MIN_TRAINING_IMAGES_PER_CLASS = 5
DEMO_MODE = "--demo" in sys.argv
DEMO_AUTOSTART = "--autostart" in sys.argv
SETTINGS_PASSWORD = os.environ.get("AVTR_SETTINGS_PASSWORD", "").strip()
SOBEL_WORKFLOW_PATH = APP_DIR / "sobel_workflow.json"
SOBEL_OUTPUT_DIR = APP_DIR / "sobel_finetune"

# Legacy route metadata remains useful for known machines, but discovery is no
# longer limited to this list. New ProductIds are added from Result CSV files.
SUPPORTED = {
    "160914002D01|LeftTable": {
        "product": "160914002D01",
        "recipe": "160914002D01.rcp",
        "table": "LeftTable",
        "expected": 12,
    },
    "204860000E-R01|RightTable": {
        "product": "204860000E-R01",
        "recipe": "204860000E-R01.rcp",
        "table": "RightTable",
        "expected": 30,
    },
}

COLORS = {
    GateState.OFFLINE: "#475569",
    GateState.READY: "#1d4ed8",
    GateState.INSPECTING: "#c2410c",
    GateState.PASS: "#15803d",
    GateState.NG: "#dc2626",
    GateState.FAULT: "#dc2626",
}

APP_STYLE = """
QWidget { background:#eef2f7; color:#172033; font-family:'Segoe UI'; font-size:10.5pt; }
QFrame#topBar { background-color:#0f172a; border:0; border-bottom:3px solid #2563eb; }
QFrame#topBar QLabel { background:transparent; color:white; border:0; }
QFrame#card { background:white; border:1px solid #d8e0ea; border-radius:10px; }
QFrame#settingsPanel { background:#e7edf4; border:1px solid #94a3b8; border-radius:8px; }
QFrame#settingsHeader { background:#eaf2ff; border:1px solid #bfdbfe; border-radius:10px; }
QFrame#settingsHeader QLabel { background:transparent; border:0; }
QFrame#settingsCard { background:white; border:1px solid #b8c4d4; border-radius:6px; }
QFrame#settingsCard QLabel { background:transparent; border:0; }
QWidget#settingsPage, QFrame#settingsPage { background:#e7edf4; border:0; }
QFrame#pageHeader { background:#17324d; border:0; border-left:5px solid #2563eb; border-radius:5px; }
QFrame#pageHeader QLabel { background:transparent; border:0; }
QLabel#pageTitle { color:white; font-size:11pt; font-weight:800; }
QLabel#pageSubtitle { color:#cbd5e1; font-size:9pt; }
QFrame#controlDeck { background:white; border:1px solid #b8c4d4; border-radius:6px; }
QFrame#parameterCard { background:#f8fafc; border:1px solid #cbd5e1; border-radius:5px; }
QFrame#sourceCard { background:white; border:1px solid #b8c4d4; border-radius:6px; }
QLabel#sourceCaption { color:#64748b; font-size:9pt; font-weight:700; }
QLabel#sourceValue { color:#0f172a; font-size:10.5pt; font-weight:800; }
QTabWidget#settingsTabs { background:#e7edf4; border:0; }
QTabWidget#settingsTabs::pane { background:#e7edf4; border:0; top:0; }
QTabWidget#settingsTabs::tab-bar { left:12px; }
QTabWidget#settingsTabs QTabBar::tab { background:white; color:#334155; border:1px solid #aebccc; border-radius:5px; padding:11px 22px; margin:7px 6px 8px 0; font-weight:800; }
QTabWidget#settingsTabs QTabBar::tab:selected { background:#1d4ed8; color:white; border-color:#1d4ed8; }
QTabWidget#settingsTabs QTabBar::tab:hover:!selected { background:#eff6ff; color:#1d4ed8; border-color:#60a5fa; }
QPushButton { background:#334155; color:white; border:0; border-radius:5px; padding:8px 13px; font-weight:700; }
QPushButton:hover { background:#1e293b; }
QPushButton:pressed { padding-top:9px; padding-bottom:7px; }
QPushButton:disabled { background:#cbd5e1; color:#f8fafc; }
QPushButton#primary { background:#16a34a; font-size:12pt; }
QPushButton#primary:hover { background:#15803d; }
QPushButton#testAction { background:#2563eb; font-size:11pt; }
QPushButton#testAction:hover { background:#1d4ed8; }
QPushButton#saveAction { background:#16a34a; font-size:11pt; }
QPushButton#saveAction:hover { background:#15803d; }
QPushButton#saveSobel { background:#16a34a; }
QPushButton#saveSobel:hover { background:#15803d; }
QPushButton#hold { background:#dc2626; font-size:12pt; }
QPushButton#hold:hover { background:#b91c1c; }
QPushButton#outline { background:white; color:#1d4ed8; border:1px solid #2563eb; }
QPushButton#outline:hover { background:#eff6ff; }
QPushButton#secondaryAction { background:#f8fafc; color:#334155; border:1px solid #94a3b8; }
QPushButton#secondaryAction:hover { background:#e2e8f0; }
QPushButton#nextWork { background:#fff7ed; color:#9a3412; border:1px solid #f59e0b; }
QPushButton#nextWork:hover { background:#ffedd5; }
QPushButton#settings { background:#e2e8f0; color:#334155; border:1px solid #cbd5e1; }
QPushButton#stepArrow { background:#e2e8f0; color:#1d4ed8; border:1px solid #b8c4d4; border-radius:6px; padding:4px; font-weight:900; }
QPushButton#stepArrow:hover { background:#dbeafe; }
QComboBox, QDoubleSpinBox, QLineEdit { background:white; border:1px solid #b8c4d4; border-radius:6px; padding:7px 10px; }
QComboBox:focus, QDoubleSpinBox:focus, QLineEdit:focus { border:2px solid #2563eb; }
QSlider::groove:horizontal { background:#cbd5e1; height:7px; border-radius:3px; }
QSlider::sub-page:horizontal { background:#2563eb; border-radius:3px; }
QSlider::handle:horizontal { background:white; border:2px solid #2563eb; width:18px; margin:-7px 0; border-radius:10px; }
QTableWidget { background:white; alternate-background-color:#f8fafc; border:1px solid #d8e0ea; border-radius:7px; gridline-color:#e7edf4; }
QTableWidget::item { padding:6px; }
QTableWidget::item:selected { background:#dbeafe; color:#172033; }
QHeaderView::section { background:#eaf0f7; color:#334155; padding:8px; border:0; border-right:1px solid #d8e0ea; font-weight:700; }
QProgressBar { background:#dbe3ed; border:0; border-radius:4px; height:8px; }
QProgressBar::chunk { background:#2563eb; border-radius:4px; }
QSplitter::handle { background:#cbd5e1; }
QSplitter::handle:horizontal { width:5px; margin:0 2px; }
QScrollBar:vertical { background:#e2e8f0; width:12px; margin:0; }
QScrollBar::handle:vertical { background:#94a3b8; border-radius:5px; min-height:32px; margin:2px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height:0; }
"""

STATE_TEXT = {
    GateState.OFFLINE: "OFFLINE",
    GateState.READY: "READY",
    GateState.INSPECTING: "INSPECTING",
    GateState.PASS: "GOOD",
    GateState.NG: "NG",
    GateState.FAULT: "NG",
}


def bgr_to_pixmap(image) -> QPixmap:
    height, width, channels = image.shape
    qimage = QImage(image.data, width, height, channels * width, QImage.Format_BGR888)
    return QPixmap.fromImage(qimage.copy())


def _letterbox(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Fit an image into a fixed dark canvas without changing its aspect ratio."""
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    if image is None or image.size == 0:
        cv2.putText(
            canvas, "UNREADABLE", (max(8, width // 2 - 54), height // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1, cv2.LINE_AA,
        )
        return canvas
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    source_h, source_w = image.shape[:2]
    scale = min(width / max(1, source_w), height / max(1, source_h))
    target_w = max(1, int(round(source_w * scale)))
    target_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    x0 = (width - target_w) // 2
    y0 = (height - target_h) // 2
    canvas[y0:y0 + target_h, x0:x0 + target_w] = resized
    return canvas


def render_prediction_overview(
    classifier,
    result: PanelDecision,
    good_confidence_min: float = 0.95,
    *,
    cell_width: int = 360,
    cell_height: int = 220,
    tile_context: dict[str, str] | None = None,
    columns: int | None = None,
) -> np.ndarray:
    """Render every cut point into one Original/Sobel/prediction overview.

    Each cut-point tile always reserves the left pane for the cropped original
    image and the right pane for the exact Sobel edge map used by the model.
    The prediction is written beside the cut-point name in the top strip so it
    never covers evidence in either image pane.
    """
    details = sorted(result.details, key=lambda item: item.index)
    count = len(details)
    if not count:
        empty = np.full((240, 720, 3), 24, dtype=np.uint8)
        cv2.putText(
            empty, "NO CUT-POINT IMAGES AVAILABLE", (118, 128),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (210, 210, 210), 2, cv2.LINE_AA,
        )
        return empty

    cell_width = max(260, int(cell_width))
    cell_height = max(180, int(cell_height))
    columns = (
        min(6, max(1, math.ceil(math.sqrt(count))))
        if columns is None else max(1, min(int(columns), count))
    )
    rows = math.ceil(count / columns)
    gap = 10
    overview_width = columns * cell_width + (columns + 1) * gap
    overview_height = rows * cell_height + (rows + 1) * gap
    overview = np.full((overview_height, overview_width, 3), 20, dtype=np.uint8)

    for slot, detail in enumerate(details):
        row, column = divmod(slot, columns)
        x0 = gap + column * (cell_width + gap)
        y0 = gap + row * (cell_height + gap)
        tile = np.full((cell_height, cell_width, 3), 32, dtype=np.uint8)

        original = cv2.imread(str(detail.path), cv2.IMREAD_COLOR)
        if original is not None:
            original = classifier.crop.apply(original)
        edge = classifier.extractor.edge_map(detail.path, classifier.crop)

        top = 31
        bottom = 24
        pane_gap = 4
        pane_width = (cell_width - pane_gap) // 2
        pane_height = cell_height - top - bottom
        tile[top:top + pane_height, :pane_width] = _letterbox(
            original, pane_width, pane_height)
        tile[top:top + pane_height, pane_width + pane_gap:] = _letterbox(
            edge, cell_width - pane_width - pane_gap, pane_height)

        label = detail.label or "UNREADABLE"
        confidence = float(detail.confidence)
        if label == "GOOD" and confidence >= good_confidence_min:
            colour = (45, 145, 55)
        elif label == "NG":
            colour = (40, 40, 215)
        else:
            colour = (0, 140, 230)

        point_text = (
            tile_context.get(detail.path, f"CUT POINT {detail.index + 1:02d} | ")
            if tile_context else f"CUT POINT {detail.index + 1:02d} | "
        )
        prediction_text = f"PREDICT: {label} {confidence:.1%}"
        if label == "GOOD" and confidence < good_confidence_min:
            prediction_text += " | BELOW THRESHOLD"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        while font_scale > 0.32:
            point_width = cv2.getTextSize(point_text, font, font_scale, 1)[0][0]
            prediction_width = cv2.getTextSize(
                prediction_text, font, font_scale, 1
            )[0][0]
            if point_width + prediction_width <= cell_width - 18:
                break
            font_scale -= 0.02
        point_width = cv2.getTextSize(point_text, font, font_scale, 1)[0][0]
        header_colour = (
            (80, 220, 95) if label == "GOOD" and confidence >= good_confidence_min
            else (70, 80, 245) if label == "NG"
            else (30, 175, 245)
        )
        cv2.putText(
            tile, point_text, (9, 22), font, font_scale,
            (238, 238, 238), 1, cv2.LINE_AA,
        )
        cv2.putText(
            tile, prediction_text, (9 + point_width, 22), font, font_scale,
            header_colour, 1, cv2.LINE_AA,
        )
        cv2.putText(
            tile, "ORIGINAL", (8, cell_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 205, 205), 1, cv2.LINE_AA,
        )
        cv2.putText(
            tile, "SOBEL EDGE", (pane_width + pane_gap + 8, cell_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (205, 205, 205), 1, cv2.LINE_AA,
        )

        cv2.rectangle(tile, (0, 0), (cell_width - 1, cell_height - 1), colour, 3)
        overview[y0:y0 + cell_height, x0:x0 + cell_width] = tile

    return overview


def panel_id(run: Run) -> str:
    return run.sn.strip() or run.run_id.strip() or run.result_file


def model_file_name(key: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return f"{MODEL_PREFIX}{safe}.pt"


def product_from_route_key(key: str) -> str:
    """Return ProductId from either ``ProductId`` or ``ProductId|Table``."""
    return str(key).split("|", 1)[0].strip()


def inferred_expected_images(runs: list[Run]) -> int:
    """Infer the normal image count while incomplete runs still fail closed."""
    counts = Counter(len(run.pictures) for run in runs if run.pictures)
    if not counts:
        return 0
    most_common = counts.most_common()
    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        return 0
    return int(most_common[0][0])


class DemoClassifier:
    """Scripted UI/interlock demo; it is deliberately not a trained model."""

    classes = ["GOOD", "NG"]

    def __init__(self):
        self.crop = CropBox()
        self.extractor = FeatureExtractor(device="cpu")
        self.calls = 0

    def predict(self, paths: list[str], progress=None, cancelled=None):
        if cancelled and cancelled():
            raise InterruptedError("Demo cancelled")
        phase = self.calls % 3
        self.calls += 1
        predictions = [("GOOD", 0.985) for _ in paths]
        if paths and phase == 1:
            predictions[len(paths) // 2] = ("NG", 0.91)
        elif paths and phase == 2:
            predictions[len(paths) // 2] = ("GOOD", 0.72)
        return predictions


def verdict_for(result: PanelDecision) -> tuple[VisionVerdict, str]:
    """Convert inspection output into the gate's deliberately strict verdict."""
    if not result.run.passed:
        return VisionVerdict.NG, result.run.message or "router reported FAIL"
    if result.status == STATUS_GOOD:
        return VisionVerdict.PASS, result.note or f"all {result.checked} images GOOD"
    if result.status == STATUS_NG:
        return VisionVerdict.NG, result.note or result.status
    return VisionVerdict.FAULT, result.note or result.status


class InspectionWorker(QThread):
    completed = Signal(object, object, float)
    failed = Signal(str)

    def __init__(self, classifier: CutClassifier, run: Run, expected_images: int,
                 good_confidence_min: float):
        super().__init__()
        self.classifier = classifier
        self.target_run = run
        self.expected_images = expected_images
        self.good_confidence_min = good_confidence_min

    def run(self) -> None:
        started = time.perf_counter()
        try:
            result = classify_panel(
                self.classifier,
                self.target_run,
                self.expected_images,
                self.good_confidence_min,
                cancelled=self.isInterruptionRequested,
            )
            overview = render_prediction_overview(
                self.classifier, result, self.good_confidence_min)
        except Exception:
            self.failed.emit(traceback.format_exc(limit=6))
            return
        self.completed.emit(
            result, overview, (time.perf_counter() - started) * 1000.0)


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
                raise RuntimeError(
                    "Vision Transformer returned an incomplete batch result."
                )
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


class ResultImageDialog(QDialog):
    """Large fit/actual-size viewer for a completed inspection montage."""

    def __init__(
        self,
        panel: str,
        result: PanelDecision,
        reason: str,
        pixmap: QPixmap,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.original_pixmap = pixmap
        self.fit_mode = True
        self.setWindowTitle(f"Inspection Images - Panel {panel}")
        self.setMinimumSize(900, 620)
        self.resize(1280, 820)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        title = QLabel(f"PANEL {panel}  |  RESULT {result.status}")
        title.setFont(QFont("Segoe UI Semibold", 16))
        title.setStyleSheet("color:#0f172a;")
        layout.addWidget(title)

        summary = QLabel(
            f"{result.checked}/{len(result.details)} cut points checked  |  {reason}"
        )
        summary.setWordWrap(True)
        summary.setStyleSheet("color:#475569; padding-bottom:4px;")
        layout.addWidget(summary)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(False)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet(
            "QScrollArea { background:#0b1220; border:1px solid #334155; "
            "border-radius:10px; }"
        )
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setStyleSheet("background:#0b1220;")
        self.scroll.setWidget(self.preview)
        layout.addWidget(self.scroll, 1)

        actions = QHBoxLayout()
        fit_button = QPushButton("FIT OVERVIEW")
        fit_button.setObjectName("outline")
        fit_button.clicked.connect(self._fit_overview)
        actual_button = QPushButton("ACTUAL SIZE")
        actual_button.setObjectName("outline")
        actual_button.clicked.connect(self._show_actual_size)
        close_button = QPushButton("CLOSE")
        close_button.clicked.connect(self.close)
        actions.addStretch(1)
        actions.addWidget(fit_button)
        actions.addWidget(actual_button)
        actions.addWidget(close_button)
        layout.addLayout(actions)
        QTimer.singleShot(0, self._fit_overview)

    def _fit_overview(self) -> None:
        self.fit_mode = True
        size = self.scroll.viewport().size()
        fitted = self.original_pixmap.scaled(
            max(1, size.width() - 6),
            max(1, size.height() - 6),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview.setPixmap(fitted)
        self.preview.resize(fitted.size())

    def _show_actual_size(self) -> None:
        self.fit_mode = False
        self.preview.setPixmap(self.original_pixmap)
        self.preview.resize(self.original_pixmap.size())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self.fit_mode:
            QTimer.singleShot(0, self._fit_overview)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            f"{APP_FULL_NAME} — SCRIPTED DEMO" if DEMO_MODE else APP_FULL_NAME
        )
        self.resize(1440, 900)
        self.setMinimumSize(960, 640)

        self.cfg = AppConfig.load(CONFIG_PATH)
        self.link = SimulatedConveyorLink()
        self.link.connect()
        self.classifier: CutClassifier | None = None
        self.active_model_path: Path | None = None
        self.model_error = "model not loaded"
        # The production classifier follows the recipe selected on the operator
        # screen.  Settings can point at a different AUO6000 export, so its trial
        # classifier must be resolved independently to avoid cross-product tests.
        self.settings_classifier: CutClassifier | DemoClassifier | None = None
        self.settings_model_error = "select an AUO6000 data source"
        self.settings_model_name = ""

        self.runs_by_key: dict[str, list[Run]] = defaultdict(list)
        self.recipe_specs: dict[str, dict[str, object]] = {}
        self.queue: list[Run] = []
        self.queue_index = 0
        self.current_run: Run | None = None
        self.current_result: PanelDecision | None = None
        self.worker: InspectionWorker | None = None
        self.settings_worker: VisionTrialWorker | None = None
        self.training_worker: ModelTrainingWorker | None = None
        self.training_paths_in_progress: list[str] = []
        self.training_model_target: Path | None = None
        self.training_last_message = ""
        self.training_last_success: bool | None = None
        self.auto_running = False
        self.image_pixmap: QPixmap | None = None
        self.settings_preview_pixmap: QPixmap | None = None
        self.trial_preview_pixmap: QPixmap | None = None
        self.trial_result: PanelDecision | None = None
        self.trial_reason = ""
        self.trial_predictions: dict[str, tuple[str, float]] = {}
        self.settings_trial_approved = False
        self.trial_input_signature = None
        self.training_input_signatures: dict[str, dict] = {}
        self.closing_requested = False
        self.settings_image_dir = self._load_sobel_source_path()
        self.settings_image_path = ""
        self.settings_image_paths: list[str] = []
        self.auo_dataset: AUO6000Dataset | None = None
        self.settings_image_metadata: dict[str, AUO6000Image] = {}
        self.sobel_records = self._load_sobel_records()
        self._loading_sobel_parameters = False
        self.sobel_preview_timer = QTimer(self)
        self.sobel_preview_timer.setSingleShot(True)
        self.sobel_preview_timer.setInterval(80)
        self.sobel_preview_timer.timeout.connect(self._preview_sobel)
        self.result_images: dict[int, tuple[PanelDecision, str, QPixmap]] = {}
        self.result_sequence = 0
        self.result_dialog: ResultImageDialog | None = None

        self._build_ui()
        if CONFIG_PATH.is_file():
            self._load_dataset()
        else:
            self.link.fault("Select a data source in SETTING before inspection")

        self.heartbeat_timer = QTimer(self)
        self.heartbeat_timer.timeout.connect(self._heartbeat)
        self.heartbeat_timer.start(500)
        self.auto_timer = QTimer(self)
        self.auto_timer.setSingleShot(True)
        self.auto_timer.timeout.connect(self._auto_tick)

        if DEMO_MODE:
            self.spin_interval.setValue(4.0)
            self.lbl_detail.setText(
                "DEMO: Sobel images are real; GOOD/NG predictions are scripted, not model output."
            )
            if DEMO_AUTOSTART:
                QTimer.singleShot(1_200, self._toggle_auto)

        self._refresh_gate()

    # -- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        page = QWidget()
        root = QVBoxLayout(page)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(58)
        header = QHBoxLayout(top_bar)
        header.setContentsMargins(20, 7, 20, 7)
        title = QLabel(
            "AVTR - Automatic Vision Tab Router | "
            "Sobel Edge Detection with Vision Transformer"
        )
        title.setFont(QFont("Segoe UI Semibold", 13))
        title.setStyleSheet("color:#ffffff; letter-spacing:1px;")
        header.addWidget(title)
        header.addStretch(1)
        self.lbl_datetime = QLabel("--:--:--  |  00-00-0000")
        self.lbl_datetime.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lbl_datetime.setFont(QFont("Segoe UI Semibold", 10))
        self.lbl_datetime.setStyleSheet("color:#cbd5e1;")
        header.addWidget(self.lbl_datetime)
        mode = QLabel(
            "SCRIPTED DEMO | NO PLC OUTPUT" if DEMO_MODE
            else "DATA REVIEW | PLC OUTPUT DISABLED"
        )
        mode.setStyleSheet(
            "background:#dbeafe; color:#1e40af; border:1px solid #60a5fa;"
            "border-radius:7px; padding:7px 10px; margin-left:12px; font-weight:800;"
        )
        header.addWidget(mode)
        root.addWidget(top_bar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(10, 8, 10, 9)
        content_layout.setSpacing(6)

        self.state_banner = QFrame()
        self.state_banner.setMinimumHeight(56)
        banner_layout = QHBoxLayout(self.state_banner)
        banner_layout.setContentsMargins(18, 7, 18, 7)
        self.lbl_state_big = QLabel("READY")
        self.lbl_state_big.setAlignment(Qt.AlignCenter)
        self.lbl_state_big.setFont(QFont("Segoe UI Semibold", 25))
        banner_layout.addWidget(self.lbl_state_big, 1)
        content_layout.addWidget(self.state_banner)

        controls = QFrame()
        controls.setObjectName("card")
        control_layout = QHBoxLayout(controls)
        control_layout.setContentsMargins(8, 5, 8, 5)
        control_layout.setSpacing(6)
        self.lbl_active_product = QLabel("Preparing product...")
        self.lbl_active_product.setStyleSheet(
            "color:#1d4ed8; font-weight:800; font-size:10.5pt; padding-right:8px;")
        self.cmb_recipe = QComboBox()
        self.cmb_recipe.setMinimumWidth(245)
        self.cmb_recipe.currentTextChanged.connect(self._recipe_changed)
        self.cmb_day = QComboBox()
        self.cmb_day.setMinimumWidth(145)
        self.cmb_day.currentTextChanged.connect(self._rebuild_queue)
        self.btn_auto = QPushButton("▶ START AUTO")
        self.btn_auto.setObjectName("primary")
        self.btn_auto.setMinimumWidth(130)
        self.btn_auto.setMinimumHeight(34)
        self.btn_auto.clicked.connect(self._toggle_auto)
        self.btn_ack = QPushButton("↻ ACK / RESET")
        self.btn_ack.setObjectName("outline")
        self.btn_ack.setMinimumHeight(34)
        self.btn_ack.clicked.connect(self._acknowledge)
        self.btn_hold = QPushButton("■ HOLD")
        self.btn_hold.setObjectName("hold")
        self.btn_hold.setMinimumHeight(34)
        self.btn_hold.clicked.connect(lambda: self._software_hold())
        self.btn_settings = QPushButton("SETTING ▾")
        self.btn_settings.setObjectName("settings")
        self.btn_settings.clicked.connect(self._toggle_settings)
        for widget in (self.btn_auto, self.btn_ack, self.btn_hold):
            control_layout.addWidget(widget)
        active_separator = QLabel("|")
        active_separator.setStyleSheet(
            "color:#94a3b8; font-size:16pt; font-weight:300; padding:0 3px;"
        )
        control_layout.addWidget(active_separator)
        control_layout.addWidget(self.lbl_active_product)

        control_layout.addStretch(1)
        control_layout.addWidget(self.btn_settings)
        content_layout.addWidget(controls)

        self.settings_panel = QFrame()
        self.settings_panel.setObjectName("settingsPanel")
        settings_layout = QVBoxLayout(self.settings_panel)
        settings_layout.setContentsMargins(12, 8, 12, 8)
        settings_layout.setSpacing(7)

        self.advanced_panel = QFrame(self.settings_panel)
        self.advanced_panel.setObjectName("settingsCard")
        advanced_layout = QVBoxLayout(self.advanced_panel)
        advanced_layout.setContentsMargins(12, 10, 12, 10)
        advanced_layout.setSpacing(8)
        advanced_title = QLabel("ADVANCED MODEL & MACHINE OPTIONS")
        advanced_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        advanced_layout.addWidget(advanced_title)
        production_controls = QHBoxLayout()
        self.btn_model = QPushButton("RELOAD MODEL")
        self.btn_model.clicked.connect(self._reload_model)
        self.btn_next = QPushButton("INSPECT ONE PANEL")
        self.btn_next.clicked.connect(self._inspect_next)
        self.spin_interval = QDoubleSpinBox()
        self.spin_interval.setRange(0.5, 30.0)
        self.spin_interval.setValue(1.0)
        self.spin_interval.setSuffix(" seconds after PASS")
        self.lbl_dataset = QLabel("")
        self.lbl_dataset.setStyleSheet("color:#4b5563;")
        self.lbl_dataset.setWordWrap(True)
        for widget in (
            QLabel("Product / Table"),
            self.cmb_recipe,
            QLabel("Data date"),
            self.cmb_day,
            self.btn_model,
            self.btn_next,
            self.spin_interval,
        ):
            production_controls.addWidget(widget)
        production_controls.addStretch(1)
        advanced_layout.addLayout(production_controls)
        self.lbl_dataset.setStyleSheet(
            "background:#f1f5f9; color:#475569; border-radius:6px; padding:6px;"
        )
        advanced_layout.addWidget(self.lbl_dataset)

        self.slider_sobel_blur = QSlider(Qt.Horizontal)
        self.slider_sobel_blur.setRange(0, 3)
        self.slider_sobel_blur.setSingleStep(1)
        self.slider_sobel_blur.setPageStep(1)
        self.slider_blur_sigma = QSlider(Qt.Horizontal)
        self.slider_blur_sigma.setRange(0, 500)
        self.slider_blur_sigma.setSingleStep(1)
        self.slider_blur_sigma.setPageStep(10)
        self.slider_sobel_kernel = QSlider(Qt.Horizontal)
        self.slider_sobel_kernel.setRange(0, 3)
        self.slider_sobel_kernel.setSingleStep(1)
        self.slider_sobel_kernel.setPageStep(1)
        self.slider_gradient_x = QSlider(Qt.Horizontal)
        self.slider_gradient_x.setRange(0, 2000)
        self.slider_gradient_x.setSingleStep(10)
        self.slider_gradient_x.setPageStep(100)
        self.slider_gradient_y = QSlider(Qt.Horizontal)
        self.slider_gradient_y.setRange(0, 2000)
        self.slider_gradient_y.setSingleStep(10)
        self.slider_gradient_y.setPageStep(100)
        self.slider_sobel_clip = QSlider(Qt.Horizontal)
        self.slider_sobel_clip.setRange(9000, 10000)
        self.slider_sobel_clip.setSingleStep(1)
        self.slider_sobel_clip.setPageStep(10)
        self.slider_edge_gain = QSlider(Qt.Horizontal)
        self.slider_edge_gain.setRange(500, 3000)
        self.slider_edge_gain.setSingleStep(10)
        self.slider_edge_gain.setPageStep(100)
        self.slider_noise_floor = QSlider(Qt.Horizontal)
        self.slider_noise_floor.setRange(0, 100)
        self.slider_noise_floor.setSingleStep(1)
        self.slider_noise_floor.setPageStep(5)
        self.lbl_sobel_blur_value = QLabel("3")
        self.lbl_blur_sigma_value = QLabel("AUTO")
        self.lbl_sobel_kernel_value = QLabel("3")
        self.lbl_gradient_x_value = QLabel("1.00")
        self.lbl_gradient_y_value = QLabel("1.00")
        self.lbl_sobel_clip_value = QLabel("99.50 %")
        self.lbl_edge_gain_value = QLabel("1.00×")
        self.lbl_noise_floor_value = QLabel("0")
        for value_label in (
            self.lbl_sobel_blur_value,
            self.lbl_blur_sigma_value,
            self.lbl_sobel_kernel_value,
            self.lbl_gradient_x_value,
            self.lbl_gradient_y_value,
            self.lbl_sobel_clip_value,
            self.lbl_edge_gain_value,
            self.lbl_noise_floor_value,
        ):
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setMinimumWidth(62)
            value_label.setStyleSheet(
                "background:#dbeafe; color:#1d4ed8; border-radius:6px; "
                "padding:4px 8px; font-weight:800;"
            )
        for slider in (
            self.slider_sobel_blur,
            self.slider_blur_sigma,
            self.slider_sobel_kernel,
            self.slider_gradient_x,
            self.slider_gradient_y,
            self.slider_sobel_clip,
            self.slider_edge_gain,
            self.slider_noise_floor,
        ):
            slider.valueChanged.connect(self._sobel_parameter_changed)
        self.spin_confidence = QDoubleSpinBox()
        self.spin_confidence.setRange(50.0, 100.0)
        self.spin_confidence.setDecimals(1)
        self.spin_confidence.setSingleStep(0.5)
        self.spin_confidence.setSuffix(" % GOOD threshold")
        self.btn_select_image = QPushButton("SELECT AUO6000 PATH")
        self.btn_select_image.clicked.connect(self._select_settings_path)
        self.btn_trial = QPushButton("TEST VISION TRANSFORMER")
        self.btn_trial.setObjectName("testAction")
        self.btn_trial.clicked.connect(self._test_vision_transformer)
        self.btn_apply_settings = QPushButton("SAVE SETTINGS")
        self.btn_apply_settings.setObjectName("saveAction")
        self.btn_apply_settings.setEnabled(False)
        self.btn_apply_settings.clicked.connect(self._apply_settings)
        self.spin_confidence.valueChanged.connect(self._invalidate_settings_trial)

        self.cmb_settings_image = QComboBox()
        self.cmb_settings_image.setEnabled(False)
        self.cmb_settings_image.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_settings_image.setMinimumContentsLength(24)
        self.cmb_settings_image.currentIndexChanged.connect(
            self._settings_image_changed
        )
        self.txt_settings_path = QLineEdit()
        self.txt_settings_path.setReadOnly(True)
        self.txt_settings_path.setPlaceholderText(
            "Select the AUO6000 export root containing Picture, Result, Log and Recipe"
        )
        self.lbl_settings_image_count = QLabel("0 PANELS | 0 IMAGES")
        self.lbl_settings_image_count.setStyleSheet(
            "background:#eaf2ff; color:#1d4ed8; border:1px solid #bfdbfe; "
            "border-radius:6px; padding:8px; font-weight:700;"
        )
        self.btn_previous_image = QPushButton("◀ PREVIOUS")
        self.btn_previous_image.setObjectName("outline")
        self.btn_previous_image.clicked.connect(self._previous_settings_image)
        self.btn_next_image = QPushButton("NEXT ▶")
        self.btn_next_image.setObjectName("outline")
        self.btn_next_image.clicked.connect(self._next_settings_image)
        self.btn_next_unsaved = QPushButton("NEXT NOT SAVED")
        self.btn_next_unsaved.setObjectName("nextWork")
        self.btn_next_unsaved.clicked.connect(self._next_unsaved_image)
        self.btn_reset_sobel = QPushButton("RESET PARAMETERS")
        self.btn_reset_sobel.setObjectName("secondaryAction")
        self.btn_reset_sobel.clicked.connect(self._load_current_image_parameters)
        self.btn_save_sobel = QPushButton("SAVE CURRENT SOBEL")
        self.btn_save_sobel.setObjectName("saveSobel")
        self.btn_save_sobel.clicked.connect(self._save_current_sobel)
        self.lbl_sobel_save_status = QLabel("SOBEL: NOT SAVED")
        self.lbl_training_status = QLabel("TRAINING: NOT TRAINED")
        for status_label in (
            self.lbl_sobel_save_status,
            self.lbl_training_status,
        ):
            status_label.setAlignment(Qt.AlignCenter)
            status_label.setMinimumWidth(145)
        self.lbl_workflow_progress = QLabel(
            "SOBEL 0/0 SAVED  |  TRAINED 0/0  |  0 LEFT"
        )
        self.lbl_workflow_progress.setStyleSheet(
            "background:#eff6ff; color:#1e40af; border:1px solid #bfdbfe; "
            "border-radius:6px; padding:6px 9px; font-weight:700;"
        )
        self.lbl_current_image_context = QLabel(
            "PANEL NOT ASSIGNED  |  CUT POINT --  |  MACHINE RESULT UNKNOWN"
        )
        self.lbl_current_image_context.setStyleSheet(
            "background:#0f172a; color:white; border-radius:6px; "
            "padding:7px 10px; font-weight:800;"
        )
        self.lbl_current_image_context.setAlignment(Qt.AlignCenter)
        for button in (
            self.btn_previous_image,
            self.btn_next_image,
            self.btn_next_unsaved,
            self.btn_reset_sobel,
            self.btn_save_sobel,
        ):
            button.setEnabled(False)
        self.lbl_settings_preview = QLabel(
            "Select an image to compare ORIGINAL and SOBEL EDGE"
        )
        self.lbl_settings_preview.setAlignment(Qt.AlignCenter)
        self.lbl_settings_preview.setMinimumHeight(280)
        self.lbl_settings_preview.setStyleSheet(
            "background:#0b1220; color:#cbd5e1; border:1px solid #334155; "
            "border-radius:7px;"
        )
        self.lbl_trial_result = QLabel("VISION TRANSFORMER: NOT TESTED")
        self.lbl_trial_result.setAlignment(Qt.AlignCenter)
        self.lbl_trial_result.setWordWrap(True)
        self.lbl_trial_result.setMinimumWidth(250)
        self.lbl_trial_result.setStyleSheet(
            "background:#e2e8f0; color:#334155; border-radius:7px; "
            "padding:10px; font-weight:800;"
        )
        self.lbl_trial_result.setMinimumHeight(54)
        self.lbl_trial_result.setMaximumHeight(92)

        self.settings_workflow = QTabWidget()
        self.settings_workflow.setObjectName("settingsTabs")
        self.settings_workflow.setDocumentMode(True)
        self.settings_workflow.tabBar().setExpanding(True)
        self.settings_workflow.tabBar().setUsesScrollButtons(False)

        source_page = QWidget()
        source_page.setObjectName("settingsPage")
        source_layout = QVBoxLayout(source_page)
        source_layout.setContentsMargins(14, 10, 14, 12)
        source_layout.setSpacing(10)
        source_header = QFrame()
        source_header.setObjectName("pageHeader")
        source_header_layout = QVBoxLayout(source_header)
        source_header_layout.setContentsMargins(12, 8, 12, 8)
        source_header_layout.setSpacing(2)
        step_one = QLabel("1. AUROTEK AUO6000 DATA PATH")
        step_one.setObjectName("pageTitle")
        step_one_help = QLabel(
            "Select one machine export root containing Picture, Result, Log and Recipe."
        )
        step_one_help.setObjectName("pageSubtitle")
        source_header_layout.addWidget(step_one)
        source_header_layout.addWidget(step_one_help)
        image_row = QHBoxLayout()
        image_row.addWidget(self.btn_select_image)
        image_row.addWidget(self.txt_settings_path, 1)
        image_row.addWidget(self.lbl_settings_image_count)
        source_layout.addWidget(source_header)
        source_layout.addLayout(image_row)

        self.lbl_auo_files = QLabel("SELECT AN AUO6000 DATA SOURCE TO CONTINUE")
        self.lbl_auo_files.setAlignment(Qt.AlignCenter)
        self.lbl_auo_files.setStyleSheet(
            "background:#e2e8f0; color:#475569; border-radius:7px; "
            "padding:10px 12px; font-weight:800;"
        )
        source_layout.addWidget(self.lbl_auo_files)

        self.auo_source_card = QFrame()
        self.auo_source_card.setObjectName("sourceCard")
        source_grid = QGridLayout(self.auo_source_card)
        source_grid.setContentsMargins(16, 12, 16, 12)
        source_grid.setHorizontalSpacing(14)
        source_grid.setVerticalSpacing(8)
        identity_title = QLabel("MACHINE INFORMATION")
        identity_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        inspection_title = QLabel("INSPECTION DATA")
        inspection_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        source_grid.addWidget(identity_title, 0, 0, 1, 2)
        source_grid.addWidget(inspection_title, 0, 2, 1, 2)

        self.lbl_auo_machine = QLabel("AUROTEK ROUTER AUO6000")
        self.lbl_auo_recipe = QLabel("--")
        self.lbl_auo_panels = QLabel("0")
        self.lbl_auo_cut_points = QLabel("--")
        self.lbl_auo_format = QLabel("--")
        self.lbl_auo_images = QLabel("0")
        detail_rows = (
            (1, "MACHINE", self.lbl_auo_machine, "PANELS", self.lbl_auo_panels),
            (2, "RECIPE", self.lbl_auo_recipe, "CUT POINTS / PANEL", self.lbl_auo_cut_points),
            (3, "IMAGE CONTENT", self.lbl_auo_format, "SOURCE IMAGES", self.lbl_auo_images),
        )
        for row, left_caption, left_value, right_caption, right_value in detail_rows:
            left_label = QLabel(left_caption)
            left_label.setObjectName("sourceCaption")
            left_value.setObjectName("sourceValue")
            right_label = QLabel(right_caption)
            right_label.setObjectName("sourceCaption")
            right_value.setObjectName("sourceValue")
            left_value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            source_grid.addWidget(left_label, row, 0)
            source_grid.addWidget(left_value, row, 1)
            source_grid.addWidget(right_label, row, 2)
            source_grid.addWidget(right_value, row, 3)
        source_grid.setColumnStretch(1, 2)
        source_grid.setColumnStretch(3, 1)
        source_layout.addWidget(self.auo_source_card)

        folders_title = QLabel("SOURCE FOLDERS")
        folders_title.setStyleSheet("color:#334155; font-weight:800;")
        source_layout.addWidget(folders_title)
        self.tbl_auo_folders = QTableWidget(4, 4)
        self.tbl_auo_folders.setHorizontalHeaderLabels(
            ["DATA", "STATUS", "FILES", "LOCATION"]
        )
        self.tbl_auo_folders.verticalHeader().setVisible(False)
        self.tbl_auo_folders.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tbl_auo_folders.setSelectionMode(QAbstractItemView.NoSelection)
        self.tbl_auo_folders.setFocusPolicy(Qt.NoFocus)
        self.tbl_auo_folders.setAlternatingRowColors(True)
        self.tbl_auo_folders.setFixedHeight(172)
        folder_header = self.tbl_auo_folders.horizontalHeader()
        folder_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        folder_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        folder_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        folder_header.setSectionResizeMode(3, QHeaderView.Stretch)
        source_layout.addWidget(self.tbl_auo_folders)

        self.lbl_auo_warning = QLabel("")
        self.lbl_auo_warning.setWordWrap(True)
        self.lbl_auo_warning.setStyleSheet(
            "background:#fff7ed; color:#9a3412; border:1px solid #fed7aa; "
            "border-radius:5px; padding:5px 8px; font-weight:700;"
        )
        self.lbl_auo_warning.setVisible(False)
        source_layout.addWidget(self.lbl_auo_warning)
        source_layout.addStretch(1)
        continue_row = QHBoxLayout()
        continue_row.addStretch(1)
        self.btn_continue_sobel = QPushButton("CONTINUE TO SOBEL TUNING")
        self.btn_continue_sobel.setObjectName("primary")
        self.btn_continue_sobel.setMinimumWidth(260)
        self.btn_continue_sobel.setEnabled(False)
        self.btn_continue_sobel.clicked.connect(
            lambda: self.settings_workflow.setCurrentIndex(1)
        )
        continue_row.addWidget(self.btn_continue_sobel)
        source_layout.addLayout(continue_row)
        self.settings_workflow.addTab(source_page, "1  DATA SOURCE")

        sobel_card = QWidget()
        sobel_card.setObjectName("settingsPage")
        sobel_layout = QVBoxLayout(sobel_card)
        sobel_layout.setContentsMargins(14, 10, 14, 12)
        sobel_layout.setSpacing(8)
        sobel_header_frame = QFrame()
        sobel_header_frame.setObjectName("pageHeader")
        sobel_header = QHBoxLayout()
        sobel_header.setContentsMargins(12, 8, 10, 8)
        step_two = QLabel("2. FINE-TUNE SOBEL EDGE DETECTION")
        step_two.setObjectName("pageTitle")
        sobel_header.addWidget(step_two)
        sobel_header.addStretch(1)
        sobel_header.addWidget(self.lbl_workflow_progress)
        sobel_header_frame.setLayout(sobel_header)
        sobel_layout.addWidget(sobel_header_frame)

        sobel_control_deck = QFrame()
        sobel_control_deck.setObjectName("controlDeck")
        sobel_control_layout = QVBoxLayout(sobel_control_deck)
        sobel_control_layout.setContentsMargins(9, 8, 9, 8)
        sobel_control_layout.setSpacing(7)
        image_navigation = QHBoxLayout()
        image_navigation.setSpacing(7)
        image_navigation.addWidget(self.cmb_settings_image, 1)
        image_navigation.addWidget(self.btn_previous_image)
        image_navigation.addWidget(self.btn_next_image)
        sobel_control_layout.addLayout(image_navigation)
        sobel_control_layout.addWidget(self.lbl_current_image_context)
        image_status = QHBoxLayout()
        image_status.setSpacing(7)
        image_status.addWidget(self.lbl_sobel_save_status)
        image_status.addWidget(self.lbl_training_status)
        image_status.addStretch(1)
        image_status.addWidget(self.btn_reset_sobel)
        image_status.addWidget(self.btn_next_unsaved)
        image_status.addWidget(self.btn_save_sobel)
        sobel_control_layout.addLayout(image_status)
        sobel_layout.addWidget(sobel_control_deck)

        sobel_workspace = QSplitter(Qt.Horizontal)
        sobel_workspace.setChildrenCollapsible(False)
        sobel_workspace.splitterMoved.connect(
            lambda *_: QTimer.singleShot(0, self._layout_sobel_parameter_cards)
        )

        preview_panel = QFrame()
        preview_panel.setObjectName("settingsCard")
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(10, 8, 10, 10)
        preview_title_row = QHBoxLayout()
        preview_title = QLabel("LIVE ORIGINAL / SOBEL COMPARISON")
        preview_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        preview_note = QLabel("Updates automatically")
        preview_note.setStyleSheet("color:#64748b; font-size:9pt;")
        preview_title_row.addWidget(preview_title)
        preview_title_row.addStretch(1)
        preview_title_row.addWidget(preview_note)
        preview_layout.addLayout(preview_title_row)
        preview_layout.addWidget(self.lbl_settings_preview, 1)

        parameter_panel = QFrame()
        parameter_panel.setObjectName("settingsCard")
        parameter_layout = QVBoxLayout(parameter_panel)
        parameter_layout.setContentsMargins(10, 8, 10, 10)
        parameter_title = QLabel("SOBEL PARAMETERS")
        parameter_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        parameter_hint = QLabel("Decimal controls use precise 0.01 steps.")
        parameter_hint.setStyleSheet("color:#64748b; font-size:9pt;")
        parameter_layout.addWidget(parameter_title)
        parameter_layout.addWidget(parameter_hint)
        parameter_scroll = QScrollArea()
        parameter_scroll.setWidgetResizable(True)
        parameter_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        parameter_scroll.setFrameShape(QFrame.NoFrame)
        parameter_content = QWidget()
        sobel_parameters = QGridLayout(parameter_content)
        sobel_parameters.setContentsMargins(0, 2, 0, 2)
        sobel_parameters.setHorizontalSpacing(8)
        sobel_parameters.setVerticalSpacing(8)
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "Blur size", self.slider_sobel_blur, self.lbl_sobel_blur_value,
                "Smooth small noise before detecting edges.",
            ),
            0,
            0,
        )
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "Blur strength", self.slider_blur_sigma,
                self.lbl_blur_sigma_value,
                "Gaussian sigma; AUTO lets OpenCV choose from the blur size.",
            ),
            0,
            1,
        )
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "Sobel kernel", self.slider_sobel_kernel,
                self.lbl_sobel_kernel_value,
                "Edge detector size: larger values produce broader edge response.",
            ),
            1,
            0,
        )
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "Edge normalization", self.slider_sobel_clip,
                self.lbl_sobel_clip_value,
                "Percentile used to normalize strong edges to white.",
            ),
            1,
            1,
        )
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "X edge weight", self.slider_gradient_x,
                self.lbl_gradient_x_value,
                "Weight for left/right intensity changes.",
            ),
            2,
            0,
        )
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "Y edge weight", self.slider_gradient_y,
                self.lbl_gradient_y_value,
                "Weight for top/bottom intensity changes.",
            ),
            2,
            1,
        )
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "Edge brightness", self.slider_edge_gain,
                self.lbl_edge_gain_value,
                "Brightness gain applied after edge normalization.",
            ),
            3,
            0,
        )
        sobel_parameters.addWidget(
            self._make_parameter_control(
                "Noise removal", self.slider_noise_floor,
                self.lbl_noise_floor_value,
                "Remove weak edge pixels below this intensity.",
            ),
            3,
            1,
        )
        self.sobel_parameter_scroll = parameter_scroll
        self.sobel_parameters_layout = sobel_parameters
        self.sobel_parameter_cards = [
            sobel_parameters.itemAtPosition(row, column).widget()
            for row in range(4)
            for column in range(2)
        ]
        self.sobel_parameter_columns = 2
        parameter_scroll.setWidget(parameter_content)
        parameter_layout.addWidget(parameter_scroll, 1)
        sobel_workspace.addWidget(preview_panel)
        sobel_workspace.addWidget(parameter_panel)
        sobel_workspace.setStretchFactor(0, 3)
        sobel_workspace.setStretchFactor(1, 2)
        sobel_workspace.setSizes([1100, 700])
        sobel_layout.addWidget(sobel_workspace, 1)
        self.settings_workflow.addTab(sobel_card, "2  SOBEL TUNING")

        decision_card = QFrame()
        decision_card.setObjectName("settingsPage")
        decision_layout = QVBoxLayout(decision_card)
        decision_layout.setContentsMargins(14, 10, 14, 12)
        decision_layout.setSpacing(8)
        decision_header = QFrame()
        decision_header.setObjectName("pageHeader")
        decision_header_layout = QVBoxLayout(decision_header)
        decision_header_layout.setContentsMargins(12, 8, 12, 8)
        decision_header_layout.setSpacing(2)
        step_three = QLabel("3. IMAGE CLASSIFICATION BY VISION TRANSFORMER")
        step_three.setObjectName("pageTitle")
        step_three_help = QLabel(
            "First review and save Sobel images in Tab 2. Then label GOOD/NG here, "
            "train one model for this ProductId, and test every image."
        )
        step_three_help.setWordWrap(True)
        step_three_help.setObjectName("pageSubtitle")
        decision_header_layout.addWidget(step_three)
        decision_header_layout.addWidget(step_three_help)
        self.lbl_trial_batch = QLabel("CURRENT BATCH: NO AUO6000 DATA SELECTED")
        self.lbl_trial_batch.setWordWrap(True)
        self.lbl_trial_batch.setStyleSheet(
            "background:#eff6ff; color:#1e40af; border:1px solid #bfdbfe; "
            "border-radius:7px; padding:9px 12px; font-weight:800;"
        )
        self.cmb_training_image = QComboBox()
        self.cmb_training_image.setMinimumContentsLength(24)
        self.cmb_training_image.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.cmb_training_image.currentIndexChanged.connect(
            self._training_image_changed
        )
        self.btn_training_previous = QPushButton("◀")
        self.btn_training_previous.setObjectName("outline")
        self.btn_training_previous.setToolTip("Previous image")
        self.btn_training_previous.clicked.connect(self._previous_training_image)
        self.btn_training_next = QPushButton("▶")
        self.btn_training_next.setObjectName("outline")
        self.btn_training_next.setToolTip("Next image")
        self.btn_training_next.clicked.connect(self._next_training_image)
        self.btn_next_unlabelled = QPushButton("NEXT UNLABELED")
        self.btn_next_unlabelled.setObjectName("outline")
        self.btn_next_unlabelled.clicked.connect(self._next_unlabelled_image)
        self.btn_label_good = QPushButton("LABEL GOOD")
        self.btn_label_good.setStyleSheet(
            "background:#16a34a; color:white; font-weight:800;"
        )
        self.btn_label_good.clicked.connect(
            lambda: self._set_training_label("GOOD")
        )
        self.btn_label_ng = QPushButton("LABEL NG")
        self.btn_label_ng.setStyleSheet(
            "background:#dc2626; color:white; font-weight:800;"
        )
        self.btn_label_ng.clicked.connect(lambda: self._set_training_label("NG"))
        self.btn_clear_label = QPushButton("CLEAR LABEL")
        self.btn_clear_label.setObjectName("outline")
        self.btn_clear_label.clicked.connect(lambda: self._set_training_label(""))
        self.lbl_training_image_context = QLabel("SELECT AN IMAGE TO LABEL")
        self.lbl_training_image_context.setWordWrap(True)
        self.lbl_training_image_context.setAlignment(Qt.AlignCenter)
        self.lbl_training_image_context.setStyleSheet(
            "background:#e2e8f0; color:#334155; border-radius:6px; "
            "padding:6px 8px; font-weight:800;"
        )
        self.lbl_label_progress = QLabel(
            "LABELS: GOOD 0 | NG 0 | UNLABELED 0 | TRAIN-READY: NO"
        )
        self.lbl_label_progress.setWordWrap(True)
        self.lbl_label_progress.setStyleSheet("color:#475569; font-weight:700;")

        self.btn_train_model = QPushButton("TRAIN & SAVE MODEL")
        self.btn_train_model.setObjectName("primary")
        self.btn_train_model.clicked.connect(self._train_settings_model)
        self.training_progress = QProgressBar()
        self.training_progress.setVisible(False)
        self.lbl_model_training_status = QLabel(
            "Label at least 5 saved images for each class."
        )
        self.lbl_model_training_status.setWordWrap(True)
        self.lbl_model_training_status.setStyleSheet("color:#64748b;")

        confidence_label = QLabel("GOOD confidence threshold")
        confidence_label.setStyleSheet("font-weight:700;")
        self.lbl_trial_preview = QLabel(
            "SELECT AN IMAGE AND ASSIGN A GOOD OR NG LABEL"
        )
        self.lbl_trial_preview.setAlignment(Qt.AlignCenter)
        self.lbl_trial_preview.setMinimumSize(1, 200)
        self.lbl_trial_preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.lbl_trial_preview.setStyleSheet(
            "background:#0b1220; color:#cbd5e1; border-radius:8px; "
            "font-weight:800;"
        )
        self.trial_preview_scroll = QScrollArea()
        self.trial_preview_scroll.setWidgetResizable(True)
        self.trial_preview_scroll.setAlignment(Qt.AlignCenter)
        self.trial_preview_scroll.setStyleSheet(
            "QScrollArea { background:#0b1220; border:1px solid #334155; "
            "border-radius:8px; }"
        )
        self.trial_preview_scroll.setWidget(self.lbl_trial_preview)
        decision_layout.addWidget(decision_header)
        decision_layout.addWidget(self.lbl_trial_batch)

        vision_workspace = QSplitter(Qt.Horizontal)
        self.vision_workspace = vision_workspace
        vision_workspace.setChildrenCollapsible(False)
        vision_workspace.splitterMoved.connect(lambda *_: self._fit_trial_preview())

        review_card = QFrame()
        review_card.setObjectName("sourceCard")
        review_layout = QVBoxLayout(review_card)
        review_layout.setContentsMargins(10, 9, 10, 10)
        review_layout.setSpacing(7)
        review_header = QHBoxLayout()
        review_title = QLabel("IMAGE REVIEW / PREDICTION RESULTS")
        review_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        review_hint = QLabel("Select an image or use the left / right arrows")
        review_title.setWordWrap(True)
        review_hint.setWordWrap(True)
        review_hint.setStyleSheet("color:#64748b; font-size:9pt;")
        review_header.addWidget(review_title)
        review_header.addStretch(1)
        review_header.addWidget(review_hint)
        review_layout.addLayout(review_header)
        review_layout.addWidget(self.lbl_trial_result)
        review_layout.addWidget(self.trial_preview_scroll, 1)

        label_card = QFrame()
        label_card.setObjectName("sourceCard")
        label_layout = QVBoxLayout(label_card)
        label_layout.setContentsMargins(11, 9, 11, 9)
        label_layout.setSpacing(6)
        label_title = QLabel("1. LABEL SOBEL IMAGES")
        label_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        label_layout.addWidget(label_title)
        label_selector = QHBoxLayout()
        label_selector.setSpacing(5)
        label_selector.addWidget(self.cmb_training_image, 1)
        label_selector.addWidget(self.btn_training_previous)
        label_selector.addWidget(self.btn_training_next)
        label_layout.addLayout(label_selector)
        label_layout.addWidget(self.lbl_training_image_context)
        label_primary_actions = QHBoxLayout()
        label_primary_actions.setSpacing(6)
        label_primary_actions.addWidget(self.btn_label_good)
        label_primary_actions.addWidget(self.btn_label_ng)
        label_layout.addLayout(label_primary_actions)
        label_secondary_actions = QHBoxLayout()
        label_secondary_actions.setSpacing(6)
        label_secondary_actions.addWidget(self.btn_clear_label)
        label_secondary_actions.addWidget(self.btn_next_unlabelled)
        label_layout.addLayout(label_secondary_actions)
        label_layout.addWidget(self.lbl_label_progress)

        train_card = QFrame()
        train_card.setObjectName("sourceCard")
        train_layout = QVBoxLayout(train_card)
        train_layout.setContentsMargins(11, 8, 11, 8)
        train_layout.setSpacing(6)
        train_title = QLabel("2. TRAIN MODEL")
        train_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        train_note = QLabel(
            "DINOv2 Small · current Sobel settings · 25% validation\n"
            "Only saved images matching the current Sobel settings are used.\n"
            "The first run may download the pretrained backbone."
        )
        train_note.setWordWrap(True)
        train_note.setStyleSheet("color:#64748b; font-size:9pt;")
        train_layout.addWidget(train_title)
        train_layout.addWidget(train_note)
        train_layout.addWidget(self.lbl_model_training_status)
        train_layout.addWidget(self.training_progress)
        train_layout.addWidget(self.btn_train_model)

        test_card = QFrame()
        test_card.setObjectName("sourceCard")
        test_layout = QVBoxLayout(test_card)
        test_layout.setContentsMargins(11, 8, 11, 8)
        test_layout.setSpacing(6)
        test_title = QLabel("3. TEST MODEL")
        test_title.setStyleSheet("color:#1d4ed8; font-weight:800;")
        test_note = QLabel("Testing uses the Sobel preprocessing saved in the model.")
        test_note.setWordWrap(True)
        test_note.setStyleSheet("color:#64748b; font-size:9pt;")
        test_layout.addWidget(test_title)
        test_layout.addWidget(confidence_label)
        test_layout.addWidget(self.spin_confidence)
        test_layout.addWidget(test_note)
        test_layout.addWidget(self.btn_trial)
        test_layout.addWidget(self.btn_apply_settings)

        workflow_content = QWidget()
        workflow_layout = QVBoxLayout(workflow_content)
        workflow_layout.setContentsMargins(1, 1, 5, 1)
        workflow_layout.setSpacing(8)
        workflow_layout.addWidget(label_card)
        workflow_layout.addWidget(train_card)
        workflow_layout.addWidget(test_card)
        workflow_layout.addStretch(1)

        workflow_scroll = QScrollArea()
        self.vision_workflow_scroll = workflow_scroll
        workflow_scroll.setWidgetResizable(True)
        workflow_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        workflow_scroll.setFrameShape(QFrame.NoFrame)
        workflow_scroll.setWidget(workflow_content)
        workflow_scroll.setMinimumWidth(440)

        vision_workspace.addWidget(review_card)
        vision_workspace.addWidget(workflow_scroll)
        vision_workspace.setStretchFactor(0, 3)
        vision_workspace.setStretchFactor(1, 2)
        vision_workspace.setSizes([1120, 680])
        decision_layout.addWidget(vision_workspace, 1)
        self.settings_workflow.addTab(decision_card, "3  VISION TRANSFORMER")
        self.settings_workflow.currentChanged.connect(
            self._settings_workflow_changed
        )
        settings_layout.addWidget(self.settings_workflow, 1)

        self.advanced_panel.setVisible(False)

        self.settings_panel.setVisible(False)
        content_layout.addWidget(self.settings_panel, 1)

        body = QHBoxLayout()
        body.setSpacing(6)
        image_card = QFrame()
        self.image_card = image_card
        image_card.setObjectName("card")
        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(12, 10, 12, 10)
        self.image = QLabel("A valid Sobel-DINOv2 model is required before inspection")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(600, 220)
        self.image.setStyleSheet(
            "background:#0b1220; color:#cbd5e1; border:1px solid #1e293b; border-radius:8px;"
        )
        image_layout.addWidget(self.image, 1)
        # Keep operational faults visible without adding a persistent summary.
        self.lbl_detail = QLabel()
        self.lbl_detail.setVisible(False)
        self.lbl_alarm = QLabel()
        self.lbl_alarm.setWordWrap(True)
        self.lbl_alarm.setStyleSheet("color:#b91c1c; font-weight:700; padding:4px;")
        self.lbl_alarm.hide()
        image_layout.addWidget(self.lbl_alarm)
        body.addWidget(image_card, 1)

        # Detailed I/O stays in the password-protected settings panel so the operator
        # view can devote its width to the complete cut-point overview.
        signal_card = QFrame()
        signal_card.setObjectName("card")
        signal_layout = QGridLayout(signal_card)
        signal_layout.setContentsMargins(10, 7, 10, 7)
        signal_layout.setHorizontalSpacing(7)
        signal_layout.setVerticalSpacing(3)
        signal_title = QLabel("INTERNAL SIGNALS")
        signal_title.setFont(QFont("Segoe UI Semibold", 9))
        signal_title.setStyleSheet("color:#1d4ed8;")
        signal_layout.addWidget(signal_title, 0, 0, 1, 7)
        self.signal_labels: dict[str, QLabel] = {}
        for column, name in enumerate(
            ("CONNECTED", "VISION HEALTHY", "HEARTBEAT", "RESULT VALID", "PASS", "NG", "RELEASE")
        ):
            caption = QLabel(name)
            caption.setAlignment(Qt.AlignCenter)
            caption.setStyleSheet("color:#64748b; font-size:8pt;")
            signal_layout.addWidget(caption, 1, column)
            value = QLabel("OFF")
            value.setAlignment(Qt.AlignCenter)
            value.setMinimumWidth(58)
            self.signal_labels[name] = value
            signal_layout.addWidget(value, 2, column)
        advanced_layout.addWidget(signal_card)
        advanced_layout.addStretch(1)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        image_layout.addWidget(self.progress)
        content_layout.addLayout(body, 1)

        log_card = QFrame()
        self.log_card = log_card
        log_card.setObjectName("card")
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 9, 12, 10)
        log_head = QLabel(
            "RECENT INSPECTION RESULTS — LAST 5 BOARDS  |  "
            "CLICK A RESULT TO VIEW IMAGES"
        )
        log_head.setStyleSheet("color:#1d4ed8; font-weight:700;")
        log_layout.addWidget(log_head)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Time", "Panel / SN", "Product", "Table", "Result", "Time ms", "Reason"]
        )
        header_view = self.table.horizontalHeader()
        for column in range(6):
            header_view.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(6, QHeaderView.Stretch)
        self.table.setAlternatingRowColors(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(False)
        self.table.cellClicked.connect(self._open_recent_result)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(25)
        log_layout.addWidget(self.table)
        self.table.setMinimumHeight(160)
        self.table.setMaximumHeight(178)
        content_layout.addWidget(log_card)

        root.addWidget(content, 1)

        self.setCentralWidget(page)

    # -- data -------------------------------------------------------------
    def _load_dataset(self, dataset: AUO6000Dataset | None = None) -> None:
        result_dir = Path(self.cfg.result_dir)
        picture_dir = Path(self.cfg.picture_dir)
        if not result_dir.exists() or not picture_dir.exists():
            self.runs_by_key.clear()
            self.recipe_specs.clear()
            self.cmb_recipe.clear()
            self.link.fault("dataset path missing; check config.json")
            QMessageBox.critical(
                self,
                "Dataset missing",
                f"Picture: {picture_dir}\nResult: {result_dir}\n\nUpdate config.json before running.",
            )
            self._refresh_gate()
            return

        previous_key = self.cfg.active_recipe or self.cmb_recipe.currentText()
        self.runs_by_key.clear()
        self.recipe_specs.clear()
        try:
            dataset = dataset or scan_auo6000_dataset(picture_dir)
            use_auo_assignments = (
                bool(dataset.result_dir)
                and Path(dataset.result_dir).resolve() == result_dir.resolve()
            )
            assigned_images: dict[str, list[str]] = defaultdict(list)
            if use_auo_assignments:
                for image in dataset.images:
                    if image.result_file:
                        assigned_images[Path(image.result_file).name].append(image.path)
            days = available_days(result_dir)
            for day in days:
                runs = load_runs(result_dir, day)
                if use_auo_assignments:
                    for run in runs:
                        run.pictures = assigned_images.get(run.result_file, [])
                else:
                    pictures = sorted(
                        str(path) for path in picture_dir.glob(f"{day}_*")
                        if path.suffix.lower() in {".bmp", ".png", ".jpg", ".jpeg"}
                    )
                    attach_pictures(runs, pictures)
                for run in runs:
                    if run.key and run.product_id.strip():
                        self.runs_by_key[run.key].append(run)
        except OSError as exc:
            self.link.fault(f"cannot load dataset: {exc}")
            self._refresh_gate()
            return

        for key, runs in self.runs_by_key.items():
            first = runs[0]
            legacy = SUPPORTED.get(key, {})
            self.recipe_specs[key] = {
                "product": first.product_id.strip(),
                "recipe": first.recipe.strip(),
                "table": first.table.strip(),
                "expected": int(self.cfg.expected_images_by_route.get(key, 0))
                or int(legacy.get("expected", 0))
                or (inferred_expected_images(runs) if DEMO_MODE else 0),
            }

        self.cmb_recipe.blockSignals(True)
        self.cmb_recipe.clear()
        self.cmb_recipe.addItems(sorted(self.runs_by_key))
        if previous_key:
            selected = self.cmb_recipe.findText(previous_key)
            if selected >= 0:
                self.cmb_recipe.setCurrentIndex(selected)
        self.cmb_recipe.blockSignals(False)
        self._recipe_changed()

    def _recipe_changed(self) -> None:
        if self.link.state in (GateState.PASS, GateState.INSPECTING):
            self.link.fault("Recipe changed; press ACK / RESET before inspection")
        key = self.cmb_recipe.currentText()
        spec = self.recipe_specs.get(key, {})
        table = str(spec.get("table", ""))
        route_text = f" · {table}" if table else ""
        self.lbl_active_product.setText(
            f"Recipe Name : {spec.get('product', key)}{route_text}"
        )
        if not key:
            self.classifier = None
            self.model_error = "no AUO6000 production runs were found"
            self._rebuild_queue()
            return
        self._load_active_model(key)
        runs = self.runs_by_key.get(key, [])
        days = sorted({run.start.strftime("%Y%m%d") for run in runs})
        # Result filenames define retained days. A run just before midnight may
        # start on the previous calendar date, so also derive from the filename.
        file_days = sorted({run.result_file[1:9] for run in runs if run.result_file.startswith("_")})
        days = file_days or days
        keep = self.cmb_day.currentText()
        self.cmb_day.blockSignals(True)
        self.cmb_day.clear()
        self.cmb_day.addItem("All retained days", "")
        for day in days:
            self.cmb_day.addItem(datetime.strptime(day, "%Y%m%d").strftime("%Y-%m-%d"), day)
        if keep:
            index = self.cmb_day.findText(keep)
            if index >= 0:
                self.cmb_day.setCurrentIndex(index)
        self.cmb_day.blockSignals(False)
        self._rebuild_queue()

    def _rebuild_queue(self) -> None:
        if self.link.state in (GateState.PASS, GateState.INSPECTING):
            self.link.fault("Inspection queue changed; press ACK / RESET")
        key = self.cmb_recipe.currentText()
        day = self.cmb_day.currentData() or ""
        runs = self.runs_by_key.get(key, [])
        if day:
            runs = [run for run in runs if run.result_file.startswith(f"_{day}_")]
        if DEMO_MODE:
            expected = int(self.recipe_specs.get(key, {}).get("expected", 0))
            runs = [run for run in runs if run.passed and len(run.pictures) == expected]
        self.queue = list(runs)
        self.queue_index = 0
        expected = int(self.recipe_specs.get(key, {}).get("expected", 0))
        complete = sum(len(run.pictures) == expected for run in runs)
        images = sum(len(run.pictures) for run in runs)
        model_text = (
            "SCRIPTED DEMO (NOT MODEL OUTPUT)" if DEMO_MODE
            else "SOBEL MODEL READY" if self.classifier
            else f"MODEL BLOCKED: {self.model_error}"
        )
        self.lbl_dataset.setText(
            f"{len(runs)} panels · {complete} complete · {images:,} images · "
            f"expected {expected} images/panel · {model_text}"
        )
        self.btn_next.setEnabled(bool(runs) and expected > 0 and self.worker is None)
        if self.classifier is None:
            self.link.fault(self.model_error)
        elif not runs:
            self.link.fault("no runs match the selected recipe/day")
        elif expected <= 0:
            self.link.fault("expected cut-point count is ambiguous; inspection is blocked")
        self._refresh_gate()

    # -- model ------------------------------------------------------------
    def _model_path(self, key: str) -> Path:
        folder = Path(self.cfg.model_dir) if self.cfg.model_dir.strip() else APP_DIR / "models"
        return folder / model_file_name(key)

    def _configured_sobel(self) -> SobelConfig:
        return SobelConfig(
            blur_ksize=self.cfg.sobel_blur_ksize,
            blur_sigma=self.cfg.sobel_blur_sigma,
            sobel_ksize=self.cfg.sobel_ksize,
            gradient_x_weight=self.cfg.sobel_gradient_x_weight,
            gradient_y_weight=self.cfg.sobel_gradient_y_weight,
            clip_percentile=self.cfg.sobel_clip_percentile,
            edge_gain=self.cfg.sobel_edge_gain,
            noise_floor=self.cfg.sobel_noise_floor,
        ).validated()

    def _settings_crop(self) -> CropBox:
        """Return only the crop belonging to the selected Settings product."""
        if isinstance(self.settings_classifier, CutClassifier):
            return self.settings_classifier.crop
        return CropBox()

    def _apply_sobel_to_classifier(self) -> None:
        """Keep scripted-demo previews in sync without mutating trained models."""
        sobel = self._configured_sobel()
        if not isinstance(self.classifier, DemoClassifier):
            return
        self.classifier.extractor.sobel = sobel

    def _settings_product_id(self) -> str:
        """Return the single ProductId represented by the selected AUO6000 path."""
        if self.auo_dataset is None:
            return ""
        products = sorted(
            {panel.product_id.strip() for panel in self.auo_dataset.panels
             if panel.product_id.strip()}
        )
        return products[0] if len(products) == 1 else ""

    def _load_settings_model(self) -> None:
        """Load only a model that belongs to the AUO6000 settings data source."""
        self._invalidate_settings_trial()
        self.settings_classifier = None
        self.settings_model_name = ""
        product = self._settings_product_id()
        if not product:
            self.settings_model_error = (
                "The selected Result data does not contain one unambiguous ProductId."
            )
            self._refresh_trial_model_status()
            return

        if DEMO_MODE:
            self.settings_classifier = DemoClassifier()
            self.settings_model_name = f"SCRIPTED DEMO FOR {product}"
            self.settings_model_error = ""
            sobel = self._configured_sobel()
            self.settings_classifier.extractor.sobel = sobel
            self._refresh_trial_model_status()
            return

        model_folder = (
            Path(self.cfg.model_dir)
            if self.cfg.model_dir.strip()
            else APP_DIR / "models"
        )
        exact_model = model_folder / model_file_name(product)
        candidates = [exact_model] if exact_model.is_file() else []
        if not candidates:
            self.settings_model_error = (
                f"No trained Sobel ViT model for ProductId {product}. "
                "Label GOOD/NG images and train the model first."
            )
            self._refresh_trial_model_status()
            return

        valid: list[tuple[Path, CutClassifier]] = []
        rejected: list[str] = []
        for path in candidates:
            try:
                classifier = CutClassifier.load(path, expected_model_key=product)
                if set(classifier.classes) != {"GOOD", "NG"}:
                    raise ValueError("classes must be exactly GOOD and NG")
                quality_error = model_quality_error(classifier.report)
                if quality_error:
                    raise ValueError(quality_error)
            except Exception as exc:
                rejected.append(f"{path.name}: {exc}")
                continue
            valid.append((path, classifier))

        if len(valid) == 1:
            path, self.settings_classifier = valid[0]
            self.settings_model_name = path.name
            self.settings_model_error = ""
        else:
            self.settings_model_error = (
                "Matching model was rejected: " + (rejected[0] if rejected else "unknown error")
            )
        self._refresh_trial_model_status()

    def _load_active_model(self, key: str) -> None:
        self.active_model_path = None
        if DEMO_MODE:
            self.classifier = DemoClassifier()
            self.model_error = ""
            self._apply_sobel_to_classifier()
            return
        self.classifier = None
        product = str(self.recipe_specs.get(key, {}).get("product", ""))
        product = product.strip() or product_from_route_key(key)
        path = self._model_path(product)
        legacy_path = self._model_path(key)
        if not path.exists() and legacy_path != path and legacy_path.exists():
            path = legacy_path
        if not path.exists():
            self.model_error = f"Sobel model missing: {path.name}; retrain before production"
            return
        try:
            classifier = CutClassifier.load(
                path, expected_model_key=key if path == legacy_path and path != self._model_path(product) else product
            )
            quality_error = model_quality_error(classifier.report)
            if quality_error:
                raise ValueError(quality_error)
        except Exception as exc:
            self.model_error = f"Sobel model rejected: {exc}"
            return
        if set(classifier.classes) != {"GOOD", "NG"}:
            self.model_error = "Sobel model classes must be exactly GOOD and NG"
            return
        self.classifier = classifier
        self.active_model_path = path
        self.model_error = ""

    def _reload_model(self, _checked: bool = False) -> None:
        if self.worker is not None or self.auto_running or self._settings_job_busy():
            return
        key = self.cmb_recipe.currentText()
        self._load_active_model(key)
        if self.classifier is None:
            self.link.fault(self.model_error)
        else:
            self.link.fault("Model reloaded; press ACK / RESET before inspection")
            if DEMO_MODE:
                self.lbl_detail.setText(
                    "DEMO reset: the scripted sequence restarts at GOOD; this is not model output."
                )
            else:
                path = self.active_model_path or self._model_path(
                    product_from_route_key(key)
                )
                self.lbl_detail.setText(
                    f"Loaded {path.name}; GOOD release threshold "
                    f"{self.cfg.good_confidence_min:.1%}"
                )
        self._rebuild_queue()
        self._refresh_trial_model_status()

    # -- inspection -------------------------------------------------------
    def _inspect_next(self) -> None:
        if self.settings_worker is not None or self.training_worker is not None:
            return
        if self.worker and self.worker.isRunning():
            return
        if not self.queue:
            self.link.fault("inspection queue is empty")
            self._refresh_gate()
            return
        key = self.cmb_recipe.currentText()
        if self.classifier is None:
            self.link.fault(self.model_error or "no valid Sobel model loaded")
            self._refresh_gate()
            return
        if self.link.state not in (GateState.READY,):
            self.lbl_detail.setText(
                "ACKNOWLEDGE / RESET the current result before inspecting the next panel."
            )
            return
        if self.queue_index >= len(self.queue):
            self._software_hold("no uninspected panels remain in the current queue")
            self.lbl_detail.setText(
                "No uninspected panels remain. Reload the data source before continuing."
            )
            self._refresh_gate()
            return
        expected = int(self.recipe_specs.get(key, {}).get("expected", 0))
        if expected <= 0:
            self._software_hold("expected cut-point count is not configured; save tested Settings first")
            return
        run = self.queue[self.queue_index]
        self.queue_index += 1
        self.current_run = run
        self.current_result = None
        identity = panel_id(run)
        self.link.begin_inspection(identity)
        self.lbl_detail.setText(
            f"Inspecting {len(run.pictures)} images; expected {expected}"
        )
        self._set_busy(True)
        self._refresh_gate()
        self.worker = InspectionWorker(
            self.classifier,
            run,
            expected,
            self.cfg.good_confidence_min,
        )
        self.worker.completed.connect(self._inspection_done)
        self.worker.failed.connect(self._worker_failed)
        self.worker.finished.connect(self._inspection_thread_finished)
        self.worker.start()

    def _inspection_done(
        self, result: PanelDecision, overview: np.ndarray, elapsed_ms: float
    ) -> None:
        self.current_result = result
        verdict, reason = verdict_for(result)
        identity = panel_id(result.run)
        self.link.publish(identity, verdict, reason)
        self._show_result_image(result, overview)
        self._append_log(result, verdict, reason, elapsed_ms)
        self._set_busy(False)
        self._refresh_gate()

        if self.auto_running:
            if verdict is VisionVerdict.PASS:
                delay = max(500, int(self.spin_interval.value() * 1000))
                self.auto_timer.start(delay)
            else:
                self.auto_running = False
                self.btn_auto.setText("▶ START AUTO")

    def _show_result_image(
        self, result: PanelDecision, overview: np.ndarray | None = None
    ) -> None:
        if overview is None:
            if self.classifier is None:
                self.image_pixmap = None
                self.image.setText("No classifier is available for this panel")
                return
            overview = render_prediction_overview(
                self.classifier, result, self.cfg.good_confidence_min)
        self.image_pixmap = bgr_to_pixmap(overview)
        self._fit_image()

    def _append_log(
        self,
        result: PanelDecision,
        verdict: VisionVerdict,
        reason: str,
        elapsed_ms: float,
    ) -> None:
        if self.image_pixmap is None:
            return
        self.result_sequence += 1
        record_id = self.result_sequence
        self.result_images[record_id] = (
            result,
            reason,
            self.image_pixmap.copy(),
        )
        self.table.insertRow(0)
        values = [
            time.strftime("%H:%M:%S"),
            panel_id(result.run),
            result.run.product_id,
            result.run.table,
            verdict.value,
            f"{elapsed_ms:.0f}",
            reason,
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.UserRole, record_id)
            if column == 4:
                item.setToolTip("Click to view the inspected images")
            if column == 6:
                item.setToolTip(str(value))
            self.table.setItem(0, column, item)
        self.table.resizeRowsToContents()
        while self.table.rowCount() > RECENT_RESULT_LIMIT:
            last_row = self.table.rowCount() - 1
            old_item = self.table.item(last_row, 0)
            if old_item is not None:
                self.result_images.pop(old_item.data(Qt.UserRole), None)
            self.table.removeRow(last_row)

    def _open_recent_result(self, row: int, column: int) -> None:
        if column != 4:
            return
        item = self.table.item(row, column)
        if item is None:
            return
        record = self.result_images.get(item.data(Qt.UserRole))
        if record is None:
            self.lbl_detail.setText("Inspection images are no longer available.")
            return
        result, reason, pixmap = record
        self._open_result_dialog(result, reason, pixmap)

    def _open_result_dialog(
        self,
        result: PanelDecision,
        reason: str,
        pixmap: QPixmap,
    ) -> None:
        if self.result_dialog is not None:
            self.result_dialog.close()
        self.result_dialog = ResultImageDialog(
            panel_id(result.run),
            result,
            reason,
            pixmap,
            self,
        )
        self.result_dialog.show()
        self.result_dialog.raise_()
        self.result_dialog.activateWindow()

    # -- automatic replay and interlock ----------------------------------
    def _toggle_auto(self) -> None:
        if self.settings_panel.isVisible() or self._settings_job_busy():
            return
        if self.auto_running:
            self._software_hold("automatic mode stopped by operator")
            return
        if self.classifier is None:
            self.link.fault(self.model_error or "no valid Sobel model; automatic mode refused")
            self._refresh_gate()
            return
        if self.link.state is not GateState.READY:
            self.lbl_detail.setText("Press ACK / RESET before starting automatic mode")
            return
        self.auto_running = True
        self.btn_auto.setText("■ STOP AUTO")
        self.cmb_recipe.setEnabled(False)
        self.cmb_day.setEnabled(False)
        self._auto_tick()

    def _auto_tick(self) -> None:
        if not self.auto_running:
            return
        if self.link.state is GateState.PASS:
            self.link.acknowledge(panel_id(self.current_run) if self.current_run else None)
            self._refresh_gate()
        if self.link.state is GateState.READY:
            self._inspect_next()

    def _acknowledge(self) -> None:
        if self.worker and self.worker.isRunning():
            self.lbl_detail.setText("Cannot ACK while vision inspection is running")
            return
        self.link.acknowledge()
        self.cmb_recipe.setEnabled(not self.auto_running)
        self.cmb_day.setEnabled(not self.auto_running)
        self._refresh_gate()

    def _software_hold(self, reason: str = "STOP / HOLD pressed by operator") -> None:
        self.auto_running = False
        self.auto_timer.stop()
        self.btn_auto.setText("▶ START AUTO")
        self.link.fault(reason)
        self.cmb_recipe.setEnabled(True)
        self.cmb_day.setEnabled(True)
        self._refresh_gate()

    def _toggle_settings(self) -> None:
        if self.auto_running or self.worker is not None or self._settings_job_busy():
            return
        if self.settings_panel.isVisible():
            self.settings_panel.setVisible(False)
            self.image_card.setVisible(True)
            self.log_card.setVisible(True)
            self.btn_settings.setText("SETTING ▾")
            return

        if not SETTINGS_PASSWORD:
            QMessageBox.warning(
                self,
                "Settings unavailable",
                "Settings access is not configured. Set the "
                "AVTR_SETTINGS_PASSWORD environment variable and restart AVTR.",
            )
            return

        password, accepted = QInputDialog.getText(
            self,
            "Settings authentication",
            "Enter settings password:",
            QLineEdit.Password,
        )
        if not accepted:
            return
        if not hmac.compare_digest(
            password.encode("utf-8"), SETTINGS_PASSWORD.encode("utf-8")
        ):
            QMessageBox.warning(self, "Access denied", "Incorrect settings password.")
            return

        self._sync_settings_controls()
        if not self.settings_image_path:
            if self.settings_image_dir and Path(self.settings_image_dir).is_dir():
                self._set_settings_path(self.settings_image_dir)
            else:
                self._use_default_settings_image()
        self.advanced_panel.setVisible(False)
        self.settings_workflow.setVisible(True)
        self.image_card.setVisible(False)
        self.log_card.setVisible(False)
        self.settings_panel.setVisible(True)
        self.btn_settings.setText("SETTING ▴")
        QTimer.singleShot(0, self._preview_sobel)
        QTimer.singleShot(0, self._layout_sobel_parameter_cards)

    def _make_parameter_control(
        self,
        title: str,
        slider: QSlider,
        value_label: QLabel,
        help_text: str = "",
    ) -> QFrame:
        card = QFrame()
        card.setObjectName("parameterCard")
        if help_text:
            card.setToolTip(help_text)
            slider.setToolTip(help_text)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(10, 7, 10, 9)
        layout.setSpacing(5)

        heading = QHBoxLayout()
        title_label = QLabel(title)
        title_label.setStyleSheet("color:#334155; font-weight:700;")
        heading.addWidget(title_label)
        heading.addStretch(1)
        heading.addWidget(value_label)
        layout.addLayout(heading)

        adjustment = QHBoxLayout()
        adjustment.setSpacing(6)
        decrease = QPushButton("◀")
        increase = QPushButton("▶")
        for button in (decrease, increase):
            button.setObjectName("stepArrow")
            button.setFixedSize(34, 30)
        decrease.setToolTip(f"Decrease {title}")
        increase.setToolTip(f"Increase {title}")
        decrease.clicked.connect(
            lambda _checked=False, control=slider: control.setValue(
                control.value() - control.singleStep()
            )
        )
        increase.clicked.connect(
            lambda _checked=False, control=slider: control.setValue(
                control.value() + control.singleStep()
            )
        )
        adjustment.addWidget(decrease)
        adjustment.addWidget(slider, 1)
        adjustment.addWidget(increase)
        layout.addLayout(adjustment)
        return card

    def _invalidate_settings_trial(self, *_args) -> None:
        self.settings_trial_approved = False
        self.trial_input_signature = None
        self.trial_predictions.clear()
        self.trial_result = None
        if hasattr(self, "btn_apply_settings"):
            self.btn_apply_settings.setEnabled(False)

    def _settings_job_busy(self) -> bool:
        return self.settings_worker is not None or self.training_worker is not None

    def _lock_settings_job(self, busy: bool) -> None:
        self.settings_workflow.setEnabled(not busy)
        self.btn_settings.setEnabled(not busy)
        self.btn_auto.setEnabled(not busy)
        self.btn_model.setEnabled(not busy)
        self.btn_next.setEnabled(not busy and bool(self.queue))
        self.cmb_recipe.setEnabled(not busy)
        self.cmb_day.setEnabled(not busy)

    def _trial_signature(self):
        """Tie a completed test to the source, model and threshold it used."""
        model_path = self._model_path(self._settings_product_id())
        return (
            self.settings_image_dir,
            id(self.settings_classifier),
            self._source_signature(str(model_path)) if model_path.is_file() else None,
            self._sobel_parameter_record(),
            self.spin_confidence.value(),
            tuple((path, self._source_signature(path)) for path in self.settings_image_paths),
        )

    def _sobel_parameter_changed(self, _value: int = 0) -> None:
        kernel_values = (1, 3, 5, 7)
        self.lbl_sobel_blur_value.setText(
            str(kernel_values[self.slider_sobel_blur.value()])
        )
        sigma = self.slider_blur_sigma.value() / 100.0
        self.lbl_blur_sigma_value.setText("AUTO" if sigma == 0.0 else f"{sigma:.2f}")
        self.lbl_sobel_kernel_value.setText(
            str(kernel_values[self.slider_sobel_kernel.value()])
        )
        self.lbl_gradient_x_value.setText(
            f"{self.slider_gradient_x.value() / 1000.0:.2f}"
        )
        self.lbl_gradient_y_value.setText(
            f"{self.slider_gradient_y.value() / 1000.0:.2f}"
        )
        self.lbl_sobel_clip_value.setText(
            f"{self.slider_sobel_clip.value() / 100.0:.2f} %"
        )
        self.lbl_edge_gain_value.setText(
            f"{self.slider_edge_gain.value() / 1000.0:.2f}×"
        )
        self.lbl_noise_floor_value.setText(str(self.slider_noise_floor.value()))
        if hasattr(self, "lbl_sobel_save_status"):
            self._update_image_workflow_status()
        if self.settings_image_path and not self._loading_sobel_parameters:
            self._invalidate_settings_trial()
            self.sobel_preview_timer.start()

    def _sync_settings_controls(self) -> None:
        kernel_values = (1, 3, 5, 7)
        blur = (
            self.cfg.sobel_blur_ksize
            if self.cfg.sobel_blur_ksize in kernel_values
            else 3
        )
        sobel = (
            self.cfg.sobel_ksize if self.cfg.sobel_ksize in kernel_values else 3
        )
        self.slider_sobel_blur.setValue(kernel_values.index(blur))
        self.slider_blur_sigma.setValue(
            int(round(self.cfg.sobel_blur_sigma * 100.0))
        )
        self.slider_sobel_kernel.setValue(kernel_values.index(sobel))
        self.slider_gradient_x.setValue(
            int(round(self.cfg.sobel_gradient_x_weight * 1000.0))
        )
        self.slider_gradient_y.setValue(
            int(round(self.cfg.sobel_gradient_y_weight * 1000.0))
        )
        self.slider_sobel_clip.setValue(
            int(round(self.cfg.sobel_clip_percentile * 100.0))
        )
        self.slider_edge_gain.setValue(
            int(round(self.cfg.sobel_edge_gain * 1000.0))
        )
        self.slider_noise_floor.setValue(self.cfg.sobel_noise_floor)
        self._sobel_parameter_changed()
        self.spin_confidence.setValue(self.cfg.good_confidence_min * 100.0)
        self._refresh_trial_model_status()

    def _refresh_trial_model_status(self) -> None:
        if not hasattr(self, "lbl_trial_result"):
            return
        if self.settings_classifier is None:
            self.lbl_trial_result.setStyleSheet(
                "background:#fee2e2; color:#b91c1c; border:1px solid #ef4444; "
                "border-radius:7px; padding:10px; font-weight:800;"
            )
            self.lbl_trial_result.setText(
                f"MODEL NOT READY  |  {self.settings_model_error}"
            )
            self.btn_trial.setToolTip(self.settings_model_error)
        elif self._settings_model_needs_retraining():
            self.lbl_trial_result.setStyleSheet(
                "background:#ffedd5; color:#9a3412; border:1px solid #f97316; "
                "border-radius:7px; padding:10px; font-weight:800;"
            )
            self.lbl_trial_result.setText(
                "MODEL RETRAIN REQUIRED  |  Saved images, labels or Sobel settings changed."
            )
            self.btn_trial.setToolTip("Retrain and save the model before testing.")
        else:
            mode = self.settings_model_name or "MODEL READY"
            self.lbl_trial_result.setStyleSheet(
                "background:#dcfce7; color:#166534; border:1px solid #22c55e; "
                "border-radius:7px; padding:10px; font-weight:800;"
            )
            self.lbl_trial_result.setText(f"MODEL READY  |  {mode}")
            self.btn_trial.setToolTip("")
        if hasattr(self, "btn_train_model"):
            self._refresh_training_workflow_status()

    def _use_default_settings_image(self) -> None:
        auo6000_root = APP_DIR.parent / "sobel"
        if (auo6000_root / "Picture").is_dir():
            self._set_settings_path(str(auo6000_root))
            return
        path = ""
        if self.current_result and self.current_result.details:
            path = self.current_result.details[0].path
        elif self.current_run and self.current_run.pictures:
            path = self.current_run.pictures[0]
        elif self.queue and self.queue[0].pictures:
            path = self.queue[0].pictures[0]
        if path:
            self._set_settings_path(str(Path(path).parent))

    def _load_sobel_source_path(self) -> str:
        try:
            raw = json.loads(SOBEL_WORKFLOW_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return ""
        path = raw.get("source_path", "") if isinstance(raw, dict) else ""
        return str(path) if path and Path(str(path)).is_dir() else ""

    def _load_sobel_records(self) -> dict[str, dict]:
        try:
            raw = json.loads(SOBEL_WORKFLOW_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}
        records = raw.get("images", {}) if isinstance(raw, dict) else {}
        return records if isinstance(records, dict) else {}

    def _save_sobel_records(self) -> None:
        SOBEL_WORKFLOW_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = SOBEL_WORKFLOW_PATH.with_suffix(".tmp")
        payload = {
            "version": 3,
            "source_path": self.settings_image_dir,
            "images": self.sobel_records,
        }
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(SOBEL_WORKFLOW_PATH)

    @staticmethod
    def _sobel_record_key(path: str) -> str:
        return os.path.normcase(str(Path(path).resolve()))

    @staticmethod
    def _source_signature(path: str) -> dict[str, int]:
        stat = Path(path).stat()
        return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _sobel_parameter_record(self) -> dict[str, int | float]:
        return self._sobel_config_record(self._sobel_from_controls())

    @staticmethod
    def _sobel_config_record(sobel: SobelConfig) -> dict[str, int | float]:
        return {
            "blur_ksize": sobel.blur_ksize,
            "blur_sigma": round(sobel.blur_sigma, 2),
            "sobel_ksize": sobel.sobel_ksize,
            "gradient_x_weight": round(sobel.gradient_x_weight, 2),
            "gradient_y_weight": round(sobel.gradient_y_weight, 2),
            "clip_percentile": round(sobel.clip_percentile, 2),
            "edge_gain": round(sobel.edge_gain, 2),
            "noise_floor": sobel.noise_floor,
        }

    def _normalise_saved_sobel_parameters(
        self, parameters: dict | None
    ) -> dict[str, int | float]:
        raw = parameters if isinstance(parameters, dict) else {}
        defaults = SobelConfig()
        sobel = SobelConfig(
            blur_ksize=raw.get("blur_ksize", defaults.blur_ksize),
            blur_sigma=raw.get("blur_sigma", defaults.blur_sigma),
            sobel_ksize=raw.get("sobel_ksize", defaults.sobel_ksize),
            gradient_x_weight=raw.get(
                "gradient_x_weight", defaults.gradient_x_weight
            ),
            gradient_y_weight=raw.get(
                "gradient_y_weight", defaults.gradient_y_weight
            ),
            clip_percentile=raw.get(
                "clip_percentile", defaults.clip_percentile
            ),
            edge_gain=raw.get("edge_gain", defaults.edge_gain),
            noise_floor=raw.get("noise_floor", defaults.noise_floor),
        ).validated()
        return self._sobel_config_record(sobel)

    def _record_matches_source(self, path: str, record: dict) -> bool:
        try:
            return record.get("source") == self._source_signature(path)
        except OSError:
            return False

    def _record_is_saved(self, path: str, *, current_parameters: bool = False) -> bool:
        record = self.sobel_records.get(self._sobel_record_key(path), {})
        output = Path(str(record.get("sobel_output", "")))
        if not output.is_file() or not self._record_matches_source(path, record):
            return False
        if current_parameters:
            saved_parameters = self._normalise_saved_sobel_parameters(
                record.get("sobel_parameters")
            )
            return (
                saved_parameters == self._sobel_parameter_record()
                and record.get("crop", asdict(CropBox())) == asdict(self._settings_crop())
            )
        return True

    def _record_is_trained(self, path: str) -> bool:
        record = self.sobel_records.get(self._sobel_record_key(path), {})
        model_name = str(record.get("trained_model", ""))
        model_folder = (
            Path(self.cfg.model_dir)
            if self.cfg.model_dir.strip()
            else APP_DIR / "models"
        )
        saved_parameters = self._normalise_saved_sobel_parameters(
            record.get("sobel_parameters")
        )
        trained_parameters = self._normalise_saved_sobel_parameters(
            record.get("trained_sobel_parameters")
        )
        model_path = model_folder / model_name
        try:
            model_current = record.get("trained_model_signature") == self._source_signature(str(model_path))
        except OSError:
            model_current = False
        return (
            bool(record.get("trained"))
            and self._record_is_saved(path)
            and bool(model_name)
            and bool(record.get("trained_sobel_parameters"))
            and record.get("trained_label") == record.get("training_label")
            and saved_parameters == trained_parameters
            and model_current
        )

    def _set_settings_path(self, directory: str) -> None:
        if self._settings_job_busy() or self.worker is not None or self.auto_running:
            return
        root = Path(directory)
        if not root.is_dir():
            return
        try:
            dataset = scan_auo6000_dataset(root)
            output_root = SOBEL_OUTPUT_DIR.resolve()
            paths = [
                image.path
                for image in dataset.images
                if output_root not in Path(image.path).resolve().parents
            ]
        except OSError as exc:
            QMessageBox.warning(self, "Path could not be read", str(exc))
            return
        self.auo_dataset = dataset
        self._invalidate_settings_trial()
        self.trial_predictions.clear()
        self.settings_image_metadata = dataset.image_index
        self.settings_image_dir = dataset.root
        self.txt_settings_path.setText(self.settings_image_dir)
        self.txt_settings_path.setToolTip(self.settings_image_dir)
        self._update_auo_source_summary()
        self.trial_preview_pixmap = None
        self.trial_result = None
        self.training_last_message = ""
        self.training_last_success = None
        self._set_settings_images(paths)
        self._load_settings_model()
        self._refresh_training_workflow_status()
        self._show_training_image()
        try:
            self._save_sobel_records()
        except OSError:
            pass
        if paths:
            self.settings_workflow.setCurrentIndex(1)

    def _update_auo_source_summary(self) -> None:
        dataset = self.auo_dataset
        if dataset is None:
            return
        panel_count = len(dataset.panels)
        image_count = len(dataset.images)
        self.lbl_auo_recipe.setText(dataset.recipe_name or "NOT FOUND")
        self.lbl_auo_recipe.setToolTip(dataset.recipe_name or "Recipe not found")
        self.lbl_auo_panels.setText(str(panel_count))
        self.lbl_auo_images.setText(str(image_count))
        self.lbl_auo_cut_points.setText(
            str(dataset.expected_cut_points) if dataset.expected_cut_points else "UNASSIGNED"
        )
        format_text = dataset.format_summary or "NOT FOUND"
        disguised_count = sum(
            1
            for image in dataset.images
            if Path(image.path).suffix.casefold() == ".bmp"
            and image.stored_format != "BMP"
        )
        if disguised_count:
            format_text += "  |  .BMP NAMES"
        self.lbl_auo_format.setText(format_text)

        matched_panels = len({image.result_file for image in dataset.images if image.result_file})
        linked_images = sum(bool(image.result_file) for image in dataset.images)
        if image_count and panel_count:
            status = (
                "READY FOR SOBEL TUNING\n"
                f"{linked_images}/{image_count} images linked to {matched_panels}/{panel_count} panels"
            )
            background = "#dcfce7"
            colour = "#166534"
        elif image_count:
            status = (
                "IMAGES FOUND - PANEL RESULTS NOT LINKED\n"
                "Review the Result folder before continuing"
            )
            background = "#ffedd5"
            colour = "#9a3412"
        else:
            status = "NO IMAGES FOUND\nSelect the AUO6000 export root or Picture folder"
            background = "#fee2e2"
            colour = "#b91c1c"
        self.lbl_auo_files.setText(status)
        self.lbl_auo_files.setStyleSheet(
            f"background:{background}; color:{colour}; border:1px solid {colour}; "
            "border-radius:7px; padding:9px 12px; font-weight:800;"
        )

        folder_rows = (
            ("Picture", image_count, dataset.picture_dir),
            ("Result", panel_count, dataset.result_dir),
            ("Recipe", len(dataset.recipe_files), dataset.recipe_dir),
            ("Log", len(dataset.log_files), dataset.log_dir),
        )
        for row, (name, count, location) in enumerate(folder_rows):
            values = (
                name,
                "READY" if location and count else "MISSING",
                f"{count:,}",
                location or "Not found",
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 3:
                    item.setToolTip(value)
                self.tbl_auo_folders.setItem(row, column, item)
            self.tbl_auo_folders.setRowHeight(row, 30)

        self.lbl_auo_warning.setText(
            "DATA NOTE  |  " + "  |  ".join(dataset.warnings)
            if dataset.warnings else ""
        )
        self.lbl_auo_warning.setVisible(bool(dataset.warnings))
        self.btn_continue_sobel.setEnabled(bool(image_count))
        self.lbl_trial_batch.setText(
            f"CURRENT BATCH: {panel_count} PANEL{'S' if panel_count != 1 else ''}  |  "
            f"{image_count} IMAGE{'S' if image_count != 1 else ''}  |  "
            f"{dataset.expected_cut_points or '--'} CUT POINTS / PANEL  |  "
            f"{dataset.recipe_name or 'RECIPE NOT FOUND'}"
        )
        self.btn_trial.setText(
            f"TEST ALL {image_count:,} IMAGES"
            if image_count else "TEST VISION TRANSFORMER"
        )

    def _set_settings_images(self, paths: list[str]) -> None:
        unique_paths: list[str] = []
        seen: set[str] = set()
        for path in paths:
            resolved = str(Path(path))
            key = os.path.normcase(os.path.abspath(resolved))
            if key not in seen:
                seen.add(key)
                unique_paths.append(resolved)

        self.settings_image_paths = unique_paths
        self.cmb_settings_image.blockSignals(True)
        self.cmb_settings_image.clear()
        for path in unique_paths:
            self.cmb_settings_image.addItem("", path)
        self.cmb_settings_image.blockSignals(False)
        self.cmb_settings_image.setEnabled(bool(unique_paths))
        panel_count = len(self.auo_dataset.panels) if self.auo_dataset else 0
        self.lbl_settings_image_count.setText(
            f"{panel_count} PANEL{'S' if panel_count != 1 else ''} | "
            f"{len(unique_paths)} IMAGE{'S' if len(unique_paths) != 1 else ''}"
        )

        if unique_paths:
            self.cmb_settings_image.setCurrentIndex(0)
            self.settings_image_path = unique_paths[0]
            self.cmb_settings_image.setToolTip(unique_paths[0])
            self._load_current_image_parameters()
        else:
            self.settings_image_path = ""
            self.cmb_settings_image.setToolTip("")
            self.settings_preview_pixmap = None
            self.lbl_settings_preview.clear()
            self.lbl_settings_preview.setText("NO IMAGES FOUND IN THE SELECTED PATH")
        self._refresh_image_selector_labels()
        self._update_image_workflow_status()
        self._set_training_images(unique_paths)

    def _settings_image_changed(self, index: int) -> None:
        path = self.cmb_settings_image.itemData(index) if index >= 0 else ""
        self.settings_image_path = str(path or "")
        self.cmb_settings_image.setToolTip(self.settings_image_path)
        if self.settings_image_path:
            self._load_current_image_parameters()
            self._preview_sobel()
        self._update_image_workflow_status()

    def _set_training_images(self, paths: list[str]) -> None:
        if not hasattr(self, "cmb_training_image"):
            return
        current_path = self._current_training_image_path()
        self.cmb_training_image.blockSignals(True)
        self.cmb_training_image.clear()
        for path in paths:
            self.cmb_training_image.addItem("", path)
        selected = 0
        if current_path in paths:
            selected = paths.index(current_path)
        elif self.settings_image_path in paths:
            selected = paths.index(self.settings_image_path)
        if paths:
            self.cmb_training_image.setCurrentIndex(selected)
        self.cmb_training_image.blockSignals(False)
        self.cmb_training_image.setEnabled(bool(paths))
        self._refresh_training_workflow_status()
        self._show_training_image()

    def _current_training_image_path(self) -> str:
        if not hasattr(self, "cmb_training_image"):
            return ""
        index = self.cmb_training_image.currentIndex()
        return str(self.cmb_training_image.itemData(index) or "") if index >= 0 else ""

    def _training_label_for(self, path: str) -> str:
        if not path:
            return ""
        record = self.sobel_records.get(self._sobel_record_key(path), {})
        if not self._record_matches_source(path, record):
            return ""
        label = str(record.get("training_label", "")).upper()
        return label if label in {"GOOD", "NG"} else ""

    def _training_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        for path in self.settings_image_paths:
            label = self._training_label_for(path)
            if label and self._record_is_saved(path, current_parameters=True):
                items.append((path, label))
        return items

    def _settings_model_needs_retraining(self) -> bool:
        if not isinstance(self.settings_classifier, CutClassifier):
            return False
        if (
            self._sobel_config_record(self.settings_classifier.sobel)
            != self._sobel_parameter_record()
        ):
            return True
        model_name = self.settings_model_name
        for path, record in self.sobel_records.items():
            if record.get("trained_model") == model_name and not self._record_is_trained(path):
                return True
        eligible = dict(self._training_items())
        for path in self.settings_image_paths:
            record = self.sobel_records.get(self._sobel_record_key(path), {})
            label = eligible.get(path, "")
            belonged_to_model = record.get("trained_model") == model_name
            if (
                (label and not self._record_is_trained(path))
                or (label and record.get("trained_model") != model_name)
                or (belonged_to_model and not self._record_is_trained(path))
            ):
                return True
        return False

    def _training_image_changed(self, _index: int) -> None:
        self._refresh_training_workflow_status()
        self._show_training_image()

    def _previous_training_image(self) -> None:
        count = self.cmb_training_image.count()
        if count:
            self.cmb_training_image.setCurrentIndex(
                (self.cmb_training_image.currentIndex() - 1) % count
            )

    def _next_training_image(self) -> None:
        count = self.cmb_training_image.count()
        if count:
            self.cmb_training_image.setCurrentIndex(
                (self.cmb_training_image.currentIndex() + 1) % count
            )

    def _next_unlabelled_image(self) -> None:
        count = self.cmb_training_image.count()
        current = self.cmb_training_image.currentIndex()
        for offset in range(1, count + 1):
            index = (current + offset) % count
            path = str(self.cmb_training_image.itemData(index) or "")
            if path and not self._training_label_for(path):
                self.cmb_training_image.setCurrentIndex(index)
                return

    def _set_training_label(self, label: str) -> None:
        if self._settings_job_busy():
            return
        path = self._current_training_image_path()
        if not path or not Path(path).is_file():
            return
        key = self._sobel_record_key(path)
        previous = self.sobel_records.get(key, {})
        record = dict(previous) if self._record_matches_source(path, previous) else {}
        old_label = str(record.get("training_label", "")).upper()
        record.update(
            {
                "source_path": str(Path(path).resolve()),
                "source": self._source_signature(path),
                "product_id": self._settings_product_id(),
            }
        )
        if label in {"GOOD", "NG"}:
            record["training_label"] = label
            record["labelled_at"] = datetime.now().isoformat(timespec="seconds")
        else:
            record.pop("training_label", None)
            record.pop("labelled_at", None)
        if old_label != label:
            record["trained"] = False
            record["trained_at"] = ""
            # Retain the previous training contract so a cleared/changed label
            # still invalidates that model after an application restart.
            self.training_last_message = ""
            self.training_last_success = None
            self._invalidate_settings_trial()
        self.sobel_records[key] = record
        try:
            self._save_sobel_records()
        except OSError as exc:
            QMessageBox.warning(self, "Label not saved", str(exc))
            return
        self._refresh_training_workflow_status()
        self._refresh_image_selector_labels()
        self._refresh_trial_model_status()
        self._show_training_image()

    def _refresh_training_workflow_status(self) -> None:
        if not hasattr(self, "cmb_training_image"):
            return
        current = self.cmb_training_image.currentIndex()
        good = ng = eligible_good = eligible_ng = 0
        self.cmb_training_image.blockSignals(True)
        for index, path in enumerate(self.settings_image_paths):
            label = self._training_label_for(path)
            saved = self._record_is_saved(path, current_parameters=True)
            if label == "GOOD":
                good += 1
                eligible_good += int(saved)
            elif label == "NG":
                ng += 1
                eligible_ng += int(saved)
            metadata = self.settings_image_metadata.get(
                os.path.normcase(os.path.abspath(path))
            )
            context = (
                f"SN {metadata.panel_sn or 'UNASSIGNED'} | CUT {metadata.cut_point:02d} | "
                if metadata and metadata.cut_point else "UNASSIGNED | "
            )
            self.cmb_training_image.setItemText(
                index,
                f"{context}LABEL {label or 'UNLABELED'} | "
                f"SOBEL {'SAVED' if saved else 'NOT SAVED'} | {Path(path).name}",
            )
        if current >= 0:
            self.cmb_training_image.setCurrentIndex(current)
        self.cmb_training_image.blockSignals(False)

        total = len(self.settings_image_paths)
        ready = (
            eligible_good >= MIN_TRAINING_IMAGES_PER_CLASS
            and eligible_ng >= MIN_TRAINING_IMAGES_PER_CLASS
            and bool(self._settings_product_id())
        )
        self.lbl_label_progress.setText(
            f"LABELS: GOOD {good} | NG {ng} | UNLABELED {total - good - ng}  ·  "
            f"SAVED + READY: GOOD {eligible_good}/{MIN_TRAINING_IMAGES_PER_CLASS} | "
            f"NG {eligible_ng}/{MIN_TRAINING_IMAGES_PER_CLASS}"
        )
        busy = bool(self.training_worker and self.training_worker.isRunning())
        trial_busy = bool(self.settings_worker and self.settings_worker.isRunning())
        self.btn_train_model.setEnabled(ready and not busy and not trial_busy and not DEMO_MODE)
        self.btn_trial.setEnabled(
            self.settings_classifier is not None
            and not self._settings_model_needs_retraining()
            and bool(self.settings_image_paths)
            and not busy
            and not trial_busy
        )
        has_image = bool(self._current_training_image_path())
        for button in (
            self.btn_training_previous,
            self.btn_training_next,
            self.btn_next_unlabelled,
        ):
            button.setEnabled(total > 1 and not busy and not trial_busy)
        for button in (self.btn_label_good, self.btn_label_ng, self.btn_clear_label):
            button.setEnabled(has_image and not busy and not trial_busy)

        if busy:
            return
        if self.training_last_message:
            colour = "#166534" if self.training_last_success else "#b91c1c"
            self.lbl_model_training_status.setStyleSheet(
                f"color:{colour}; font-weight:800;"
            )
            self.lbl_model_training_status.setText(self.training_last_message)
            self.btn_train_model.setToolTip(
                "" if self.training_last_success else self.training_last_message
            )
            return
        if DEMO_MODE:
            status = "TRAINING DISABLED IN SCRIPTED DEMO MODE"
        elif ready:
            status = (
                f"READY TO TRAIN {eligible_good + eligible_ng} SAVED IMAGES FOR "
                f"{self._settings_product_id()}"
            )
        else:
            need_good = max(0, MIN_TRAINING_IMAGES_PER_CLASS - eligible_good)
            need_ng = max(0, MIN_TRAINING_IMAGES_PER_CLASS - eligible_ng)
            status = f"NEED {need_good} MORE SAVED GOOD AND {need_ng} MORE SAVED NG IMAGES"
        status_colour = "#166534" if ready and not DEMO_MODE else "#64748b"
        self.lbl_model_training_status.setStyleSheet(
            f"color:{status_colour}; font-weight:700;"
        )
        self.lbl_model_training_status.setText(status)
        self.btn_train_model.setToolTip("" if ready else status)

    def _show_selected_trial_prediction(self, path: str) -> bool:
        prediction = self.trial_predictions.get(path)
        if prediction is None or self.settings_classifier is None:
            return False
        label, confidence = prediction
        try:
            image_index = self.settings_image_paths.index(path)
        except ValueError:
            image_index = 0
        metadata = self.settings_image_metadata.get(
            os.path.normcase(os.path.abspath(path))
        )
        context = (
            f"SN {metadata.panel_sn or 'UNASSIGNED'} | CUT {metadata.cut_point:02d} | "
            if metadata and metadata.cut_point
            else f"CUT POINT {image_index + 1:02d} | "
        )
        selected = PanelDecision(
            run=self.current_run,
            status="GOOD" if label == "GOOD" else "NG",
            checked=1,
            details=[ImageDecision(image_index, path, label, float(confidence))],
        )
        overview = render_prediction_overview(
            self.settings_classifier,
            selected,
            self.spin_confidence.value() / 100.0,
            cell_width=1120,
            cell_height=500,
            tile_context={path: context},
            columns=1,
        )
        self.trial_preview_pixmap = bgr_to_pixmap(overview)
        threshold_state = (
            "ACCEPTED"
            if label == "GOOD" and confidence >= self.spin_confidence.value() / 100.0
            else "BELOW THRESHOLD"
            if label == "GOOD"
            else "REJECTED"
        )
        self.lbl_training_image_context.setText(
            f"{context}PREDICT {label} {confidence:.1%} | {threshold_state} | "
            f"{Path(path).name}"
        )
        self._fit_trial_preview()
        return True

    def _show_training_image(self) -> None:
        if not hasattr(self, "lbl_trial_preview"):
            return
        path = self._current_training_image_path()
        if not path or not Path(path).is_file():
            self.trial_preview_pixmap = None
            self.lbl_trial_preview.setMinimumSize(1, 200)
            self.lbl_trial_preview.clear()
            self.lbl_trial_preview.setText("SELECT AN IMAGE TO LABEL")
            return
        if self._show_selected_trial_prediction(path):
            return
        original = cv2.imread(path, cv2.IMREAD_COLOR)
        if original is None:
            self.trial_preview_pixmap = None
            self.lbl_trial_preview.setText("IMAGE COULD NOT BE READ")
            return
        crop = self._settings_crop()
        cropped = crop.apply(original)
        record = self.sobel_records.get(self._sobel_record_key(path), {})
        sobel = (
            SobelConfig(**self._normalise_saved_sobel_parameters(record.get("sobel_parameters")))
            if self._record_is_saved(path) else self._sobel_from_controls()
        )
        extractor = FeatureExtractor(device="cpu", sobel=sobel)
        edge = extractor.edge_map(path, crop)
        if edge is None:
            self.trial_preview_pixmap = None
            self.lbl_trial_preview.setText("SOBEL EDGE COULD NOT BE GENERATED")
            return

        pane_width, pane_height, header_height = 620, 420, 38
        canvas = np.full(
            (pane_height + header_height, pane_width * 2 + 8, 3), 11, dtype=np.uint8
        )
        canvas[header_height:, :pane_width] = _letterbox(cropped, pane_width, pane_height)
        canvas[header_height:, pane_width + 8:] = _letterbox(edge, pane_width, pane_height)
        label = self._training_label_for(path) or "UNLABELED"
        metadata = self.settings_image_metadata.get(
            os.path.normcase(os.path.abspath(path))
        )
        context = (
            f"SN {metadata.panel_sn or 'UNASSIGNED'} | CUT {metadata.cut_point:02d} | "
            if metadata and metadata.cut_point else ""
        )
        colour = (
            (70, 210, 85) if label == "GOOD" else
            (60, 70, 235) if label == "NG" else (210, 210, 210)
        )
        cv2.putText(
            canvas, f"{context}MANUAL LABEL: {label}", (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "ORIGINAL", (8, header_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "SOBEL EDGE", (pane_width + 16, header_height + 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.52, (225, 225, 225), 1, cv2.LINE_AA,
        )
        self.trial_preview_pixmap = bgr_to_pixmap(canvas)
        self.trial_result = None
        self.lbl_training_image_context.setText(
            f"{context}LABEL {label} | {Path(path).name}"
        )
        context_background = (
            "#dcfce7" if label == "GOOD" else
            "#fee2e2" if label == "NG" else "#e2e8f0"
        )
        context_colour = (
            "#166534" if label == "GOOD" else
            "#b91c1c" if label == "NG" else "#334155"
        )
        self.lbl_training_image_context.setStyleSheet(
            f"background:{context_background}; color:{context_colour}; "
            "border-radius:6px; padding:6px 8px; font-weight:800;"
        )
        self._fit_trial_preview()

    def _settings_workflow_changed(self, index: int) -> None:
        if index == 1:
            current_path = self._current_training_image_path()
            for item in range(self.cmb_settings_image.count()):
                if str(self.cmb_settings_image.itemData(item) or "") == current_path:
                    self.cmb_settings_image.setCurrentIndex(item)
                    break
            QTimer.singleShot(0, self._layout_sobel_parameter_cards)
            return
        if index != 2:
            return
        QTimer.singleShot(0, self._layout_vision_workspace)
        current_path = self.settings_image_path
        for item in range(self.cmb_training_image.count()):
            if str(self.cmb_training_image.itemData(item) or "") == current_path:
                self.cmb_training_image.setCurrentIndex(item)
                break
        self._refresh_training_workflow_status()
        self._show_training_image()

    def _select_settings_path(self) -> None:
        start = (
            self.settings_image_dir
            if self.settings_image_dir
            else self.cfg.picture_dir
        )
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select image path",
            start,
        )
        if directory:
            self._set_settings_path(directory)
            self.settings_workflow.setCurrentIndex(0)

    def _load_current_image_parameters(self) -> None:
        record = self.sobel_records.get(
            self._sobel_record_key(self.settings_image_path), {}
        )
        parameters = (
            self._normalise_saved_sobel_parameters(
                record.get("sobel_parameters")
            )
            if self._record_matches_source(self.settings_image_path, record)
            else {}
        )
        kernel_values = (1, 3, 5, 7)
        blur = int(parameters.get("blur_ksize", self.cfg.sobel_blur_ksize))
        sigma = float(parameters.get("blur_sigma", self.cfg.sobel_blur_sigma))
        sobel = int(parameters.get("sobel_ksize", self.cfg.sobel_ksize))
        x_weight = float(
            parameters.get(
                "gradient_x_weight", self.cfg.sobel_gradient_x_weight
            )
        )
        y_weight = float(
            parameters.get(
                "gradient_y_weight", self.cfg.sobel_gradient_y_weight
            )
        )
        clip = float(
            parameters.get("clip_percentile", self.cfg.sobel_clip_percentile)
        )
        gain = float(parameters.get("edge_gain", self.cfg.sobel_edge_gain))
        noise_floor = int(
            parameters.get("noise_floor", self.cfg.sobel_noise_floor)
        )
        blur = blur if blur in kernel_values else 3
        sobel = sobel if sobel in kernel_values else 3
        self._loading_sobel_parameters = True
        try:
            self.slider_sobel_blur.setValue(kernel_values.index(blur))
            self.slider_blur_sigma.setValue(int(round(sigma * 100.0)))
            self.slider_sobel_kernel.setValue(kernel_values.index(sobel))
            self.slider_gradient_x.setValue(int(round(x_weight * 1000.0)))
            self.slider_gradient_y.setValue(int(round(y_weight * 1000.0)))
            self.slider_sobel_clip.setValue(int(round(clip * 100.0)))
            self.slider_edge_gain.setValue(int(round(gain * 1000.0)))
            self.slider_noise_floor.setValue(noise_floor)
        finally:
            self._loading_sobel_parameters = False
        self._sobel_parameter_changed()

    def _previous_settings_image(self) -> None:
        count = self.cmb_settings_image.count()
        if count:
            self.cmb_settings_image.setCurrentIndex(
                (self.cmb_settings_image.currentIndex() - 1) % count
            )

    def _next_settings_image(self) -> None:
        count = self.cmb_settings_image.count()
        if count:
            self.cmb_settings_image.setCurrentIndex(
                (self.cmb_settings_image.currentIndex() + 1) % count
            )

    def _next_unsaved_image(self) -> None:
        count = self.cmb_settings_image.count()
        current = self.cmb_settings_image.currentIndex()
        for offset in range(1, count + 1):
            index = (current + offset) % count
            path = str(self.cmb_settings_image.itemData(index) or "")
            if path and not self._record_is_saved(path, current_parameters=True):
                self.cmb_settings_image.setCurrentIndex(index)
                return

    def _refresh_image_selector_labels(self) -> None:
        current = self.cmb_settings_image.currentIndex()
        saved_count = 0
        trained_count = 0
        self.cmb_settings_image.blockSignals(True)
        for index, path in enumerate(self.settings_image_paths):
            is_saved = self._record_is_saved(path)
            is_trained = self._record_is_trained(path)
            saved_count += int(is_saved)
            trained_count += int(is_trained)
            saved = "SAVED" if is_saved else "NOT SAVED"
            trained = "TRAINED" if is_trained else "NOT TRAINED"
            metadata = self.settings_image_metadata.get(
                os.path.normcase(os.path.abspath(path))
            )
            if metadata and metadata.cut_point:
                source = (
                    f"SN {metadata.panel_sn or 'UNASSIGNED'}  |  "
                    f"CUT POINT {metadata.cut_point:02d}/{metadata.cut_point_total:02d}  |  "
                    f"ROUTER {metadata.machine_result}  |  "
                )
            else:
                source = "UNASSIGNED IMAGE  |  "
            self.cmb_settings_image.setItemText(
                index,
                f"{source}{saved}  |  {trained}  |  {Path(path).name}",
            )
        if current >= 0:
            self.cmb_settings_image.setCurrentIndex(current)
        self.cmb_settings_image.blockSignals(False)
        total = len(self.settings_image_paths)
        self.lbl_workflow_progress.setText(
            f"SOBEL {saved_count:,}/{total:,} SAVED  |  "
            f"TRAINED {trained_count:,}/{total:,}  |  "
            f"{total - saved_count:,} LEFT"
        )

    def _update_image_workflow_status(self) -> None:
        count = len(self.settings_image_paths)
        has_image = bool(self.settings_image_path and Path(self.settings_image_path).is_file())
        for button in (
            self.btn_previous_image,
            self.btn_next_image,
            self.btn_next_unsaved,
        ):
            button.setEnabled(count > 1)
        self.btn_reset_sobel.setEnabled(has_image)
        self.btn_save_sobel.setEnabled(has_image)

        saved = has_image and self._record_is_saved(
            self.settings_image_path, current_parameters=True
        )
        trained = saved and self._record_is_trained(self.settings_image_path)
        metadata = self.settings_image_metadata.get(
            os.path.normcase(os.path.abspath(self.settings_image_path))
        ) if self.settings_image_path else None
        if metadata and metadata.cut_point:
            captured = (
                metadata.captured_at.strftime("%d-%m-%Y %H:%M:%S")
                if metadata.captured_at else "TIME UNKNOWN"
            )
            self.lbl_current_image_context.setText(
                f"PANEL SN {metadata.panel_sn or 'UNASSIGNED'}  |  "
                f"CUT POINT {metadata.cut_point:02d} OF {metadata.cut_point_total:02d}  |  "
                f"ROUTER RESULT {metadata.machine_result}  |  {captured}"
            )
            context_colour = (
                "#166534" if metadata.machine_result == "GOOD" else
                "#b91c1c" if metadata.machine_result == "NG" else "#334155"
            )
        else:
            self.lbl_current_image_context.setText(
                "PANEL NOT ASSIGNED  |  CUT POINT --  |  ROUTER RESULT UNKNOWN"
            )
            context_colour = "#334155"
        self.lbl_current_image_context.setStyleSheet(
            f"background:{context_colour}; color:white; border-radius:6px; "
            "padding:7px 10px; font-weight:800;"
        )
        self.lbl_sobel_save_status.setText(
            "SOBEL: SAVED" if saved else "SOBEL: NOT SAVED"
        )
        self.lbl_sobel_save_status.setStyleSheet(
            "background:#dcfce7; color:#166534; border-radius:6px; "
            "padding:7px; font-weight:800;"
            if saved else
            "background:#ffedd5; color:#9a3412; border-radius:6px; "
            "padding:7px; font-weight:800;"
        )
        self.lbl_training_status.setText(
            "TRAINING: TRAINED" if trained else "TRAINING: NOT TRAINED"
        )
        self.lbl_training_status.setStyleSheet(
            "background:#dcfce7; color:#166534; border-radius:6px; "
            "padding:7px; font-weight:800;"
            if trained else
            "background:#e2e8f0; color:#475569; border-radius:6px; "
            "padding:7px; font-weight:800;"
        )
        self.lbl_training_status.setToolTip(
            "TRAINED is set only after this saved image is included in a completed "
            "model-training workflow."
        )

    def _save_current_sobel(self) -> None:
        if self._settings_job_busy():
            return
        path = self.settings_image_path
        if not path or not Path(path).is_file():
            return
        crop = self._settings_crop()
        extractor = FeatureExtractor(device="cpu", sobel=self._sobel_from_controls())
        edge = extractor.edge_map(path, crop)
        if edge is None:
            QMessageBox.warning(
                self, "Sobel not saved", "Sobel edge could not be generated."
            )
            return
        try:
            check_write_target(SOBEL_OUTPUT_DIR, self._settings_protected_roots())
            SOBEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        except (OSError, ProtectedPathError) as exc:
            QMessageBox.warning(self, "Sobel not saved", str(exc))
            return
        digest = hashlib.sha256(
            self._sobel_record_key(path).encode("utf-8")
        ).hexdigest()[:12]
        output = SOBEL_OUTPUT_DIR / f"{Path(path).stem}_{digest}_sobel.png"
        if not cv2.imwrite(str(output), edge):
            QMessageBox.warning(self, "Sobel not saved", "Output file could not be written.")
            return
        key = self._sobel_record_key(path)
        previous = self.sobel_records.get(key, {})
        same_source = self._record_matches_source(path, previous)
        same_processing = (
            self._normalise_saved_sobel_parameters(
                previous.get("sobel_parameters")
            ) == self._sobel_parameter_record()
            and same_source
            and previous.get("crop", asdict(CropBox())) == asdict(crop)
        )
        self._invalidate_settings_trial()
        self.sobel_records[key] = {
            "source_path": str(Path(path).resolve()),
            "source": self._source_signature(path),
            "sobel_parameters": self._sobel_parameter_record(),
            "crop": asdict(crop),
            "sobel_output": str(output.resolve()),
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "trained": bool(previous.get("trained")) if same_processing else False,
            "trained_at": previous.get("trained_at", "") if same_processing else "",
            "trained_model": previous.get("trained_model", ""),
            "trained_label": previous.get("trained_label", ""),
            "trained_model_signature": previous.get("trained_model_signature"),
            "trained_sobel_parameters": (
                previous.get("trained_sobel_parameters", {})
            ),
            "training_label": previous.get("training_label", "") if same_source else "",
            "labelled_at": previous.get("labelled_at", "") if same_source else "",
            "product_id": previous.get("product_id", "") if same_source else "",
        }
        try:
            self._save_sobel_records()
        except OSError as exc:
            QMessageBox.warning(self, "Sobel status not saved", str(exc))
            return
        self._refresh_image_selector_labels()
        self._update_image_workflow_status()
        if hasattr(self, "cmb_training_image"):
            self._refresh_training_workflow_status()
            self._refresh_trial_model_status()

    def _sobel_from_controls(self) -> SobelConfig:
        kernel_values = (1, 3, 5, 7)
        return SobelConfig(
            blur_ksize=kernel_values[self.slider_sobel_blur.value()],
            blur_sigma=self.slider_blur_sigma.value() / 100.0,
            sobel_ksize=kernel_values[self.slider_sobel_kernel.value()],
            gradient_x_weight=self.slider_gradient_x.value() / 1000.0,
            gradient_y_weight=self.slider_gradient_y.value() / 1000.0,
            clip_percentile=self.slider_sobel_clip.value() / 100.0,
            edge_gain=self.slider_edge_gain.value() / 1000.0,
            noise_floor=self.slider_noise_floor.value(),
        ).validated()

    def _preview_sobel(self) -> bool:
        path = self.settings_image_path
        if not path or not Path(path).is_file():
            self.lbl_trial_result.setText("SELECT A VALID TEST IMAGE")
            return False
        original = cv2.imread(path, cv2.IMREAD_COLOR)
        if original is None:
            self.lbl_trial_result.setText("IMAGE COULD NOT BE READ")
            return False
        crop = self._settings_crop()
        cropped = crop.apply(original)
        extractor = FeatureExtractor(device="cpu", sobel=self._sobel_from_controls())
        edge = extractor.edge_map(path, crop)
        if edge is None:
            self.lbl_trial_result.setText("SOBEL EDGE COULD NOT BE GENERATED")
            return False

        pane_width, pane_height, header_height = 560, 520, 28
        canvas = np.full(
            (pane_height + header_height, pane_width * 2 + 8, 3), 11, dtype=np.uint8
        )
        canvas[header_height:, :pane_width] = _letterbox(
            cropped, pane_width, pane_height
        )
        canvas[header_height:, pane_width + 8:] = _letterbox(
            edge, pane_width, pane_height
        )
        cv2.putText(
            canvas, "ORIGINAL", (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (220, 225, 232), 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, "SOBEL EDGE", (pane_width + 16, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (220, 225, 232), 1, cv2.LINE_AA,
        )
        self.settings_preview_pixmap = bgr_to_pixmap(canvas)
        self._fit_settings_preview()
        return True

    def _train_settings_model(self) -> None:
        if self.worker is not None or self.auto_running:
            self.lbl_model_training_status.setText("STOP AUTO INSPECTION BEFORE TRAINING")
            return
        if DEMO_MODE:
            self.lbl_model_training_status.setText(
                "TRAINING IS DISABLED IN SCRIPTED DEMO MODE"
            )
            return
        if self.training_worker is not None:
            return
        if self.settings_worker is not None:
            self.lbl_model_training_status.setText(
                "WAIT FOR THE CURRENT MODEL TEST TO FINISH"
            )
            return
        product = self._settings_product_id()
        if not product:
            self.lbl_model_training_status.setText(
                "TRAINING BLOCKED: THE DATA SOURCE HAS NO SINGLE PRODUCT ID"
            )
            return
        items = self._training_items()
        counts = {
            label: sum(item_label == label for _, item_label in items)
            for label in ("GOOD", "NG")
        }
        if any(counts[label] < MIN_TRAINING_IMAGES_PER_CLASS for label in counts):
            self.lbl_model_training_status.setText(
                f"TRAINING BLOCKED: NEED {MIN_TRAINING_IMAGES_PER_CLASS} SAVED "
                f"IMAGES PER CLASS (GOOD {counts['GOOD']}, NG {counts['NG']})"
            )
            return

        target = self._model_path(product)
        try:
            check_write_target(target, self._settings_protected_roots())
            signatures = {path: self._source_signature(path) for path, _ in items}
        except (OSError, ProtectedPathError) as exc:
            self._settings_training_failed(str(exc))
            return
        if target.exists():
            answer = QMessageBox.question(
                self,
                "Replace trained model",
                f"Retrain and replace {target.name}?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        resume = (
            self.settings_classifier
            if isinstance(self.settings_classifier, CutClassifier)
            else None
        )
        self.training_paths_in_progress = [path for path, _ in items]
        self.training_input_signatures = signatures
        self._invalidate_settings_trial()
        self.training_model_target = target
        self.training_last_message = ""
        self.training_last_success = None
        self.training_progress.setVisible(True)
        self.training_progress.setRange(0, len(items))
        self.training_progress.setValue(0)
        self.lbl_model_training_status.setStyleSheet(
            "color:#1d4ed8; font-weight:800;"
        )
        self.lbl_model_training_status.setText(
            f"STARTING DINOv2 TRAINING FOR {product}..."
        )
        self.training_worker = ModelTrainingWorker(
            items,
            self._sobel_from_controls(),
            resume_from=resume,
        )
        self.training_worker.progress_changed.connect(self._training_progress_changed)
        self.training_worker.completed.connect(self._settings_training_done)
        self.training_worker.failed.connect(self._settings_training_failed)
        self.training_worker.finished.connect(self._settings_training_finished)
        self._lock_settings_job(True)
        self.training_worker.start()
        self._refresh_training_workflow_status()

    def _training_progress_changed(self, done: int, total: int, message: str) -> None:
        self.training_progress.setRange(0, max(1, total))
        self.training_progress.setValue(max(0, min(done, max(1, total))))
        self.lbl_model_training_status.setText(message.upper())

    def _settings_training_done(self, classifier: CutClassifier, report) -> None:
        if self.closing_requested:
            return
        target = self.training_model_target
        if target is None:
            self._settings_training_failed("Model target was not prepared.")
            return
        quality_error = model_quality_error(report)
        if quality_error:
            self._settings_training_failed(f"MODEL NOT ACCEPTED: {quality_error}")
            return
        try:
            if any(
                self._source_signature(path) != signature
                for path, signature in self.training_input_signatures.items()
            ):
                raise ValueError("Training images changed during training; run training again.")
            classifier.model_key = self._settings_product_id()
            classifier.save(target, protected=self._settings_protected_roots())
        except Exception as exc:
            self._settings_training_failed(f"Model could not be saved: {exc}")
            return

        trained_at = datetime.now().isoformat(timespec="seconds")
        sobel_parameters = self._sobel_config_record(classifier.sobel)
        for record in self.sobel_records.values():
            if record.get("trained_model") == target.name:
                record["trained"] = False
                record["trained_model"] = ""
        for path in self.training_paths_in_progress:
            key = self._sobel_record_key(path)
            record = dict(self.sobel_records.get(key, {}))
            record["trained"] = True
            record["trained_at"] = trained_at
            record["trained_model"] = target.name
            record["trained_label"] = record.get("training_label", "")
            record["trained_sobel_parameters"] = sobel_parameters
            record["trained_model_signature"] = self._source_signature(str(target))
            self.sobel_records[key] = record
        try:
            self._save_sobel_records()
        except OSError as exc:
            self._settings_training_failed(
                f"Model saved, but image training status could not be saved: {exc}"
            )
            return

        self.settings_classifier = classifier
        self.settings_model_name = target.name
        self.settings_model_error = ""
        self.training_progress.setRange(0, 100)
        self.training_progress.setValue(100)
        validation = (
            f"{report.val_acc:.1%}"
            if np.isfinite(report.val_acc) else "NOT AVAILABLE"
        )
        self.lbl_model_training_status.setStyleSheet(
            "color:#166534; font-weight:800;"
        )
        self.training_last_success = True
        self.training_last_message = (
            f"TRAINING COMPLETE | VALIDATION {validation} | {target.name}"
        )
        self.lbl_model_training_status.setText(self.training_last_message)
        self._refresh_image_selector_labels()
        self._refresh_trial_model_status()
        self._show_training_image()

    def _settings_training_failed(self, message: str) -> None:
        self.training_last_success = False
        self.training_last_message = f"TRAINING FAILED | {message}"
        self.lbl_model_training_status.setStyleSheet(
            "color:#b91c1c; font-weight:800;"
        )
        self.lbl_model_training_status.setText(self.training_last_message)

    def _settings_training_finished(self) -> None:
        worker = self.training_worker
        if worker is not None:
            worker.deleteLater()
            self.training_worker = None
        self.training_paths_in_progress = []
        self.training_model_target = None
        self.training_input_signatures = {}
        self._lock_settings_job(False)
        self._refresh_training_workflow_status()

    def _settings_protected_roots(self) -> list[Path]:
        roots = protected_roots(self.cfg)
        if self.auo_dataset is not None:
            roots.append(Path(self.auo_dataset.root).resolve())
        return roots

    def _apply_settings(self, _checked: bool = False) -> bool:
        if self.worker is not None or self.auto_running or self._settings_job_busy():
            return False
        try:
            approved = (
                self.settings_trial_approved
                and self.trial_input_signature == self._trial_signature()
                and not self._settings_model_needs_retraining()
            )
        except OSError:
            approved = False
        if not approved:
            self._invalidate_settings_trial()
            self.lbl_trial_result.setText("RUN A COMPLETE MODEL TEST BEFORE SAVING SETTINGS")
            return False
        dataset = self.auo_dataset
        product = self._settings_product_id()
        if dataset is None or not product or not dataset.result_dir:
            self.lbl_trial_result.setText("SELECT A VALID AUO6000 DATA SOURCE")
            return False
        if dataset.expected_cut_points <= 0 or any(not image.result_file for image in dataset.images):
            self.lbl_trial_result.setText("IMAGE / PANEL ASSIGNMENT IS AMBIGUOUS; SETTINGS NOT SAVED")
            return False
        sobel = self._sobel_from_controls()
        source_runs = [
            run for day in available_days(dataset.result_dir)
            for run in load_runs(dataset.result_dir, day)
        ]
        expected = dict(self.cfg.expected_images_by_route)
        for run in source_runs:
            expected[run.key] = dataset.expected_cut_points
        route = source_runs[0].key if source_runs else product
        candidate = replace(
            self.cfg,
            picture_dir=dataset.picture_dir,
            result_dir=dataset.result_dir,
            recipe_dir=dataset.recipe_dir,
            eqp_cfg_path=str(Path(dataset.root) / "Config" / "Eqp.cfg"),
            active_recipe=route,
            expected_images_by_route=expected,
            sobel_blur_ksize=sobel.blur_ksize,
            sobel_blur_sigma=sobel.blur_sigma,
            sobel_ksize=sobel.sobel_ksize,
            sobel_gradient_x_weight=sobel.gradient_x_weight,
            sobel_gradient_y_weight=sobel.gradient_y_weight,
            sobel_clip_percentile=sobel.clip_percentile,
            sobel_edge_gain=sobel.edge_gain,
            sobel_noise_floor=sobel.noise_floor,
            good_confidence_min=self.spin_confidence.value() / 100.0,
        )
        try:
            candidate.save(CONFIG_PATH)
        except (OSError, ValueError, ProtectedPathError) as exc:
            QMessageBox.critical(self, "Settings not saved", str(exc))
            return False
        self.cfg = candidate
        self.link.fault("Settings changed; press ACK / RESET before inspection")
        self._load_dataset(dataset)
        self.lbl_trial_result.setStyleSheet(
            "background:#dbeafe; color:#1d4ed8; border-radius:7px; "
            "padding:10px; font-weight:800;"
        )
        self.lbl_trial_result.setText(
            f"SETTINGS APPLIED\nGOOD THRESHOLD {self.cfg.good_confidence_min:.1%}"
        )
        self.lbl_detail.setText(
            f"Production confidence threshold set to {self.cfg.good_confidence_min:.1%}."
        )
        return True

    def _test_vision_transformer(self) -> None:
        if (self.worker and self.worker.isRunning()) or self.auto_running:
            self.lbl_trial_result.setText("STOP AUTO INSPECTION BEFORE RUNNING A TRIAL")
            return
        if self._settings_job_busy():
            return
        self._invalidate_settings_trial()
        if self.settings_classifier is None:
            self.lbl_trial_result.setStyleSheet(
                "background:#fee2e2; color:#b91c1c; border:1px solid #ef4444; "
                "border-radius:7px; padding:10px; font-weight:800;"
            )
            self.lbl_trial_result.setText(
                f"MODEL NOT READY  |  {self.settings_model_error}"
            )
            self.trial_preview_pixmap = None
            self.lbl_trial_preview.clear()
            self.lbl_trial_preview.setMinimumSize(1, 200)
            self.lbl_trial_preview.setText(
                "PREDICTION IMAGES ARE NOT AVAILABLE YET\n\n"
                "Complete GOOD/NG labeling and model training in steps 1 and 2 above."
            )
            return
        if self._settings_model_needs_retraining():
            self._refresh_trial_model_status()
            return
        if not self.settings_image_paths:
            self.lbl_trial_result.setText("SELECT IMAGES BEFORE TESTING")
            return
        if not self._preview_sobel():
            return
        try:
            self.trial_input_signature = self._trial_signature()
        except OSError as exc:
            self._vision_trial_failed(str(exc))
            return
        if isinstance(self.settings_classifier, DemoClassifier):
            trial_sobel = self._sobel_from_controls()
            self.settings_classifier.extractor.sobel = trial_sobel
        self.btn_trial.setEnabled(False)
        self.btn_apply_settings.setEnabled(False)
        self.lbl_trial_result.setStyleSheet(
            "background:#ffedd5; color:#9a3412; border-radius:7px; "
            "padding:10px; font-weight:800;"
        )
        self.lbl_trial_result.setText("VISION TRANSFORMER: RUNNING...")
        self.trial_preview_pixmap = None
        self.lbl_trial_preview.clear()
        self.lbl_trial_preview.setMinimumSize(1, 200)
        self.lbl_trial_preview.setText(
            f"ANALYZING {len(self.settings_image_paths)} IMAGES..."
        )
        self.settings_worker = VisionTrialWorker(
            self.settings_classifier, self.settings_image_paths
        )
        self.settings_worker.completed.connect(self._vision_trial_done)
        self.settings_worker.failed.connect(self._vision_trial_failed)
        self.settings_worker.finished.connect(self._vision_trial_finished)
        self._lock_settings_job(True)
        self.settings_worker.start()
        self._refresh_training_workflow_status()

    def _vision_trial_done(self, results: list[tuple[str, str, float]]) -> None:
        if self.closing_requested:
            return
        try:
            if self.trial_input_signature != self._trial_signature():
                raise ValueError("Test inputs changed. Run the model test again.")
            if not results or [item[0] for item in results] != self.settings_image_paths:
                raise ValueError("The model returned an incomplete or mismatched image set.")
            for _, label, confidence in results:
                if label not in {"GOOD", "NG"} or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError("The model returned an invalid label or confidence.")
        except (OSError, TypeError, ValueError) as exc:
            self._vision_trial_failed(str(exc))
            return
        threshold = self.spin_confidence.value() / 100.0
        details: list[ImageDecision] = []
        passed_count = 0
        model_good_count = 0
        model_ng_count = 0
        below_threshold_count = 0
        for index, (image_path, label, confidence) in enumerate(results):
            passed = label == "GOOD" and confidence >= threshold
            passed_count += int(passed)
            model_good_count += int(label == "GOOD")
            model_ng_count += int(label == "NG")
            below_threshold_count += int(label == "GOOD" and not passed)
            details.append(
                ImageDecision(index, image_path, label, float(confidence))
            )
        failed_count = len(results) - passed_count
        batch_passed = bool(results) and failed_count == 0
        final = "GOOD" if batch_passed else "NG"
        colour = "#15803d" if batch_passed else "#dc2626"
        note = "SCRIPTED DEMO" if DEMO_MODE else "VISION TRANSFORMER"
        self.trial_reason = (
            f"MODEL GOOD {model_good_count}  |  MODEL NG {model_ng_count}  |  "
            f"BELOW THRESHOLD {below_threshold_count}  |  "
            f"ACCEPTED {passed_count}/{len(results)}  |  THRESHOLD {threshold:.1%}"
        )
        self.trial_result = PanelDecision(
            run=self.current_run,
            status=final,
            note=self.trial_reason,
            checked=len(details),
            details=details,
        )
        # Keep the result set in memory, but render only the selected image.
        # A monolithic 14k-image canvas can exceed 4 GiB before Qt copies it.
        self.trial_predictions = {
            image_path: (label, float(confidence))
            for image_path, label, confidence in results
        }
        self.settings_trial_approved = True
        self.btn_apply_settings.setEnabled(not self._settings_job_busy())
        self._show_training_image()
        self.lbl_trial_result.setStyleSheet(
            f"background:{colour}; color:white; border-radius:7px; "
            "padding:10px; font-weight:700;"
        )
        self.lbl_trial_result.setText(
            f"{final}  |  {note}  |  {self.trial_reason}"
        )

    def _vision_trial_failed(self, message: str) -> None:
        self._invalidate_settings_trial()
        self.lbl_trial_result.setStyleSheet(
            "background:#dc2626; color:white; border-radius:7px; "
            "padding:10px; font-weight:800;"
        )
        self.lbl_trial_result.setText(f"VISION TRANSFORMER ERROR\n{message}")
        self.trial_preview_pixmap = None
        self.lbl_trial_preview.clear()
        self.lbl_trial_preview.setMinimumSize(1, 200)
        self.lbl_trial_preview.setText("PREDICTION IMAGES COULD NOT BE GENERATED")

    def _vision_trial_finished(self) -> None:
        worker = self.settings_worker
        if worker is not None:
            worker.deleteLater()
            self.settings_worker = None
        self._lock_settings_job(False)
        self.btn_apply_settings.setEnabled(self.settings_trial_approved)
        self._refresh_training_workflow_status()

    def _heartbeat(self) -> None:
        self.link.toggle_heartbeat()
        now = datetime.now()
        self.lbl_datetime.setText(now.strftime("%H:%M:%S  |  %d-%m-%Y"))
        self._refresh_signals()

    def _refresh_gate(self) -> None:
        state = self.link.state
        colour = COLORS[state]
        self.state_banner.setStyleSheet(
            f"QFrame {{ background:{colour}; border:0; border-radius:10px; }}"
            "QLabel { background:transparent; color:white; border:0; }"
        )
        self.lbl_state_big.setText(STATE_TEXT[state])
        self.lbl_alarm.setText(self.link.reason)
        self.lbl_alarm.setVisible(state is GateState.FAULT)
        self._refresh_signals()

    def _refresh_signals(self) -> None:
        signals = self.link.signals
        values = {
            "CONNECTED": signals.connected,
            "VISION HEALTHY": signals.healthy,
            "HEARTBEAT": signals.heartbeat,
            "RESULT VALID": signals.result_valid,
            "PASS": signals.pass_signal,
            "NG": signals.ng_signal,
            "RELEASE": signals.release,
        }
        for name, enabled in values.items():
            label = self.signal_labels[name]
            label.setText("ON" if enabled else "OFF")
            if enabled and name == "NG":
                on_colour = "#dc2626"
            elif enabled and name in ("RESULT VALID", "HEARTBEAT"):
                on_colour = "#2563eb"
            else:
                on_colour = "#15803d"
            label.setStyleSheet(
                f"background:{on_colour if enabled else '#e5e7eb'};"
                f"color:{'white' if enabled else '#6b7280'};"
                "border-radius:4px; padding:4px; font-weight:700;"
            )

    def _set_busy(self, busy: bool) -> None:
        self.btn_model.setEnabled(not busy)
        self.btn_next.setEnabled(not busy and bool(self.queue))
        self.cmb_recipe.setEnabled(not busy and not self.auto_running)
        self.cmb_day.setEnabled(not busy and not self.auto_running)
        self.btn_settings.setEnabled(not busy and not self.auto_running)

    def _worker_failed(self, message: str) -> None:
        self.auto_running = False
        self.auto_timer.stop()
        self.btn_auto.setText("▶ START AUTO")
        self.progress.setVisible(False)
        self._set_busy(False)
        reason = "vision worker failed: " + (message.strip().splitlines() or ["unknown error"])[-1]
        self.link.fault(reason)
        self.lbl_detail.setText(message)
        if self.current_run is not None:
            result = PanelDecision(run=self.current_run, note=reason)
            self._append_log(result, VisionVerdict.FAULT, reason, 0.0)
        self._refresh_gate()

    def _inspection_thread_finished(self) -> None:
        worker = self.worker
        if worker is not None:
            worker.deleteLater()
            self.worker = None
        self._set_busy(False)

    def _fit_image(self) -> None:
        if self.image_pixmap is None:
            return
        self.image.setPixmap(
            self.image_pixmap.scaled(
                max(10, self.image.width() - 12),
                max(10, self.image.height() - 12),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _fit_settings_preview(self) -> None:
        if self.settings_preview_pixmap is None:
            return
        self.lbl_settings_preview.setPixmap(
            self.settings_preview_pixmap.scaled(
                max(10, self.lbl_settings_preview.width() - 8),
                max(10, self.lbl_settings_preview.height() - 8),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def _layout_sobel_parameter_cards(self) -> None:
        """Keep tuning controls readable instead of allowing horizontal scroll."""
        scroll = getattr(self, "sobel_parameter_scroll", None)
        cards = getattr(self, "sobel_parameter_cards", None)
        layout = getattr(self, "sobel_parameters_layout", None)
        if scroll is None or not cards or layout is None:
            return
        columns = 2 if scroll.viewport().width() >= 620 else 1
        if columns == self.sobel_parameter_columns:
            return
        for card in cards:
            layout.removeWidget(card)
        for index, card in enumerate(cards):
            layout.addWidget(card, index // columns, index % columns)
        self.sobel_parameter_columns = columns
        layout.invalidate()

    def _fit_trial_preview(self) -> None:
        if self.trial_preview_pixmap is None:
            return
        viewport = self.trial_preview_scroll.viewport()
        target_width = max(1, viewport.width() - 10)
        pixmap = self.trial_preview_pixmap
        readable = pixmap.scaled(
            target_width, max(1, viewport.height() - 10),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.lbl_trial_preview.setPixmap(readable)

    def _layout_vision_workspace(self) -> None:
        workspace = getattr(self, "vision_workspace", None)
        if workspace is None:
            return
        orientation = Qt.Vertical if workspace.width() < 1150 else Qt.Horizontal
        if workspace.orientation() != orientation:
            workspace.setOrientation(orientation)
            workspace.setSizes([270, 360] if orientation == Qt.Vertical else [760, 500])
        self._fit_trial_preview()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._fit_image()
        self._fit_settings_preview()
        self._fit_trial_preview()
        self._layout_sobel_parameter_cards()
        self._layout_vision_workspace()

    def closeEvent(self, event) -> None:  # noqa: N802
        self.closing_requested = True
        self.auto_running = False
        self.auto_timer.stop()
        self.heartbeat_timer.stop()
        self.link.disconnect("application closing; conveyor held")
        running = False
        for worker in (self.worker, self.settings_worker, self.training_worker):
            if worker and worker.isRunning():
                worker.requestInterruption()
                running = True
        if running:
            self.setWindowTitle("AVTR - Waiting for the current operation to stop...")
            event.ignore()
            QTimer.singleShot(250, self.close)
            return
        event.accept()


def main() -> int:
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName(APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    window = MainWindow()
    window.showMaximized()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
