"""Review original/Sobel images and select the PCB side; no model-input edits."""

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from .guard import ProtectedPathError
from .machine_reference import sha256
from .model import CropBox, FeatureExtractor
from .training_edge import load_edge_review, saved_edge

EDGE_REVIEW_STYLE = """
QDialog#pcbEdgeReview { background:#f2f5fa; }
QWidget { font-family:'Segoe UI'; font-size:10pt; }
QLabel { background:transparent; border:0; color:#172b4d; }
QFrame#reviewHeader, QFrame#reviewSelection, QFrame#reviewPane {
    background:white; border:1px solid #dce4ef; border-radius:12px;
}
QLabel#reviewEyebrow { color:#2563eb; font-size:9pt; font-weight:700; }
QLabel#reviewTitle { color:#14243e; font-size:19pt; font-weight:700; }
QLabel#reviewMuted { color:#64748b; font-size:9pt; }
QLabel#reviewCut { color:#1d4ed8; background:#edf4ff; border:1px solid #d6e6ff;
    border-radius:9px; padding:11px 16px; font-size:12pt; font-weight:700; }
QLabel#reviewPaneTitle { color:#233653; font-size:11pt; font-weight:700; }
QLabel#reviewLegend { color:#047caa; font-size:9pt; }
QLabel#reviewImage { background:#0c1422; border-radius:6px; }
QLabel#reviewStatus { color:#596b83; background:#f0f4f9; border-radius:7px;
    padding:9px 12px; font-size:9pt; }
QLabel#reviewStatus[state="saved"] { color:#166534; background:#edf9f1; }
QLabel#reviewStatus[state="changed"] { color:#925d0b; background:#fff8e9; }
QLabel#reviewStatus[state="error"] { color:#b42318; background:#fff0ee; }
QPushButton#reviewLoad, QPushButton#reviewCancel { background:white; color:#465a76;
    border:1px solid #ccd7e5; border-radius:7px; padding:8px 14px; font-size:9pt; }
QPushButton#reviewLoad:hover, QPushButton#reviewCancel:hover { background:#edf4ff; border-color:#8eb5ee; }
QPushButton#reviewSave { background:#2563eb; color:white; border:0; border-radius:8px;
    padding:10px 22px; font-size:10pt; font-weight:700; }
QPushButton#reviewSave:hover { background:#1d4ed8; }
QPushButton#reviewSave:disabled { background:#cdd8e6; color:#f8fafc; }
QPushButton#reviewView { background:transparent; color:#64748b; border:1px solid #d5dfec;
    border-radius:6px; padding:5px 12px; font-size:9pt; }
QPushButton#reviewView:checked { background:#e7efff; color:#1d4ed8; border-color:#a9c4f5; }
QPushButton#reviewView:disabled { color:#a4b0c1; background:#f2f5f9; }
QComboBox#reviewChoice { background:#f8fbff; color:#172b4d; border:1px solid #b5c9e3;
    border-radius:8px; padding:10px 14px; font-size:11pt; font-weight:600; }
QComboBox#reviewChoice:focus { border:2px solid #2563eb; }
QComboBox#reviewChoice:disabled { color:#a0aec0; background:#f1f4f8; }
QComboBox#reviewChoice QAbstractItemView { background:white; color:#172b4d;
    selection-background-color:#e7efff; selection-color:#1d4ed8; }
"""


class TrainingEdgeDialog(QDialog):
    def __init__(
        self,
        image_path,
        inputs,
        choices,
        sobel,
        manifest_path=None,
        parent=None,
        save_callback=None,
    ):
        super().__init__(parent)
        self.setObjectName("pcbEdgeReview")
        self.setStyleSheet(EDGE_REVIEW_STYLE)
        self.setWindowTitle("AVTR | Select PCB edge for this cut point")
        screen = self.screen().availableGeometry()
        self.setMinimumSize(820, 580)
        self.resize(min(1250, screen.width() - 50), min(780, screen.height() - 50))
        self.image_path, self.inputs, self.choices = image_path, inputs, choices
        self.save_callback = save_callback
        self.manifest_path = str(manifest_path or "")
        self.review = None
        self._saved_edge = None
        self.detail_requested = True
        self.display_roi = None
        self.preview_pixmaps = []
        self.preview_array = None
        self.image_hash = sha256(image_path)
        self.image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError("Image could not be decoded.")
        self.sobel = FeatureExtractor(device="cpu", sobel=sobel).edge_map(
            image_path, CropBox(0, 0, self.image.shape[1], self.image.shape[0])
        )
        if self.sobel is None or sha256(image_path) != self.image_hash:
            raise ValueError("Sobel edge could not be generated.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)
        header = QFrame()
        header.setObjectName("reviewHeader")
        header_row = QHBoxLayout(header)
        header_row.setContentsMargins(18, 12, 18, 12)
        heading = QVBoxLayout()
        heading.setSpacing(3)
        eyebrow = QLabel("IMAGE TRAINING / EDGE REVIEW")
        eyebrow.setObjectName("reviewEyebrow")
        title = QLabel("Select the PCB edge")
        title.setObjectName("reviewTitle")
        context = QLabel(f"{Path(image_path).name}  |  {inputs['product_id']}")
        context.setObjectName("reviewMuted")
        context.setWordWrap(True)
        context.setTextInteractionFlags(Qt.TextSelectableByMouse)
        for label in (eyebrow, title, context):
            heading.addWidget(label)
        header_row.addLayout(heading, 1)
        cut = QLabel(f"CUT {inputs['cut_point']:02d}")
        cut.setObjectName("reviewCut")
        header_row.addWidget(cut)
        layout.addWidget(header)

        view_row = QHBoxLayout()
        hint = QLabel("Compare the same area in both images")
        hint.setObjectName("reviewMuted")
        view_row.addWidget(hint, 1)
        self.detail_button = QPushButton("EDGE DETAIL")
        self.full_button = QPushButton("FULL IMAGE")
        for button in (self.detail_button, self.full_button):
            button.setCheckable(True)
            button.setObjectName("reviewView")
            view_row.addWidget(button)
        self.detail_button.clicked.connect(lambda: self.set_detail_view(True))
        self.full_button.clicked.connect(lambda: self.set_detail_view(False))
        layout.addLayout(view_row)

        panes = QHBoxLayout()
        panes.setSpacing(12)
        self.original_preview = self.make_preview_pane(panes, "Original", "Source image")
        self.preview = self.make_preview_pane(panes, "Sobel + A/B", "A · upper     B · lower")
        layout.addLayout(panes, 1)

        selection = QFrame()
        selection.setObjectName("reviewSelection")
        selection_row = QHBoxLayout(selection)
        selection_row.setContentsMargins(16, 12, 16, 12)
        selection_row.setSpacing(16)
        explanation = QVBoxLayout()
        explanation.setSpacing(3)
        question = QLabel("PCB side to inspect")
        question.setObjectName("reviewPaneTitle")
        note = QLabel("A: upper PCB  ·  B: lower PCB")
        note.setObjectName("reviewMuted")
        explanation.addWidget(question)
        explanation.addWidget(note)
        selection_row.addLayout(explanation)
        self.load_button = QPushButton("LOAD REFERENCE DATA")
        self.load_button.setObjectName("reviewLoad")
        self.load_button.clicked.connect(self.choose_manifest)
        self.choice = QComboBox()
        self.choice.setObjectName("reviewChoice")
        self.choice.setMinimumWidth(210)
        for text, value in (("NOT SURE", "UNSURE"), ("A - UPPER PCB", "A"), ("B - LOWER PCB", "B")):
            self.choice.addItem(text, value)
        self.choice.currentIndexChanged.connect(self.selection_changed)
        selection_row.addWidget(self.choice)
        self.status = QLabel()
        self.status.setObjectName("reviewStatus")
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(0)
        self.status.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        selection_row.addWidget(self.status, 1)
        layout.addWidget(selection)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.load_button)
        actions.addStretch(1)
        self.cancel_button = QPushButton("CANCEL")
        self.cancel_button.setObjectName("reviewCancel")
        self.cancel_button.setMinimumWidth(105)
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.cancel_button)
        self.save_button = QPushButton("SAVE EDGE CHOICE")
        self.save_button.setObjectName("reviewSave")
        self.save_button.setMinimumHeight(42)
        self.save_button.clicked.connect(self.confirm)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        scope = QLabel(
            "Saved for this recipe and cut point. GOOD / NG stays separate.  ·  Calibration unverified."
        )
        scope.setObjectName("reviewMuted")
        scope.setWordWrap(True)
        scope.setToolTip(
            "The choice is shared only within the same product, recipe version, table, program and cut point. "
            "Each image uses its own recorded alignment. Reference position and mm calibration remain experimental."
        )
        layout.addWidget(scope)
        self.load_reference(self.manifest_path)

    def make_preview_pane(self, row, title, subtitle):
        pane = QFrame()
        pane.setObjectName("reviewPane")
        content = QVBoxLayout(pane)
        content.setContentsMargins(8, 10, 8, 8)
        content.setSpacing(9)
        caption = QHBoxLayout()
        caption.setContentsMargins(6, 0, 6, 0)
        heading = QLabel(title)
        heading.setObjectName("reviewPaneTitle")
        caption.addWidget(heading)
        caption.addStretch(1)
        detail = QLabel(subtitle)
        detail.setObjectName("reviewLegend" if title.startswith("Sobel") else "reviewMuted")
        caption.addWidget(detail)
        content.addLayout(caption)
        preview = QLabel()
        preview.setObjectName("reviewImage")
        preview.setAlignment(Qt.AlignCenter)
        preview.setMinimumSize(1, 140)
        preview.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        content.addWidget(preview, 1)
        row.addWidget(pane, 1)
        return preview

    def set_status(self, text, state="pending"):
        self.status.setText(text)
        self.status.setProperty("state", state)
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)

    def selection_changed(self, *_):
        if self.review:
            value = self.choice.currentData()
            if value == self._saved_edge:
                self.set_status(f"Saved choice: {value}", "saved")
            else:
                self.set_status("Not saved · confirm with SAVE EDGE CHOICE.", "changed")
        self.render()

    def set_detail_view(self, enabled):
        self.detail_requested = enabled
        self.render()

    def choose_manifest(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recorded A/B reference data", self.manifest_path, "Reference data (*.json)"
        )
        if path:
            self.load_reference(path)

    def load_reference(self, path):
        self.review = None
        self._saved_edge = None
        try:
            if not path:
                raise ValueError("Load matching reference data to display A/B. No line is guessed.")
            review = load_edge_review(path, self.image_path, **self.inputs)
            if review["A"]["provenance"]["image_sha256"] != self.image_hash:
                raise ValueError("Image changed. Close and reopen the edge review.")
            self.review, self.manifest_path = review, str(path)
            value = saved_edge(self.choices, review)
            self._saved_edge = value
            self.choice.blockSignals(True)
            self.choice.setCurrentIndex(self.choice.findData(value or "UNSURE"))
            self.choice.blockSignals(False)
            self.set_status(
                f"Saved choice: {value}" if value else "Not reviewed · choose A, B or NOT SURE.",
                "saved" if value else "pending",
            )
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            self.set_status(str(exc), "error")
        self.choice.setEnabled(self.review is not None)
        self.save_button.setEnabled(self.review is not None)
        self.render()

    def confirm(self):
        # Reject source changes while the trainer was reviewing the picture.
        try:
            fresh = load_edge_review(self.manifest_path, self.image_path, **self.inputs)
            if self.review is None or fresh["manifest_sha256"] != self.review["manifest_sha256"]:
                raise ValueError("Reference data changed. Reload and review before saving.")
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            self.set_status(str(exc), "error")
            self.review = None
            self.choice.setEnabled(False)
            self.save_button.setEnabled(False)
            self.render()
            return
        if self.save_callback is not None:
            try:
                self.save_callback(fresh, self.choice.currentData())
            except (OSError, ValueError, KeyError, TypeError, ProtectedPathError) as exc:
                self.set_status(f"Edge choice not saved: {exc}", "error")
                return
        self.accept()

    def render(self, *_):
        drawn = cv2.cvtColor(self.sobel, cv2.COLOR_GRAY2BGR)
        selected = self.choice.currentData()
        if self.review:
            for name in ("A", "B"):
                p, q = np.asarray(self.review[name]["line"])
                cv2.line(
                    drawn,
                    tuple(np.rint(p * 256).astype(int)),
                    tuple(np.rint(q * 256).astype(int)),
                    (255, 190, 0),
                    4 if selected == name else 2,
                    cv2.LINE_AA,
                    shift=8,
                )
                label = name
                x = min(drawn.shape[1] - 40, max(0, round(q[0]) + 14))
                y = min(drawn.shape[0] - 14, max(28, round(q[1]) - 8))
                cv2.rectangle(drawn, (x - 4, y - 23), (x + 25, y + 5), (35, 25, 12), -1)
                cv2.putText(
                    drawn,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 190, 0),
                    2,
                    cv2.LINE_AA,
                )
        # One display-only ROI for both panes; all reference coordinates stay full-size.
        h, w = self.image.shape[:2]
        x0, y0, x1, y1 = 0, 0, w, h
        detail = bool(self.review and self.detail_requested)
        self.detail_button.setEnabled(self.review is not None)
        self.detail_button.setChecked(detail)
        self.full_button.setChecked(not detail)
        if detail:
            points = np.vstack([self.review["A"]["line"], self.review["B"]["line"]])
            lo = np.floor(points.min(axis=0) - [120, 120]).astype(int)
            hi = np.ceil(points.max(axis=0) + [120, 120]).astype(int)
            x0, y0 = max(0, lo[0]), max(0, lo[1])
            x1, y1 = min(w, hi[0]), min(h, hi[1])
        self.display_roi = (int(x0), int(y0), int(x1), int(y1))
        frames = [self.image[y0:y1, x0:x1], drawn[y0:y1, x0:x1]]
        self.preview_array = np.hstack(frames)
        self.preview_pixmaps = []
        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            self.preview_pixmaps.append(
                QPixmap.fromImage(
                    QImage(rgb.data, width, height, rgb.strides[0], QImage.Format_RGB888).copy()
                )
            )
        self.fit_preview()

    def fit_preview(self):
        if not self.preview_pixmaps:
            return
        for preview, pixmap in zip((self.original_preview, self.preview), self.preview_pixmaps):
            preview.setPixmap(
                pixmap.scaled(
                    max(1, preview.width()),
                    max(1, preview.height()),
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
            )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "preview_array"):
            self.fit_preview()
