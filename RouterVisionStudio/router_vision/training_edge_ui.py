"""Review original/Sobel images and select the PCB side; no model-input edits."""

from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QEvent, QSignalBlocker, Qt
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
from .training_edge import (
    confirmation_key,
    confirmation_matches,
    edge_snapshot,
    load_edge_review,
    make_confirmation,
    measure_image_edge,
    saved_edge,
)

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
        confirmations=None,
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
        self.confirmations = confirmations if isinstance(confirmations, dict) else {}
        self.snapshot = None
        self.clicked_edge_point = None
        self.sobel_settings = asdict(sobel.validated())
        self.manifest_path = str(manifest_path or "")
        self.review = None
        self.measurement = None
        self.measurement_profile = None
        self.measurement_requested = False
        self._edge_measurements = {}
        self._saved_edge = None
        self.detail_requested = True
        self.display_roi = None
        self.base_roi = None
        self.zoom_roi = None
        self.pan_start = None
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
        hint = QLabel("Select A/B or click a detected edge in the Sobel image")
        hint.setObjectName("reviewMuted")
        view_row.addWidget(hint, 1)
        self.machine_button = QPushButton("MACHINE REFERENCE")
        self.machine_button.setCheckable(True)
        self.machine_button.setObjectName("reviewView")
        self.machine_button.setToolTip(
            "Show recorded machine tangents for comparison. Measurements use the image baseline."
        )
        self.machine_button.toggled.connect(self.render)
        view_row.addWidget(self.machine_button)
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
        self.preview = self.make_preview_pane(
            panes, "Sobel + selected edge", "Cyan: edge · Blue: guide · Green: PCB zero"
        )
        self.preview.setCursor(Qt.CrossCursor)
        self.preview.installEventFilter(self)
        layout.addLayout(panes, 1)

        zoom_row = QHBoxLayout()
        zoom_hint = QLabel("Wheel: zoom · Right-drag: pan · Left-click: select edge")
        zoom_hint.setObjectName("reviewMuted")
        zoom_hint.setWordWrap(True)
        zoom_row.addWidget(zoom_hint, 1)
        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("+")
        self.zoom_reset_button = QPushButton("FIT")
        self.inner_focus_button = QPushButton("MAX INNER")
        self.outer_focus_button = QPushButton("MAX OUTER")
        self.zoom_label = QLabel("1.0×")
        zoom_row.addWidget(self.zoom_label)
        for button in (
            self.zoom_out_button,
            self.zoom_in_button,
            self.zoom_reset_button,
            self.inner_focus_button,
            self.outer_focus_button,
        ):
            button.setObjectName("reviewLoad")
            zoom_row.addWidget(button)
        self.zoom_out_button.clicked.connect(lambda: self.zoom_preview(1 / 1.25))
        self.zoom_in_button.clicked.connect(lambda: self.zoom_preview(1.25))
        self.zoom_reset_button.clicked.connect(self.reset_zoom)
        self.inner_focus_button.clicked.connect(lambda: self.focus_maximum("inner"))
        self.outer_focus_button.clicked.connect(lambda: self.focus_maximum("outer"))
        layout.addLayout(zoom_row)

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
        self.edge_side_note = note
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

        self.measurement_label = QLabel("Select A or B to measure inner / outer MAX.")
        self.measurement_label.setWordWrap(True)
        self.measurement_label.setStyleSheet(
            "background:#edf4ff; color:#233653; border-radius:7px; padding:7px 10px; font-size:9pt;"
        )
        self.measurement_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        measurement_row = QHBoxLayout()
        self.measure_button = QPushButton("MEASUREMENT EDGE")
        self.measure_button.setObjectName("reviewLoad")
        self.measure_button.setMinimumHeight(42)
        self.measure_button.setToolTip(
            "Show maximum Inner Cut / Outer Cut for the selected PCB side in mm."
        )
        self.measure_button.clicked.connect(self.show_edge_measurement)
        measurement_row.addWidget(self.measure_button)
        measurement_row.addWidget(self.measurement_label, 1)
        layout.addLayout(measurement_row)
        self.cut_measurement_label = QLabel()
        self.cut_measurement_label.setWordWrap(True)
        self.cut_measurement_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.cut_measurement_label.setStyleSheet(
            "background:white; color:#172b4d; border:1px solid #b5c9e3; "
            "border-radius:7px; padding:10px; font-size:12pt; font-weight:600;"
        )
        self.cut_measurement_label.hide()
        layout.addWidget(self.cut_measurement_label)

        actions = QHBoxLayout()
        actions.setSpacing(10)
        actions.addWidget(self.load_button)
        actions.addStretch(1)
        self.cancel_button = QPushButton("CANCEL")
        self.cancel_button.setObjectName("reviewCancel")
        self.cancel_button.setMinimumWidth(105)
        self.cancel_button.clicked.connect(self.reject)
        actions.addWidget(self.cancel_button)
        self.save_button = QPushButton("SAVE EDGE")
        self.save_button.setObjectName("reviewSave")
        self.save_button.setMinimumHeight(42)
        self.save_button.clicked.connect(self.confirm)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        scope = QLabel(
            "A/B follows the cut point. Edge confirmation is for this image only. GOOD / NG stays separate."
        )
        scope.setObjectName("reviewMuted")
        scope.setWordWrap(True)
        scope.setToolTip(
            "The choice is shared only within the same product, recipe version, table, program and cut point. "
            "Every image gets a newly detected edge. Confirmations are annotations, not a trained edge model. "
            "Machine pixel scale is reused; physical measurement accuracy remains unverified."
        )
        layout.addWidget(scope)
        self.load_reference(self.manifest_path)

    def make_preview_pane(self, row, title, subtitle):
        pane = QFrame()
        pane.setObjectName("reviewPane")
        pane.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Expanding)
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
        self.clicked_edge_point = None
        if self.review:
            value = self.choice.currentData()
            if value == self._saved_edge:
                self.set_status(f"Saved side: {value} · review this image's edge.")
            else:
                self.set_status("Not saved · review the selected edge, then SAVE EDGE.", "changed")
        self.update_measurement()
        self.render()

    def update_measurement(self):
        self.measurement = None
        self.measurement_profile = None
        self.snapshot = None
        self.measurement_label.setText("Select A or B to measure inner / outer MAX.")
        edge = self.choice.currentData()
        if self.review is None or edge not in ("A", "B"):
            self.update_save_state()
            return
        try:
            fresh = load_edge_review(self.manifest_path, self.image_path, **self.inputs)
            if (
                fresh["manifest_sha256"] != self.review["manifest_sha256"]
                or fresh[edge]["provenance"]["image_sha256"] != self.image_hash
            ):
                raise ValueError("Reference or image changed. Reload and review before measuring.")
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            self._edge_measurements.clear()
            self.review = None
            self.choice.setEnabled(False)
            self.save_button.setEnabled(False)
            self.measurement_label.setText(f"MEASUREMENT UNAVAILABLE · {exc}")
            self.set_status(str(exc), "error")
            self.update_save_state()
            return
        try:
            self.measurement, self.measurement_profile = self.cached_edge_measurement(fresh, edge)
        except (ValueError, KeyError, TypeError, cv2.error) as exc:
            self.measurement_label.setText(
                f"EDGE REVIEW UNAVAILABLE · {exc}\nChoose NOT SURE if the edge cannot be confirmed."
            )
            self.update_save_state()
            return
        result = self.measurement
        cut_path = result["cut_path"]
        path_text = f"Cut travel {cut_path['center_travel_mm']:.3f} mm + bit {cut_path['bit_diameter_mm']:.3f} mm"

        state = result["measurement_status"]
        if state == "EDGE_ONLY":
            guide = (
                "Blue reference: visual estimate only"
                if result["preview_reference"] is not None
                else "Reference unavailable: too few side-edge samples"
            )
            self.measurement_label.setText(
                f"EDGE {edge} · {guide} · Read coverage {result['coverage']:.1%} · {path_text}\n"
                f"Inner / outer mm unavailable: {result['baseline_reason']}\n"
                "SAVE EDGE records the highlighted samples only. Gaps and mm remain unconfirmed."
            )
            self.snapshot = edge_snapshot(
                fresh, result, self.measurement_profile, self.sobel_settings
            )
            if confirmation_matches(self.confirmations.get(confirmation_key(fresh)), self.snapshot):
                self.set_status("Selected edge saved for this image.", "saved")
            self.update_save_state()
            return
        scope = "accepted samples only" if state == "PARTIAL" else "selected segment only"
        if state == "NO_EDGE":
            scope = "no reliable edge samples; N/A is not zero"
        self.measurement_label.setText(
            f"EDGE {edge} · ESTIMATED · {state} · Coverage {result['coverage']:.1%} · {path_text}\n"
            f"{scope}. Zero: normal PCB Sobel edge outside cut. Blue: visual guide only."
        )
        self.snapshot = edge_snapshot(fresh, result, self.measurement_profile, self.sobel_settings)
        previously_confirmed = confirmation_matches(
            self.confirmations.get(confirmation_key(fresh)), self.snapshot
        )
        if previously_confirmed:
            self.set_status("Confirmed edge saved for this image.", "saved")
        self.update_save_state()

    def cached_edge_measurement(self, review, edge):
        """Reuse immutable image analysis within this dialog, after source validation."""
        key = (
            review["manifest_sha256"],
            review["key"],
            self.image_hash,
            edge,
            tuple(review[edge]["scales"]),
        )
        if key not in self._edge_measurements:
            self._edge_measurements[key] = measure_image_edge(self.image, review, edge)
        return self._edge_measurements[key]

    def update_save_state(self, *_):
        if not hasattr(self, "save_button"):
            return
        unsure = self.choice.currentData() == "UNSURE"
        self.save_button.setText("SAVE NOT SURE" if unsure else "SAVE EDGE")
        self.save_button.setEnabled(bool(self.review and (unsure or self.snapshot is not None)))
        self.measure_button.setEnabled(bool(self.review and not unsure))
        for name, button in (
            ("inner", self.inner_focus_button),
            ("outer", self.outer_focus_button),
        ):
            button.setEnabled(
                bool(self.measurement and self.measurement["max_indices"][name] is not None)
            )
        self.update_cut_measurement()

    def show_edge_measurement(self):
        self.measurement_requested = True
        # Revalidate sources before reusing this image's analysis.
        self.update_measurement()
        self.render()

    def update_cut_measurement(self):
        self.cut_measurement_label.setVisible(self.measurement_requested)
        if not self.measurement_requested:
            return
        result = self.measurement
        if result is None or result["inner_line_max_mm"] is None:
            reason = (
                "Normal PCB edge outside the cut could not be established; blue guide is visual only."
                if result is not None and result["measurement_status"] == "EDGE_ONLY"
                else "Select a readable edge with matching reference data."
            )
            self.cut_measurement_label.setText(
                f"Inner Cut (MAX): N/A mm   |   Outer Cut (MAX): N/A mm\n{reason}"
            )
            return
        scope = (
            "Partial edge: maxima cover detected samples only."
            if result["measurement_status"] == "PARTIAL"
            else "Maxima over cut travel including the bit radius at both ends."
        )
        self.cut_measurement_label.setText(
            f"EDGE {result['edge']}   |   "
            f"Inner Cut (MAX): {result['inner_line_max_mm']:.4f} mm   |   "
            f"Outer Cut (MAX): {result['outer_line_max_mm']:.4f} mm\n"
            f"{scope} Estimated using the machine pixel scale."
        )

    def set_detail_view(self, enabled):
        self.detail_requested = enabled
        self.zoom_roi = None
        self.render()

    def reset_zoom(self):
        self.zoom_roi = None
        self.render()

    def set_zoom_roi(self, center, width, height):
        h, w = self.image.shape[:2]
        width, height = min(w, max(20, int(round(width)))), min(h, max(20, int(round(height))))
        x = min(w - width, max(0, int(round(center[0] - width / 2))))
        y = min(h - height, max(0, int(round(center[1] - height / 2))))
        self.zoom_roi = (x, y, x + width, y + height)

    def zoom_preview(self, factor, point=None):
        if self.display_roi is None or self.base_roi is None:
            return
        x0, y0, x1, y1 = self.display_roi
        bx0, by0, bx1, by1 = self.base_roi
        current = (bx1 - bx0) / (x1 - x0)
        target = float(np.clip(current * factor, 1, 16))
        if target <= 1:
            self.reset_zoom()
            return
        width, height = (bx1 - bx0) / target, (by1 - by0) / target
        point = np.asarray(point if point is not None else [(x0 + x1) / 2, (y0 + y1) / 2])
        uv = (point - [x0, y0]) / [x1 - x0, y1 - y0]
        self.set_zoom_roi(point + (0.5 - uv) * [width, height], width, height)
        self.render()

    def focus_maximum(self, name):
        self.measurement_requested = True
        self.update_measurement()
        if self.measurement is None:
            self.render()
            return
        index = self.measurement["max_indices"][name]
        if index is None:
            return
        profile = self.measurement_profile
        center = (profile.points[index] + profile.anchors[index]) / 2
        x0, y0, x1, y1 = self.base_roi
        height = max((y1 - y0) / 4, abs(profile.points[index, 1] - profile.anchors[index, 1]) + 60)
        width = max((x1 - x0) / 4, height * (x1 - x0) / (y1 - y0))
        self.set_zoom_roi(center, width, height)
        self.render()

    def source_point_from_preview(self, position):
        pixmap = self.preview.pixmap()
        if pixmap is None or pixmap.isNull() or self.display_roi is None:
            return None
        rect = self.preview.contentsRect()
        pw, ph = (
            pixmap.width() / pixmap.devicePixelRatio(),
            pixmap.height() / pixmap.devicePixelRatio(),
        )
        left, top = rect.x() + (rect.width() - pw) / 2, rect.y() + (rect.height() - ph) / 2
        u, v = (position.x() - left) / pw, (position.y() - top) / ph
        if not 0 <= u < 1 or not 0 <= v < 1:
            return None
        x0, y0, x1, y1 = self.display_roi
        return np.array([x0 + u * (x1 - x0), y0 + v * (y1 - y0)])

    def select_edge_at(self, point):
        if self.review is None:
            return
        candidates = []
        failures = {}
        for side in ("A", "B"):
            try:
                _, profile = self.cached_edge_measurement(self.review, side)
            except (ValueError, KeyError, TypeError, cv2.error) as exc:
                failures[side] = str(exc)
                continue
            points = profile.points[profile.accepted]
            if len(points):
                distances = np.linalg.norm(points - point, axis=1)
                index = int(np.argmin(distances))
                candidates.append((float(distances[index]), side, points[index]))
        if not candidates or min(c[0] for c in candidates) > 20:
            reason = " ".join(
                f"Edge {side} unavailable: {error}" for side, error in failures.items()
            )
            self.set_status(
                "Click within 20 image pixels of the detected PCB edge inside the cut span. "
                + reason,
                "changed",
            )
            return
        _, side, selected_point = min(candidates, key=lambda candidate: candidate[0])
        with QSignalBlocker(self.choice):
            self.choice.setCurrentIndex(self.choice.findData(side))
        # Revalidate sources even when the click selects the already-active side.
        self.update_measurement()
        if self.snapshot is not None:
            self.clicked_edge_point = selected_point
            self.set_status(
                f"Selected edge {side} · review the blue edge, then SAVE EDGE.", "changed"
            )
        self.render()

    def eventFilter(self, watched, event):
        if watched is self.preview:
            if event.type() == QEvent.Wheel:
                point = self.source_point_from_preview(event.position())
                if point is not None and event.angleDelta().y():
                    self.zoom_preview(1.25 if event.angleDelta().y() > 0 else 1 / 1.25, point)
                return True
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
                if (
                    self.zoom_roi is not None
                    and self.source_point_from_preview(event.position()) is not None
                ):
                    self.pan_start = (
                        event.position(),
                        self.display_roi,
                        self.preview.pixmap().deviceIndependentSize(),
                    )
                    self.preview.setCursor(Qt.ClosedHandCursor)
                return True
            if event.type() == QEvent.MouseMove and self.pan_start is not None:
                start, (x0, y0, x1, y1), size = self.pan_start
                delta = event.position() - start
                center = [
                    (x0 + x1) / 2 - delta.x() * (x1 - x0) / size.width(),
                    (y0 + y1) / 2 - delta.y() * (y1 - y0) / size.height(),
                ]
                self.set_zoom_roi(center, x1 - x0, y1 - y0)
                self.render()
                return True
            if event.type() == QEvent.MouseButtonRelease and event.button() == Qt.RightButton:
                self.pan_start = None
                self.preview.setCursor(Qt.CrossCursor)
                return True
        if (
            watched is self.preview
            and event.type() == QEvent.MouseButtonPress
            and event.button() == Qt.LeftButton
        ):
            point = self.source_point_from_preview(event.position())
            if point is not None:
                self.select_edge_at(point)
            return True
        return super().eventFilter(watched, event)

    def choose_manifest(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open recorded A/B reference data", self.manifest_path, "Reference data (*.json)"
        )
        if path:
            self.load_reference(path)

    def load_reference(self, path):
        self._edge_measurements.clear()
        self.review = None
        self._saved_edge = None
        try:
            if not path:
                raise ValueError("Load matching reference data to display A/B. No line is guessed.")
            review = load_edge_review(path, self.image_path, **self.inputs)
            if review["A"]["provenance"]["image_sha256"] != self.image_hash:
                raise ValueError("Image changed. Close and reopen the edge review.")
            self.review, self.manifest_path = review, str(path)
            for side in ("A", "B"):
                self.choice.setItemText(
                    self.choice.findData(side),
                    f"{side} - {review[side]['pcb_side_name'].upper()} PCB",
                )
            self.edge_side_note.setText(
                f"A: {review['A']['pcb_side_name']} PCB  ·  B: {review['B']['pcb_side_name']} PCB"
            )
            value = saved_edge(self.choices, review)
            self._saved_edge = value
            self.choice.blockSignals(True)
            self.choice.setCurrentIndex(self.choice.findData(value or "UNSURE"))
            self.choice.blockSignals(False)
            self.set_status(
                f"Saved side: {value} · review this image's edge."
                if value
                else "Not reviewed · choose A, B or NOT SURE.",
                "pending",
            )
        except (OSError, ValueError, KeyError, TypeError, cv2.error) as exc:
            self.set_status(str(exc), "error")
        self.choice.setEnabled(self.review is not None)
        self.save_button.setEnabled(self.review is not None)
        self.update_measurement()
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
            self.measurement = None
            self.measurement_profile = None
            self.snapshot = None
            self.measurement_label.setText("MEASUREMENT UNAVAILABLE · Source data changed.")
            self.update_save_state()
            self.render()
            return
        value = self.choice.currentData()
        confirmation = None
        if value in ("A", "B"):
            if self.snapshot is None:
                self.set_status("Select a supported PCB edge before saving.", "error")
                return
            confirmation = make_confirmation(self.snapshot)
        if self.save_callback is not None:
            try:
                self.save_callback(fresh, value, confirmation)
            except (OSError, ValueError, KeyError, TypeError, ProtectedPathError) as exc:
                self.set_status(f"Edge choice not saved: {exc}", "error")
                return
        self.accept()

    def render(self, *_):
        drawn = cv2.cvtColor(self.sobel, cv2.COLOR_GRAY2BGR)
        selected = self.choice.currentData()
        self.machine_button.setEnabled(self.review is not None and selected in ("A", "B"))
        if self.review and self.machine_button.isChecked():
            for name in (selected,) if selected in ("A", "B") else ():
                p, q = np.asarray(self.review[name]["line"])
                cv2.line(
                    drawn,
                    tuple(np.rint(p * 256).astype(int)),
                    tuple(np.rint(q * 256).astype(int)),
                    (180, 180, 180),
                    4 if selected == name else 2,
                    cv2.LINE_AA,
                    shift=8,
                )
                label = "Machine " + name
                x = min(drawn.shape[1] - 40, max(0, round(q[0]) + 14))
                y = min(drawn.shape[0] - 14, max(28, round(q[1]) - 8))
                cv2.rectangle(drawn, (x - 4, y - 23), (x + 25, y + 5), (35, 25, 12), -1)
                cv2.putText(
                    drawn,
                    label,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )
        if self.review and self.measurement and self.measurement_profile is not None:
            profile = self.measurement_profile
            for index in np.flatnonzero(profile.accepted):
                point = tuple(np.rint(profile.points[index]).astype(int))
                cv2.circle(drawn, point, 2, (255, 190, 0), -1)
                if index and profile.accepted[index - 1]:
                    previous = tuple(np.rint(profile.points[index - 1]).astype(int))
                    cv2.line(drawn, previous, point, (255, 190, 0), 2, cv2.LINE_AA)
            guide = self.measurement["preview_reference"]
            if guide is not None:
                ends = np.rint(np.asarray(guide["line_px"])).astype(int)
                cv2.line(drawn, tuple(ends[0]), tuple(ends[1]), (255, 100, 0), 1, cv2.LINE_8)
            if self.measurement_requested and self.measurement["reference_line_px"] is not None:
                ends = np.rint(np.asarray(self.measurement["reference_line_px"])).astype(int)
                cv2.line(drawn, tuple(ends[0]), tuple(ends[1]), (80, 210, 80), 1, cv2.LINE_8)
                for point in self.measurement["normal_pcb_points_px"]:
                    cv2.circle(drawn, tuple(np.rint(point).astype(int)), 1, (80, 210, 80), -1)
            if self.clicked_edge_point is not None:
                cv2.circle(
                    drawn,
                    tuple(np.rint(self.clicked_edge_point).astype(int)),
                    8,
                    (255, 190, 0),
                    2,
                    cv2.LINE_AA,
                )
            for name, index in self.measurement["max_indices"].items():
                if index is None:
                    continue
                colour = (0, 0, 255) if name == "inner" else (0, 255, 255)
                anchor = tuple(np.rint(profile.anchors[index]).astype(int))
                point = tuple(np.rint(profile.points[index]).astype(int))
                cv2.line(drawn, anchor, point, colour, 2, cv2.LINE_AA)
                cv2.circle(drawn, point, 6, colour, 2, cv2.LINE_AA)
                if self.measurement_requested:
                    if self.zoom_roi is not None:
                        zx0, zy0, zx1, zy1 = self.zoom_roi
                        if not (zx0 <= point[0] < zx1 and zy0 <= point[1] < zy1):
                            continue
                    label = f"MAX {name.upper()}: {self.measurement[name + '_line_max_mm']:.4f} mm"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    tx = max(3, min(drawn.shape[1] - tw - 3, point[0] - tw // 2))
                    ty = int(
                        np.clip(
                            point[1] + (-18 if name == "outer" else 30), th + 4, drawn.shape[0] - 5
                        )
                    )
                    cv2.rectangle(
                        drawn, (tx - 2, ty - th - 3), (tx + tw + 2, ty + 3), (20, 20, 20), -1
                    )
                    cv2.putText(
                        drawn,
                        label,
                        (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        1,
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
            if self.measurement_profile is not None:
                profile = self.measurement_profile
                points = np.vstack([points, profile.points[profile.accepted]])
                guide = self.measurement["preview_reference"]
                if guide is not None:
                    points = np.vstack([points, guide["line_px"]])
            lo = np.floor(points.min(axis=0) - [120, 120]).astype(int)
            hi = np.ceil(points.max(axis=0) + [120, 120]).astype(int)
            x0, y0 = max(0, lo[0]), max(0, lo[1])
            x1, y1 = min(w, hi[0]), min(h, hi[1])
        self.base_roi = (int(x0), int(y0), int(x1), int(y1))
        if self.zoom_roi is not None:
            x0, y0, x1, y1 = self.zoom_roi
        self.display_roi = (int(x0), int(y0), int(x1), int(y1))
        self.zoom_label.setText(f"{(self.base_roi[2] - self.base_roi[0]) / (x1 - x0):.1f}×")
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
