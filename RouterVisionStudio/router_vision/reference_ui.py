"""Interactive, explicitly non-production reference-line measurement editor."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QLocale, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .guard import ProtectedPathError
from .recipe_reference import validate_recipe_reference
from .recipe_reference_ui import RecipeLineDialog
from .reference import ReferenceStore, image_signature, measure_reference, normal_scale


class ReferenceCanvas(QGraphicsView):
    line_changed = Signal()

    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor("#0b1220"))
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.scene().addPixmap(
            QPixmap.fromImage(
                QImage(
                    rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
                ).copy()
            )
        )
        self.image_rect = self.scene().itemsBoundingRect()
        self.setSceneRect(self.image_rect)
        self.line = None
        self.drawing = False
        self.line_locked = False
        self.drag_endpoint = None
        self.overlays = []
        self.zoomed = False
        self.setMinimumSize(200, 200)

    def fit_image(self):
        self.fitInView(self.image_rect, Qt.KeepAspectRatio)
        self.zoomed = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.zoomed:
            self.fit_image()

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 1 / 1.2
        if 0.03 <= self.transform().m11() * factor <= 30:
            self.scale(factor, factor)
            self.zoomed = True
        event.accept()

    def _point(self, event):
        p = self.mapToScene(event.position().toPoint())
        return [
            float(np.clip(p.x(), 0, self.image_rect.width() - 1)),
            float(np.clip(p.y(), 0, self.image_rect.height() - 1)),
        ]

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and not self.line_locked:
            if self.drawing:
                self.line = [self._point(event), self._point(event)]
                self.drag_endpoint = 1
                event.accept()
                return
            if self.line is not None:
                for index, point in enumerate(self.line):
                    screen = self.mapFromScene(QPointF(*point))
                    if (screen - event.position().toPoint()).manhattanLength() <= 16:
                        self.drag_endpoint = index
                        event.accept()
                        return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drag_endpoint is not None:
            self.line[self.drag_endpoint] = self._point(event)
            self.line_changed.emit()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drag_endpoint is not None:
            self.line[self.drag_endpoint] = self._point(event)
            self.drag_endpoint = None
            self.drawing = False
            self.setCursor(Qt.OpenHandCursor)
            self.line_changed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def segment(self, a, b, color, width=1, dashed=False):
        pen = QPen(QColor(color), width)
        pen.setCosmetic(True)
        if dashed:
            pen.setStyle(Qt.DashLine)
        item = self.scene().addLine(float(a[0]), float(a[1]), float(b[0]), float(b[1]), pen)
        self.overlays.append(item)

    def overlay(self, profile, side, radius, inward_limit, protrusion_limit):
        for item in self.overlays:
            self.scene().removeItem(item)
        self.overlays.clear()
        if self.line is None:
            return
        ends = np.asarray(self.line, dtype=float)
        vector = ends[1] - ends[0]
        length = np.linalg.norm(vector)
        if length < 1:
            return
        normal = side * np.array([-vector[1], vector[0]]) / length
        for offset in (-radius, radius):
            self.segment(
                ends[0] + normal * offset, ends[1] + normal * offset, "#64748b", dashed=True
            )
        if profile is not None:
            for i in range(1, len(profile.points)):
                if not (profile.accepted[i - 1] and profile.accepted[i]):
                    continue
                value = profile.deviations[i]
                color = (
                    "#ef4444"
                    if value > inward_limit
                    else "#f59e0b"
                    if value < -protrusion_limit
                    else "#22c55e"
                )
                self.segment(profile.points[i - 1], profile.points[i], color, 2)
            for positive, color in ((True, "#ef4444"), (False, "#f59e0b")):
                indices = np.flatnonzero(profile.accepted)
                if len(indices):
                    selected = indices[
                        np.argmax(profile.deviations[indices])
                        if positive
                        else np.argmin(profile.deviations[indices])
                    ]
                    self.segment(profile.anchors[selected], profile.points[selected], color, 3)
        self.segment(*ends, "#38bdf8", 2)
        center = ends.mean(axis=0)
        tip = center + normal * max(15, radius * 0.8)
        self.segment(center, tip, "#e879f9", 3)
        tangent = vector / length
        for sign in (-1, 1):
            self.segment(tip, tip - normal * 6 + tangent * sign * 4, "#e879f9", 2)
        # Endpoint handles remain the same screen size at all zoom levels.
        radius_px = 5 / max(0.01, self.transform().m11())
        for x, y in ends:
            self.overlays.append(
                self.scene().addEllipse(
                    x - radius_px,
                    y - radius_px,
                    2 * radius_px,
                    2 * radius_px,
                    QPen(QColor("#ffffff")),
                    QColor("#0284c7"),
                )
            )


class ReferenceMeasurementDialog(QDialog):
    def __init__(
        self,
        image_path,
        store_path,
        protected=(),
        parent=None,
        *,
        recipe_dir=None,
        recipe_path=None,
    ):
        super().__init__(parent)
        self.setLocale(QLocale(QLocale.English, QLocale.UnitedStates))
        self.setWindowTitle("AVTR | Reference edge measurement - EXPERIMENTAL")
        screen = self.screen().availableGeometry()
        self.resize(min(1250, screen.width() - 40), min(820, screen.height() - 80))
        self.image_path = str(image_path)
        self.source = image_signature(image_path)
        self.image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if self.image is None:
            raise ValueError("Image could not be read.")
        if self.source != image_signature(image_path):
            raise ValueError("Image changed while opening; try again.")
        self.store = ReferenceStore(Path(store_path), protected)
        self.recipe_dir = recipe_dir
        self.recipe_path = recipe_path
        self.recipe_reference = None
        self.side = 1
        self.profile = None
        self.display_scale = 1.0
        self.dirty = False
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(90)
        self.timer.timeout.connect(self.measure)
        layout = QVBoxLayout(self)
        title = QLabel("REFERENCE EDGE MEASUREMENT  |  TRIAL ONLY - DOES NOT CHANGE GOOD / NG")
        title.setWordWrap(True)
        title.setStyleSheet("color:#1d4ed8; font-weight:800; padding:6px;")
        layout.addWidget(title)
        name = QLabel(Path(image_path).name + "  |  Original-resolution image")
        name.setWordWrap(True)
        layout.addWidget(name)
        body = QSplitter(Qt.Horizontal)
        layout.addWidget(body, 1)
        self.canvas = ReferenceCanvas(self.image)
        body.addWidget(self.canvas)
        controls = QWidget()
        column = QVBoxLayout(controls)
        instructions = QLabel(
            "Choose a recipe line, check it on the photo, then save.\n"
            "Blue = reference. Pink arrow = PCB material side."
        )
        instructions.setWordWrap(True)
        column.addWidget(instructions)
        self.recipe_button = QPushButton("CHOOSE RECIPE LINE")
        self.recipe_button.clicked.connect(self.open_recipe)
        column.addWidget(self.recipe_button)
        self.reference_source = QLabel("Reference source: none")
        self.reference_source.setWordWrap(True)
        column.addWidget(self.reference_source)
        self.advanced_button = QPushButton("ADVANCED MEASUREMENT SETTINGS")
        self.advanced_button.setCheckable(True)
        self.advanced_panel = QWidget()
        advanced = QVBoxLayout(self.advanced_panel)
        self.advanced_panel.hide()
        self.advanced_button.toggled.connect(self.advanced_panel.setVisible)
        actions = QHBoxLayout()
        self.draw_button = QPushButton("MANUAL LINE")
        self.draw_button.clicked.connect(self.draw_line)
        fit = QPushButton("FIT IMAGE")
        fit.clicked.connect(self.canvas.fit_image)
        actions.addWidget(self.draw_button)
        column.addWidget(fit)
        advanced.addLayout(actions)
        flip = QPushButton("SWITCH PCB SIDE")
        flip.clicked.connect(self.flip_side)
        column.addWidget(flip)
        nudge = QHBoxLayout()
        self.nudge_buttons = []
        for text, direction in (("MOVE -1 px", -1), ("MOVE +1 px", 1)):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, sign=direction: self.nudge(sign))
            nudge.addWidget(button)
            self.nudge_buttons.append(button)
        advanced.addLayout(nudge)
        form = QFormLayout()
        self.radius = QSpinBox()
        self.radius.setRange(4, 200)
        self.radius.setValue(24)
        self.radius.setSuffix(" px")
        self.contrast = QDoubleSpinBox()
        self.contrast.setRange(0.1, 100)
        self.contrast.setValue(5)
        self.polarity = QComboBox()
        self.polarity.addItems(["PCB brighter than slot", "PCB darker than slot"])
        self.inward = self.spin(0, 100000, 2, 3)
        self.protrusion = self.spin(0, 100000, 2, 3)
        form.addRow("Search each side", self.radius)
        form.addRow("Minimum edge contrast", self.contrast)
        form.addRow("Boundary polarity", self.polarity)
        advanced.addLayout(form)
        limits = QFormLayout()
        limits.addRow("Allowed inward cut", self.inward)
        limits.addRow("Allowed protrusion", self.protrusion)
        column.addLayout(limits)
        self.aligned = QCheckBox("I checked the nominal reference alignment")
        column.addWidget(self.aligned)
        scale_note = QLabel(
            "mm is disabled until you verify X/Y scale for this camera, image resolution and PCB plane."
        )
        scale_note.setWordWrap(True)
        advanced.addWidget(scale_note)
        scale_form = QFormLayout()
        self.scale_x = self.spin(0, 10, 0, 8)
        self.scale_y = self.spin(0, 10, 0, 8)
        scale_form.addRow("X scale (mm / px)", self.scale_x)
        scale_form.addRow("Y scale (mm / px)", self.scale_y)
        advanced.addLayout(scale_form)
        self.calibrated = QCheckBox("Scale verified against a known dimension")
        advanced.addWidget(self.calibrated)
        self.result = QLabel(
            "Choose a recipe line to begin. If camera setup is missing, the next screen will explain what is needed."
        )
        self.result.setWordWrap(True)
        self.result.setMinimumHeight(90)
        self.result.setStyleSheet(
            "background:#e2e8f0; color:#0f172a; padding:10px; font-weight:700;"
        )
        layout.insertWidget(2, self.result)
        legend = QLabel(
            "Blue: reference | Pink: PCB side\nRed: inward over limit | Orange: protrusion over limit\nGreen: detected edge within trial limits\nGray dashed: search band"
        )
        legend.setWordWrap(True)
        column.addWidget(legend)
        note = QLabel(
            "2-D visible edge only; not burr height. Reflections or solder-mask edges may be mistaken for the PCB edge. "
            "Recipe mapping must match this camera view. Saved for this image only. Not production-qualified."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#92400e;")
        advanced.addWidget(note)
        column.addWidget(self.advanced_button)
        column.addWidget(self.advanced_panel)
        self.save_button = QPushButton("SAVE REFERENCE FOR THIS IMAGE")
        self.save_button.clicked.connect(self.save_reference)
        self.saved_status = QLabel("NOT SAVED")
        self.saved_status.setWordWrap(True)
        column.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(360)
        body.addWidget(scroll)
        body.setSizes([850, 380])
        body.setChildrenCollapsible(False)
        close = QPushButton("CLOSE")
        close.clicked.connect(self.close)
        footer = QHBoxLayout()
        footer.addWidget(self.save_button)
        footer.addWidget(self.saved_status, 1)
        footer.addWidget(close)
        layout.addLayout(footer)
        self.canvas.line_changed.connect(self.line_edited)
        for control in (self.radius, self.contrast, self.inward, self.protrusion):
            control.valueChanged.connect(self.changed)
        self.polarity.currentIndexChanged.connect(self.changed)
        self.aligned.toggled.connect(self.measure)
        self.calibrated.toggled.connect(self.measure)
        self.scale_x.valueChanged.connect(self.scale_changed)
        self.scale_y.valueChanged.connect(self.scale_changed)
        try:
            record = self.store.load(image_path)
            if record:
                self.restore(record)
                self.saved_status.setText("SAVED REFERENCE LOADED - RECHECK ALIGNMENT / SCALE")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self.saved_status.setText(str(exc))
        self.dirty = False
        self.measure()
        QTimer.singleShot(0, self.canvas.fit_image)

    @staticmethod
    def spin(minimum, maximum, value, decimals):
        widget = QDoubleSpinBox()
        widget.setDecimals(decimals)
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(0.001 if decimals <= 3 else 0.0001)
        return widget

    def changed(self, *_):
        self.dirty = True
        self.saved_status.setText("NOT SAVED - changes apply to this image only")
        self.timer.start()

    def line_edited(self):
        self.aligned.setChecked(False)
        self.changed()

    def draw_line(self):
        self.set_recipe_source(None)
        self.canvas.drawing = True
        self.canvas.setCursor(Qt.CrossCursor)
        self.result.setText(
            "Drag from the start to the end of the nominal edge; do not follow a defect."
        )

    def set_recipe_source(self, record):
        self.recipe_reference = record
        self.canvas.line_locked = record is not None
        self.canvas.drawing = False
        self.canvas.drag_endpoint = None
        self.scale_x.setEnabled(record is None)
        self.scale_y.setEnabled(record is None)
        for button in self.nudge_buttons:
            button.setEnabled(record is None)
        self.reference_source.setText(
            f"Recipe: {Path(record['path']).name}\nEdge offset: {record['offset_mm']:+.6f} mm | Endpoints locked"
            if record
            else "Reference source: manual"
        )

    def open_recipe(self):
        dialog = RecipeLineDialog(
            self.image.shape,
            self.recipe_dir,
            self,
            previous=self.recipe_reference,
            image=self.image,
        )
        try:
            if self.recipe_reference is None and self.recipe_path:
                dialog.load_recipe(self.recipe_path)
            if dialog.exec() == QDialog.Accepted:
                self.apply_recipe_reference(dialog.reference)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            QMessageBox.warning(self, "Recipe reference not available", str(exc))
        finally:
            dialog.deleteLater()

    def apply_recipe_reference(self, record):
        line = validate_recipe_reference(record, self.image.shape)
        self.canvas.line = line
        self.set_recipe_source(record)
        self.scale_x.setValue(record["mapping"]["scale_x"])
        self.scale_y.setValue(record["mapping"]["scale_y"])
        self.calibrated.setChecked(False)
        self.line_edited()
        self.measure()

    def flip_side(self):
        self.side *= -1
        self.line_edited()

    def nudge(self, amount):
        if self.canvas.line is None or self.recipe_reference is not None:
            return
        ends = np.array(self.canvas.line)
        vector = ends[1] - ends[0]
        length = np.linalg.norm(vector)
        if length >= 1:
            ends += amount * self.side * np.array([-vector[1], vector[0]]) / length
            h, w = self.image.shape[:2]
            if (ends >= 0).all() and (ends[:, 0] < w).all() and (ends[:, 1] < h).all():
                self.canvas.line = ends.tolist()
                self.line_edited()

    def scale_changed(self):
        self.calibrated.setChecked(False)
        self.changed()

    def measure(self, *_):
        self.timer.stop()
        line = self.canvas.line
        if line is None:
            self.save_button.setEnabled(False)
            return
        self.profile = measure_reference(
            self.image,
            line,
            side=self.side,
            radius=self.radius.value(),
            min_contrast=self.contrast.value(),
            polarity=1 if self.polarity.currentIndex() == 0 else -1,
        )
        scale = 1.0
        unit = "px"
        if self.calibrated.isChecked():
            try:
                scale = normal_scale(
                    self.profile.normal, self.scale_x.value(), self.scale_y.value()
                )
                unit = "mm"
            except ValueError:
                self.calibrated.blockSignals(True)
                self.calibrated.setChecked(False)
                self.calibrated.blockSignals(False)
        # Keep limits physically equivalent when switching px <-> verified mm.
        if unit != ("mm" if self.inward.suffix() == " mm" else "px"):
            ratio = scale / self.display_scale
            for spin in (self.inward, self.protrusion):
                spin.blockSignals(True)
                spin.setDecimals(6 if unit == "mm" else 3)
                spin.setValue(spin.value() * ratio)
                spin.blockSignals(False)
        self.display_scale = scale
        for spin in (self.inward, self.protrusion):
            spin.setSuffix(" " + unit)
        self.canvas.overlay(
            self.profile,
            self.side,
            self.radius.value(),
            self.inward.value() / scale,
            self.protrusion.value() / scale,
        )
        inward, protrusion = self.profile.inward_px * scale, self.profile.protrusion_px * scale
        over = inward > self.inward.value() or protrusion > self.protrusion.value()
        state = (
            "MEASUREMENT NOT VALID"
            if not self.profile.valid
            else "REFERENCE ALIGNMENT NOT CONFIRMED"
            if not self.aligned.isChecked()
            else "OUTSIDE TRIAL LIMITS"
            if over
            else "WITHIN TRIAL LIMITS"
        )
        measurements = (
            f"Observed inward max: {inward:.3f} {unit}  |  Observed protrusion max: {protrusion:.3f} {unit}"
            if self.profile.accepted.any()
            else "Observed inward / protrusion: N/A - no reliable edge"
        )
        self.result.setText(
            f"{state}  |  Edge coverage: {self.profile.coverage:.0%}\n{measurements}\n"
            + (
                "Pixels only - scale not verified."
                if unit == "px"
                else "mm uses operator-verified scale; accuracy is not certified."
            )
            + (
                "  Adjust line / search band / contrast."
                if not self.profile.valid
                else "  Selected segment only."
            )
        )
        ends = np.asarray(line)
        h, w = self.image.shape[:2]
        geometry_valid = (
            np.isfinite(ends).all()
            and np.linalg.norm(np.diff(ends, axis=0)) >= 12
            and (ends >= 0).all()
            and (ends[:, 0] < w).all()
            and (ends[:, 1] < h).all()
        )
        self.save_button.setEnabled(bool(geometry_valid))

    def settings(self):
        settings = {
            "line": self.canvas.line,
            "side": self.side,
            "radius": self.radius.value(),
            "contrast": self.contrast.value(),
            "polarity": self.polarity.currentIndex(),
            "inward_px": self.inward.value() / self.display_scale,
            "protrusion_px": self.protrusion.value() / self.display_scale,
            "scale_x": self.scale_x.value(),
            "scale_y": self.scale_y.value(),
        }
        if self.recipe_reference is not None:
            settings["recipe_reference"] = self.recipe_reference
        return settings

    def restore(self, record):
        line = np.asarray(record["line"], dtype=float)
        if line.shape != (2, 2) or not np.isfinite(line).all() or record["side"] not in (-1, 1):
            raise ValueError("Saved reference geometry is invalid.")
        h, w = self.image.shape[:2]
        if (
            (line < 0).any()
            or (line[:, 0] >= w).any()
            or (line[:, 1] >= h).any()
            or np.linalg.norm(line[1] - line[0]) < 12
        ):
            raise ValueError("Saved reference is outside this image or too short.")
        for key, low, high in (
            ("radius", 4, 200),
            ("contrast", 0.1, 100),
            ("polarity", 0, 1),
            ("inward_px", 0, 100000),
            ("protrusion_px", 0, 100000),
            ("scale_x", 0, 10),
            ("scale_y", 0, 10),
        ):
            value = record.get(key, 0)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not low <= value <= high
            ):
                raise ValueError(f"Saved reference {key} is invalid.")
        if not float(record["radius"]).is_integer() or not float(record["polarity"]).is_integer():
            raise ValueError("Saved reference search / polarity is invalid.")
        recipe = record.get("recipe_reference")
        if recipe is not None:
            projected = validate_recipe_reference(recipe, self.image.shape)
            if not np.allclose(projected, line, atol=1e-6, rtol=0):
                raise ValueError("Saved line no longer matches its recipe mapping.")
        self.canvas.line = line.tolist()
        self.set_recipe_source(recipe)
        self.side = record["side"]
        self.radius.setValue(int(record["radius"]))
        self.contrast.setValue(record["contrast"])
        self.polarity.setCurrentIndex(int(record["polarity"]))
        self.inward.setValue(record["inward_px"])
        self.protrusion.setValue(record["protrusion_px"])
        self.scale_x.setValue(record.get("scale_x", 0))
        self.scale_y.setValue(record.get("scale_y", 0))

    def save_reference(self):
        if self.canvas.line is None:
            return
        self.measure()
        if not self.save_button.isEnabled():
            return
        try:
            if self.recipe_reference is not None:
                projected = validate_recipe_reference(self.recipe_reference, self.image.shape)
                if not np.allclose(projected, self.canvas.line, atol=1e-6, rtol=0):
                    raise ValueError(
                        "Reference line was changed independently of the recipe. Select the recipe again."
                    )
            self.store.save(self.image_path, self.settings(), self.source)
        except (OSError, ValueError, ProtectedPathError) as exc:
            QMessageBox.warning(self, "Reference not saved", str(exc))
            return
        self.dirty = False
        self.saved_status.setText(
            "SAVED FOR THIS IMAGE ONLY - does not change model / production settings"
        )

    def reject(self):
        if self.dirty:
            answer = QMessageBox.question(
                self,
                "Unsaved reference",
                "Discard unsaved reference changes?",
                QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Cancel,
            )
            if answer != QMessageBox.Discard:
                return
        self.timer.stop()
        super().reject()


def main():
    """Open the reference editor independently, without starting inspection."""
    import argparse

    from production_app import APP_STYLE
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication

    parser = argparse.ArgumentParser(description="AVTR experimental reference edge editor")
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--recipe", type=Path, help="Open a Router recipe; camera mapping is still required."
    )
    parser.add_argument("--line", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--pcb-side", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--search-radius", type=int, default=24)
    parser.add_argument("--min-contrast", type=float, default=5)
    args = parser.parse_args()
    QLocale.setDefault(QLocale(QLocale.English, QLocale.UnitedStates))
    app = QApplication([])
    app.setFont(QFont("Segoe UI", 10))
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    image_path = args.image.resolve()
    try:
        dialog = ReferenceMeasurementDialog(
            image_path,
            Path(__file__).resolve().parent.parent / "reference_lines.json",
            [image_path.parent],
            recipe_dir=image_path.parent.parent / "Recipe",
            recipe_path=args.recipe,
        )
    except (OSError, ValueError, cv2.error) as exc:
        parser.error(str(exc))
    if args.line:
        dialog.canvas.line = [args.line[:2], args.line[2:]]
        dialog.side = args.pcb_side
        dialog.radius.setValue(args.search_radius)
        dialog.contrast.setValue(args.min_contrast)
        dialog.line_edited()
        dialog.measure()
    if args.recipe:
        QTimer.singleShot(0, dialog.open_recipe)
    dialog.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
