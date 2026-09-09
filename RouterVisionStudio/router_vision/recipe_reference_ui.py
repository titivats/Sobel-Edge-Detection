"""Select recipe geometry and supply camera mapping without drawing a new line."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .recipe_reference import project_segment, recipe_segments, recipe_signature


class RecipeLineDialog(QDialog):
    def __init__(self, image_shape, recipe_dir=None, parent=None, previous=None, image=None):
        super().__init__(parent)
        self.setWindowTitle("AVTR | Choose Recipe Line")
        screen = self.screen().availableGeometry()
        self.resize(min(1050, screen.width() - 40), min(760, screen.height() - 60))
        self.image_shape = image_shape
        self.recipe_dir = str(recipe_dir or "")
        self.path = None
        self.digest = None
        self.candidates = []
        self.reference = None
        self.line = None
        self.image = image
        self.photo_line = None
        layout = QVBoxLayout(self)
        heading = QLabel("CHOOSE A RECIPE LINE")
        heading.setStyleSheet("font-size:20px; font-weight:800; color:#1d4ed8; padding:8px;")
        layout.addWidget(heading)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        body = QVBoxLayout(content)
        note = QLabel("1. Choose the recipe.   2. Select a line.   3. Check it on the photo.")
        note.setWordWrap(True)
        body.addWidget(note)
        choose = QPushButton("CHOOSE RECIPE")
        choose.clicked.connect(self.choose_recipe)
        body.addWidget(choose)
        self.filename = QLabel("No recipe selected")
        self.filename.setWordWrap(True)
        body.addWidget(self.filename)
        self.segment = QComboBox()
        self.segment.currentIndexChanged.connect(self.selection_changed)
        navigation = QHBoxLayout()
        previous_button = QPushButton("PREVIOUS")
        previous_button.clicked.connect(lambda: self.step_segment(-1))
        next_button = QPushButton("NEXT")
        next_button.clicked.connect(lambda: self.step_segment(1))
        navigation.addWidget(previous_button)
        navigation.addWidget(self.segment, 1)
        navigation.addWidget(next_button)
        body.addLayout(navigation)
        views = QHBoxLayout()
        recipe_column = QVBoxLayout()
        recipe_column.addWidget(QLabel("RECIPE — selected line in blue"))
        self.scene = QGraphicsScene(self)
        self.preview = QGraphicsView(self.scene)
        self.preview.setMinimumSize(220, 230)
        recipe_column.addWidget(self.preview)
        views.addLayout(recipe_column, 1)
        photo_column = QVBoxLayout()
        photo_column.addWidget(QLabel("ORIGINAL PHOTO"))
        self.photo_scene = QGraphicsScene(self)
        self.photo = QGraphicsView(self.photo_scene)
        self.photo.setMinimumSize(220, 230)
        if image is not None:
            import cv2

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            picture = QImage(
                rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
            ).copy()
            self.photo_scene.addPixmap(QPixmap.fromImage(picture))
        else:
            self.photo_scene.addText("No image preview")
        photo_column.addWidget(self.photo)
        views.addLayout(photo_column, 1)
        body.addLayout(views)
        self.setup_button = QPushButton("CAMERA SETUP — for the setup engineer")
        self.setup_button.setCheckable(True)
        body.addWidget(self.setup_button)
        self.setup_panel = QWidget()
        setup_layout = QVBoxLayout(self.setup_panel)
        setup_note = QLabel(
            "This image needs a verified camera-to-recipe mapping before a line can be placed. "
            "Use the inspection camera pose, including camera/tool offsets. "
            "Check candidate geometry against the Router program; a cutter centerline may need an edge offset."
        )
        setup_note.setWordWrap(True)
        setup_layout.addWidget(setup_note)
        form = QFormLayout()
        self.fields = {}
        for key, label, low, high, decimals in (
            ("center_x_mm", "Machine X at image center (mm)", -5000, 5000, 6),
            ("center_y_mm", "Machine Y at image center (mm)", -5000, 5000, 6),
            ("scale_x", "Verified image X scale (mm / px)", 0, 10, 8),
            ("scale_y", "Verified image Y scale (mm / px)", 0, 10, 8),
            ("image_x_axis_deg", "Image +X angle in machine XY (degrees)", -360, 360, 6),
            ("offset_mm", "Edge offset: +left / -right of A to B (mm)", -100, 100, 6),
        ):
            field = QDoubleSpinBox()
            field.setDecimals(decimals)
            field.setRange(low, high)
            field.setSingleStep(0.001)
            field.valueChanged.connect(self.inputs_changed)
            self.fields[key] = field
            form.addRow(label, field)
        self.y_up = QCheckBox("Machine +Y points up in the image at zero angle")
        self.y_up.setChecked(True)
        self.y_up.toggled.connect(self.inputs_changed)
        form.addRow(self.y_up)
        setup_layout.addLayout(form)
        hint = QLabel(
            "Use the inspection camera pose, including camera/tool offsets; spindle coordinates alone may differ. "
            "Zero edge offset means the selected recipe line itself is the nominal edge. "
            "Only its visible portion will be measured. Geometry is never stretched to fit the image."
        )
        hint.setWordWrap(True)
        setup_layout.addWidget(hint)
        body.addWidget(self.setup_panel)
        self.setup_panel.hide()
        self.setup_button.toggled.connect(self.setup_panel.setVisible)
        self.verified = QCheckBox(
            "I verified the segment, edge offset and camera mapping for this image"
        )
        setup_layout.addWidget(self.verified)
        self.verified.toggled.connect(self.refresh_state)
        self.status = QLabel("Choose a recipe to begin.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(
            "background:#fff7ed; color:#92400e; padding:12px; font-weight:700;"
        )
        scroll.setWidget(content)
        layout.addWidget(scroll)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        self.use_button = QPushButton("USE THIS LINE")
        self.use_button.clicked.connect(self.use_reference)
        self.use_button.setEnabled(False)
        cancel = QPushButton("BACK")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(self.use_button)
        buttons.addWidget(cancel)
        layout.addLayout(buttons)
        if previous:
            try:
                self.load_recipe(previous["path"])
                for key, field in self.fields.items():
                    field.setValue(
                        previous["offset_mm"] if key == "offset_mm" else previous["mapping"][key]
                    )
                self.y_up.setChecked(previous["mapping"]["y_up"])
                for i, segment in enumerate(self.candidates):
                    if segment == previous["segment"]:
                        self.segment.setCurrentIndex(i)
                        break
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.status.setText(str(exc))
        self.refresh_state()

    def step_segment(self, direction):
        if self.segment.count():
            self.segment.setCurrentIndex(
                (self.segment.currentIndex() + direction) % self.segment.count()
            )

    def showEvent(self, event):
        super().showEvent(event)
        self.fit_views()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "photo"):
            self.fit_views()

    def fit_views(self):
        if self.candidates:
            bounds = self.scene.itemsBoundingRect()
            margin = max(10.0, bounds.width() * 0.06)
            self.preview.fitInView(
                bounds.adjusted(-margin, -margin, margin, margin), Qt.KeepAspectRatio
            )
        self.photo.fitInView(self.photo_scene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def refresh_state(self, *_):
        if not hasattr(self, "use_button"):
            return
        self.use_button.setEnabled(False)
        if self.photo_line is not None:
            self.photo_scene.removeItem(self.photo_line)
            self.photo_line = None
        if self.path is None:
            self.status.setText("Choose a recipe to begin.")
            return
        mapping = {k: v.value() for k, v in self.fields.items() if k != "offset_mm"}
        mapping["y_up"] = self.y_up.isChecked()
        if mapping["scale_x"] <= 0 or mapping["scale_y"] <= 0:
            self.status.setText(
                "CAMERA SETUP NEEDED\nYou can browse the recipe lines. Placing a line on the photo is not ready yet. "
                "The setup engineer needs to link this camera view to the recipe."
            )
            return
        try:
            line = project_segment(
                self.candidates[self.segment.currentIndex()]["points_mm"],
                mapping,
                self.image_shape,
                self.fields["offset_mm"].value(),
            )
        except ValueError as exc:
            self.status.setText("CAMERA SETUP NEEDS ATTENTION\n" + str(exc))
            return
        if not self.verified.isChecked():
            self.status.setText(
                "CAMERA SETUP NEEDS REVIEW\nOpen Camera Setup to verify the mapping for this image and line."
            )
            return
        if self.image is not None:
            a, b = line
            pen = QPen(QColor("#0284c7"), 3)
            pen.setCosmetic(True)
            self.photo_line = self.photo_scene.addLine(*a, *b, pen)
        self.use_button.setEnabled(True)
        self.status.setText(
            "CHECK THE BLUE LINE ON THE PHOTO\nIf it marks the intended PCB edge, choose USE THIS LINE."
        )

    def inputs_changed(self, *_):
        if hasattr(self, "verified"):
            self.verified.setChecked(False)
        self.refresh_state()

    def choose_recipe(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Router recipe", self.recipe_dir, "Router recipe (*.rcp)"
        )
        if path:
            try:
                self.load_recipe(path)
            except (OSError, ValueError) as exc:
                self.status.setText(str(exc))

    def load_recipe(self, path):
        digest = recipe_signature(path)
        candidates = recipe_segments(path)
        if recipe_signature(path) != digest:
            raise ValueError("Recipe changed while opening; select it again.")
        if not candidates:
            raise ValueError(
                "No straight geometry candidates found in the supported recipe header."
            )
        self.path, self.digest, self.candidates = Path(path).resolve(), digest, candidates
        self.segment.blockSignals(True)
        self.segment.clear()
        for i, candidate in enumerate(candidates):
            a, b = candidate["points_mm"]
            self.segment.addItem(f"Recipe line {i + 1:02d} of {len(candidates)}")
            self.segment.setItemData(
                i, f"A ({a[0]:.3f}, {a[1]:.3f}) -> B ({b[0]:.3f}, {b[1]:.3f}) mm", Qt.ToolTipRole
            )
        self.segment.blockSignals(False)
        self.filename.setText(self.path.name)
        self.filename.setToolTip(str(self.path))
        self.selection_changed()
        self.refresh_state()

    def selection_changed(self, *_):
        self.inputs_changed()
        self.scene.clear()
        for i, candidate in enumerate(self.candidates):
            a, b = candidate["points_mm"]
            pen = QPen(QColor("#0284c7" if i == self.segment.currentIndex() else "#94a3b8"))
            pen.setWidth(3 if i == self.segment.currentIndex() else 1)
            pen.setCosmetic(True)
            self.scene.addLine(a[0], -a[1], b[0], -b[1], pen)
            cx, cy = (a[0] + b[0]) / 2, -(a[1] + b[1]) / 2
            marker = self.scene.addEllipse(-3, -3, 6, 6, pen, QBrush(pen.color()))
            marker.setPos(cx, cy)
            marker.setFlag(QGraphicsItem.ItemIgnoresTransformations)
            number = self.scene.addSimpleText(f"{i + 1:02d}")
            number.setBrush(QBrush(pen.color()))
            number.setPos(cx + 4, cy + 6)
            number.setFlag(QGraphicsItem.ItemIgnoresTransformations)
        if self.candidates:
            self.fit_views()

    def use_reference(self):
        try:
            if self.path is None or self.segment.currentIndex() < 0:
                raise ValueError("Select a Router recipe segment first.")
            if not self.verified.isChecked():
                raise ValueError(
                    "Verify the recipe segment, nominal edge offset and camera mapping first."
                )
            if recipe_signature(self.path) != self.digest:
                raise ValueError("Recipe changed while open; select it again.")
            mapping = {k: v.value() for k, v in self.fields.items() if k != "offset_mm"}
            mapping["y_up"] = self.y_up.isChecked()
            candidate = self.candidates[self.segment.currentIndex()]
            offset = self.fields["offset_mm"].value()
            self.line = project_segment(candidate["points_mm"], mapping, self.image_shape, offset)
            self.reference = {
                "path": str(self.path),
                "sha256": self.digest,
                "segment": candidate,
                "mapping": mapping,
                "offset_mm": offset,
            }
        except (OSError, ValueError) as exc:
            self.status.setText(str(exc))
            return
        self.accept()
