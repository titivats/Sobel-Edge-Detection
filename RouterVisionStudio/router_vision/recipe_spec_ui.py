"""Recipe-specific Inner / Outer acceptance settings."""

from __future__ import annotations

import math

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
)


def validated_limits(inner, outer) -> dict[str, float]:
    result = {}
    for key, caption, raw in (
        ("inner_max_mm", "Max Inner", inner),
        ("outer_max_mm", "Max Outer", outer),
    ):
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError):
            raise ValueError(f"{caption}: enter a non-negative number in mm.") from None
        if isinstance(raw, bool) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{caption}: enter a finite, non-negative number in mm.")
        result[key] = value
    return result


class RecipeSpecDialog(QDialog):
    def __init__(self, recipes, saved, active_recipe, save_callback, parent=None):
        super().__init__(parent)
        self.setWindowTitle("AVTR | Recipe SPEC")
        self.setMinimumWidth(560)
        self.resize(640, 360)
        self.save_callback = save_callback
        self.drafts = {}
        self.original = {}
        self.current_key = None
        saved = saved if isinstance(saved, dict) else {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)
        title = QLabel("RECIPE SPEC")
        title.setStyleSheet("font-size:18pt; font-weight:800; color:#17324d;")
        layout.addWidget(title)
        form = QFormLayout()
        form.setSpacing(14)
        self.recipe = QComboBox()
        for key, caption in recipes.items():
            self.recipe.addItem(caption, key)
            self.recipe.setItemData(self.recipe.count() - 1, key, Qt.ToolTipRole)
            record = saved.get(key, {})
            record = record if isinstance(record, dict) else {}
            self.original[key] = tuple(
                str(record[field]) if field in record else ""
                for field in ("inner_max_mm", "outer_max_mm")
            )
        self.inner = QLineEdit()
        self.outer = QLineEdit()
        for editor in (self.inner, self.outer):
            editor.setPlaceholderText("Not set — enter mm")
            editor.setClearButtonEnabled(True)
        form.addRow("Recipe / Program", self.recipe)
        form.addRow("Max Inner (mm)  ≤", self.inner)
        form.addRow("Max Outer (mm)  ≤", self.outer)
        layout.addLayout(form)
        note = QLabel("Save limits per recipe. Values equal to the limit are acceptable.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.status = QLabel("Production checks these limits after the model passes.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton("CANCEL")
        cancel.setObjectName("secondaryAction")
        cancel.clicked.connect(self.reject)
        self.save_button = QPushButton("SAVE SPEC")
        self.save_button.setObjectName("saveAction")
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(cancel)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        index = self.recipe.findData(active_recipe)
        if index >= 0:
            self.recipe.setCurrentIndex(index)
        self.recipe.currentIndexChanged.connect(self._select_recipe)
        self._select_recipe()
        if not recipes:
            self.status.setText("No production recipes available. Load the data source first.")
            self.save_button.setEnabled(False)
            self.inner.setEnabled(False)
            self.outer.setEnabled(False)

    def _remember(self):
        if self.current_key is not None:
            self.drafts[self.current_key] = (self.inner.text().strip(), self.outer.text().strip())

    def _select_recipe(self, *_):
        self._remember()
        self.current_key = self.recipe.currentData()
        values = self.drafts.get(self.current_key, self.original.get(self.current_key, ("", "")))
        self.inner.setText(values[0])
        self.outer.setText(values[1])

    def save(self):
        if self.current_key is None:
            return
        self._remember()
        changes = {}
        try:
            for key, values in self.drafts.items():
                if values != self.original[key]:
                    try:
                        changes[key] = validated_limits(*values)
                    except ValueError as exc:
                        self.recipe.setCurrentIndex(self.recipe.findData(key))
                        raise ValueError(f"{self.recipe.currentText()}: {exc}") from exc
            if not changes:
                self.status.setText("Enter or change the limits before saving.")
                return
            self.save_callback(changes, self.current_key)
        except Exception as exc:
            self.status.setStyleSheet("color:#b91c1c;")
            self.status.setText(f"SPEC NOT SAVED: {exc}")
            return
        self.accept()
