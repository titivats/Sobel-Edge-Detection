"""Shared dashboard chrome: the red header bar and the KPI cards."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from . import theme


class StatusDot(QWidget):
    """Small filled circle used as an online / busy / alert indicator."""

    def __init__(self, colour: str = theme.ONLINE_GREEN):
        super().__init__()
        self._colour = QColor(colour)
        self.setFixedSize(18, 18)

    def set_colour(self, colour: str) -> None:
        self._colour = QColor(colour)
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(self._colour)
        p.setPen(QColor("#ffffff"))
        p.drawEllipse(2, 2, 14, 14)


class HeaderBar(QFrame):
    """Dark red title bar with a live status readout on the right."""

    def __init__(self, title: str):
        super().__init__()
        self.setStyleSheet(f"background: {theme.SIDEBAR};")
        self.setFixedHeight(74)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(24, 12, 24, 12)

        self.title = QLabel(title)
        self.title.setFont(QFont("Segoe UI Semibold", 19))
        self.title.setStyleSheet(f"color:#ffffff; background:{theme.SIDEBAR};")
        layout.addWidget(self.title)
        layout.addStretch(1)

        self.dot = StatusDot()
        self.dot.setStyleSheet(f"background:{theme.SIDEBAR};")
        layout.addWidget(self.dot)

        text_box = QVBoxLayout()
        text_box.setSpacing(2)
        self.status = QLabel("Ready.")
        self.status.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.status.setStyleSheet(f"color:#ffffff; background:{theme.SIDEBAR};")
        self.status.setAlignment(Qt.AlignRight)
        self.detail = QLabel("")
        self.detail.setFont(QFont("Segoe UI", 9))
        self.detail.setStyleSheet(f"color:{theme.HEADER_MUTED}; background:{theme.SIDEBAR};")
        self.detail.setAlignment(Qt.AlignRight)
        text_box.addWidget(self.status)
        text_box.addWidget(self.detail)
        layout.addLayout(text_box)

    def set_status(
        self, message: str, colour: str = theme.ONLINE_GREEN, detail: str | None = None
    ) -> None:
        self.status.setText(message)
        self.dot.set_colour(colour)
        if detail is not None:
            self.detail.setText(detail)


class KpiCard(QFrame):
    """One number with a caption, in the dashboard's summary row."""

    def __init__(self, caption: str, colour: str = theme.TEXT, value: str = "0"):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background:{theme.PANEL_2}; border:1px solid {theme.BORDER};"
            f"border-radius:4px; }}"
        )
        self.setFixedWidth(112)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(0)

        self.caption = QLabel(caption)
        self.caption.setFont(QFont("Segoe UI", 8, QFont.Bold))
        self.caption.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        self.caption.setAlignment(Qt.AlignCenter)

        self.value = QLabel(value)
        self.value.setFont(QFont("Segoe UI Semibold", 21))
        self.value.setStyleSheet(f"color:{colour}; border:none; background:transparent;")
        self.value.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.caption)
        layout.addWidget(self.value)

    def set_value(self, value) -> None:
        self.value.setText(str(value))


class Card(QFrame):
    """White bordered panel used to group a section of the dashboard."""

    def __init__(self, title: str = ""):
        super().__init__()
        self.setStyleSheet(
            f"QFrame {{ background:{theme.PANEL}; border:1px solid {theme.BORDER};"
            f"border-radius:4px; }}"
        )
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(14, 12, 14, 12)
        if title:
            label = QLabel(title)
            label.setFont(QFont("Segoe UI Semibold", 13))
            label.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
            self.body.addWidget(label)


class PosCard(QFrame):
    """One inspection position on the live dashboard.

    The machine shoots the same spots on every panel in the same order, so a
    card per position stays put while panels come and go: the operator watches
    one tile drift instead of hunting for a position in a scrolling table.
    """

    THUMB_W = 168
    THUMB_H = 116

    clicked = Signal(int)
    double_clicked = Signal(int)

    def __init__(self, pos: int):
        super().__init__()
        self.pos = pos
        self._thumb_path = ""
        self._tint = theme.PANEL
        self._selected = False
        self.setFixedWidth(self.THUMB_W + 24)
        self.setCursor(Qt.PointingHandCursor)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(3)

        head = QHBoxLayout()
        head.setSpacing(6)
        self.title = QLabel(f"Pos {pos}" if pos >= 0 else "Unmatched")
        self.title.setFont(QFont("Segoe UI Semibold", 12))
        self.title.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
        self.verdict = QLabel("-")
        self.verdict.setFont(QFont("Segoe UI", 9, QFont.Bold))
        self.verdict.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        head.addWidget(self.title)
        head.addStretch(1)
        head.addWidget(self.verdict)
        layout.addLayout(head)

        self.caption = QLabel("")
        self.caption.setFont(QFont("Segoe UI", 8))
        self.caption.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        layout.addWidget(self.caption)

        self.thumb = QLabel("no image")
        self.thumb.setAlignment(Qt.AlignCenter)
        self.thumb.setFixedSize(self.THUMB_W, self.THUMB_H)
        self.thumb.setStyleSheet(
            f"color:{theme.MUTED}; background:{theme.PANEL_2};"
            f"border:1px solid {theme.BORDER}; border-radius:3px;"
        )
        layout.addWidget(self.thumb)

        self.width_label = QLabel("-")
        self.width_label.setFont(QFont("Segoe UI Semibold", 14))
        self.width_label.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
        layout.addWidget(self.width_label)

        self.dev = QLabel("no baseline")
        self.dev.setFont(QFont("Consolas", 8))
        self.dev.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        layout.addWidget(self.dev)

        self.foot = QLabel("")
        self.foot.setFont(QFont("Segoe UI", 8))
        self.foot.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        layout.addWidget(self.foot)

        self._apply_frame()

    # ------------------------------------------------------------------
    def set_selected(self, selected: bool) -> None:
        if selected != self._selected:
            self._selected = selected
            self._apply_frame()

    def _apply_frame(self) -> None:
        border = f"2px solid {theme.ACCENT}" if self._selected else f"1px solid {theme.BORDER}"
        pad = "9px" if self._selected else "10px"
        self.setStyleSheet(
            f"PosCard {{ background:{self._tint}; border:{border};"
            f"border-radius:4px; margin:{pad}0px; }}"
        )

    def show_record(
        self,
        *,
        caption: str,
        width_text: str,
        dev_text: str,
        verdict: str,
        verdict_colour: str,
        foot: str,
        tint: str,
        image_path: str,
    ) -> None:
        self.caption.setText(caption)
        self.width_label.setText(width_text)
        self.dev.setText(dev_text)
        self.verdict.setText(verdict)
        self.verdict.setStyleSheet(f"color:{verdict_colour}; border:none; background:transparent;")
        self.foot.setText(foot)
        if tint != self._tint:
            self._tint = tint
            self._apply_frame()
        if image_path != self._thumb_path:
            self._thumb_path = image_path
            pix = QPixmap(image_path)
            if pix.isNull():
                self.thumb.setText("no image")
            else:
                self.thumb.setPixmap(
                    pix.scaled(
                        self.THUMB_W, self.THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
                    )
                )

    # ------------------------------------------------------------------
    def mousePressEvent(self, event):  # noqa: N802
        self.clicked.emit(self.pos)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        self.double_clicked.emit(self.pos)
        super().mouseDoubleClickEvent(event)


# ---------------------------------------------------------------------------
# key to the marks drawn on an inspection picture
# ---------------------------------------------------------------------------

# As drawn by vision.render_overlay(), in the order it paints them.
OVERLAY_MARKS = (
    (
        "line",
        "#00ffff",
        "Cyan lines",
        "the edges of the slot as measured in this picture; a side that is a tab or "
        "a contour rather than a straight edge is left out",
    ),
    (
        "dots",
        "#ffff00",
        "Yellow dots",
        "where each scan row found the edge - they should form a straight line",
    ),
)

SWATCH_W = 34
SWATCH_H = 12
PICTURE_GREY = "#4a4a4a"  # inspection pictures are dark, so is the key


def mark_swatch(style: str, colour: str, w: int = SWATCH_W, h: int = SWATCH_H) -> QPixmap:
    """A small sample of one overlay mark, drawn the way the overlay draws it."""
    pix = QPixmap(w, h)
    pix.fill(QColor(PICTURE_GREY))
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    mid = h // 2
    if style == "dots":
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(colour))
        for x in range(3, w, 7):
            p.drawEllipse(x - 2, mid - 2, 4, 4)
    else:
        pen = QPen(QColor(colour), 2)
        if style == "dash":
            pen.setStyle(Qt.CustomDashLine)
            pen.setDashPattern([3, 3])
        p.setPen(pen)
        p.drawLine(2, mid, w - 2, mid)
    p.end()
    return pix


class OverlayLegend(QFrame):
    """Key to the marks on an inspection picture, with a link to the long story.

    Nothing on the picture itself says which mark is which, so this sits under
    every place a measured picture is shown.
    """

    def __init__(self, parent_window=None):
        super().__init__()
        self._parent_window = parent_window
        self.setStyleSheet(
            f"QFrame {{ background:{theme.PANEL_2}; border:1px solid {theme.BORDER};"
            f"border-radius:3px; }}"
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(10, 6, 10, 6)
        outer.setSpacing(4)

        top = QHBoxLayout()
        heading = QLabel("What the marks on the picture mean")
        heading.setFont(QFont("Segoe UI Semibold", 9))
        heading.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
        top.addWidget(heading)
        top.addStretch(1)
        self.btn_help = QPushButton("How is this measured?")
        self.btn_help.setCursor(Qt.PointingHandCursor)
        self.btn_help.clicked.connect(self._open_help)
        top.addWidget(self.btn_help)
        outer.addLayout(top)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(2)
        for row, (style, colour, name, detail) in enumerate(OVERLAY_MARKS):
            swatch = QLabel()
            swatch.setPixmap(mark_swatch(style, colour))
            swatch.setStyleSheet("border:none; background:transparent;")
            title = QLabel(name)
            title.setFont(QFont("Segoe UI", 9, QFont.Bold))
            title.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
            text = QLabel(detail)
            text.setFont(QFont("Segoe UI", 9))
            text.setWordWrap(True)
            text.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
            grid.addWidget(swatch, row, 0)
            grid.addWidget(title, row, 1)
            grid.addWidget(text, row, 2)
        grid.setColumnStretch(2, 1)
        outer.addLayout(grid)

    def _open_help(self):
        OverlayHelpDialog(self._parent_window or self.window()).exec()


class OverlayDiagram(QWidget):
    """A drawn stand-in for an inspection picture, with every mark labelled.

    A real picture is dark and noisy, which is exactly why the marks are hard
    to tell apart on it. This draws the same marks on an idealised slot so the
    operator can match what is on screen to what it means.
    """

    W, H = 560, 300
    PIC = (30, 26, 500, 190)  # x, y, w, h of the pretend picture
    LEFT_EDGE, RIGHT_EDGE = 205, 355  # measured edges, px in this widget

    def __init__(self):
        super().__init__()
        self.setFixedSize(self.W, self.H)

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(theme.PANEL))

        x, y, w, h = self.PIC
        # the picture: two strips of green solder mask with the cut between them
        p.fillRect(x, y, w, h, QColor("#3f3f3f"))
        p.fillRect(x, y, self.LEFT_EDGE - x, h, QColor("#1e4d3b"))
        p.fillRect(self.RIGHT_EDGE, y, x + w - self.RIGHT_EDGE, h, QColor("#1e4d3b"))
        p.fillRect(self.LEFT_EDGE, y, self.RIGHT_EDGE - self.LEFT_EDGE, h, QColor("#161616"))
        p.setPen(QPen(QColor(theme.BORDER), 1))
        p.drawRect(x, y, w, h)

        # scan rows: the tool walks outward from the middle of the picture
        p.setPen(QPen(QColor("#6b7280"), 1, Qt.DotLine))
        rows = list(range(y + 16, y + h - 8, 22))
        for ry in rows:
            p.drawLine(x + 6, ry, x + w - 6, ry)

        # yellow dots: one hit per scan row, jittering around the real edge
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#ffff00"))
        jitter = (0, 1, -1, 2, 0, -1, 1, 0, -2, 1)
        for i, ry in enumerate(rows):
            for edge, sign in ((self.LEFT_EDGE, 1), (self.RIGHT_EDGE, -1)):
                p.drawEllipse(edge + sign * jitter[i % len(jitter)] - 2, ry - 2, 5, 5)

        # cyan: the median of those dots
        p.setPen(QPen(QColor("#00ffff"), 2))
        for edge in (self.LEFT_EDGE, self.RIGHT_EDGE):
            p.drawLine(edge, y, edge, y + h)

        p.setFont(QFont("Segoe UI", 8))
        p.setPen(QColor("#e5e7eb"))
        p.drawText(x + 8, y + 14, "solder mask")
        p.drawText(self.RIGHT_EDGE + 8, y + 14, "solder mask")
        p.drawText(self.LEFT_EDGE + 6, y + h - 8, "the cut")

        # ---- measurements called out under the picture --------------------
        base = y + h + 26
        p.setFont(QFont("Segoe UI", 9))

        p.setPen(QPen(QColor("#0e7490"), 2))
        self._span(p, self.LEFT_EDGE, self.RIGHT_EDGE, base)
        p.setPen(QColor(theme.TEXT))
        p.drawText(self.LEFT_EDGE + 8, base - 6, "Width mm")

        p.setPen(QColor(theme.MUTED))
        p.setFont(QFont("Segoe UI", 8))
        p.drawText(x, base + 34, "dWidth and dShift compare these two numbers with the calibrated")
        p.drawText(
            x, base + 50, "baseline, which is held as a number and not drawn on the picture."
        )

    @staticmethod
    def _span(p, x_from: int, x_to: int, y: int) -> None:
        """A double-headed arrow between two verticals."""
        if x_to < x_from:
            x_from, x_to = x_to, x_from
        p.drawLine(x_from, y, x_to, y)
        for end, direction in ((x_from, 1), (x_to, -1)):
            p.drawLine(end, y, end + 5 * direction, y - 4)
            p.drawLine(end, y, end + 5 * direction, y + 4)
            p.drawLine(end, y - 6, end, y + 6)


class OverlayHelpDialog(QDialog):
    """The long version: how a picture turns into Width mm, dWidth and dShift."""

    STEPS = (
        (
            "1. Find the board",
            None,
            "Solder mask is green, so every pixel that is clearly more green than red "
            "counts as board and everything else counts as cut, shadow or debris. The "
            "three green thresholds on the Configuration tab decide how strict that is.",
        ),
        (
            "2. Scan row by row",
            None,
            "Nothing is scanned by hardware here - the picture is already taken, and a "
            "'scan row' is simply one horizontal row of its pixels. Every row of the "
            "picture is scanned, about a thousand of them, which is what the Scan Y "
            "and Scan row step settings on the Configuration tab control. On each row "
            "the tool starts at the middle of the picture and walks outward, one pixel "
            "at a time, until it meets board on the left and on the right. The slot "
            "therefore has to sit near the middle of the frame, and a row that runs off "
            "the edge without finding board is thrown away - that is the difference "
            "between the two numbers in 'scan  n/m rows'.",
        ),
        (
            "3. Mark every hit",
            "dots",
            "Each of those two hits is drawn as a yellow dot - a sample of them, since "
            "one dot per row would merge into a solid bar. Two straight columns of "
            "dots mean a clean edge. Dots that wander or thin out mean burrs, debris or "
            "weak contrast - the 'scan  n/m rows  spread' line under the picture puts a "
            "number on it, and below 'Min valid rows' the picture is not measured at all.",
        ),
        (
            "4. Take the median",
            "line",
            "The cyan lines are the median of the dots on each side, which is what makes "
            "one burr or one speck harmless. The gap between them is the slot width: "
            "'Width mm' on the tile. When one side is not a straight edge - a breakaway "
            "tab or a contour cut - it is dropped from both the drawing and the sum, the "
            "tile says 'left edge only', and there is no width to report; only how far "
            "that straight edge has moved.",
        ),
        (
            "5. Compare with the baseline",
            None,
            "Calibration measured this same position on known-good panels and stored where "
            "its edges sat. That baseline is a stored number, not drawn on the picture. "
            "dWidth is measured width minus baseline width - the slot got wider or "
            "narrower. dShift is the measured edge minus the baseline edge - the cut moved "
            "sideways. 'no baseline yet' on a tile means this position was never "
            "calibrated; run calibration on the Configuration tab.",
        ),
        (
            "6. Judge it",
            None,
            "A tile turns red when dWidth or dShift passes the tolerance set on the "
            "Configuration tab, or when the trained model calls the picture NG. Green "
            "means both agree it is fine, amber means there is nothing to compare against.",
        ),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("How the slot is measured")
        self.resize(660, 780)

        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        title = QLabel("Reading the marks on an inspection picture")
        title.setFont(QFont("Segoe UI Semibold", 14))
        title.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
        layout.addWidget(title)

        lead = QLabel(
            "Every picture is measured the same way, with no model involved: it is "
            "plain colour thresholding plus a median. The marks drawn on the picture "
            "are the working of that sum."
        )
        lead.setWordWrap(True)
        lead.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        layout.addWidget(lead)

        layout.addWidget(OverlayDiagram())

        for heading, style, body in self.STEPS:
            row = QHBoxLayout()
            row.setSpacing(8)
            if style:
                colour = next(c for st, c, *_ in OVERLAY_MARKS if st == style)
                swatch = QLabel()
                swatch.setPixmap(mark_swatch(style, colour, 30, 11))
                swatch.setAlignment(Qt.AlignTop)
                swatch.setStyleSheet("border:none; background:transparent;")
                row.addWidget(swatch, 0, Qt.AlignTop)
            else:
                row.addSpacing(30)

            block = QVBoxLayout()
            block.setSpacing(1)
            head = QLabel(heading)
            head.setFont(QFont("Segoe UI", 10, QFont.Bold))
            head.setStyleSheet(f"color:{theme.ACCENT_DARK}; border:none; background:transparent;")
            text = QLabel(body)
            text.setWordWrap(True)
            text.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
            block.addWidget(head)
            block.addWidget(text)
            row.addLayout(block, 1)
            layout.addLayout(row)

        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(close)

        shell = QVBoxLayout(self)
        shell.setContentsMargins(0, 0, 10, 10)
        shell.addWidget(scroll, 1)
        shell.addLayout(buttons)
