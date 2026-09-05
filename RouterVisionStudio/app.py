"""Router Vision Studio.

One place to watch the router line, measure every cut, label what went wrong,
train a DINOv2 classifier on those labels, and put that model straight back
into the live view.

    python app.py
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import cv2
from PySide6.QtCore import QRectF, QSize, Qt, QThread, QTimer, Signal
from PySide6.QtGui import (QBrush, QColor, QFont, QIcon, QImage, QPainter, QPen,
                           QPixmap)
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QProgressBar,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router_vision import (  # noqa: E402
    AppConfig, BaselineStore, CropBox, CutClassifier, DEFAULT_CLASSES,
    LabelStore, PositionLabels, ProtectedPathError, available_days, calibrate,
    check_write_target, find_recipe, index_pictures, inspect, load_day,
    fit_bounds, load_toolpath, pick_device, protected_roots, read_machine_flags,
    read_pixel_size, render_overlay,
)
from router_vision.vision import draw_overlay  # noqa: E402
from router_vision import theme  # noqa: E402
from router_vision.changes import baseline_is_stale, find_bit_events  # noqa: E402
from router_vision.analysis import MIN_CALIB_RUNS, _measure  # noqa: E402
from router_vision.config import DEFAULT_BASELINE_NAME, DEFAULT_CONFIG_NAME  # noqa: E402
from router_vision.machine import PICTURE_STAMP, attach_pictures, load_runs  # noqa: E402
from router_vision.widgets import (Card, HeaderBar, KpiCard,  # noqa: E402
                                   OverlayLegend, PosCard)

APP_DIR = Path(__file__).resolve().parent
CONFIG_PATH = APP_DIR / DEFAULT_CONFIG_NAME
BASELINE_PATH = APP_DIR / DEFAULT_BASELINE_NAME
POSITION_PATH = APP_DIR / "positions.json"
LABEL_PATH = APP_DIR / "labels.json"
MODEL_DIR = APP_DIR / "models"          # used when the config names no folder
MODEL_PREFIX = "cut_classifier_"


def model_file_name(key: str) -> str:
    """File name for one ProductId|table, e.g. cut_classifier_PROD_LeftTable.pt."""
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return f"{MODEL_PREFIX}{safe}.pt"

THUMB = QSize(210, 158)
# A baseline built from a handful of panels is not worth having, so automatic
# calibration waits until a product/table has produced at least this many.
MIN_AUTO_CALIB_PANELS = 10
COLOR_OK = QColor(theme.TINT_OK)
COLOR_NG = QColor(theme.TINT_NG)
COLOR_WARN = QColor(theme.TINT_WARN)
COLOR_GREY = QColor(theme.TINT_GREY)
STATUS_COLORS = {
    "OK": None, "NG WIDTH": COLOR_NG, "NG SHIFT": COLOR_NG,
    "WARN ALIGN": COLOR_WARN, "NO BASELINE": COLOR_GREY,
    "NO DATA": COLOR_GREY, "SKIP": COLOR_GREY,
}
CLASS_COLORS = {"GOOD": COLOR_OK, "NG": COLOR_NG}


def bgr_to_pixmap(bgr) -> QPixmap:
    h, w, ch = bgr.shape
    return QPixmap.fromImage(QImage(bgr.data, w, h, ch * w, QImage.Format_BGR888).copy())


class Worker(QThread):
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self._fn, self._cancelled = fn, False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self) -> bool:
        return self._cancelled

    def run(self):
        try:
            result = self._fn(self)
        except InterruptedError:
            self.failed.emit("Cancelled.")
        except (ValueError, OSError) as exc:
            # expected data problems - the operator needs the sentence, not a stack
            self.failed.emit(str(exc))
        except Exception:
            self.failed.emit(traceback.format_exc(limit=5))
        else:
            self.done.emit(result)


class DayPicker(QWidget):
    """Year / month / day, as three plain pickers.

    A day is still handled as one YYYYMMDD string everywhere else, so this
    offers the slice of QComboBox that the day-picker plumbing drives it
    through and does the splitting on screen only.
    """

    def __init__(self):
        super().__init__()
        self._days: list[str] = []
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.cmb_year = QComboBox()
        self.cmb_year.setFixedWidth(74)
        self.cmb_month = QComboBox()
        self.cmb_month.setFixedWidth(54)
        self.cmb_day = QComboBox()
        self.cmb_day.setFixedWidth(54)
        # Months and days are the whole calendar, not only what the folder holds:
        # picking a date with no pictures simply reports that it has none.
        self.cmb_month.addItems([str(m) for m in range(1, 13)])
        self.cmb_day.addItems([str(d) for d in range(1, 32)])
        for caption, box in (("Y", self.cmb_year), ("M", self.cmb_month), ("D", self.cmb_day)):
            tag = QLabel(caption)
            tag.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
            layout.addWidget(tag)
            layout.addWidget(box)
        self._fill_years()

    def _fill_years(self) -> None:
        """From the earliest year in the data through to this year."""
        this_year = datetime.now().year
        first = min((int(d[:4]) for d in self._days), default=this_year)
        years = [str(y) for y in range(min(first, this_year), this_year + 1)]
        keep = self.cmb_year.currentText()
        blocked = self.cmb_year.blockSignals(True)
        self.cmb_year.clear()
        self.cmb_year.addItems(years)
        self.cmb_year.setCurrentIndex(years.index(keep) if keep in years else len(years) - 1)
        self.cmb_year.blockSignals(blocked)

    # -- the part of QComboBox the rest of the app calls --------------------
    def clear(self) -> None:
        self.addItems([])

    def addItems(self, days) -> None:
        self._days = list(days)
        self._fill_years()

    def count(self) -> int:
        return len(self._days)

    def itemText(self, index: int) -> str:
        return self._days[index] if 0 <= index < len(self._days) else ""

    def currentText(self) -> str:
        year = self.cmb_year.currentText()
        month = self.cmb_month.currentText()
        day = self.cmb_day.currentText()
        if not (year and month and day):
            return ""
        return f"{year}{int(month):02d}{int(day):02d}"

    def setCurrentIndex(self, index: int) -> None:
        day = self.itemText(index)
        if len(day) != 8:
            return
        if self.cmb_year.findText(day[:4]) < 0:
            self.cmb_year.addItem(day[:4])
        self.cmb_year.setCurrentText(day[:4])
        self.cmb_month.setCurrentText(str(int(day[4:6])))
        self.cmb_day.setCurrentText(str(int(day[6:8])))

    def blockSignals(self, block: bool) -> bool:
        for box in (self.cmb_year, self.cmb_month, self.cmb_day):
            box.blockSignals(block)
        return super().blockSignals(block)


class ThumbnailLoader(QThread):
    ready = Signal(int, object)

    def __init__(self, paths: list[str], cfg=None):
        super().__init__()
        self.paths, self._stop = paths, False
        # With a config the thumbnails carry the measurement overlay: on the
        # Label tab that is the whole point, since the operator is judging the
        # cut and needs to see what the tool measured before calling it PASS/NG.
        self.cfg = cfg

    def stop(self):
        self._stop = True

    def run(self):
        for i, path in enumerate(self.paths):
            if self._stop:
                return
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            s = min(THUMB.width() / img.shape[1], THUMB.height() / img.shape[0])
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if self.cfg is not None:
                # drawn after the resize, so the marks stay one crisp pixel wide
                small = draw_overlay(small, _measure(self.cfg, path), scale=s)
            self.ready.emit(i, bgr_to_pixmap(small))


# ---------------------------------------------------------------------------
# toolpath drawing
# ---------------------------------------------------------------------------

class ToolpathView(QWidget):
    """Draws the cut geometry read out of the recipe."""

    def __init__(self):
        super().__init__()
        self.toolpath = None
        self.setMinimumHeight(360)
        self.setStyleSheet("background:#ffffff;")

    def set_toolpath(self, toolpath):
        self.toolpath = toolpath
        self.update()

    def paintEvent(self, event):  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#ffffff"))
        if not self.toolpath or not self.toolpath.segments:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Select a recipe to see its cutting path.")
            return

        # Fit to the body of the geometry: one mis-parsed segment out in the
        # weeds would otherwise push the real cut path into a corner.
        (x0, y0, x1, y1), outliers = fit_bounds(self.toolpath.segments)
        span_x, span_y = max(x1 - x0, 1e-6), max(y1 - y0, 1e-6)
        margin = 34
        w = max(self.width() - 2 * margin, 10)
        h = max(self.height() - 2 * margin, 10)
        scale = min(w / span_x, h / span_y)
        off_x = margin + (w - span_x * scale) / 2
        off_y = margin + (h - span_y * scale) / 2

        def to_px(pt):
            # machine Y grows upward, screen Y grows downward
            return (off_x + (pt[0] - x0) * scale,
                    off_y + (y1 - pt[1]) * scale)

        # frame
        p.setPen(QPen(QColor("#d1d5db"), 1))
        p.drawRect(QRectF(off_x, off_y, span_x * scale, span_y * scale))

        p.setClipRect(self.rect())
        outlier_set = set(outliers)
        for index, seg in enumerate(self.toolpath.segments):
            pts = [to_px(pt) for pt in seg.points]
            if index in outlier_set:
                # Off the fitted view: drawn in grey where it lands, so a bad
                # parse is visible as junk instead of silently reshaping the page.
                p.setPen(QPen(QColor("#9ca3af"), 1, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                for i in range(len(pts) - 1):
                    a, b = pts[i], pts[i + 1]
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            elif seg.is_slot:
                p.setPen(QPen(QColor("#b91c1c"), 2))
                p.setBrush(QBrush(QColor(185, 28, 28, 60)))
                cx = sum(q[0] for q in pts) / len(pts)
                cy = sum(q[1] for q in pts) / len(pts)
                p.drawEllipse(QRectF(cx - 5, cy - 5, 10, 10))
                for i in range(len(pts)):
                    a, b = pts[i], pts[(i + 1) % len(pts)]
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
            else:
                p.setPen(QPen(QColor("#2563eb"), 2))
                p.setBrush(Qt.NoBrush)
                for i in range(len(pts) - 1):
                    a, b = pts[i], pts[i + 1]
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

        p.setPen(QColor("#374151"))
        p.setFont(QFont("Segoe UI", 9))
        p.drawText(margin, 20, f"{self.toolpath.recipe.name}   {self.toolpath.summary()}")
        legend = "red = routed slot / breakaway tab        blue = cut segment"
        if outliers:
            legend += (f"        grey = {len(outliers)} segment(s) outside the "
                       "fitted view, check the recipe")
        p.drawText(margin, self.height() - 10, legend)


# ---------------------------------------------------------------------------
# image viewer
# ---------------------------------------------------------------------------

class ImageViewer(QDialog):
    def __init__(self, cfg, path, baseline_index=None, caption="", prediction=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"{caption} - {Path(path).name}" if caption else Path(path).name)
        self.resize(1040, 900)
        meas = _measure(cfg, path)
        bl_l = baseline_index.left_mean if baseline_index else None
        bl_w = baseline_index.width_mean if baseline_index else None
        pix = (bgr_to_pixmap(render_overlay(path, meas)) if cfg.show_overlay
               else QPixmap(path))

        view = QLabel()
        view.setAlignment(Qt.AlignCenter)
        view.setPixmap(pix.scaled(1000, 740, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        info = QLabel()
        info.setFont(QFont("Consolas", 9))
        lines = []
        if meas.valid:
            rough = meas.rough_side or (baseline_index.rough_side if baseline_index else "")
            side = "right" if rough == "left" else "left"
            edge_px = meas.slot_right if side == "right" else meas.slot_left
            if rough:
                lines.append(f"measured   {side} edge {edge_px} px   "
                             f"({rough} edge is a tab or contour, not judged)")
            else:
                lines.append(f"measured   width {meas.width} px = "
                             f"{cfg.px_to_mm(meas.width):.4f} mm"
                             f"   left {meas.slot_left} px   right {meas.slot_right} px")
            lines.append(f"scan rows  {meas.rows}/{meas.attempted}   spread "
                         f"{cfg.px_to_mm(meas.width_sd):.4f} mm")
            if baseline_index and rough:
                shift = cfg.px_to_mm(edge_px - baseline_index.edge(side))
                lines.append(f"vs baseline  shift {shift:+.4f} mm ({side} edge)")
            elif baseline_index:
                lines.append(f"vs baseline  width {cfg.px_to_mm(meas.width - bl_w):+.4f} mm"
                             f"   shift {cfg.px_to_mm(meas.slot_left - bl_l):+.4f} mm")
        else:
            lines.append(f"No routed slot visible ({meas.rows}/{meas.attempted} scan rows).")
        if prediction and prediction[0]:
            lines.append(f"model      {prediction[0]}  ({prediction[1]:.1%} confidence)")
        info.setText("\n".join(lines))

        layout = QVBoxLayout(self)
        layout.addWidget(view, 1)
        layout.addWidget(OverlayLegend())
        layout.addWidget(info)


# ---------------------------------------------------------------------------
# main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Router Vision Studio")
        self.resize(1480, 940)

        self.cfg = AppConfig.load(CONFIG_PATH)
        roots = protected_roots(self.cfg)
        self.store = BaselineStore(BASELINE_PATH, protected=roots)
        self.labels = PositionLabels(POSITION_PATH, protected=roots)
        self.label_store = LabelStore(LABEL_PATH, protected=roots)
        # One model per ProductId. None in the cache means "looked, none there",
        # so a product without a model is not hunted for on every poll.
        self.classifiers: dict[str, CutClassifier | None] = {}
        self.picture_index: dict[str, list[str]] = {}
        self.results: list = []
        self.visible_results: list = []
        self.current = None
        self.worker: Worker | None = None
        self.thumbs: ThumbnailLoader | None = None
        self.label_paths: list[str] = []
        self.label_runs: dict[str, object] = {}   # picture path -> its Run, for ProductId
        self.seen_live: set[str] = set()
        self.live_rows: list[dict] = []
        self._run_cache: dict[str, list] = {}

        self.header = HeaderBar("Router Vision Studio")
        tabs = QTabWidget()
        # Ordered by how often the shop floor reaches for them: watch the line,
        # label what it saw, retrain, and only then the occasional deep dives.
        tabs.addTab(self._build_dashboard_tab(), "Dashboard Real-time")
        tabs.addTab(self._build_label_tab(), "Label")
        tabs.addTab(self._build_train_tab(), "Train Model")
        tabs.addTab(self._build_inspect_tab(), "Inspect Panels")
        tabs.addTab(self._build_toolpath_tab(), "Cut Path")
        tabs.addTab(self._build_config_tab(), "Configuration")

        shell = QWidget()
        shell_layout = QVBoxLayout(shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)
        shell_layout.addWidget(self.header)
        wrap = QWidget()
        wrap_layout = QVBoxLayout(wrap)
        wrap_layout.setContentsMargins(14, 12, 14, 12)
        wrap_layout.addWidget(tabs)
        shell_layout.addWidget(wrap, 1)
        self.setCentralWidget(shell)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

        self.live_timer = QTimer(self)
        self.live_timer.timeout.connect(self._live_poll)
        self.cal_timer = QTimer(self)
        self.cal_timer.timeout.connect(self._auto_calibrate)
        # The index is built for you: once shortly after the window appears, then
        # on a timer so pictures written during the shift keep showing up.
        self.index_timer = QTimer(self)
        self.index_timer.timeout.connect(lambda: self._refresh_index(quiet=True))
        self.index_timer.start(self.spin_index_every.value() * 60_000)
        QTimer.singleShot(300, lambda: self._refresh_index(quiet=False))
        # Auto calibration comes back on if that is how the config was saved,
        # but the first sweep is left to the timer so opening the app stays quick.
        if self.chk_auto_cal.isChecked():
            self._toggle_auto_calibrate(run_now=False)

        self._refresh_model_label()
        self.status("Ready. Config -> Calibrate -> Inspect. Label and Train when you want a model.")

    # -- helpers ------------------------------------------------------------
    def status(self, text, colour=theme.ONLINE_GREEN, detail=None):
        self.statusBar().showMessage(text)
        self.header.set_status(text, colour, detail)

    def _busy(self, busy):
        self.progress.setVisible(busy)
        for b in (self.btn_calibrate, self.btn_inspect, self.btn_list_recipes,
                  self.btn_index, self.btn_train, self.btn_label_load):
            b.setEnabled(not busy)

    def _on_progress(self, done, total, label):
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        self.status(f"{label}  {done}/{total}" if total else label)

    def _fail(self, message):
        self._busy(False)
        self.status("Failed.", theme.ALERT_RED)
        QMessageBox.critical(self, "Router Vision Studio", message)

    def _run_worker(self, job, on_done):
        self.worker = Worker(job)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(on_done)
        self.worker.failed.connect(self._fail)
        self.worker.start()

    def _resync_day_combos(self):
        """Re-fill every day picker after the index changed, without a dialog."""
        try:
            shown, _ = self._days_with_pictures()
        except OSError:
            return
        for name in ("cmb_ins_day", "cmb_lbl_day", "cmb_cal_day"):
            combo = getattr(self, name, None)
            if combo is not None and [combo.itemText(i) for i in range(combo.count())] != shown:
                self._fill_day_combo(combo, shown)

    def _ensure_index(self) -> bool:
        """Guarantee the index exists, building it here if nobody has yet.

        Scanning the folder takes a fraction of a second even at 140k files, so
        it is cheaper to just do it than to make the operator go and press a
        button on another tab.
        """
        if self.picture_index:
            return True
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.picture_index = index_pictures(self.cfg.picture_dir)
        except OSError as exc:
            QMessageBox.warning(self, "Picture index",
                                f"Cannot read the picture folder:\n{exc}")
            return False
        finally:
            QApplication.restoreOverrideCursor()
        self._report_index()
        return bool(self.picture_index)

    def _report_index(self):
        days = len(self.picture_index)
        total = sum(len(v) for v in self.picture_index.values())
        text = (f"{total:,} images over {days} days" if days
                else "no images found in the picture folder")
        if hasattr(self, "lbl_index"):
            self.lbl_index.setText(f"index: {text}  -  {time.strftime('%H:%M:%S')}")
        return text

    def _refresh_index(self, quiet: bool = True):
        """Rescan the picture folder in the background so new images show up."""
        if self.worker is not None and self.worker.isRunning():
            return
        cfg = self._collect()

        def job(worker):
            return index_pictures(
                cfg.picture_dir,
                progress=lambda n: worker.progress.emit(n, n, "Indexing pictures"))

        def finished(result):
            self.picture_index = result
            self._run_cache.clear()      # picture lists per run are now stale
            self._busy(False)
            self._resync_day_combos()    # days whose pictures just landed appear now
            text = self._report_index()
            if not quiet:
                self.status(f"Picture index rebuilt: {text}")

        self._busy(True)
        self._run_worker(job, finished)

    def _days_with_pictures(self) -> tuple[list[str], int]:
        """Days a tab may offer, plus how many run days are still without pictures.

        Every day picker feeds work that measures images, so a day whose pictures
        have not arrived yet is left out and joins the list by itself once they do.
        """
        days = available_days(self._collect().result_dir)
        if not self.picture_index:
            return days, 0
        shown = [d for d in days if self.picture_index.get(d)]
        return shown, len(days) - len(shown)

    def _fill_day_combo(self, combo, shown: list[str]) -> None:
        keep = combo.currentText()
        blocked = combo.blockSignals(True)
        combo.clear()
        combo.addItems(shown)
        if shown:
            combo.setCurrentIndex(shown.index(keep) if keep in shown else len(shown) - 1)
        combo.blockSignals(blocked)

    def _refresh_days(self, combo):
        self._ensure_index()
        try:
            shown, hidden = self._days_with_pictures()
        except OSError as exc:
            QMessageBox.warning(self, "Refresh days", str(exc))
            return
        self._fill_day_combo(combo, shown)
        if not shown:
            self.status(f"No day has pictures yet - {hidden} run days waiting.",
                        theme.ALERT_RED)
        else:
            note = f"  ({hidden} without pictures hidden)" if hidden else ""
            self.status(f"Found {len(shown)} days with pictures{note}.")

    # ======================================================================
    # Dashboard - live view
    # ======================================================================
    def _build_dashboard_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(4, 4, 4, 4)

        # ---- summary card: title, KPI row, controls, SN search ----
        summary = Card()
        title_row = QHBoxLayout()
        title = QLabel("Dashboard Real-time  |  slot measurement and model verdict")
        title.setFont(QFont("Segoe UI Semibold", 14))
        title.setStyleSheet(f"color:{theme.TEXT}; border:none; background:transparent;")
        title_row.addWidget(title)
        title_row.addStretch(1)

        self.kpi = {}
        for caption, colour in (("PICTURES", theme.TEXT), ("PANELS", theme.ACCENT_DARK),
                                ("PASS", theme.ONLINE_GREEN), ("NG", theme.ALERT_RED),
                                ("UNJUDGED", theme.RUNNING_ORANGE), ("MEAN mm", theme.TEXT)):
            card = KpiCard(caption, colour)
            self.kpi[caption] = card
            title_row.addWidget(card)
        summary.body.addLayout(title_row)

        controls = QHBoxLayout()
        self.chk_live = QCheckBox("Watch picture folder")
        self.chk_live.stateChanged.connect(self._toggle_live)
        self.spin_poll = QSpinBox()
        self.spin_poll.setRange(1, 120)
        self.spin_poll.setValue(5)
        self.spin_poll.setSuffix(" s")
        btn_once = QPushButton("Check now")
        btn_once.setProperty("primary", True)
        btn_once.clicked.connect(self._live_poll)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._clear_live)
        self.chk_overlay_dash = QCheckBox("Overlay")
        self.chk_overlay_dash.setToolTip(
            "Draw the measured edges on the picture. Turn it off to look at the "
            "bare cut - the measurement itself is unaffected either way.")
        self.chk_overlay_dash.setChecked(self.cfg.show_overlay)
        self.chk_overlay_dash.toggled.connect(self._toggle_overlay)
        controls.addWidget(self.chk_live)
        controls.addWidget(QLabel("every"))
        controls.addWidget(self.spin_poll)
        controls.addWidget(btn_once)
        controls.addWidget(btn_clear)
        controls.addWidget(self.chk_overlay_dash)
        controls.addSpacing(24)

        lbl_sn = QLabel("Search SN")
        lbl_sn.setFont(QFont("Segoe UI", 10, QFont.Bold))
        lbl_sn.setStyleSheet("border:none; background:transparent;")
        self.edit_sn = QLineEdit()
        self.edit_sn.setPlaceholderText("panel serial, e.g. 32078")
        self.edit_sn.setFixedWidth(230)
        self.edit_sn.returnPressed.connect(self._filter_live)
        btn_find = QPushButton("Find")
        btn_find.setProperty("primary", True)
        btn_find.clicked.connect(self._filter_live)
        btn_clear_sn = QPushButton("Clear")
        btn_clear_sn.clicked.connect(lambda: (self.edit_sn.clear(), self._filter_live()))
        for w in (lbl_sn, self.edit_sn, btn_find, btn_clear_sn):
            controls.addWidget(w)
        controls.addStretch(1)
        self.lbl_model = QLabel("model: none")
        self.lbl_model.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        controls.addWidget(self.lbl_model)
        summary.body.addLayout(controls)

        note = QLabel(
            "Each new image is measured against the calibrated baseline and, when a model "
            "is trained, classified GOOD/NG from its Sobel edge map. Machine data is only "
            "ever read.")
        note.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        note.setWordWrap(True)
        summary.body.addWidget(note)
        outer.addWidget(summary)

        # ---- result card: a tile per position left, picture and detail right ----
        result = Card()
        split = QSplitter(Qt.Horizontal)

        # One card per inspection position, newest picture of that position on
        # top. Positions keep their place as panels come and go, so a spot that
        # drifts stays under the same tile instead of scrolling away.
        self.pos_cards: dict[int, PosCard] = {}
        self.live_by_pos: dict[int, dict] = {}
        self.selected_pos: int | None = None
        self._grid_keys: tuple[int, ...] = ()
        self._grid_cols = 0

        self.pos_holder = QWidget()
        self.pos_grid = QGridLayout(self.pos_holder)
        self.pos_grid.setContentsMargins(4, 4, 4, 4)
        self.pos_grid.setSpacing(8)
        self.pos_grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.lbl_no_cards = QLabel("Waiting for the first pictures of a panel...")
        self.lbl_no_cards.setStyleSheet(
            f"color:{theme.MUTED}; border:none; background:transparent;")
        self.pos_grid.addWidget(self.lbl_no_cards, 0, 0)

        self.pos_scroll = QScrollArea()
        self.pos_scroll.setWidgetResizable(True)
        self.pos_scroll.setWidget(self.pos_holder)
        self.pos_scroll.setStyleSheet(
            f"QScrollArea {{ background:{theme.PANEL}; border:1px solid {theme.BORDER};"
            "border-radius:3px; }")
        split.addWidget(self.pos_scroll)
        split.splitterMoved.connect(lambda *_: self._layout_pos_cards(force=True))

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.live_image = QLabel("Waiting for images...")
        self.live_image.setAlignment(Qt.AlignCenter)
        self.live_image.setMinimumSize(520, 380)
        self.live_image.setStyleSheet(
            f"background:{theme.PANEL_2}; border:1px solid {theme.BORDER}; border-radius:3px;")
        rl.addWidget(self.live_image, 1)
        self.legend_dash = OverlayLegend()
        rl.addWidget(self.legend_dash)
        self.live_verdict = QLabel("")
        self.live_verdict.setAlignment(Qt.AlignCenter)
        self.live_verdict.setFont(QFont("Segoe UI Semibold", 15))
        self.live_verdict.setStyleSheet("border:none; background:transparent;")
        rl.addWidget(self.live_verdict)
        self.live_info = QLabel("")
        self.live_info.setFont(QFont("Consolas", 9))
        self.live_info.setStyleSheet(
            f"background:{theme.PANEL_2}; border:1px solid {theme.BORDER};"
            "border-radius:3px; padding:8px;")
        self.live_info.setWordWrap(True)
        rl.addWidget(self.live_info)
        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 5)
        result.body.addWidget(split)
        outer.addWidget(result, 1)
        return page

    def _clear_live(self):
        self.live_rows = []
        self.seen_live.clear()
        self.selected_pos = None
        self._refresh_pos_cards()
        self.live_image.setText("Waiting for images...")
        self.live_verdict.setText("")
        self.live_info.setText("")
        self._refresh_kpi()
        self.status("Feed cleared.")

    def _filter_live(self):
        needle = self.edit_sn.text().strip().lower()
        shown = self._refresh_pos_cards()
        self.status(f"{shown} position(s) match SN '{needle}'." if needle
                    else f"{shown} position(s).")

    # ---- the position grid ------------------------------------------------
    def _visible_records(self) -> dict[int, dict]:
        """Newest picture per position, honouring the SN search box.

        live_rows is in arrival order, so a later record for the same position
        simply overwrites the earlier one.
        """
        needle = self.edit_sn.text().strip().lower()
        latest: dict[int, dict] = {}
        for record in getattr(self, "live_rows", []):
            # pos -1 means the picture matched no run; it still gets a tile so a
            # stream of unmatched images is visible instead of silently dropped.
            if needle and needle not in record["sn"].lower():
                continue
            latest[record["pos"]] = record
        return latest

    def _refresh_pos_cards(self) -> int:
        """Rebuild the tiles from the current feed. Returns how many are shown."""
        cfg = self.cfg
        self.live_by_pos = self._visible_records()

        for pos in [p for p in self.pos_cards if p not in self.live_by_pos]:
            card = self.pos_cards.pop(pos)
            self.pos_grid.removeWidget(card)
            card.deleteLater()

        for pos, record in self.live_by_pos.items():
            card = self.pos_cards.get(pos)
            if card is None:
                card = PosCard(pos)
                card.clicked.connect(self._select_pos)
                card.double_clicked.connect(self._open_live_image)
                self.pos_cards[pos] = card

            run = record["run"]
            name = self.labels.label(run.key, pos) if run else ""
            dev_w, dev_e = record["dev_w"], record["dev_e"]
            rough, side = record.get("rough", ""), record.get("side", "left")
            over = ((dev_w is not None and abs(dev_w) > cfg.width_tol_mm)
                    or (dev_e is not None and abs(dev_e) > cfg.edge_tol_mm))
            if record["cls"] == "NG" or over:
                tint = theme.TINT_NG
            elif record["cls"] == "GOOD":
                tint = theme.TINT_OK
            elif dev_e is None:
                tint = theme.TINT_WARN
            else:
                tint = theme.PANEL
            if record["cls"]:
                verdict = record["cls"]
            elif over:
                verdict = "OUT OF TOL"
            elif not record.get("has_model", True):
                verdict = "no model"      # nothing trained for this ProductId yet
            else:
                verdict = "-"
            colour = (theme.ALERT_RED if verdict in ("NG", "OUT OF TOL")
                      else theme.ONLINE_GREEN if verdict == "GOOD" else theme.MUTED)
            if dev_e is None:
                dev_text = "no baseline yet"
            elif rough:
                # only one straight edge here, so there is no width to report
                dev_text = f"dS {dev_e:+.4f}   ({side} edge)"
            else:
                dev_text = f"dW {dev_w:+.4f}   dS {dev_e:+.4f}"
            if record["width_mm"] is not None:
                width_text = f"{record['width_mm']:.4f} mm"
            elif rough:
                width_text = f"{side} edge only"
            else:
                width_text = "not measurable"
            stamp = record["time"]
            conf = f"   {record['conf']:.0%}" if record["cls"] else ""
            card.show_record(
                caption=name or (f"SN {record['sn']}" if record["sn"] else "unmatched"),
                width_text=width_text,
                dev_text=dev_text, verdict=verdict, verdict_colour=colour,
                foot=f"{stamp[:2]}:{stamp[2:4]}:{stamp[4:]}{conf}",
                tint=tint, image_path=record["path"])
            card.set_selected(pos == self.selected_pos)

        self._layout_pos_cards()
        self.lbl_no_cards.setVisible(not self.pos_cards)
        return len(self.pos_cards)

    def _layout_pos_cards(self, force: bool = False) -> None:
        """Place the tiles in as many columns as the pane is wide enough for."""
        if not self.pos_cards:
            return
        step = PosCard.THUMB_W + 24 + self.pos_grid.spacing()
        cols = max(1, (self.pos_scroll.viewport().width() - 8) // step)
        keys = tuple(sorted(self.pos_cards, key=lambda p: (p < 0, p)))
        if not force and keys == self._grid_keys and cols == self._grid_cols:
            return
        self._grid_keys, self._grid_cols = keys, cols
        for i, pos in enumerate(keys):
            self.pos_grid.addWidget(self.pos_cards[pos], i // cols, i % cols)

    def _select_pos(self, pos: int) -> None:
        self.selected_pos = pos
        for key, card in self.pos_cards.items():
            card.set_selected(key == pos)
        self._show_live_selection()

    def _refresh_kpi(self):
        rows = getattr(self, "live_rows", [])
        self.kpi["PICTURES"].set_value(len(rows))
        self.kpi["PANELS"].set_value(len({r["sn"] for r in rows if r["sn"]}))
        self.kpi["PASS"].set_value(sum(1 for r in rows if r["cls"] == "GOOD"))
        self.kpi["NG"].set_value(sum(1 for r in rows if r["cls"] == "NG"))
        self.kpi["UNJUDGED"].set_value(sum(1 for r in rows if not r["cls"]))
        widths = [r["width_mm"] for r in rows if r["width_mm"] is not None]
        self.kpi["MEAN mm"].set_value(f"{sum(widths)/len(widths):.3f}" if widths else "-")

    def _toggle_live(self):
        if self.chk_live.isChecked():
            # keep `seen_live` - clearing it would re-add every picture already
            # in the feed as if it were new
            self.live_timer.start(self.spin_poll.value() * 1000)
            self.status("Watching the picture folder for new images.")
            self._live_poll()
        else:
            self.live_timer.stop()
            self.status("Stopped watching.", theme.MUTED)

    def _runs_for(self, day: str):
        """Runs for one day, cached, so the live feed can name the panel.

        This deliberately reads the result CSVs directly rather than relying on
        the picture index: the live view has to work the moment the app opens,
        before anyone has pressed 'Build picture index'.
        """
        if day not in self._run_cache:
            try:
                self._run_cache[day] = load_runs(self.cfg.result_dir, day)
            except OSError:
                self._run_cache[day] = []
        return self._run_cache[day]

    def _locate_picture(self, path: Path):
        """Find which panel a picture belongs to, and its position within it.

        Matching is by timestamp against each run's [start, end] window, the
        same rule the batch join uses. The position is how many pictures of that
        run precede this one, counted from the folder itself.
        """
        try:
            when = datetime.strptime(path.stem, PICTURE_STAMP)
        except ValueError:
            return None, -1
        day = path.stem[:8]
        for run in self._runs_for(day):
            if run.start <= when <= run.end:
                if not run.pictures:
                    self._attach_day_pictures(day)
                for i, p in enumerate(run.pictures):
                    if Path(p).name == path.name:
                        return run, i
                return run, -1
        return None, -1

    def _attach_day_pictures(self, day: str) -> None:
        """Fill in each run's picture list for one day, straight from the folder."""
        runs = self._run_cache.get(day)
        if not runs:
            return
        folder = Path(self.cfg.picture_dir)
        try:
            paths = sorted(str(p) for p in folder.glob(f"{day}_*.bmp"))
        except OSError:
            return
        for run in runs:
            run.pictures.clear()
        attach_pictures(runs, paths)

    def _live_poll(self):
        cfg = self._collect()
        folder = Path(cfg.picture_dir)
        if not folder.exists():
            self.status(f"Picture folder not found: {folder}", theme.ALERT_RED)
            return
        try:
            newest = sorted(folder.glob("*.bmp"))[-12:]
        except OSError as exc:
            self.status(str(exc), theme.ALERT_RED)
            return
        fresh = [p for p in newest if str(p) not in self.seen_live]
        if not fresh:
            self.header.set_status("Watching", theme.ONLINE_GREEN,
                                   f"no new images  -  checked {time.strftime('%H:%M:%S')}")
            return

        if not hasattr(self, "live_rows"):
            self.live_rows = []

        newest_pos = None
        for path in fresh:
            self.seen_live.add(str(path))
            meas = _measure(cfg, str(path))
            run, pos = self._locate_picture(path)

            dev_w = dev_e = None
            rough = meas.rough_side
            side = meas.ref_side
            if run is not None and meas.valid:
                baseline = self.store.get(run.key)
                if baseline:
                    ib = baseline.indices.get(str(pos))
                    if ib:
                        # A scalloped side is dropped: no width to compare, only
                        # how far the straight edge has moved.
                        rough = rough or ib.rough_side
                        side = "right" if rough == "left" else "left"
                        edge_px = meas.slot_right if side == "right" else meas.slot_left
                        if not rough:
                            dev_w = cfg.px_to_mm(meas.width - ib.width_mean)
                        dev_e = cfg.px_to_mm(edge_px - ib.edge(side))

            cls, conf = "", 0.0
            classifier = self._classifier_for(run.key if run else "")
            if classifier is not None:
                try:
                    cls, conf = classifier.predict_one(str(path))
                except Exception:
                    cls, conf = "error", 0.0

            record = {
                "path": str(path), "name": path.name,
                "time": path.stem.split("_")[-1],
                "sn": run.sn if run else "", "product": run.product_id if run else "",
                "table": run.table.replace("Table", "") if run else "",
                "pos": pos, "run": run,
                "width_mm": cfg.px_to_mm(meas.width) if meas.width_usable else None,
                "dev_w": dev_w, "dev_e": dev_e, "cls": cls, "conf": conf,
                "rough": rough, "side": side, "meas": meas,
                "has_model": classifier is not None,
            }
            self.live_rows.append(record)
            newest_pos = pos

        self._refresh_kpi()
        shown = self._refresh_pos_cards()
        # Follow the newest position unless the operator has picked one that is
        # still on screen, so the detail pane keeps up with production.
        if newest_pos is not None and (self.selected_pos not in self.live_by_pos):
            self._select_pos(newest_pos)
        elif self.selected_pos in self.live_by_pos:
            self._show_live_selection()
        self.status(f"{len(fresh)} new image(s)", theme.ONLINE_GREEN,
                    f"last update {time.strftime('%H:%M:%S')}  -  "
                    f"{len(self.live_rows)} in feed  -  {shown} position(s)")

    # ---- the overlay switch ----------------------------------------------
    def _picture_pixmap(self, path: str, meas=None):
        """The picture as it should appear: measured, or plain if the switch is off."""
        if not self.cfg.show_overlay:
            return QPixmap(path)
        if meas is None:
            meas = _measure(self.cfg, path)
        return bgr_to_pixmap(render_overlay(path, meas))

    def _toggle_overlay(self, on: bool):
        """One setting, two switches: whichever is clicked, both tabs follow."""
        self.cfg.show_overlay = bool(on)
        for box in (self.chk_overlay_dash, self.chk_overlay_lbl):
            if box.isChecked() != bool(on):
                box.blockSignals(True)
                box.setChecked(bool(on))
                box.blockSignals(False)
        if self._current_live_record() is not None:
            self._show_live_selection()
        if self.lbl_gallery.currentItem() is not None:
            self._show_label_selection()
        # the gallery bakes the overlay into its thumbnails, so they are redrawn
        if self.label_paths:
            self._start_label_thumbnails()
        self.status(f"Overlay {'on' if on else 'off'}.")

    def _current_live_record(self):
        return self.live_by_pos.get(self.selected_pos)

    def _show_live_selection(self):
        record = self._current_live_record()
        if record is None:
            return
        cfg = self.cfg
        meas = record["meas"]
        run, pos = record["run"], record["pos"]
        baseline = self.store.get(run.key) if run else None
        ib = baseline.indices.get(str(pos)) if baseline else None

        pix = self._picture_pixmap(record["path"], meas)
        self.live_image.setPixmap(pix.scaled(self.live_image.width() - 10,
                                             self.live_image.height() - 10,
                                             Qt.KeepAspectRatio, Qt.SmoothTransformation))

        if record["cls"]:
            colour = theme.ONLINE_GREEN if record["cls"] == "GOOD" else theme.ALERT_RED
            self.live_verdict.setText(f"{record['cls']}   {record['conf']:.0%}")
            self.live_verdict.setStyleSheet(
                f"color:{colour}; border:none; background:transparent;")
        else:
            self.live_verdict.setText(
                f"no model for {run.key}" if run else "no model trained")
            self.live_verdict.setStyleSheet(
                f"color:{theme.MUTED}; border:none; background:transparent;")

        lines = [f"image      {record['name']}"]
        if run is not None:
            lines += [
                f"panel      SN {run.sn}   {run.product_id}   {run.table}"
                f"   {'PASS' if run.passed else 'FAIL'}",
                f"recipe     {run.recipe}",
                f"position   {pos} of {len(run.pictures)}   "
                f"{self.labels.label(run.key, pos)}",
                f"panel run  {run.start:%Y-%m-%d %H:%M:%S} -> {run.end:%H:%M:%S}"
                f"   tact {run.tact_time}s   sub-boards {run.sub_boards}",
                f"alignment  X {run.offset_x:+.4f} mm   Y {run.offset_y:+.4f} mm"
                f"   rotate {run.rotate_angle:+.5f}",
            ]
            if run.message:
                lines.append(f"message    {run.message}")
        else:
            lines.append("panel      not matched to a run result")

        rough, side = record.get("rough", ""), record.get("side", "left")
        if meas.valid:
            if rough:
                lines.append(f"slot       {side} edge {meas.ref_edge} px   "
                             f"width not measured")
            else:
                lines.append(f"slot       width {meas.width} px = "
                             f"{cfg.px_to_mm(meas.width):.4f} mm   "
                             f"left {meas.slot_left}  right {meas.slot_right}")
            lines.append(f"scan       {meas.rows}/{meas.attempted} rows   spread "
                         f"{cfg.px_to_mm(meas.width_sd):.4f} mm")
            lines.append(f"edges      left wanders {cfg.px_to_mm(meas.left_spread):.4f} mm"
                         f"   right {cfg.px_to_mm(meas.right_spread):.4f} mm")
            if rough:
                lines.append(f"dropped    {rough} edge is not straight - it is a tab or "
                             f"contour, so only the {side} edge is judged")
        else:
            lines.append(f"slot       not measurable "
                         f"({meas.rows}/{meas.attempted} rows found edges)")
        if ib and not rough:
            lines.append(f"baseline   width {ib.width_mean:.2f} px "
                         f"({cfg.px_to_mm(ib.width_mean):.4f} mm)   sd "
                         f"{cfg.px_to_mm(ib.width_sd):.4f} mm   n={ib.samples}")
        elif ib:
            lines.append(f"baseline   {side} edge {ib.edge(side):.2f} px   sd "
                         f"{cfg.px_to_mm(ib.edge_sd(side)):.4f} mm   n={ib.samples}")
        if record["dev_e"] is None:
            lines.append("deviation  no baseline for this position - calibrate first")
        elif rough:
            verdict = ("within tolerance" if abs(record["dev_e"]) <= cfg.edge_tol_mm
                       else "OUT OF TOLERANCE")
            lines.append(f"deviation  shift {record['dev_e']:+.4f} mm "
                         f"({side} edge)   -> {verdict}")
        else:
            verdict = ("within tolerance"
                       if abs(record["dev_w"]) <= cfg.width_tol_mm
                       and abs(record["dev_e"]) <= cfg.edge_tol_mm else "OUT OF TOLERANCE")
            lines.append(f"deviation  width {record['dev_w']:+.4f} mm   "
                         f"shift {record['dev_e']:+.4f} mm   -> {verdict}")
        lines.append(f"label      {self.label_store.cls_of(record['path']) or '-'}")
        self.live_info.setText("\n".join(lines))

    def _open_live_image(self, pos=None):
        if isinstance(pos, int) and pos in self.live_by_pos:
            self._select_pos(pos)
        record = self._current_live_record()
        if record is None:
            return
        run, pos = record["run"], record["pos"]
        baseline = self.store.get(run.key) if run else None
        ib = baseline.indices.get(str(pos)) if baseline else None
        caption = self.labels.label(run.key, pos) if run else ""
        pred = (record["cls"], record["conf"]) if record["cls"] else None
        ImageViewer(self.cfg, record["path"], ib, caption, pred, self).exec()

    def _model_dir(self) -> Path:
        """The folder from the config, or the app's own when that is left empty."""
        return Path(self.cfg.model_dir) if self.cfg.model_dir.strip() else MODEL_DIR

    def _model_path(self, key: str) -> Path:
        return self._model_dir() / model_file_name(key)

    def _model_files(self) -> list[str]:
        folder = self._model_dir()
        if not folder.exists():
            return []
        return sorted(p.stem[len(MODEL_PREFIX):] for p in folder.glob(f"{MODEL_PREFIX}*.pt"))

    def _classifier_for(self, key: str):
        """The model trained for this ProductId|table, or None if there is none.

        A model from another product or the other table is never substituted:
        it learned that combination's cuts and would be guessing here.
        """
        if not key or key.startswith("|"):
            return None
        if key not in self.classifiers:
            path = self._model_path(key)
            try:
                self.classifiers[key] = (CutClassifier.load(path)
                                         if path.exists() else None)
            except Exception:
                self.classifiers[key] = None
        return self.classifiers[key]

    def _refresh_model_label(self):
        trained = self._model_files()
        if not trained:
            self.lbl_model.setText("models: none - label and train per product/table")
        else:
            self.lbl_model.setText(
                f"models: {len(trained)} - {', '.join(trained[:2])}"
                + (" ..." if len(trained) > 2 else ""))

    # ======================================================================
    # Inspect
    # ======================================================================
    def _build_inspect_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        top = QHBoxLayout()
        self.cmb_ins_day = QComboBox()
        self.cmb_ins_day.setMinimumWidth(130)
        btn_days = QPushButton("Refresh days")
        btn_days.clicked.connect(lambda: self._refresh_days(self.cmb_ins_day))
        self.btn_inspect = QPushButton("Inspect day")
        self.btn_inspect.clicked.connect(self._run_inspection)
        self.chk_flagged = QCheckBox("Flagged only")
        self.chk_flagged.stateChanged.connect(self._fill_run_table)
        btn_names = QPushButton("Name positions...")
        btn_names.clicked.connect(self._name_positions)
        btn_export = QPushButton("Export CSV")
        btn_export.clicked.connect(self._export)
        for w in (QLabel("Day"), self.cmb_ins_day, btn_days, self.btn_inspect,
                  self.chk_flagged, btn_names, btn_export):
            top.addWidget(w)
        top.addStretch(1)
        outer.addLayout(top)

        split = QSplitter(Qt.Horizontal)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Panels produced (click one)"))
        self.tbl_runs = QTableWidget(0, 6)
        self.tbl_runs.setHorizontalHeaderLabels(["SN", "Time", "Product", "Table", "Result", "Status"])
        self.tbl_runs.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_runs.setSelectionBehavior(QTableWidget.SelectRows)
        self.tbl_runs.verticalHeader().setVisible(False)
        self.tbl_runs.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tbl_runs.itemSelectionChanged.connect(self._show_selected_run)
        ll.addWidget(self.tbl_runs, 1)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.lbl_panel = QLabel("Select a panel to see its inspection pictures.")
        self.lbl_panel.setFont(QFont("Consolas", 10))
        self.lbl_panel.setStyleSheet("background:#f5f5f5; border:1px solid #ddd;"
                                     "border-radius:4px; padding:8px;")
        self.lbl_panel.setWordWrap(True)
        rl.addWidget(self.lbl_panel)
        self.gallery = QListWidget()
        self.gallery.setViewMode(QListWidget.IconMode)
        self.gallery.setIconSize(THUMB)
        self.gallery.setGridSize(QSize(THUMB.width() + 30, THUMB.height() + 74))
        self.gallery.setResizeMode(QListWidget.Adjust)
        self.gallery.setMovement(QListWidget.Static)
        self.gallery.setWordWrap(True)
        self.gallery.setSpacing(6)
        self.gallery.itemDoubleClicked.connect(self._open_thumb)
        rl.addWidget(self.gallery, 1)
        rl.addWidget(QLabel("Double-click a picture to open it full size."))
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 7)
        outer.addWidget(split, 1)
        return page

    def _run_inspection(self):
        day = self.cmb_ins_day.currentText()
        if not day or not self._ensure_index():
            self.status("Pick a day first.")
            return
        cfg = self._collect()
        self._busy(True)

        def job(worker):
            runs = load_day(cfg.result_dir, day, self.picture_index)[: cfg.max_runs_per_scan]
            return inspect(cfg, runs, self.store,
                           progress=lambda d, t: worker.progress.emit(d, t, "Inspecting panel"),
                           cancelled=worker.is_cancelled)

        def finished(results):
            self.results = results
            self._fill_run_table()
            counts = {}
            for r in results:
                counts[r.status] = counts.get(r.status, 0) + 1
            self._busy(False)
            self.status(f"{day}: {len(results)} panels - "
                        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
            if self.tbl_runs.rowCount():
                self.tbl_runs.selectRow(0)

        self._run_worker(job, finished)

    def _fill_run_table(self):
        rows = self.results
        if self.chk_flagged.isChecked():
            rows = [r for r in rows if r.status != "OK"]
        self.visible_results = rows
        self.tbl_runs.setRowCount(0)
        for r in rows:
            row = self.tbl_runs.rowCount()
            self.tbl_runs.insertRow(row)
            values = [r.run.sn or "-", f"{r.run.start:%H:%M:%S}", r.run.product_id,
                      r.run.table.replace("Table", ""),
                      "PASS" if r.run.passed else "FAIL", r.status]
            colour = STATUS_COLORS.get(r.status)
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setToolTip(r.note or r.status)
                if colour:
                    item.setBackground(colour)
                self.tbl_runs.setItem(row, col, item)
        self.gallery.clear()

    def _show_selected_run(self):
        row = self.tbl_runs.currentRow()
        if row < 0 or row >= len(self.visible_results):
            return
        result = self.visible_results[row]
        self.current = result
        run = result.run
        by_index = {d.index: d for d in result.details}
        self.lbl_panel.setText(
            f"SN {run.sn or '-'}    {run.product_id}    {run.table}    "
            f"{'PASS' if run.passed else 'FAIL'}    -> {result.status}\n"
            f"start {run.start:%Y-%m-%d %H:%M:%S}    tact {run.tact_time}s    "
            f"sub-boards {run.sub_boards}    pictures {len(run.pictures)}\n"
            f"alignment X {run.offset_x:+.4f} mm   Y {run.offset_y:+.4f} mm"
            + (f"\n{result.note}" if result.note else ""))

        self.gallery.clear()
        if self.thumbs:
            self.thumbs.stop()
            self.thumbs.wait(1200)
        placeholder = QPixmap(THUMB)
        placeholder.fill(QColor(250, 250, 250))
        for i, path in enumerate(run.pictures):
            detail = by_index.get(i)
            name = self.labels.label(run.key, i)
            if detail is None or not detail.ok:
                caption, colour = f"{name}\n(no slot visible)", COLOR_GREY
            else:
                bad = ((detail.width_judged
                        and abs(detail.width_dev_mm) > self.cfg.width_tol_mm)
                       or abs(detail.edge_dev_mm) > self.cfg.edge_tol_mm)
                head = (f"width {detail.width_dev_mm:+.3f} mm" if detail.width_judged
                        else f"{detail.rough_side} edge dropped")
                caption = (f"{name}\n{head}"
                           f"{'  NG' if bad else ''}\nshift {detail.edge_dev_mm:+.3f} mm")
                colour = COLOR_NG if bad else COLOR_OK
            item = QListWidgetItem(QIcon(placeholder), caption)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setBackground(colour)
            item.setData(Qt.UserRole, (i, path))
            self.gallery.addItem(item)
        self.thumbs = ThumbnailLoader(list(run.pictures))
        self.thumbs.ready.connect(
            lambda i, pm: self.gallery.item(i).setIcon(QIcon(pm))
            if i < self.gallery.count() else None)
        self.thumbs.start()

    def _open_thumb(self, item):
        index, path = item.data(Qt.UserRole)
        baseline = self.store.get(self.current.run.key) if self.current else None
        bl_idx = baseline.indices.get(str(index)) if baseline else None
        pred = None
        classifier = self._classifier_for(self.current.run.key if self.current else "")
        if classifier is not None:
            try:
                pred = classifier.predict_one(path)
            except Exception:
                pred = None
        ImageViewer(self.cfg, path, bl_idx,
                    self.labels.label(self.current.run.key, index), pred, self).exec()

    def _name_positions(self):
        if not self.current:
            self.status("Select a panel first.")
            return
        QMessageBox.information(
            self, "Name positions",
            "Position names are edited on the Label tab, where you can see each "
            "picture while you name it.")

    def _export(self):
        if not self.results:
            self.status("Run an inspection first.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export", "cut_inspection.csv", "CSV (*.csv)")
        if not path:
            return
        try:
            check_write_target(path, protected_roots(self.cfg))
        except ProtectedPathError as exc:
            QMessageBox.critical(self, "Write refused", str(exc))
            return
        with open(path, "w", newline="", encoding="utf-8-sig") as fh:
            wr = csv.writer(fh)
            wr.writerow(["SN", "Start", "Product", "Recipe", "Table", "Result",
                         "OffsetX", "OffsetY", "PanelStatus", "Note",
                         "Position", "PositionName", "WidthDevMm", "EdgeDevMm", "Label"])
            for r in self.results:
                if not r.details:
                    wr.writerow([r.run.sn, f"{r.run.start:%Y-%m-%d %H:%M:%S}", r.run.product_id,
                                 r.run.recipe, r.run.table, "PASS" if r.run.passed else "FAIL",
                                 f"{r.run.offset_x:.6f}", f"{r.run.offset_y:.6f}",
                                 r.status, r.note, "", "", "", "", ""])
                for d in r.details:
                    wr.writerow([r.run.sn, f"{r.run.start:%Y-%m-%d %H:%M:%S}", r.run.product_id,
                                 r.run.recipe, r.run.table, "PASS" if r.run.passed else "FAIL",
                                 f"{r.run.offset_x:.6f}", f"{r.run.offset_y:.6f}",
                                 r.status, r.note, d.index,
                                 self.labels.label(r.run.key, d.index),
                                 d.width_dev_mm if d.ok else "", d.edge_dev_mm if d.ok else "",
                                 self.label_store.cls_of(d.path)])
        self.status(f"Exported to {path}")

    # ======================================================================
    # Cut path
    # ======================================================================
    def _build_toolpath_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        top = QHBoxLayout()
        self.cmb_recipe = QComboBox()
        self.cmb_recipe.setMinimumWidth(320)
        btn_scan = QPushButton("List recipes")
        btn_scan.clicked.connect(self._scan_recipes)
        btn_show = QPushButton("Show cutting path")
        btn_show.clicked.connect(self._show_toolpath)
        for w in (QLabel("Recipe"), self.cmb_recipe, btn_scan, btn_show):
            top.addWidget(w)
        top.addStretch(1)
        outer.addLayout(top)

        self.toolpath_view = ToolpathView()
        outer.addWidget(self.toolpath_view, 1)

        self.tbl_slots = QTableWidget(0, 5)
        self.tbl_slots.setHorizontalHeaderLabels(
            ["#", "Centre X mm", "Centre Y mm", "Width mm", "Length mm"])
        self.tbl_slots.setEditTriggers(QTableWidget.NoEditTriggers)
        self.tbl_slots.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_slots.setMaximumHeight(220)
        outer.addWidget(self.tbl_slots)
        return page

    def _scan_recipes(self):
        cfg = self._collect()
        root = Path(cfg.recipe_dir)
        if not root.exists():
            QMessageBox.warning(self, "Recipes", f"Not found: {root}")
            return
        self.cmb_recipe.clear()
        for p in sorted(root.rglob("*.rcp")):
            self.cmb_recipe.addItem(f"{p.parent.name}/{p.name}", str(p))
        self.status(f"{self.cmb_recipe.count()} recipes found.")

    def _show_toolpath(self):
        path = self.cmb_recipe.currentData()
        if not path:
            self.status("Pick a recipe first.")
            return
        try:
            tp = load_toolpath(path)
        except OSError as exc:
            QMessageBox.warning(self, "Cut path", str(exc))
            return
        self.toolpath_view.set_toolpath(tp)
        self.tbl_slots.setRowCount(0)
        for i, seg in enumerate(tp.slots, start=1):
            cx, cy = seg.centre
            w, h = seg.size_mm
            row = self.tbl_slots.rowCount()
            self.tbl_slots.insertRow(row)
            for col, value in enumerate([str(i), f"{cx:.2f}", f"{cy:.2f}",
                                         f"{w:.3f}", f"{h:.3f}"]):
                self.tbl_slots.setItem(row, col, QTableWidgetItem(value))
        self.status(tp.summary())

    # ======================================================================
    # Label
    # ======================================================================
    def _build_label_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)
        top = QHBoxLayout()
        self.cmb_lbl_day = DayPicker()
        btn_days = QPushButton("Refresh days")
        btn_days.clicked.connect(lambda: self._refresh_days(self.cmb_lbl_day))
        self.btn_label_load = QPushButton("Load pictures")
        self.btn_label_load.clicked.connect(self._load_label_pictures)
        self.spin_lbl_max = QSpinBox()
        # 0 shows as "all": the whole day, however many pictures that is.
        self.spin_lbl_max.setRange(0, 20000)
        self.spin_lbl_max.setSingleStep(20)
        self.spin_lbl_max.setSpecialValueText("all")
        self.spin_lbl_max.setValue(300)
        self.spin_lbl_max.setToolTip("How many of the day's pictures to load. "
                                     "Step below 20 to reach 'all'.")
        self.lbl_counts = QLabel("")
        self.lbl_counts.setStyleSheet("color:#555;")
        self.chk_overlay_lbl = QCheckBox("Overlay")
        self.chk_overlay_lbl.setToolTip(
            "Draw the measured edges on the picture and on every thumbnail. "
            "Turn it off to judge the bare cut.")
        self.chk_overlay_lbl.setChecked(self.cfg.show_overlay)
        self.chk_overlay_lbl.toggled.connect(self._toggle_overlay)
        for w in (QLabel("Day"), self.cmb_lbl_day, btn_days,
                  QLabel("max"), self.spin_lbl_max, self.btn_label_load,
                  self.chk_overlay_lbl, self.lbl_counts):
            top.addWidget(w)
        top.addStretch(1)
        outer.addLayout(top)

        # Three panes: the thumbnails to pick from, the picture at a size you can
        # actually judge, and the buttons. Clicking a thumbnail is enough - no
        # double click, because judging every picture through a dialog is slow.
        split = QSplitter(Qt.Horizontal)

        self.lbl_gallery = QListWidget()
        self.lbl_gallery.setViewMode(QListWidget.IconMode)
        self.lbl_gallery.setIconSize(THUMB)
        self.lbl_gallery.setGridSize(QSize(THUMB.width() + 26, THUMB.height() + 56))
        self.lbl_gallery.setResizeMode(QListWidget.Adjust)
        self.lbl_gallery.setMovement(QListWidget.Static)
        self.lbl_gallery.setSelectionMode(QListWidget.ExtendedSelection)
        self.lbl_gallery.setSpacing(5)
        # room for two columns of thumbnails, or picking through a day is painful
        self.lbl_gallery.setMinimumWidth(2 * (THUMB.width() + 26) + 24)
        self.lbl_gallery.itemSelectionChanged.connect(self._show_label_selection)
        self.lbl_gallery.itemDoubleClicked.connect(self._open_label_image)
        split.addWidget(self.lbl_gallery)

        middle = QWidget()
        ml = QVBoxLayout(middle)
        ml.setContentsMargins(0, 0, 0, 0)
        self.lbl_preview = QLabel("Load a day, then click a picture.")
        self.lbl_preview.setAlignment(Qt.AlignCenter)
        self.lbl_preview.setMinimumSize(460, 340)
        self.lbl_preview.setStyleSheet(
            f"background:{theme.PANEL_2}; border:1px solid {theme.BORDER}; border-radius:3px;")
        ml.addWidget(self.lbl_preview, 1)
        self.legend_lbl = OverlayLegend()
        ml.addWidget(self.legend_lbl)
        self.lbl_preview_verdict = QLabel("")
        self.lbl_preview_verdict.setAlignment(Qt.AlignCenter)
        self.lbl_preview_verdict.setFont(QFont("Segoe UI Semibold", 14))
        self.lbl_preview_verdict.setStyleSheet("border:none; background:transparent;")
        ml.addWidget(self.lbl_preview_verdict)
        self.lbl_preview_info = QLabel("")
        self.lbl_preview_info.setFont(QFont("Consolas", 9))
        self.lbl_preview_info.setStyleSheet(
            f"background:{theme.PANEL_2}; border:1px solid {theme.BORDER};"
            "border-radius:3px; padding:8px;")
        self.lbl_preview_info.setWordWrap(True)
        ml.addWidget(self.lbl_preview_info)
        split.addWidget(middle)

        side = QWidget()
        sl = QVBoxLayout(side)
        note = QLabel("Click a picture to see it full size, then click GOOD or NG.\n"
                      "Ctrl or Shift selects several at once and labels them together.\n"
                      "If a picture cannot be judged, leave it unlabelled - "
                      "unlabelled images are simply not trained on.")
        note.setWordWrap(True)
        sl.addWidget(note)
        self.class_buttons = []
        for cls in DEFAULT_CLASSES:
            b = QPushButton(cls)
            b.clicked.connect(lambda _=False, c=cls: self._apply_label(c))
            colour = CLASS_COLORS.get(cls)
            if colour:
                b.setStyleSheet(f"background:{colour.name()}; padding:14px;"
                                "font-weight:bold; font-size:15px;")
            sl.addWidget(b)
            self.class_buttons.append(b)
        btn_clear = QPushButton("Clear label")
        btn_clear.clicked.connect(lambda: self._apply_label(None))
        sl.addWidget(btn_clear)
        sl.addStretch(1)
        self.lbl_summary = QTextEdit()
        self.lbl_summary.setReadOnly(True)
        self.lbl_summary.setFont(QFont("Consolas", 9))
        self.lbl_summary.setMaximumHeight(200)
        sl.addWidget(self.lbl_summary)
        side.setMaximumWidth(300)
        split.addWidget(side)

        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 7)
        split.setStretchFactor(2, 0)
        # size hints alone leave the gallery one column wide, so say it outright
        split.setSizes([520, 660, 280])
        outer.addWidget(split, 1)
        self._refresh_label_summary()
        return page

    def _show_label_selection(self):
        """Put the picture the operator just clicked on screen, measured."""
        item = self.lbl_gallery.currentItem()
        if item is None:
            return
        path = item.data(Qt.UserRole)
        cfg = self.cfg
        meas = _measure(cfg, path)
        pix = self._picture_pixmap(path, meas)
        self.lbl_preview.setPixmap(pix.scaled(self.lbl_preview.width() - 10,
                                              self.lbl_preview.height() - 10,
                                              Qt.KeepAspectRatio, Qt.SmoothTransformation))

        cls = self.label_store.cls_of(path)
        chosen = len(self.lbl_gallery.selectedItems())
        colour = (theme.ONLINE_GREEN if cls == "GOOD"
                  else theme.ALERT_RED if cls == "NG" else theme.MUTED)
        self.lbl_preview_verdict.setText(
            f"label: {cls or 'not labelled'}"
            + (f"    ({chosen} pictures selected)" if chosen > 1 else ""))
        self.lbl_preview_verdict.setStyleSheet(
            f"color:{colour}; border:none; background:transparent;")

        lines = [f"image      {Path(path).name}"]
        if meas.valid:
            if meas.rough_side:
                lines.append(f"slot       {meas.ref_side} edge {meas.ref_edge} px   "
                             f"({meas.rough_side} edge is a tab or contour, not measured)")
            else:
                lines.append(f"slot       width {meas.width} px = "
                             f"{cfg.px_to_mm(meas.width):.4f} mm   "
                             f"left {meas.slot_left}  right {meas.slot_right}")
            lines.append(f"scan       {meas.rows}/{meas.attempted} rows   spread "
                         f"{cfg.px_to_mm(meas.width_sd):.4f} mm")
            lines.append(f"edges      left wanders {cfg.px_to_mm(meas.left_spread):.4f} mm"
                         f"   right {cfg.px_to_mm(meas.right_spread):.4f} mm")
        else:
            lines.append(f"slot       not measurable "
                         f"({meas.rows}/{meas.attempted} rows found edges)")
        classifier = self._classifier_for(self._key_of(path))
        if classifier is not None:
            try:
                predicted, conf = classifier.predict_one(path)
                lines.append(f"model      says {predicted} ({conf:.0%}) - your label wins")
            except Exception:
                pass
        self.lbl_preview_info.setText("\n".join(lines))

    def _load_label_pictures(self):
        day = self.cmb_lbl_day.currentText()
        if not day or not self._ensure_index():
            self.status("Pick a day first.")
            return
        cfg = self._collect()
        runs = load_day(cfg.result_dir, day, self.picture_index)
        every = [p for r in runs for p in r.pictures]
        limit = self.spin_lbl_max.value()
        paths = every if limit == 0 else every[:limit]
        self.label_runs = {p: r for r in runs for p in r.pictures}
        self.label_paths = paths
        self.lbl_gallery.clear()
        self.lbl_preview.setText("Click a picture to see it full size.")
        self.lbl_preview_verdict.setText("")
        self.lbl_preview_info.setText("")
        placeholder = QPixmap(THUMB)
        placeholder.fill(QColor(250, 250, 250))
        for path in paths:
            cls = self.label_store.cls_of(path)
            item = QListWidgetItem(QIcon(placeholder), f"{Path(path).stem}\n{cls or '-'}")
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setData(Qt.UserRole, path)
            if cls and cls in CLASS_COLORS:
                item.setBackground(CLASS_COLORS[cls])
            self.lbl_gallery.addItem(item)
        self._start_label_thumbnails()
        self.status(f"{len(paths)} pictures loaded for labelling.")

    def _start_label_thumbnails(self):
        """(Re)draw the gallery thumbnails, with or without the overlay."""
        if self.thumbs:
            self.thumbs.stop()
            self.thumbs.wait(1200)
        self.thumbs = ThumbnailLoader(
            self.label_paths, self.cfg if self.cfg.show_overlay else None)
        self.thumbs.ready.connect(
            lambda i, pm: self.lbl_gallery.item(i).setIcon(QIcon(pm))
            if i < self.lbl_gallery.count() else None)
        self.thumbs.start()

    def _apply_label(self, cls):
        items = self.lbl_gallery.selectedItems()
        if not items:
            self.status("Select pictures first.")
            return
        for item in items:
            path = item.data(Qt.UserRole)
            if cls is None:
                self.label_store.unset(path)
                item.setBackground(QBrush())
                item.setText(f"{Path(path).stem}\n-")
            else:
                run = self.label_runs.get(path)
                self.label_store.set(
                    path, cls,
                    product=run.product_id if run else "",
                    recipe=run.recipe if run else "",
                    table=run.table if run else "",
                    sn=run.sn if run else "")
                item.setBackground(CLASS_COLORS.get(cls, COLOR_GREY))
                item.setText(f"{Path(path).stem}\n{cls}")
        try:
            self.label_store.save()
        except ProtectedPathError as exc:
            QMessageBox.critical(self, "Write refused", str(exc))
            return
        self._refresh_label_summary()
        self._show_label_selection()      # the preview shows the label it just got
        self.status(f"{len(items)} picture(s) labelled {cls or 'cleared'}.")

    def _key_of(self, path: str) -> str:
        """ProductId|table of a picture: from its run, or from its stored label."""
        run = self.label_runs.get(path)
        if run is not None:
            return run.key
        label = self.label_store.get(path)
        return self.label_store.key_of(label) if label else ""

    def _open_label_image(self, item):
        path = item.data(Qt.UserRole)
        pred = None
        classifier = self._classifier_for(self._key_of(path))
        if classifier is not None:
            try:
                pred = classifier.predict_one(path)
            except Exception:
                pred = None
        ImageViewer(self.cfg, path, None, self.label_store.cls_of(path), pred, self).exec()

    def _refresh_label_summary(self):
        counts = self.label_store.counts()
        total = sum(counts.values())
        ok, msg = self.label_store.usable_for_training()
        lines = [f"labelled: {total}", ""]
        for cls in DEFAULT_CLASSES:
            lines.append(f"  {cls:<11}{counts.get(cls, 0):>5}")
        lines += ["", "training: " + ("READY" if ok else "not yet"), msg]
        self.lbl_summary.setPlainText("\n".join(lines))
        self.lbl_counts.setText(f"labelled {total}")

    # ======================================================================
    # Train
    # ======================================================================
    def _build_train_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)

        box = QGroupBox("DINOv2 classifier")
        grid = QGridLayout(box)
        self.cmb_backbone = QComboBox()
        self.cmb_backbone.addItems(["dinov2_vits14", "dinov2_vitb14"])
        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(50, 5000)
        self.spin_epochs.setValue(300)
        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.1, 0.5)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.25)
        self.crop_spins = {}
        for i, (key, default) in enumerate([("x0", 400), ("y0", 290), ("x1", 1040), ("y1", 930)]):
            s = QSpinBox()
            s.setRange(0, 4000)
            s.setValue(default)
            self.crop_spins[key] = s
            grid.addWidget(QLabel(f"crop {key}"), 1, i * 2)
            grid.addWidget(s, 1, i * 2 + 1)
        grid.addWidget(QLabel("backbone"), 0, 0)
        grid.addWidget(self.cmb_backbone, 0, 1)
        grid.addWidget(QLabel("epochs"), 0, 2)
        grid.addWidget(self.spin_epochs, 0, 3)
        grid.addWidget(QLabel("validation split"), 0, 4)
        grid.addWidget(self.spin_val, 0, 5)
        grid.addWidget(QLabel(f"device: {pick_device()}"), 0, 6)
        outer.addWidget(box)

        pick = QHBoxLayout()
        lbl_pick = QLabel("Product / table")
        lbl_pick.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.cmb_train_product = QComboBox()
        self.cmb_train_product.setMinimumWidth(300)
        self.cmb_train_product.currentIndexChanged.connect(self._refresh_train_product)
        btn_products = QPushButton("Refresh list")
        btn_products.clicked.connect(self._reload_train_products)
        self.lbl_train_counts = QLabel("")
        self.lbl_train_counts.setStyleSheet(f"color:{theme.MUTED};")
        for w in (lbl_pick, self.cmb_train_product, btn_products, self.lbl_train_counts):
            pick.addWidget(w)
        pick.addStretch(1)
        outer.addLayout(pick)

        row = QHBoxLayout()
        self.btn_train = QPushButton("Train on labelled pictures")
        self.btn_train.clicked.connect(self._train_model)
        btn_save = QPushButton("Save model")
        btn_save.clicked.connect(self._save_model)
        btn_load = QPushButton("Load model")
        btn_load.clicked.connect(self._load_model)
        for b in (self.btn_train, btn_save, btn_load):
            row.addWidget(b)
        row.addStretch(1)
        outer.addLayout(row)

        self.txt_train = QTextEdit()
        self.txt_train.setReadOnly(True)
        self.txt_train.setFont(QFont("Consolas", 9))
        self.txt_train.setPlainText(
            "The backbone stays frozen; only a small linear head is trained on top of its\n"
            "embeddings. That is the right size of model for a few dozen labelled images -\n"
            "fine-tuning the whole network would just memorise them.\n\n"
            "Label pictures on the Label tab first, then train here. Saving the model makes\n"
            "the Dashboard tab classify every new image as it arrives.\n\n"
            "Training a product/table that already has a model carries on from that model\n"
            "rather than starting over, so a round of extra labels refines what is there.\n"
            "A different backbone or a different set of classes cannot be carried on and\n"
            "starts from a fresh head instead - the training report says which happened.")
        outer.addWidget(self.txt_train, 1)
        self._reload_train_products()
        return page

    def _reload_train_products(self):
        """Fill the picker from the ProductId|table combinations that have labels."""
        current = self.cmb_train_product.currentText()
        keys = self.label_store.keys()
        self.cmb_train_product.blockSignals(True)
        self.cmb_train_product.clear()
        self.cmb_train_product.addItems(keys)
        if current in keys:
            self.cmb_train_product.setCurrentText(current)
        self.cmb_train_product.blockSignals(False)
        self._refresh_train_product()

    def _refresh_train_product(self):
        """Say how many labels this product/table has, and whether it has a model."""
        key = self.cmb_train_product.currentText()
        if not key:
            self.lbl_train_counts.setText(
                "no labelled pictures yet - label some on the Label tab")
            return
        counts = self.label_store.counts(key)
        detail = "  ".join(f"{c} {counts.get(c, 0)}" for c in DEFAULT_CLASSES)
        model = self._model_path(key)
        self.lbl_train_counts.setText(
            f"{detail}   -   {model.name}: {'trained' if model.exists() else 'not trained yet'}")

    def _train_model(self):
        product = self.cmb_train_product.currentText()
        if not product:
            QMessageBox.information(self, "Pick a product and table",
                                    "Label some pictures first - a model is trained "
                                    "for one ProductId and table at a time.")
            return
        ok, msg = self.label_store.usable_for_training(key=product)
        if not ok:
            QMessageBox.information(self, "Not enough labels", f"{product}: {msg}")
            return
        items = self.label_store.training_items(product)
        crop = CropBox(**{k: s.value() for k, s in self.crop_spins.items()})
        backbone = self.cmb_backbone.currentText()
        epochs = self.spin_epochs.value()
        val = self.spin_val.value()
        self._busy(True)
        self.txt_train.setPlainText("Training...")

        def job(worker):
            clf = CutClassifier(backbone=backbone, crop=crop)
            previous = None
            previous_path = self._model_path(product)
            if previous_path.exists():
                try:
                    previous = CutClassifier.load(previous_path)
                except (OSError, RuntimeError, ValueError):
                    previous = None
            report = clf.train(items, epochs=epochs, val_fraction=val,
                               progress=lambda d, t, m: worker.progress.emit(d, t, m),
                               cancelled=worker.is_cancelled,
                               resume_from=previous)
            return clf, report

        def finished(result):
            clf, report = result
            self.classifiers[product] = clf
            self._refresh_model_label()
            self._refresh_train_product()
            self._busy(False)
            self.txt_train.setPlainText(
                f"Product {product}\n\n"
                + report.text()
                + "\n\nNOTE: accuracy here is measured on a slice of the same labelled set."
                  "\nIt tells you the model learned what you taught it - not that the labels"
                  "\nthemselves were right. Check a few predictions by eye before trusting it."
                  "\n\nPress 'Save model' to use it on the Dashboard tab.")
            self.status(f"Trained {product}: val accuracy {report.val_acc:.1%} "
                        f"in {report.seconds:.1f}s")

        self._run_worker(job, finished)

    def _save_model(self):
        product = self.cmb_train_product.currentText()
        clf = self.classifiers.get(product)
        if clf is None:
            self.status("Train a model for this product first.")
            return
        target = self._model_path(product)
        try:
            clf.save(target, protected=protected_roots(self.cfg))
        except (ProtectedPathError, RuntimeError) as exc:
            QMessageBox.critical(self, "Save model", str(exc))
            return
        self._refresh_train_product()
        self._refresh_model_label()
        self.status(f"Model for {product} saved to {target}")

    def _load_model(self):
        product = self.cmb_train_product.currentText()
        if not product:
            self.status("Pick a product/table first - a model belongs to one of each.")
            return
        path, _ = QFileDialog.getOpenFileName(self, f"Load model for {product}",
                                              str(self._model_path(product)),
                                              "PyTorch model (*.pt)")
        if not path:
            return
        try:
            self.classifiers[product] = CutClassifier.load(Path(path))
        except Exception as exc:
            QMessageBox.critical(self, "Load model", str(exc))
            return
        self._refresh_model_label()
        report = self.classifiers[product].report
        if report:
            self.txt_train.setPlainText(f"Product {product}\n\n" + report.text())
        self.status(f"Loaded model for {product} from {path}")

    # ======================================================================
    # Config
    # ======================================================================
    def _build_config_tab(self):
        page = QWidget()
        outer = QVBoxLayout(page)

        paths = QGroupBox("Data locations (read-only)")
        grid = QGridLayout(paths)
        self.path_edits = {}
        for row, (key, label, is_dir) in enumerate([
                ("picture_dir", "Picture folder", True),
                ("result_dir", "Result folder", True),
                ("recipe_dir", "Recipe folder", True),
                ("eqp_cfg_path", "Eqp.cfg file", False),
                ("model_dir", "Model folder (blank = app folder)", True)]):
            edit = QLineEdit(getattr(self.cfg, key))
            btn = QPushButton("Browse...")
            btn.clicked.connect(lambda _=False, k=key, d=is_dir: self._browse(k, d))
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(btn, row, 2)
            self.path_edits[key] = edit
        grid.setColumnStretch(1, 1)
        outer.addWidget(paths)

        scale = QGroupBox("Scale")
        sl = QHBoxLayout(scale)
        self.spin_pixel = QDoubleSpinBox()
        self.spin_pixel.setDecimals(9)
        self.spin_pixel.setRange(0.000001, 1.0)
        self.spin_pixel.setValue(self.cfg.pixel_size_mm)
        self.spin_pixel.valueChanged.connect(self._update_scale_label)
        btn_px = QPushButton("Load from Eqp.cfg")
        btn_px.clicked.connect(self._load_pixel_size)
        self.lbl_scale = QLabel()
        self.lbl_scale.setStyleSheet("color:#555;")
        for w in (QLabel("PixelSize (mm/px)"), self.spin_pixel, btn_px, self.lbl_scale):
            sl.addWidget(w)
        sl.setStretch(3, 1)
        outer.addWidget(scale)
        self._update_scale_label()

        limits = QGroupBox("Judgement, scan and calibration")
        form = QGridLayout(limits)
        self.spins = {}
        specs = [
            ("good_confidence_min", "Production GOOD confidence", float, 0.5, 1.0, 0.01),
            ("width_tol_mm", "Width tolerance (mm)", float, 0.0, 5.0, 0.01),
            ("edge_tol_mm", "Edge shift tolerance (mm)", float, 0.0, 5.0, 0.01),
            ("sigma_k", "Sigma limit K", float, 0.0, 20.0, 0.5),
            ("align_warn_mm", "Alignment warn (mm)", float, 0.0, 5.0, 0.01),
            ("scan_y0", "Scan Y from (px)", int, 0, 4000, 10),
            ("scan_y1", "Scan Y to (px)", int, 1, 4000, 10),
            ("scan_step", "Scan row step (px)", int, 1, 200, 1),
            ("min_valid_rows", "Min valid rows", int, 1, 500, 1),
            ("green_delta", "Mask G-R threshold", int, 0, 128, 1),
            ("green_min", "Mask G min", int, 0, 255, 1),
            ("green_max", "Mask G max", int, 0, 255, 1),
            ("calib_max_runs", "Calibration panels", int, 5, 2000, 5),
            ("max_runs_per_scan", "Max panels per scan", int, 1, 5000, 10),
        ]
        for i, (key, label, kind, lo, hi, step) in enumerate(specs):
            widget = QDoubleSpinBox() if kind is float else QSpinBox()
            if kind is float:
                widget.setDecimals(3)
            widget.setRange(lo, hi)
            widget.setSingleStep(step)
            widget.setValue(getattr(self.cfg, key))
            form.addWidget(QLabel(label), i % 7, (i // 7) * 2)
            form.addWidget(widget, i % 7, (i // 7) * 2 + 1)
            self.spins[key] = widget
        outer.addWidget(limits)

        cal = QGroupBox("Calibrate the measurement baseline")
        cal_rows = QVBoxLayout(cal)
        cl = QHBoxLayout()
        self.cmb_cal_day = QComboBox()
        self.cmb_cal_day.setMinimumWidth(120)
        btn_cal_days = QPushButton("Refresh days")
        btn_cal_days.clicked.connect(lambda: self._refresh_days(self.cmb_cal_day))
        self.cmb_cal_recipe = QComboBox()
        self.cmb_cal_recipe.setMinimumWidth(300)
        self.btn_list_recipes = QPushButton("List products")
        self.btn_list_recipes.clicked.connect(self._list_recipes)
        self.btn_calibrate = QPushButton("Run calibration")
        self.btn_calibrate.setProperty("primary", True)
        self.btn_calibrate.clicked.connect(self._run_calibration)
        for w in (QLabel("Day"), self.cmb_cal_day, btn_cal_days,
                  self.cmb_cal_recipe, self.btn_list_recipes, self.btn_calibrate):
            cl.addWidget(w)
        cl.addStretch(1)
        cal_rows.addLayout(cl)

        auto = QHBoxLayout()
        self.chk_auto_cal = QCheckBox("Auto calibrate")
        self.chk_auto_cal.setToolTip(
            "Every interval, look at the newest production day and calibrate any\n"
            "product/table that has pictures but no baseline yet.")
        self.chk_auto_cal.setChecked(self.cfg.auto_calibrate)
        self.chk_auto_cal.stateChanged.connect(lambda _: self._toggle_auto_calibrate())
        self.spin_cal_every = QSpinBox()
        self.spin_cal_every.setRange(1, 720)
        self.spin_cal_every.setValue(self.cfg.calibrate_every_min)
        self.spin_cal_every.setSuffix(" min")
        self.spin_cal_every.valueChanged.connect(self._retime_auto_calibrate)
        self.chk_cal_refresh = QCheckBox("rebuild after a bit change")
        self.chk_cal_refresh.setChecked(self.cfg.calibrate_on_bit_change)
        self.chk_cal_refresh.setToolTip(
            "Rebuild a baseline only when the machine has logged a bit-wear alarm\n"
            "since that baseline was measured. A baseline that keeps following\n"
            "production would turn a slow drift into the new normal, so it is\n"
            "never rebuilt just because time has passed.")
        self.lbl_auto_cal = QLabel("auto calibrate is off")
        self.lbl_auto_cal.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        for w in (self.chk_auto_cal, QLabel("every"), self.spin_cal_every,
                  self.chk_cal_refresh, self.lbl_auto_cal):
            auto.addWidget(w)
        auto.addStretch(1)
        cal_rows.addLayout(auto)
        outer.addWidget(cal)

        buttons = QHBoxLayout()
        btn_save = QPushButton("Save config")
        btn_save.clicked.connect(self._save_config)
        btn_check = QPushButton("Check data + machine flags")
        btn_check.clicked.connect(self._check_data)
        self.btn_index = QPushButton("Rebuild picture index")
        self.btn_index.clicked.connect(lambda: self._refresh_index(quiet=False))
        self.spin_index_every = QSpinBox()
        self.spin_index_every.setRange(1, 720)
        self.spin_index_every.setValue(self.cfg.index_refresh_min)
        self.spin_index_every.setSuffix(" min")
        self.spin_index_every.valueChanged.connect(
            lambda v: self.index_timer.setInterval(v * 60_000))
        self.lbl_index = QLabel("index: building...")
        self.lbl_index.setStyleSheet(f"color:{theme.MUTED}; border:none; background:transparent;")
        for b in (btn_save, btn_check, self.btn_index,
                  QLabel("auto refresh every"), self.spin_index_every, self.lbl_index):
            buttons.addWidget(b)
        buttons.addStretch(1)
        outer.addLayout(buttons)

        notice = QLabel(
            "READ-ONLY on machine data. This tool never creates, changes or deletes "
            "anything in the picture, result, recipe or Eqp.cfg folders.\nIt writes "
            f"only into {APP_DIR} (config, baselines, labels), the model folder set "
            "above, plus any CSV you export.")
        notice.setWordWrap(True)
        notice.setStyleSheet("color:#1b5e20; background:#e8f5e9; border:1px solid #a5d6a7;"
                             "border-radius:4px; padding:6px;")
        outer.addWidget(notice)

        self.txt_check = QTextEdit()
        self.txt_check.setReadOnly(True)
        self.txt_check.setFont(QFont("Consolas", 9))
        outer.addWidget(self.txt_check, 1)
        return page

    def _browse(self, key, is_dir):
        current = self.path_edits[key].text()
        picked = (QFileDialog.getExistingDirectory(self, "Select folder", current) if is_dir
                  else QFileDialog.getOpenFileName(self, "Select Eqp.cfg", current,
                                                   "Config (*.cfg);;All (*.*)")[0])
        if picked:
            self.path_edits[key].setText(picked)

    def _update_scale_label(self):
        v = self.spin_pixel.value()
        if v > 0:
            self.lbl_scale.setText(f"1 px = {v:.6f} mm  ->  {1/v:.2f} px/mm   |   "
                                   f"1440x1080 px = {1440*v:.2f} x {1080*v:.2f} mm")

    def _collect(self):
        for key, edit in self.path_edits.items():
            setattr(self.cfg, key, edit.text())
        self.cfg.pixel_size_mm = self.spin_pixel.value()
        for key, widget in self.spins.items():
            setattr(self.cfg, key, widget.value())
        self.cfg.show_overlay = self.chk_overlay_dash.isChecked()
        self.cfg.index_refresh_min = self.spin_index_every.value()
        self.cfg.auto_calibrate = self.chk_auto_cal.isChecked()
        self.cfg.calibrate_every_min = self.spin_cal_every.value()
        self.cfg.calibrate_on_bit_change = self.chk_cal_refresh.isChecked()
        roots = protected_roots(self.cfg)
        self.store.protected = roots
        self.labels.protected = roots
        self.label_store.protected = roots
        return self.cfg

    def _save_config(self):
        try:
            self._collect().save(CONFIG_PATH)
        except ProtectedPathError as exc:
            QMessageBox.critical(self, "Write refused", str(exc))
            return
        self.status(f"Config saved to {CONFIG_PATH}")

    def _load_pixel_size(self):
        try:
            x, y = read_pixel_size(self.path_edits["eqp_cfg_path"].text())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Load PixelSize", str(exc))
            return
        self.spin_pixel.setValue(x)
        self.status(f"PixelSize loaded: {x} mm/px"
                    + ("" if abs(x - y) < 1e-9 else f"  (WARNING: Y differs: {y})"))

    def _check_data(self):
        cfg = self._collect()
        lines = []
        for key in ("picture_dir", "result_dir", "recipe_dir", "eqp_cfg_path"):
            value = getattr(cfg, key)
            lines.append(f"{key:<14} {'OK  ' if Path(value).exists() else 'MISS'}  {value}")
        try:
            days = available_days(cfg.result_dir)
            lines.append(f"\nRun result days : {len(days)}"
                         + (f"   {days[0]} .. {days[-1]}" if days else ""))
        except OSError as exc:
            lines.append(f"\nRun result days : {exc}")
        flags = read_machine_flags(cfg.eqp_cfg_path)
        if flags:
            lines.append("\n--- machine flags from Eqp.cfg ---")
            for key, value in flags.items():
                lines.append(f"{key:<36} = {value}")
            if flags.get("EnableAfterCuttingTakePicture") != "True":
                lines.append("\nWARNING: EnableAfterCuttingTakePicture is not True - the machine\n"
                             "         is NOT capturing new images, so the Dashboard tab will\n"
                             "         stay empty until it is switched back on.")
            try:
                if float(flags.get("AlignMaxOffset", "0")) > 0.3:
                    lines.append(f"NOTE: AlignMaxOffset is {flags['AlignMaxOffset']} mm - loose "
                                 "enough that a sustained drift can pass without alarming.")
            except ValueError:
                pass
        self.txt_check.setPlainText("\n".join(lines))
        self.status("Check complete.")

    def _list_recipes(self):
        day = self.cmb_cal_day.currentText()
        if not day or not self._ensure_index():
            self.status("Pick a day first.")
            return
        cfg = self._collect()
        runs = load_day(cfg.result_dir, day, self.picture_index)
        groups = {}
        for r in runs:
            groups.setdefault(r.key, []).append(r)
        self.cmb_cal_recipe.clear()
        hidden = 0
        for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            with_pics = sum(1 for r in items if r.pictures)
            if not with_pics:
                hidden += 1          # nothing to measure yet - show it once pictures land
                continue
            self.cmb_cal_recipe.addItem(f"{key}  [{len(items)} panels, {with_pics} with pictures]",
                                        key)
        note = f", {hidden} without pictures hidden" if hidden else ""
        if self.cmb_cal_recipe.count():
            self.status(f"{day}: {len(runs)} panels, "
                        f"{self.cmb_cal_recipe.count()} product/table combinations{note}.")
        else:
            self.status(f"{day}: no product/table has pictures yet.", theme.ALERT_RED)

    # -- automatic calibration ---------------------------------------------
    def _day_runs_with_pictures(self, day: str):
        """Runs for a day with their pictures attached, without needing the index."""
        runs = load_runs(self.cfg.result_dir, day)
        if not runs:
            return []
        try:
            paths = sorted(str(p) for p in Path(self.cfg.picture_dir).glob(f"{day}_*.bmp"))
        except OSError:
            return runs
        attach_pictures(runs, paths)
        return runs

    def _toggle_auto_calibrate(self, run_now: bool = True):
        if self.chk_auto_cal.isChecked():
            self.cal_timer.start(self.spin_cal_every.value() * 60_000)
            self.lbl_auto_cal.setText(
                f"on - checking every {self.spin_cal_every.value()} min")
            if run_now:
                self._auto_calibrate()
        else:
            self.cal_timer.stop()
            self.lbl_auto_cal.setText("auto calibrate is off")

    def _retime_auto_calibrate(self, minutes: int):
        """A new interval takes effect at once, not only after a re-tick."""
        if not self.cal_timer.isActive():
            return
        self.cal_timer.setInterval(minutes * 60_000)
        self.lbl_auto_cal.setText(f"on - checking every {minutes} min")

    def _auto_calibrate(self):
        """Fill in baselines for whatever ran recently and has none yet."""
        if self.worker is not None and self.worker.isRunning():
            return                       # a job is already using the worker slot
        cfg = self._collect()
        refresh = self.chk_cal_refresh.isChecked()

        try:
            days = available_days(cfg.result_dir)
        except OSError as exc:
            self.lbl_auto_cal.setText(f"cannot read results: {exc}")
            return
        if not days:
            self.lbl_auto_cal.setText("no run results found")
            return

        # newest day that actually produced pictures
        target_day = None
        for day in reversed(days[-14:]):
            try:
                if next(Path(cfg.picture_dir).glob(f"{day}_*.bmp"), None) is not None:
                    target_day = day
                    break
            except OSError:
                break
        if target_day is None:
            self.lbl_auto_cal.setText("no pictures on any recent day")
            return

        runs = self._day_runs_with_pictures(target_day)
        groups: dict[str, list] = {}
        for run in runs:
            if run.passed and run.pictures:
                groups.setdefault(run.key, []).append(run)

        pending = []
        reasons: dict[str, str] = {}
        for key, items in groups.items():
            if len(items) < MIN_AUTO_CALIB_PANELS:
                continue
            existing = self.store.get(key)
            if existing is None:
                pending.append((key, items))
                reasons[key] = "no baseline yet"
            elif refresh:
                stale, why = baseline_is_stale(existing.calibrated_at, cfg.result_dir)
                if stale:
                    pending.append((key, items))
                    reasons[key] = why

        if not pending:
            self.lbl_auto_cal.setText(
                f"on - {target_day}: nothing to calibrate "
                f"({len(groups)} product/table already covered)")
            return

        names = ", ".join(k.split("|")[0] for k, _ in pending)
        self.lbl_auto_cal.setText(f"calibrating {len(pending)}: {names}")
        for key in reasons:
            self.status(f"Auto calibrate {key}: {reasons[key]}")
        self._busy(True)

        def job(worker):
            done = []
            for n, (key, items) in enumerate(pending, start=1):
                if worker.is_cancelled():
                    break
                worker.progress.emit(n, len(pending), f"Auto calibrating {key}")
                try:
                    baseline = calibrate(cfg, items, cancelled=worker.is_cancelled)
                except InterruptedError:
                    raise
                except (ValueError, OSError) as exc:
                    done.append((key, None, str(exc)))
                    continue
                done.append((key, baseline, ""))
            return target_day, done

        def finished(result):
            day, done = result
            good = 0
            for key, baseline, error in done:
                if baseline is not None:
                    self.store.put(key, baseline)
                    good += 1
            self._busy(False)
            failed = [f"{k.split('|')[0]}: {e}" for k, b, e in done if b is None]
            summary = f"on - {day}: calibrated {good} of {len(done)}"
            if failed:
                summary += "  |  skipped " + "; ".join(failed[:2])
            self.lbl_auto_cal.setText(summary)
            self.status(f"Auto calibration: {good} baseline(s) added from {day}.")

        self._run_worker(job, finished)

    def _run_calibration(self):
        key = self.cmb_cal_recipe.currentData()
        day = self.cmb_cal_day.currentText()
        if not key or not day or not self._ensure_index():
            self.status("Pick a day and recipe first.")
            return
        cfg = self._collect()

        try:
            runs = [r for r in load_day(cfg.result_dir, day, self.picture_index)
                    if r.key == key]
            days = [d for d in available_days(cfg.result_dir) if self.picture_index.get(d)]
        except OSError as exc:
            QMessageBox.warning(self, "Run calibration", str(exc))
            return
        ready = [r for r in runs if r.passed and r.pictures]
        if len(ready) < MIN_CALIB_RUNS:
            if days:
                hint = "Days with pictures: " + ", ".join(days[-5:]) + "."
            else:
                hint = "No day in the picture folder has images for these results."
            QMessageBox.information(
                self, "Run calibration",
                f"{day} - {key}\n\n"
                f"{len(runs)} panels, {sum(1 for r in runs if r.pictures)} with pictures, "
                f"{len(ready)} passing with pictures.\n"
                f"Calibration needs at least {MIN_CALIB_RUNS} passing panels that have "
                f"pictures.\n\n{hint}")
            self.status(f"{day} {key}: not enough panels with pictures to calibrate.",
                        theme.ALERT_RED)
            return

        self._busy(True)

        def job(worker):
            return calibrate(cfg, runs,
                             progress=lambda d, t: worker.progress.emit(d, t, "Calibrating"),
                             cancelled=worker.is_cancelled)

        def finished(baseline):
            self.store.put(key, baseline)
            self._busy(False)
            self.status(f"Calibrated {key}: {len(baseline.indices)} usable positions "
                        f"of {baseline.pics_per_run}, from {baseline.runs_used} panels.")

        self._run_worker(job, finished)

    # ======================================================================
    def closeEvent(self, event):  # noqa: N802
        self.live_timer.stop()
        self.cal_timer.stop()
        self.index_timer.stop()
        if self.thumbs:
            self.thumbs.stop()
            self.thumbs.wait(1200)
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(theme.STYLESHEET)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
