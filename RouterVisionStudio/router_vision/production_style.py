"""Production UI stylesheet and gate-state presentation (not the legacy theme)."""

from .interlock import GateState

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
