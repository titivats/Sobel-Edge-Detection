"""Enterprise red/white theme, matching the Aurotek dashboard look.

Same palette as the existing shop-floor tool so the two applications sit
together without looking like they came from different vendors.
"""

from __future__ import annotations

BG = "#f3f4f6"
PANEL = "#ffffff"
PANEL_2 = "#fff7f7"
SIDEBAR = "#991b1b"
ACCENT = "#b91c1c"
ACCENT_DARK = "#7f1d1d"
TEXT = "#111827"
MUTED = "#6b7280"
BORDER = "#d1d5db"
HEADER_MUTED = "#fecaca"
ONLINE_GREEN = "#22c55e"
RUNNING_ORANGE = "#f59e0b"
ALERT_RED = "#dc2626"

# Row tints used in tables and galleries
TINT_OK = "#e8f5e9"
TINT_NG = "#fee2e2"
TINT_WARN = "#fef3c7"
TINT_GREY = "#eeeeee"


STYLESHEET = f"""
QWidget {{
    background: {BG};
    color: {TEXT};
    font-family: "Segoe UI";
    font-size: 10pt;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    background: {BG};
    top: -1px;
}}
QTabBar::tab {{
    background: {PANEL_2};
    color: {ACCENT_DARK};
    border: 1px solid {BORDER};
    border-bottom: none;
    padding: 11px 24px;
    font-weight: bold;
    font-size: 11pt;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background: {ACCENT};
    color: #ffffff;
}}
QTabBar::tab:hover:!selected {{
    background: {HEADER_MUTED};
}}

QGroupBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    margin-top: 14px;
    padding: 12px 10px 10px 10px;
    font-weight: bold;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT_DARK};
}}

QPushButton {{
    background: {PANEL};
    color: {ACCENT_DARK};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 6px 14px;
    font-weight: bold;
}}
QPushButton:hover {{ background: {HEADER_MUTED}; }}
QPushButton:pressed {{ background: {ACCENT}; color: #ffffff; }}
QPushButton:disabled {{ color: #9ca3af; background: #f9fafb; }}
QPushButton[primary="true"] {{
    background: {ACCENT};
    color: #ffffff;
    border: 1px solid {ACCENT_DARK};
}}
QPushButton[primary="true"]:hover {{ background: {ACCENT_DARK}; }}

QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 6px;
    selection-background-color: {ACCENT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}

QTableWidget, QListWidget, QTextEdit {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 3px;
}}
QHeaderView::section {{
    background: {PANEL_2};
    color: {ACCENT_DARK};
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    font-weight: bold;
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background: {HEADER_MUTED};
    color: {TEXT};
}}

QProgressBar {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 3px;
    text-align: center;
    height: 16px;
}}
QProgressBar::chunk {{ background: {ACCENT}; }}

QStatusBar {{ background: {PANEL_2}; border-top: 1px solid {BORDER}; }}
QSplitter::handle {{ background: {BORDER}; }}
QCheckBox {{ spacing: 6px; }}
"""
