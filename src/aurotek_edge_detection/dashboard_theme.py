from __future__ import annotations

from tkinter import ttk


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


def configure_dashboard_style(root) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")
    style.layout(
        "Dashboard.TNotebook.Tab",
        [
            (
                "Notebook.tab",
                {
                    "sticky": "nswe",
                    "children": [
                        (
                            "Notebook.padding",
                            {
                                "side": "top",
                                "sticky": "nswe",
                                "children": [
                                    ("Notebook.label", {"side": "top", "sticky": ""}),
                                ],
                            },
                        ),
                    ],
                },
            ),
        ],
    )
    style.configure("Dashboard.TNotebook", background=BG, borderwidth=0)
    style.configure(
        "Dashboard.TNotebook.Tab",
        padding=(22, 11),
        font=("Segoe UI", 11, "bold"),
        background=PANEL_2,
        foreground=ACCENT_DARK,
        borderwidth=1,
    )
    style.map(
        "Dashboard.TNotebook.Tab",
        background=[("selected", ACCENT), ("active", "#fecaca")],
        foreground=[("selected", "#ffffff"), ("active", ACCENT_DARK)],
        expand=[("selected", (0, 0, 0, 0))],
        padding=[("selected", (22, 11)), ("!selected", (22, 11))],
    )
