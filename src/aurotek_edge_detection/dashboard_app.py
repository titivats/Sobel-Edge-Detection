from __future__ import annotations

import csv
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2

from .image_file_discovery import find_images
from .dashboard_paths import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_CSV_DIR,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    PROJECT_ROOT,
)
from .dashboard_theme import (
    ACCENT,
    ACCENT_DARK,
    ALERT_RED,
    BG,
    BORDER,
    HEADER_MUTED,
    MUTED,
    ONLINE_GREEN,
    PANEL,
    PANEL_2,
    RUNNING_ORANGE,
    SIDEBAR,
    TEXT,
    configure_dashboard_style,
)
from .dashboard_utils import timestamp_from_name, tk_image_from_bgr, tk_image_from_bgr_fixed
from .image_classifier import YoloImageClassifier
from .sobel_edge_detection import create_sobel_edge_masks, thicken_edge_for_display


FINETUNE_PREVIEW_SIZE = (670, 560)
FINETUNE_CARD_BG = "#f8fafc"
FINETUNE_PREVIEW_FRAME_BG = "#111827"
FINETUNE_PREVIEW_IMAGE_BG = "#000000"
FINETUNE_PREVIEW_BORDER = "#334155"
FINETUNE_OPTION_VALUES = ("Combined", "Sobel X", "Sobel Y")
FINETUNE_PARAMETER_HELP = {
    "Edge Mode": "Combined, X, or Y edge view.",
    "Sobel Threshold": "Higher = fewer, stronger edges.",
    "Blur Kernel": "Smooth noise before edge detect.",
    "Sobel Kernel": "Larger = broader edge response.",
    "Cleanup Kernel": "Close small gaps in edges.",
    "Display Thickness": "Preview/save line thickness.",
}
DASHBOARD_POINT_PREVIEW_SIZE = (330, 230)
DASHBOARD_POINT_DETAIL_WRAP = 660
DASHBOARD_STARTUP_DELAY_MS = 5000
DASHBOARD_REFRESH_MS = 2500
DASHBOARD_UNMATCHED_SESSION = "__unmatched_product_info__"


class AurotekEdgeDashboard:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Aurotek Edge Detection Dashboard")
        self.root.geometry("1920x1080")
        self.root.minsize(1366, 768)
        self.root.configure(background=BG)

        self.input_var = tk.StringVar(value=str(DEFAULT_INPUT_DIR))
        self.csv_import_var = tk.StringVar(value=str(DEFAULT_CSV_DIR))
        self.output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.model_var = tk.StringVar(value=str(DEFAULT_CLASSIFICATION_MODEL))
        self.config_input_var = tk.StringVar(value=str(DEFAULT_INPUT_DIR))
        self.config_csv_import_var = tk.StringVar(value=str(DEFAULT_CSV_DIR))
        self.config_output_var = tk.StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.config_model_var = tk.StringVar(value=str(DEFAULT_CLASSIFICATION_MODEL))
        self.pixels_per_mm_var = tk.StringVar(value="1")
        self.black_threshold_var = tk.StringVar(value="55")
        self.sobel_threshold_var = tk.StringVar(value="0.12")
        self.orientation_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="Ready for image classification.")
        self.summary_var = tk.StringVar(value="Processed: 0 image(s)")
        self.csv_var = tk.StringVar(value="CSV: -")
        self.sobel_var = tk.StringVar(value="Sobel images: -")
        self.annotated_var = tk.StringVar(value="Annotated images: -")
        self.finetune_count_var = tk.StringVar(value="0 / 0")
        self.finetune_state_var = tk.StringVar(value="Not Save")
        self.display_thickness_var = tk.StringVar(value="2")
        self.blur_kernel_var = tk.StringVar(value="5")
        self.sobel_kernel_var = tk.StringVar(value="3")
        self.cleanup_kernel_var = tk.StringVar(value="3")
        self.edge_mode_var = tk.StringVar(value="Combined")
        self.dashboard_total_var = tk.StringVar(value="0")
        self.dashboard_pass_var = tk.StringVar(value="0")
        self.dashboard_ng_var = tk.StringVar(value="0")
        self.dashboard_unknown_var = tk.StringVar(value="0")
        self.dashboard_boards_var = tk.StringVar(value="0")
        self.dashboard_update_var = tk.StringVar(value="Last update: -")
        self.dashboard_search_var = tk.StringVar(value="")
        self.dashboard_session_summary_var = tk.StringVar(value="Session: -")

        self.is_running = False
        self.status_dot: int | None = None
        self.status_canvas: tk.Canvas | None = None
        self.result_card_photos: list[tuple[tk.PhotoImage, tk.PhotoImage]] = []
        self.dashboard_sessions: list[dict[str, object]] = []
        self.dashboard_filtered_sessions: list[dict[str, object]] = []
        self.dashboard_selected_session_key = ""
        self.dashboard_active_search_query = ""
        self.dashboard_data_signature: tuple[int, int, int] | None = None
        self.dashboard_refresh_after_id: str | None = None
        self.dashboard_finding = False
        self.dashboard_spinner_after_id: str | None = None
        self.dashboard_spinner_angle = 0
        self.csv_row_cache: dict[Path, tuple[int, list[dict[str, str]]]] = {}
        self.classification_cache: dict[tuple[Path, int, Path, int], tuple[str, float | None]] = {}
        self.finetune_images: list[Path] = []
        self.finetune_index = 0
        self.finetune_current_image = None
        self.finetune_photos: list[tk.PhotoImage] = []
        self.finetune_save_states: dict[str, str] = {}
        self.classifier = YoloImageClassifier()

        self._configure_style()
        self._build_layout()

    def run(self) -> None:
        self.root.mainloop()

    def _configure_style(self) -> None:
        configure_dashboard_style(self.root)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        header = tk.Frame(self.root, background=SIDEBAR, padx=24, pady=16)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Aurotek Edge Detection Dashboard",
            anchor="w",
            font=("Segoe UI Semibold", 24),
            foreground="#ffffff",
            background=SIDEBAR,
        ).grid(row=0, column=0, sticky="ew")
        status = tk.Frame(header, background=SIDEBAR)
        status.grid(row=0, column=1, sticky="e")
        self.status_canvas = tk.Canvas(
            status,
            width=18,
            height=18,
            background=SIDEBAR,
            highlightthickness=0,
            bd=0,
        )
        self.status_canvas.pack(side="left", padx=(0, 8))
        self.status_dot = self.status_canvas.create_oval(
            3,
            3,
            15,
            15,
            fill=ONLINE_GREEN,
            outline="#ffffff",
        )
        status_text = tk.Frame(status, background=SIDEBAR)
        status_text.pack(side="left")
        tk.Label(
            status_text,
            textvariable=self.status_var,
            anchor="e",
            font=("Segoe UI", 10, "bold"),
            foreground="#ffffff",
            background=SIDEBAR,
            wraplength=440,
        ).pack(anchor="e")
        tk.Label(
            status_text,
            textvariable=self.dashboard_update_var,
            anchor="e",
            font=("Segoe UI", 9),
            foreground="#fecaca",
            background=SIDEBAR,
            wraplength=440,
        ).pack(anchor="e", pady=(3, 0))

        self.notebook = ttk.Notebook(self.root, style="Dashboard.TNotebook")
        self.notebook.grid(row=1, column=0, sticky="nsew", padx=14, pady=(12, 14))

        self.dashboard_tab = tk.Frame(self.notebook, background=BG)
        self.configuration_tab = tk.Frame(self.notebook, background=BG)
        self.finetune_tab = tk.Frame(self.notebook, background=BG)
        self.notebook.add(self.dashboard_tab, text="Dashboard Real-time")
        self.notebook.add(self.configuration_tab, text="Configuration")
        self.notebook.add(self.finetune_tab, text="Fine-Tune Model")

        self._build_dashboard_tab()
        self._build_configuration_tab()
        self._build_finetune_tab()
        self.root.after(DASHBOARD_STARTUP_DELAY_MS, self._refresh_dashboard_on_startup)

    def _build_configuration_tab(self) -> None:
        config_panel = tk.Frame(
            self.configuration_tab,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=20,
            pady=18,
        )
        config_panel.pack(fill="x", padx=4, pady=4)
        config_panel.columnconfigure(1, weight=1)
        config_panel.columnconfigure(4, weight=1)

        tk.Label(
            config_panel,
            text="Configuration for Data Import/Export",
            anchor="w",
            font=("Segoe UI Semibold", 17),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=0, column=0, columnspan=6, sticky="ew", pady=(0, 14))

        self._add_path_row(
            config_panel,
            row=1,
            label="Input Image",
            variable=self.config_input_var,
            button_text="Browse",
            command=lambda: self._browse_directory(self.config_input_var),
        )
        self._add_path_row(
            config_panel,
            row=2,
            label="Import .CSV",
            variable=self.config_csv_import_var,
            button_text="Browse",
            command=lambda: self._browse_directory(self.config_csv_import_var),
        )
        self._add_path_row(
            config_panel,
            row=3,
            label="Models Select",
            variable=self.config_model_var,
            button_text="Browse",
            command=lambda: self._browse_file(
                self.config_model_var,
                [("PyTorch model", "*.pt"), ("All files", "*.*")],
            ),
        )
        self._add_path_row(
            config_panel,
            row=4,
            label="Output",
            variable=self.config_output_var,
            button_text="Browse",
            command=lambda: self._browse_directory(self.config_output_var),
        )

        action_row = tk.Frame(config_panel, background=PANEL)
        action_row.grid(row=5, column=0, columnspan=6, sticky="e", pady=(12, 0))
        tk.Button(
            action_row,
            text="Save",
            command=self._save_configuration,
            background=ACCENT,
            foreground="#ffffff",
            activebackground=ACCENT_DARK,
            activeforeground="#ffffff",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 11, "bold"),
            padx=22,
            pady=8,
        ).pack(side="left", padx=(0, 10))
        tk.Button(
            action_row,
            text="Reset to Default",
            command=self._reset_configuration_defaults,
            background=PANEL_2,
            foreground=ACCENT_DARK,
            activebackground="#fecaca",
            activeforeground=ACCENT_DARK,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 11),
            padx=18,
            pady=8,
        ).pack(side="left")

    def _build_dashboard_tab(self) -> None:
        self.dashboard_tab.columnconfigure(0, weight=1)
        self.dashboard_tab.rowconfigure(0, weight=1)

        body = tk.Frame(self.dashboard_tab, background=BG)
        body.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)

        summary_panel = tk.Frame(
            body,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=16,
            pady=16,
        )
        summary_panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        summary_panel.columnconfigure(0, weight=1)

        title_block = tk.Frame(summary_panel, background=PANEL)
        title_block.grid(row=0, column=0, sticky="nw")
        tk.Label(
            title_block,
            text="Dashboard Real-time Predict by Sobel Edge Detection",
            anchor="w",
            font=("Segoe UI Semibold", 16),
            foreground=TEXT,
            background=PANEL,
        ).pack(anchor="w")

        kpi_row = tk.Frame(summary_panel, background=PANEL)
        kpi_row.grid(row=0, column=1, rowspan=2, sticky="ne")
        self._add_kpi_card(kpi_row, "POINTS", self.dashboard_total_var, TEXT)
        self._add_kpi_card(kpi_row, "BOARDS", self.dashboard_boards_var, ACCENT_DARK)
        self._add_kpi_card(kpi_row, "PASS", self.dashboard_pass_var, ONLINE_GREEN)
        self._add_kpi_card(kpi_row, "NG", self.dashboard_ng_var, ALERT_RED)
        self._add_kpi_card(kpi_row, "UNKNOWN", self.dashboard_unknown_var, RUNNING_ORANGE)

        search_panel = tk.Frame(summary_panel, background=PANEL)
        search_panel.grid(row=1, column=0, sticky="ew", pady=(6, 0))
        search_panel.columnconfigure(0, weight=1)

        search_controls = tk.Frame(search_panel, background=PANEL)
        search_controls.grid(row=0, column=0, sticky="w")
        tk.Label(
            search_controls,
            text="Search SN",
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).pack(side="left", padx=(0, 12))
        tk.Entry(
            search_controls,
            textvariable=self.dashboard_search_var,
            background=FINETUNE_CARD_BG,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
            width=34,
        ).pack(side="left", ipady=3)
        tk.Button(
            search_controls,
            text="Find",
            command=self._find_dashboard_search,
            background=ACCENT,
            foreground="#ffffff",
            activebackground=ACCENT_DARK,
            activeforeground="#ffffff",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=3,
        ).pack(side="left", padx=(8, 0))
        tk.Button(
            search_controls,
            text="Clear",
            command=self._clear_dashboard_search,
            background=PANEL,
            foreground=ACCENT_DARK,
            activebackground="#e5e7eb",
            activeforeground=ACCENT_DARK,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9, "bold"),
            padx=12,
            pady=3,
        ).pack(side="left", padx=(8, 0))
        self.dashboard_spinner_canvas = tk.Canvas(
            search_controls,
            width=24,
            height=24,
            background=PANEL,
            highlightthickness=0,
            bd=0,
        )
        self.dashboard_spinner_arc = self.dashboard_spinner_canvas.create_arc(
            4,
            4,
            20,
            20,
            start=0,
            extent=270,
            style="arc",
            width=3,
            outline=ACCENT,
            state="hidden",
        )
        self.dashboard_spinner_canvas.pack(side="left", padx=(8, 0))
        tk.Label(
            search_panel,
            textvariable=self.dashboard_session_summary_var,
            anchor="w",
            justify="left",
            font=("Segoe UI", 8),
            foreground=MUTED,
            background=PANEL,
            wraplength=1200,
        ).grid(row=1, column=0, sticky="ew", pady=(6, 0))

        result_panel = tk.Frame(
            body,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        result_panel.grid(row=1, column=0, sticky="nsew")
        result_panel.columnconfigure(0, weight=1)
        result_panel.rowconfigure(0, weight=1)

        self.cards_view = tk.Frame(result_panel, background=PANEL)
        self.cards_view.grid(row=0, column=0, sticky="nsew")
        self.cards_view.columnconfigure(0, weight=1)
        self.cards_view.rowconfigure(0, weight=1)
        self.cards_canvas = tk.Canvas(
            self.cards_view,
            background=PANEL,
            highlightthickness=0,
            bd=0,
            yscrollincrement=32,
        )
        self.cards_scrollbar = tk.Scrollbar(
            self.cards_view,
            orient="vertical",
            command=self.cards_canvas.yview,
        )
        self.cards_content = tk.Frame(self.cards_canvas, background=PANEL)
        self.cards_content.bind("<Configure>", self._update_cards_scroll_region)
        self.cards_window = self.cards_canvas.create_window(
            (0, 0),
            window=self.cards_content,
            anchor="nw",
        )
        self.cards_canvas.configure(yscrollcommand=self.cards_scrollbar.set)
        self.cards_canvas.bind("<Configure>", self._resize_cards_window)
        self.root.bind_all("<MouseWheel>", self._on_cards_mousewheel, add="+")
        self.cards_canvas.grid(row=0, column=0, sticky="nsew")
        self.cards_scrollbar.grid(row=0, column=1, sticky="ns")

    def _update_cards_scroll_region(self, _event: tk.Event | None = None) -> None:
        self.cards_canvas.configure(scrollregion=self.cards_canvas.bbox("all"))

    def _resize_cards_window(self, event: tk.Event) -> None:
        self.cards_canvas.itemconfigure(self.cards_window, width=event.width)
        self._update_cards_scroll_region()

    def _on_cards_mousewheel(self, event: tk.Event) -> str:
        if not self._event_is_inside_cards_view(event):
            return ""
        steps = int(-event.delta / 120)
        if steps == 0:
            steps = -1 if event.delta > 0 else 1
        self.cards_canvas.yview_scroll(steps * 2, "units")
        return "break"

    def _event_is_inside_cards_view(self, event: tk.Event) -> bool:
        widget = self.root.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget == self.cards_view:
                return True
            widget = widget.master
        return False

    def _add_kpi_card(
        self,
        parent: tk.Widget,
        title: str,
        variable: tk.StringVar,
        color: str,
    ) -> None:
        card = tk.Frame(
            parent,
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=18,
            pady=10,
        )
        card.pack(side="left", padx=(10, 0))
        tk.Label(
            card,
            textvariable=variable,
            anchor="center",
            width=5,
            font=("Segoe UI Semibold", 22),
            foreground=color,
            background=PANEL_2,
        ).pack()
        tk.Label(
            card,
            text=title,
            anchor="center",
            font=("Segoe UI", 9, "bold"),
            foreground=ACCENT_DARK,
            background=PANEL_2,
        ).pack()

    def _build_finetune_tab(self) -> None:
        self.finetune_tab.columnconfigure(0, weight=4)
        self.finetune_tab.columnconfigure(1, weight=1, minsize=430)
        self.finetune_tab.rowconfigure(0, weight=1)

        tuning_panel = tk.Frame(
            self.finetune_tab,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=18,
            pady=18,
        )
        tuning_panel.grid(row=0, column=0, sticky="nsew", padx=(4, 10), pady=4)
        tuning_panel.rowconfigure(1, weight=1)
        tuning_panel.columnconfigure(0, weight=1)
        preview_header = tk.Frame(tuning_panel, background=PANEL)
        preview_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        preview_header.columnconfigure(0, weight=1)
        tk.Label(
            preview_header,
            text="Fine-Tune Model | Sobel Edge Detection",
            anchor="w",
            font=("Segoe UI Semibold", 18),
            foreground=ACCENT_DARK,
            background=PANEL,
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            preview_header,
            text="Live preview",
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=MUTED,
            background=PANEL,
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        preview_grid = tk.Frame(tuning_panel, background=PANEL)
        preview_grid.grid(row=1, column=0, sticky="nsew")
        preview_grid.columnconfigure(0, weight=1, uniform="finetune_preview")
        preview_grid.columnconfigure(1, weight=1, uniform="finetune_preview")
        preview_grid.rowconfigure(0, weight=1)
        self.original_preview_label = self._add_finetune_preview(
            preview_grid,
            0,
            "Original Image",
        )
        self.sobel_preview_label = self._add_finetune_preview(
            preview_grid,
            1,
            "Sobel Edge Detection",
        )

        parameter_panel = tk.Frame(
            self.finetune_tab,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=18,
            pady=18,
        )
        parameter_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 4), pady=4)
        parameter_panel.columnconfigure(0, weight=1)
        parameter_panel.rowconfigure(5, weight=1)
        parameter_header = tk.Frame(parameter_panel, background=PANEL)
        parameter_header.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        tk.Label(
            parameter_header,
            text="Sobel Fine-Tune Parameters",
            anchor="w",
            font=("Segoe UI Semibold", 16),
            foreground=ACCENT_DARK,
            background=PANEL,
        ).pack(anchor="w")

        state_content = tk.Frame(
            parameter_panel,
            background=FINETUNE_CARD_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=10,
        )
        state_content.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        self.finetune_state_canvas = tk.Canvas(
            state_content, width=18, height=18, background=FINETUNE_CARD_BG, highlightthickness=0
        )
        self.finetune_state_canvas.pack(side="left", padx=(0, 8))
        self.finetune_state_dot = self.finetune_state_canvas.create_oval(
            3, 3, 15, 15, fill=RUNNING_ORANGE, outline=BORDER
        )
        tk.Label(
            state_content,
            textvariable=self.finetune_state_var,
            foreground=TEXT,
            background=FINETUNE_CARD_BG,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")
        tk.Label(
            state_content,
            textvariable=self.finetune_count_var,
            foreground=MUTED,
            background=FINETUNE_CARD_BG,
            font=("Segoe UI", 10),
            anchor="e",
            justify="right",
            wraplength=230,
        ).pack(side="right", fill="x", expand=True)

        adjustment_panel = tk.Frame(
            parameter_panel,
            background=FINETUNE_CARD_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        adjustment_panel.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        adjustment_panel.columnconfigure(1, weight=1)
        adjustment_panel.columnconfigure(2, weight=0)
        adjust_header = tk.Frame(adjustment_panel, background=FINETUNE_CARD_BG)
        adjust_header.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        adjust_header.columnconfigure(0, weight=1)
        tk.Label(
            adjust_header,
            text="Adjust",
            anchor="w",
            font=("Segoe UI Semibold", 12),
            foreground=ACCENT_DARK,
            background=FINETUNE_CARD_BG,
        ).grid(row=0, column=0, sticky="w")
        tk.Button(
            adjust_header,
            text="Reset",
            command=self._reset_finetune_parameters,
            background=PANEL,
            foreground=ACCENT_DARK,
            activebackground="#e5e7eb",
            activeforeground=ACCENT_DARK,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=3,
        ).grid(row=0, column=1, sticky="e")

        self._add_finetune_option_row(
            adjustment_panel,
            1,
            "Edge Mode",
            self.edge_mode_var,
            FINETUNE_OPTION_VALUES,
            FINETUNE_PARAMETER_HELP["Edge Mode"],
        )

        self._add_finetune_slider_row(
            adjustment_panel,
            2,
            "Sobel Threshold",
            self.sobel_threshold_var,
            0.01,
            0.50,
            0.01,
            FINETUNE_PARAMETER_HELP["Sobel Threshold"],
        )
        self._add_finetune_slider_row(
            adjustment_panel,
            3,
            "Blur Kernel",
            self.blur_kernel_var,
            1,
            15,
            2,
            FINETUNE_PARAMETER_HELP["Blur Kernel"],
        )
        self._add_finetune_slider_row(
            adjustment_panel,
            4,
            "Sobel Kernel",
            self.sobel_kernel_var,
            1,
            7,
            2,
            FINETUNE_PARAMETER_HELP["Sobel Kernel"],
        )
        self._add_finetune_slider_row(
            adjustment_panel,
            5,
            "Cleanup Kernel",
            self.cleanup_kernel_var,
            1,
            9,
            2,
            FINETUNE_PARAMETER_HELP["Cleanup Kernel"],
        )
        self._add_finetune_slider_row(
            adjustment_panel,
            6,
            "Display Thickness",
            self.display_thickness_var,
            1,
            8,
            1,
            FINETUNE_PARAMETER_HELP["Display Thickness"],
        )

        image_actions = tk.Frame(
            parameter_panel,
            background=FINETUNE_CARD_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=10,
            pady=6,
        )
        image_actions.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        image_actions.grid_propagate(False)
        image_actions.configure(height=112)
        image_actions.columnconfigure(0, weight=1, uniform="finetune_actions")
        image_actions.columnconfigure(1, weight=1, uniform="finetune_actions")
        tk.Label(
            image_actions,
            text="Images",
            anchor="w",
            font=("Segoe UI Semibold", 11),
            foreground=ACCENT_DARK,
            background=FINETUNE_CARD_BG,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self._add_finetune_button(
            image_actions,
            "Load Images",
            self._load_finetune_images,
            ACCENT,
            "#ffffff",
            1,
            0,
            2,
        )
        self._add_finetune_button(
            image_actions,
            "Previous",
            self._previous_finetune_image,
            PANEL,
            ACCENT_DARK,
            2,
            0,
            1,
        )
        self._add_finetune_button(
            image_actions,
            "Next",
            self._next_finetune_image,
            PANEL,
            ACCENT_DARK,
            2,
            1,
            1,
        )

        decision_actions = tk.Frame(
            parameter_panel,
            background=FINETUNE_CARD_BG,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=10,
            pady=6,
        )
        decision_actions.grid(row=4, column=0, sticky="ew")
        decision_actions.grid_propagate(False)
        decision_actions.configure(height=72)
        decision_actions.columnconfigure(0, weight=1, uniform="finetune_actions")
        decision_actions.columnconfigure(1, weight=1, uniform="finetune_actions")
        tk.Label(
            decision_actions,
            text="Decision",
            anchor="w",
            font=("Segoe UI Semibold", 11),
            foreground=ACCENT_DARK,
            background=FINETUNE_CARD_BG,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        self._add_finetune_button(
            decision_actions,
            "Save",
            self._save_finetune_current,
            ONLINE_GREEN,
            "#ffffff",
            1,
            0,
            1,
        )
        self._add_finetune_button(
            decision_actions,
            "Not Save",
            self._not_save_finetune_current,
            RUNNING_ORANGE,
            "#ffffff",
            1,
            1,
            1,
        )

    def _add_finetune_preview(
        self,
        parent: tk.Widget,
        column: int,
        title: str,
    ) -> tk.Label:
        panel = tk.Frame(
            parent,
            background=FINETUNE_PREVIEW_FRAME_BG,
            highlightthickness=1,
            highlightbackground=FINETUNE_PREVIEW_BORDER,
            padx=10,
            pady=10,
        )
        panel.grid(
            row=0,
            column=column,
            sticky="n",
            padx=(0, 8) if column == 0 else (8, 0),
        )
        panel.rowconfigure(1, weight=1)
        panel.columnconfigure(0, weight=1)
        tk.Label(
            panel,
            text=title,
            anchor="center",
            justify="center",
            font=("Segoe UI Semibold", 10),
            foreground="#f8fafc",
            background=FINETUNE_PREVIEW_FRAME_BG,
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        image_slot = tk.Frame(
            panel,
            background=FINETUNE_PREVIEW_IMAGE_BG,
            width=FINETUNE_PREVIEW_SIZE[0],
            height=FINETUNE_PREVIEW_SIZE[1],
        )
        image_slot.grid(row=1, column=0)
        image_slot.grid_propagate(False)
        image_slot.rowconfigure(0, weight=1)
        image_slot.columnconfigure(0, weight=1)
        image_label = tk.Label(
            image_slot,
            text="No image loaded",
            anchor="center",
            font=("Segoe UI", 12, "bold"),
            foreground="#ffffff",
            background=FINETUNE_PREVIEW_IMAGE_BG,
            compound="center",
        )
        image_label.grid(row=0, column=0, sticky="nsew")
        return image_label

    def _add_finetune_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        background: str,
        foreground: str,
        row: int,
        column: int,
        span: int,
    ) -> None:
        if span > 1:
            padx = (0, 0)
        elif column == 0:
            padx = (0, 4)
        else:
            padx = (4, 0)
        tk.Button(
            parent,
            text=text,
            command=command,
            background=background,
            foreground=foreground,
            activebackground=ACCENT_DARK if foreground == "#ffffff" else "#fecaca",
            activeforeground="#ffffff" if foreground == "#ffffff" else ACCENT_DARK,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9, "bold" if foreground == "#ffffff" else "normal"),
            pady=3,
        ).grid(
            row=row,
            column=column,
            columnspan=span,
            sticky="ew",
            padx=padx,
            pady=(0, 2),
        )

    def _add_path_row(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        variable: tk.StringVar,
        button_text: str,
        command,
    ) -> None:
        tk.Label(
            parent,
            text=label,
            anchor="w",
            width=14,
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=row, column=0, sticky="w", pady=(0, 10))
        tk.Entry(
            parent,
            textvariable=variable,
            background=PANEL_2,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 10),
        ).grid(row=row, column=1, columnspan=4, sticky="ew", padx=(8, 10), pady=(0, 10), ipady=7)
        tk.Button(
            parent,
            text=button_text,
            command=command,
            background=PANEL_2,
            foreground=ACCENT_DARK,
            activebackground="#fecaca",
            activeforeground=ACCENT_DARK,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 10),
            padx=12,
            pady=6,
        ).grid(row=row, column=5, sticky="ew", pady=(0, 10))

    def _add_entry(
        self,
        parent: tk.Widget,
        row: int,
        column: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        tk.Label(
            parent,
            text=label,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=row, column=column, sticky="w", pady=(6, 0))
        tk.Entry(
            parent,
            textvariable=variable,
            background=PANEL_2,
            foreground=TEXT,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 10),
            width=12,
        ).grid(row=row, column=column + 1, sticky="w", padx=(8, 20), pady=(6, 0), ipady=6)

    def _add_tuning_row(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        tk.Label(
            parent,
            text=label,
            anchor="w",
            width=18,
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=row, column=0, sticky="w", pady=(0, 12))
        tk.Entry(
            parent,
            textvariable=variable,
            background=PANEL_2,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 11),
            width=18,
        ).grid(row=row, column=1, sticky="w", pady=(0, 12), ipady=7)

    def _add_parameter_entry(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        variable: tk.StringVar,
    ) -> None:
        tk.Label(
            parent,
            text=label,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=row, column=0, sticky="w", pady=(0, 12))
        tk.Entry(
            parent,
            textvariable=variable,
            background=PANEL_2,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 11),
        ).grid(row=row, column=1, sticky="ew", pady=(0, 12), ipady=7)

    def _add_finetune_slider_row(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        variable: tk.StringVar,
        from_value: float,
        to_value: float,
        resolution: float,
        description: str,
    ) -> None:
        background = str(parent.cget("background"))
        control_row = row * 2 - 1
        description_row = control_row + 1
        tk.Label(
            parent,
            text=label,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=background,
        ).grid(row=control_row, column=0, sticky="w")
        tk.Scale(
            parent,
            from_=from_value,
            to=to_value,
            resolution=resolution,
            orient="horizontal",
            variable=variable,
            command=lambda _value: self._commit_finetune_parameters(show_error=False),
            background=background,
            highlightthickness=0,
            troughcolor="#e5e7eb",
            activebackground=ACCENT,
            showvalue=False,
        ).grid(row=control_row, column=1, sticky="ew", padx=(12, 8))
        value_controls = tk.Frame(parent, background=background)
        value_controls.grid(row=control_row, column=2, sticky="e")
        self._add_finetune_step_button(
            value_controls,
            "-",
            lambda _variable=variable, _step=-resolution: self._step_finetune_parameter(
                _variable,
                _step,
            ),
        )
        entry_frame = tk.Frame(
            value_controls,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=TEXT,
            width=44,
            height=24,
        )
        entry_frame.pack(side="left", padx=3)
        entry_frame.pack_propagate(False)
        entry = tk.Entry(
            entry_frame,
            textvariable=variable,
            background=PANEL,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="flat",
            borderwidth=0,
            font=("Consolas", 10),
            justify="center",
        )
        entry.pack(fill="both", expand=True, padx=1, pady=1)
        self._add_finetune_step_button(
            value_controls,
            "+",
            lambda _variable=variable, _step=resolution: self._step_finetune_parameter(
                _variable,
                _step,
            ),
        )
        entry.bind("<Return>", lambda _event: self._commit_finetune_parameters())
        entry.bind("<FocusOut>", lambda _event: self._commit_finetune_parameters(show_error=False))
        tk.Label(
            parent,
            text=description,
            anchor="w",
            justify="left",
            font=("Segoe UI", 8),
            foreground=MUTED,
            background=background,
            wraplength=300,
        ).grid(row=description_row, column=0, columnspan=3, sticky="ew", pady=(0, 1))

    def _add_finetune_step_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
    ) -> None:
        tk.Button(
            parent,
            text=text,
            command=command,
            background=PANEL,
            foreground=ACCENT_DARK,
            activebackground="#e5e7eb",
            activeforeground=ACCENT_DARK,
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9, "bold"),
            width=2,
            padx=0,
            pady=1,
        ).pack(side="left")

    def _add_finetune_option_row(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        description: str,
    ) -> None:
        background = str(parent.cget("background"))
        control_row = row * 2 - 1
        description_row = control_row + 1
        tk.Label(
            parent,
            text=label,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=background,
        ).grid(row=control_row, column=0, sticky="w")
        combobox = ttk.Combobox(
            parent,
            textvariable=variable,
            values=values,
            state="readonly",
            font=("Segoe UI", 10),
            justify="center",
            width=16,
        )
        combobox.grid(
            row=control_row,
            column=1,
            sticky="w",
            padx=(12, 0),
            ipady=5,
        )
        combobox.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._commit_finetune_parameters(show_error=False),
        )
        tk.Label(
            parent,
            text=description,
            anchor="w",
            justify="left",
            font=("Segoe UI", 8),
            foreground=MUTED,
            background=background,
            wraplength=300,
        ).grid(row=description_row, column=0, columnspan=3, sticky="ew", pady=(0, 1))

    def _browse_directory(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or str(PROJECT_ROOT))
        if selected:
            variable.set(selected)

    def _browse_file(self, variable: tk.StringVar, filetypes: list[tuple[str, str]]) -> None:
        selected = filedialog.askopenfilename(
            initialdir=str(Path(variable.get()).expanduser().parent)
            if variable.get()
            else str(PROJECT_ROOT),
            filetypes=filetypes,
        )
        if selected:
            variable.set(selected)

    def _save_configuration(self) -> None:
        previous_model = self.model_var.get()
        self.input_var.set(self.config_input_var.get())
        self.csv_import_var.set(self.config_csv_import_var.get())
        self.model_var.set(self.config_model_var.get())
        self.output_var.set(self.config_output_var.get())
        self.classifier.reset()
        self.csv_row_cache.clear()
        self.classification_cache.clear()
        self.dashboard_data_signature = None
        Path(self.output_var.get()).expanduser().mkdir(parents=True, exist_ok=True)
        if previous_model != self.model_var.get():
            self._set_status("Configuration saved. Model will reload on next prediction.", ONLINE_GREEN)
        else:
            self._set_status("Configuration saved.", ONLINE_GREEN)

    def _reset_configuration_defaults(self) -> None:
        self.config_input_var.set(str(DEFAULT_INPUT_DIR))
        self.config_csv_import_var.set(str(DEFAULT_CSV_DIR))
        self.config_model_var.set(str(DEFAULT_CLASSIFICATION_MODEL))
        self.config_output_var.set(str(DEFAULT_OUTPUT_DIR))
        self._set_status("Default configuration staged. Press Save to apply.", RUNNING_ORANGE)

    def _start_measurement(self) -> None:
        if self.is_running:
            return
        try:
            input_path = Path(self.input_var.get()).expanduser()
            output_path = Path(self.output_var.get()).expanduser()
            pixels_per_mm = float(self.pixels_per_mm_var.get())
            black_threshold = int(self.black_threshold_var.get())
            sobel_threshold = float(self.sobel_threshold_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid Configuration", str(exc))
            return

        if not input_path.exists():
            messagebox.showerror("Input Error", f"Input path not found:\n{input_path}")
            return
        if pixels_per_mm <= 0:
            messagebox.showerror("Invalid Configuration", "Internal calibration value must be greater than 0.")
            return
        output_path.mkdir(parents=True, exist_ok=True)

        self.is_running = True
        self._set_status("Running image classification...", RUNNING_ORANGE)
        self._clear_log()
        self.summary_var.set("Processed: running...")
        self.csv_var.set("CSV: -")
        self.sobel_var.set("Sobel images: -")
        self.annotated_var.set("Annotated images: -")

        command = [
            sys.executable,
            str(PROJECT_ROOT / "router_intrusion_measure.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--pixels-per-mm",
            str(pixels_per_mm),
            "--black-threshold",
            str(black_threshold),
            "--sobel-threshold-ratio",
            str(sobel_threshold),
            "--display-edge-thickness",
            str(int(float(getattr(self, "display_thickness_var", tk.StringVar(value="2")).get()))),
            "--orientation",
            self.orientation_var.get(),
        ]
        threading.Thread(target=self._run_worker, args=(command,), daemon=True).start()

    def _run_worker(self, command: list[str]) -> None:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
        )
        output = "\n".join(part for part in (process.stdout, process.stderr) if part)
        self.root.after(0, self._finish_measurement, process.returncode, output)

    def _finish_measurement(self, return_code: int, output: str) -> None:
        self.is_running = False
        self._append_log(output or "(No process output)")
        if return_code == 0:
            self._set_status("Classification complete.", ONLINE_GREEN)
            self._parse_summary(output)
            self._load_point_cards()
        else:
            self._set_status("Classification failed. Check configuration.", ALERT_RED)

    def _parse_summary(self, output: str) -> None:
        for line in output.splitlines():
            if line.startswith("Processed "):
                self.summary_var.set(line)
            elif line.startswith("CSV:"):
                self.csv_var.set(line)
            elif line.startswith("Sobel edge images:"):
                self.sobel_var.set(line)
            elif line.startswith("Annotated images:"):
                self.annotated_var.set(line)

    def _refresh_dashboard_on_startup(self) -> None:
        self._set_status("Loading production dashboard...", RUNNING_ORANGE)
        self._load_point_cards()
        self._schedule_dashboard_refresh()

    def _load_point_cards(self, preserve_selection: bool = True) -> None:
        self.dashboard_data_signature = self._dashboard_source_signature()
        csv_path = Path(self.output_var.get()).expanduser() / "intrusion_measurements.csv"
        if not csv_path.exists():
            self.dashboard_sessions = []
            self.dashboard_filtered_sessions = []
            self.dashboard_selected_session_key = ""
            self._set_dashboard_counts(total=0, boards=0, passed=0, failed=0, unknown=0)
            self.dashboard_session_summary_var.set("Session: -")
            self._clear_dashboard_cards()
            self._show_dashboard_placeholder(
                "No processed output found yet. Please check the Output path in Configuration."
            )
            self._set_status("Dashboard ready. No output found.", RUNNING_ORANGE)
            return

        rows = self._read_csv_rows(csv_path)
        self.dashboard_sessions = self._build_dashboard_sessions(rows)
        if not self.dashboard_sessions:
            self.dashboard_filtered_sessions = []
            self.dashboard_selected_session_key = ""
            self._set_dashboard_counts(total=0, boards=0, passed=0, failed=0, unknown=0)
            self.dashboard_session_summary_var.set("Session: -")
            self._clear_dashboard_cards()
            self._show_dashboard_placeholder(
                "Processed CSV was found, but no matching images/Sobel previews could be displayed."
            )
            self._set_status("Dashboard ready. No displayable images found.", RUNNING_ORANGE)
            return

        preferred_key = self.dashboard_selected_session_key if preserve_selection else ""
        self._apply_dashboard_search(preferred_key=preferred_key)

        if self.dashboard_filtered_sessions:
            self._set_status("Production dashboard ready.", ONLINE_GREEN)
        else:
            self._set_status("Dashboard ready. No search results.", RUNNING_ORANGE)

    def _schedule_dashboard_refresh(self) -> None:
        if self.dashboard_refresh_after_id is not None:
            self.root.after_cancel(self.dashboard_refresh_after_id)
        self.dashboard_refresh_after_id = self.root.after(
            DASHBOARD_REFRESH_MS,
            self._poll_dashboard_updates,
        )

    def _poll_dashboard_updates(self) -> None:
        self.dashboard_refresh_after_id = None
        signature = self._dashboard_source_signature()
        if signature != self.dashboard_data_signature:
            self._load_point_cards(preserve_selection=True)
        self._schedule_dashboard_refresh()

    def _dashboard_source_signature(self) -> tuple[int, int, int]:
        output_csv = Path(self.output_var.get()).expanduser() / "intrusion_measurements.csv"
        output_mtime = self._file_mtime_ns(output_csv)
        csv_dir = Path(self.csv_import_var.get()).expanduser()
        latest_product_info_mtime = 0
        product_info_count = 0
        if csv_dir.exists():
            for csv_path in csv_dir.rglob("*.csv"):
                product_info_count += 1
                latest_product_info_mtime = max(
                    latest_product_info_mtime,
                    self._file_mtime_ns(csv_path),
                )
        return output_mtime, latest_product_info_mtime, product_info_count

    def _file_mtime_ns(self, path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except OSError:
            return 0

    def _build_dashboard_sessions(self, rows: list[dict[str, str]]) -> list[dict[str, object]]:
        sessions_by_key: dict[str, dict[str, object]] = {}
        for index, row in enumerate(rows, start=1):
            image_value = self._clean_csv_value(row.get("image"))
            if not image_value:
                continue
            image_path = Path(image_value)
            csv_path, csv_rows = self._match_csv_for_image(image_path)
            key = str(csv_path.resolve()) if csv_path else DASHBOARD_UNMATCHED_SESSION
            if key not in sessions_by_key:
                sessions_by_key[key] = self._new_dashboard_session(
                    key=key,
                    csv_path=csv_path,
                    csv_rows=csv_rows,
                    first_image_path=image_path,
                )
            points = sessions_by_key[key]["points"]
            if isinstance(points, list):
                points.append(
                    {
                        "source_index": index,
                        "row": row,
                        "image_path": image_path,
                    }
                )

        sessions = list(sessions_by_key.values())
        for session in sessions:
            points = session.get("points")
            if isinstance(points, list):
                points.sort(
                    key=lambda point: (
                        timestamp_from_name(point["image_path"]) or datetime.min,
                        point["source_index"],
                    )
                )
            product_row = session.get("product_row")
            point_count = len(points) if isinstance(points, list) else 0
            session["point_count"] = point_count
            session["board_count"] = self._session_board_count(product_row, point_count)
            session["search_text"] = self._session_search_text(session)

        return sorted(
            sessions,
            key=lambda session: (
                session.get("end_time") or session.get("start_time") or datetime.min,
                str(session.get("csv_path") or ""),
            ),
            reverse=True,
        )

    def _new_dashboard_session(
        self,
        key: str,
        csv_path: Path | None,
        csv_rows: list[dict[str, str]],
        first_image_path: Path,
    ) -> dict[str, object]:
        product_row = self._primary_product_row(csv_rows)
        start_time, end_time = self._session_time_range(csv_path, csv_rows, first_image_path)
        return {
            "key": key,
            "csv_path": csv_path,
            "csv_rows": csv_rows,
            "product_row": product_row,
            "start_time": start_time,
            "end_time": end_time,
            "points": [],
            "point_count": 0,
            "board_count": 0,
            "search_text": "",
        }

    def _apply_dashboard_search(self, preferred_key: str = "") -> bool:
        query = self.dashboard_active_search_query.lower()
        if query:
            filtered = [
                session
                for session in self.dashboard_sessions
                if query in str(session.get("search_text") or "")
            ]
        else:
            filtered = list(self.dashboard_sessions)

        self.dashboard_filtered_sessions = filtered
        if not filtered:
            self.dashboard_selected_session_key = ""
            self.dashboard_session_summary_var.set("Session: no matching SN")
            self._set_dashboard_counts(total=0, boards=0, passed=0, failed=0, unknown=0)
            self._clear_dashboard_cards()
            self._show_dashboard_placeholder("No matching SN found.")
            return False

        available_keys = {str(session.get("key")) for session in filtered}
        selected_key = preferred_key or self.dashboard_selected_session_key
        if selected_key not in available_keys:
            selected_key = str(filtered[0].get("key"))
        return self._select_dashboard_session(selected_key)

    def _select_dashboard_session(self, key: str) -> bool:
        self.dashboard_selected_session_key = key
        selected_session = None
        for session in self.dashboard_filtered_sessions:
            if str(session.get("key")) == key:
                selected_session = session
                break
        if selected_session is None:
            return False
        self._render_dashboard_session(selected_session)
        return True

    def _clear_dashboard_search(self) -> None:
        self.dashboard_search_var.set("")
        self.dashboard_active_search_query = ""
        self._apply_dashboard_search(preferred_key="")

    def _find_dashboard_search(self) -> None:
        if self.dashboard_finding:
            return
        self.dashboard_finding = True
        self._set_status("Finding Product Info...", RUNNING_ORANGE)
        self._show_dashboard_spinner()
        self.root.after(100, self._perform_dashboard_find)

    def _perform_dashboard_find(self) -> None:
        found = False
        try:
            self.dashboard_active_search_query = self._clean_csv_value(
                self.dashboard_search_var.get()
            )
            if not self.dashboard_sessions:
                self._load_point_cards(preserve_selection=False)
                found = bool(self.dashboard_filtered_sessions)
            else:
                found = self._apply_dashboard_search(preferred_key="")
        finally:
            self.dashboard_finding = False
            self._hide_dashboard_spinner()

        if found:
            self._set_status("Finding complete.", ONLINE_GREEN)
            messagebox.showinfo("Finding Complete", "Finding Complete")
        else:
            self._set_status("Finding incomplete.", RUNNING_ORANGE)
            messagebox.showwarning("Finding Incomplete", "Finding Incomplete")

    def _show_dashboard_spinner(self) -> None:
        if not hasattr(self, "dashboard_spinner_canvas"):
            return
        self.dashboard_spinner_canvas.itemconfigure(self.dashboard_spinner_arc, state="normal")
        self._animate_dashboard_spinner()

    def _hide_dashboard_spinner(self) -> None:
        if self.dashboard_spinner_after_id is not None:
            self.root.after_cancel(self.dashboard_spinner_after_id)
            self.dashboard_spinner_after_id = None
        if hasattr(self, "dashboard_spinner_canvas"):
            self.dashboard_spinner_canvas.itemconfigure(self.dashboard_spinner_arc, state="hidden")

    def _animate_dashboard_spinner(self) -> None:
        if not self.dashboard_finding or not hasattr(self, "dashboard_spinner_canvas"):
            return
        self.dashboard_spinner_angle = (self.dashboard_spinner_angle + 25) % 360
        self.dashboard_spinner_canvas.itemconfigure(
            self.dashboard_spinner_arc,
            start=self.dashboard_spinner_angle,
        )
        self.dashboard_spinner_after_id = self.root.after(80, self._animate_dashboard_spinner)

    def _render_dashboard_session(self, session: dict[str, object]) -> None:
        self._clear_dashboard_cards()
        self.dashboard_session_summary_var.set(self._session_summary_text(session))

        counts = {"PASS": 0, "NG": 0, "UNKNOWN": 0}
        rendered = 0
        points = session.get("points")
        if not isinstance(points, list):
            points = []
        for point_number, point in enumerate(points, start=1):
            image_path = point.get("image_path")
            row = point.get("row")
            if not isinstance(image_path, Path) or not isinstance(row, dict):
                continue
            sobel_path = Path(self.output_var.get()).expanduser() / "sobel_edges" / (
                f"{image_path.stem}_sobel_edge.png"
            )
            classification = self._add_point_card(
                point_number=point_number,
                image_path=image_path,
                sobel_path=sobel_path,
                row=row,
                session=session,
            )
            if classification is None:
                continue
            rendered += 1
            if classification in {"PASS", "NG"}:
                counts[classification] += 1
            else:
                counts["UNKNOWN"] += 1

        boards = int(session.get("board_count") or 0)
        self._set_dashboard_counts(
            total=rendered,
            boards=boards,
            passed=counts["PASS"],
            failed=counts["NG"],
            unknown=counts["UNKNOWN"],
        )
        if rendered == 0:
            self._show_dashboard_placeholder(
                "This Product Info session has no displayable image/Sobel preview."
            )

    def _set_dashboard_counts(
        self,
        total: int,
        boards: int,
        passed: int,
        failed: int,
        unknown: int,
    ) -> None:
        self.dashboard_total_var.set(str(total))
        self.dashboard_boards_var.set(str(boards))
        self.dashboard_pass_var.set(str(passed))
        self.dashboard_ng_var.set(str(failed))
        self.dashboard_unknown_var.set(str(unknown))
        self.dashboard_update_var.set(
            f"Last update: {datetime.now().strftime('%d %B %Y %H:%M:%S')}"
        )

    def _show_dashboard_placeholder(self, message: str) -> None:
        canvas_width = max(self.cards_canvas.winfo_width(), self.cards_view.winfo_width(), 900)
        canvas_height = max(self.cards_canvas.winfo_height(), self.cards_view.winfo_height(), 420)
        placeholder = tk.Frame(
            self.cards_content,
            background=PANEL,
            width=canvas_width,
            height=canvas_height,
        )
        placeholder.grid(row=0, column=0, columnspan=2, sticky="nsew")
        placeholder.grid_propagate(False)
        placeholder.columnconfigure(0, weight=1)
        if message == "No matching SN found.":
            placeholder.rowconfigure(0, weight=3)
            placeholder.rowconfigure(2, weight=2)
        else:
            placeholder.rowconfigure(0, weight=1)
            placeholder.rowconfigure(2, weight=1)
        tk.Label(
            placeholder,
            text=message,
            anchor="center",
            justify="center",
            foreground=MUTED,
            background=PANEL,
            font=("Segoe UI", 15, "bold"),
        ).grid(row=1, column=0, sticky="ew")

    def _clear_dashboard_cards(self) -> None:
        for child in self.cards_content.winfo_children():
            child.destroy()
        self.result_card_photos.clear()

    def _primary_product_row(self, csv_rows: list[dict[str, str]]) -> dict[str, str] | None:
        for row in csv_rows:
            if self._clean_csv_value(row.get("SN")):
                return row
        for row in csv_rows:
            if self._clean_csv_value(row.get("ProductId")):
                return row
        return csv_rows[0] if csv_rows else None

    def _session_board_count(self, product_row: object, fallback_count: int) -> int:
        if isinstance(product_row, dict):
            for field in ("SubBoardCount", "BoardCount", "PanelBoardCount"):
                value = self._clean_csv_value(product_row.get(field))
                if not value:
                    continue
                try:
                    count = int(float(value))
                except ValueError:
                    continue
                if count > 0:
                    return count
        return fallback_count

    def _session_time_range(
        self,
        csv_path: Path | None,
        csv_rows: list[dict[str, str]],
        fallback_image_path: Path,
    ) -> tuple[datetime | None, datetime | None]:
        start_times: list[datetime] = []
        end_times: list[datetime] = []
        for row in csv_rows:
            start_time = self._parse_csv_datetime(
                row.get("Start_time") or row.get("StartTime") or row.get("Start") or row.get("Time")
            )
            end_time = self._parse_csv_datetime(
                row.get("End_time") or row.get("EndTime") or row.get("End") or row.get("Time")
            )
            if start_time:
                start_times.append(start_time)
            if end_time:
                end_times.append(end_time)

        csv_time = timestamp_from_name(csv_path) if csv_path else None
        image_time = timestamp_from_name(fallback_image_path)
        start_time = min(start_times) if start_times else (csv_time or image_time)
        end_time = max(end_times) if end_times else (csv_time or image_time)
        return start_time, end_time

    def _session_search_text(self, session: dict[str, object]) -> str:
        product_row = session.get("product_row")
        csv_path = session.get("csv_path")
        fields = []
        if isinstance(product_row, dict):
            fields.extend(
                self._clean_csv_value(product_row.get(field))
                for field in (
                    "SN",
                    "ProductId",
                    "CutedTable",
                    "Recipe_Name",
                    "ID",
                    "Barcode",
                    "FixtureBarcode",
                    "MachineID",
                )
            )
        if isinstance(csv_path, Path):
            fields.append(csv_path.name)
            fields.append(str(csv_path))
        fields.append(self._format_session_time_range(session.get("start_time"), session.get("end_time")))
        return " ".join(field for field in fields if field).lower()

    def _session_summary_text(self, session: dict[str, object]) -> str:
        product_row = session.get("product_row")
        csv_path = session.get("csv_path")
        csv_name = csv_path.name if isinstance(csv_path, Path) else "Unmatched CSV"
        return (
            f"Session: {csv_name} | "
            f"SN: {self._product_field(product_row, 'SN') or '-'} | "
            f"ProductId: {self._product_field(product_row, 'ProductId') or '-'} | "
            f"Table: {self._product_field(product_row, 'CutedTable') or '-'} | "
            f"Boards: {session.get('board_count') or 0} | "
            f"Points: {session.get('point_count') or 0} | "
            f"Time: {self._format_session_time_range(session.get('start_time'), session.get('end_time'))}"
        )

    def _product_field(self, product_row: object, field: str) -> str:
        if not isinstance(product_row, dict):
            return ""
        return self._clean_csv_value(product_row.get(field))

    def _format_session_time_range(
        self,
        start_time: object,
        end_time: object,
        compact: bool = False,
    ) -> str:
        if not isinstance(start_time, datetime) and not isinstance(end_time, datetime):
            return "-"
        date_format = "%H:%M:%S" if compact else "%Y-%m-%d %H:%M:%S"
        start_text = start_time.strftime(date_format) if isinstance(start_time, datetime) else "-"
        end_text = end_time.strftime(date_format) if isinstance(end_time, datetime) else "-"
        if start_text == end_text:
            return start_text
        return f"{start_text} - {end_text}"

    def _add_point_card(
        self,
        point_number: int,
        image_path: Path,
        sobel_path: Path,
        row: dict[str, str],
        session: dict[str, object],
    ) -> str | None:
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        sobel = cv2.imread(str(sobel_path), cv2.IMREAD_COLOR)
        if original is None or sobel is None:
            return None
        original_photo = tk_image_from_bgr(original, DASHBOARD_POINT_PREVIEW_SIZE)
        sobel_photo = tk_image_from_bgr(sobel, DASHBOARD_POINT_PREVIEW_SIZE)
        self.result_card_photos.append((original_photo, sobel_photo))
        classification_detail, classification = self._classification_detail(
            image_path,
            sobel_path,
            session,
        )

        card = tk.Frame(
            self.cards_content,
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        row_index = (point_number - 1) // 2
        column_index = (point_number - 1) % 2
        card.grid(row=row_index, column=column_index, sticky="nsew", padx=8, pady=8)
        self.cards_content.columnconfigure(0, weight=1)
        self.cards_content.columnconfigure(1, weight=1)

        top = tk.Frame(card, background=PANEL_2)
        top.pack(fill="x", pady=(0, 10))
        tk.Label(
            top,
            text=f"POINT {point_number:03d}",
            foreground="#ffffff",
            background=ACCENT,
            font=("Segoe UI", 11, "bold"),
            padx=10,
            pady=4,
        ).pack(side="left")
        tk.Label(
            top,
            text=classification,
            foreground="#ffffff",
            background=self._classification_color(classification),
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=4,
        ).pack(side="right")
        tk.Label(
            top,
            text="Image Classification",
            foreground=MUTED,
            background=PANEL_2,
            font=("Segoe UI", 10),
        ).pack(side="right", padx=(0, 8))

        preview = tk.Frame(card, background="#111827", padx=8, pady=8)
        preview.pack(fill="x")
        self._add_preview(preview, "Original Image", original_photo)
        self._add_preview(preview, "Sobel Edge Detection", sobel_photo)

        details = (
            f"Image: {image_path.name}\n"
            f"{classification_detail}"
        )
        tk.Label(
            card,
            text=details,
            justify="left",
            anchor="nw",
            foreground=TEXT,
            background=PANEL_2,
            font=("Segoe UI", 10),
            wraplength=DASHBOARD_POINT_DETAIL_WRAP,
        ).pack(fill="x", pady=(10, 0))
        return classification

    def _classification_color(self, classification: str) -> str:
        if classification == "PASS":
            return ONLINE_GREEN
        if classification == "NG":
            return ALERT_RED
        return RUNNING_ORANGE

    def _add_preview(self, parent: tk.Widget, title: str, photo: tk.PhotoImage) -> None:
        panel = tk.Frame(parent, background="#111827")
        panel.pack(side="left", fill="both", expand=True, padx=4)
        tk.Label(
            panel,
            text=title,
            foreground=HEADER_MUTED,
            background="#111827",
            font=("Segoe UI", 9, "bold"),
        ).pack(fill="x", pady=(0, 6))
        tk.Label(panel, image=photo, background="#111827").pack()

    def _classification_detail(
        self,
        image_path: Path,
        sobel_path: Path,
        session: dict[str, object],
    ) -> tuple[str, str]:
        image_time = timestamp_from_name(image_path)
        csv_path = session.get("csv_path")
        csv_rows = session.get("csv_rows")
        if not isinstance(csv_rows, list):
            csv_rows = []
        classification, confidence = self._predict_classification_cached(sobel_path)
        time_range = self._format_session_time_range(
            session.get("start_time"),
            session.get("end_time"),
        )
        image_time_text = image_time.strftime("%Y-%m-%d %H:%M:%S") if image_time else "-"
        csv_file_text = csv_path.name if isinstance(csv_path, Path) else "-"
        confidence_text = f" ({confidence:.1%})" if confidence is not None else ""
        classification_group = classification if classification in {"PASS", "NG"} else "UNKNOWN"
        detail = (
            f"Classification: {classification}{confidence_text}\n"
            f"Image time: {image_time_text}\n"
            f"Matched CSV: {csv_file_text} ({time_range})\n"
            f"Boards in panel: {session.get('board_count') or 0}"
        )
        csv_detail = self._format_machine_csv_detail(csv_rows)
        if csv_detail:
            detail = f"{detail}\nProduct info: {csv_detail}"
        return detail, classification_group

    def _format_machine_csv_detail(self, csv_rows: list[dict[str, str]] | None) -> str:
        if not csv_rows:
            return ""

        csv_row = self._primary_product_row(csv_rows)
        if not csv_row:
            return ""
        fields = ("SN", "ProductId", "CutedTable", "Recipe_Name")
        return " | ".join(
            f"{field}: {self._clean_csv_value(csv_row.get(field)) or '-'}"
            for field in fields
        )

    def _clean_csv_value(self, value: object | None) -> str:
        return str(value or "").strip()

    def _parse_csv_datetime(self, value: object | None) -> datetime | None:
        text = self._clean_csv_value(value)
        if not text:
            return None
        for date_format in (
            "%Y/%m/%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M",
            "%Y-%m-%d %H:%M",
        ):
            try:
                return datetime.strptime(text, date_format)
            except ValueError:
                continue
        return None

    def _predict_classification(self, image_path: Path) -> tuple[str, float | None]:
        model_path = Path(self.model_var.get()).expanduser()
        result = self.classifier.predict(image_path=image_path, model_path=model_path)
        return result.label, result.confidence

    def _predict_classification_cached(self, image_path: Path) -> tuple[str, float | None]:
        model_path = Path(self.model_var.get()).expanduser()
        cache_key = (
            image_path.resolve() if image_path.exists() else image_path,
            self._file_mtime_ns(image_path),
            model_path.resolve() if model_path.exists() else model_path,
            self._file_mtime_ns(model_path),
        )
        cached = self.classification_cache.get(cache_key)
        if cached is not None:
            return cached
        result = self.classifier.predict(image_path=image_path, model_path=model_path)
        prediction = (result.label, result.confidence)
        self.classification_cache[cache_key] = prediction
        return prediction

    def _match_csv_for_image(self, image_path: Path) -> tuple[Path | None, list[dict[str, str]]]:
        csv_dir = Path(self.csv_import_var.get()).expanduser()
        image_time = timestamp_from_name(image_path)
        if image_time is None or not csv_dir.exists():
            return None, []

        range_candidates: list[tuple[datetime, float, Path, list[dict[str, str]]]] = []
        filename_candidates: list[tuple[datetime, Path]] = []
        for csv_path in csv_dir.rglob("*.csv"):
            rows = self._read_csv_rows(csv_path)
            for row in rows:
                start_time = self._parse_csv_datetime(
                    row.get("Start_time") or row.get("StartTime") or row.get("Start")
                )
                end_time = self._parse_csv_datetime(
                    row.get("End_time") or row.get("EndTime") or row.get("End")
                )
                if start_time and end_time and start_time <= image_time <= end_time:
                    duration_seconds = max((end_time - start_time).total_seconds(), 0.0)
                    range_candidates.append((start_time, -duration_seconds, csv_path, rows))

            csv_time = timestamp_from_name(csv_path)
            if csv_time is not None and csv_time <= image_time:
                filename_candidates.append((csv_time, csv_path))

        if range_candidates:
            _start_time, _duration, csv_path, rows = max(
                range_candidates,
                key=lambda item: (item[0], item[1]),
            )
            return csv_path, rows

        if not filename_candidates:
            return None, []

        _csv_time, csv_path = max(filename_candidates, key=lambda item: item[0])
        return csv_path, self._read_csv_rows(csv_path)

    def _read_csv_rows(self, csv_path: Path) -> list[dict[str, str]]:
        try:
            modified_time = csv_path.stat().st_mtime_ns
            cached = self.csv_row_cache.get(csv_path)
            if cached is not None and cached[0] == modified_time:
                return cached[1]
            with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                rows = [
                    {str(key or ""): str(value or "") for key, value in row.items()}
                    for row in reader
                ]
            self.csv_row_cache[csv_path] = (modified_time, rows)
            return rows
        except OSError:
            return []

    def _load_finetune_images(self) -> None:
        try:
            self.finetune_images = find_images(Path(self.input_var.get()).expanduser())
        except (FileNotFoundError, OSError) as exc:
            messagebox.showerror("Input Error", str(exc))
            return
        if not self.finetune_images:
            messagebox.showerror("Input Error", "No supported images found.")
            return
        self.finetune_index = 0
        self._load_finetune_current()

    def _load_finetune_current(self) -> None:
        if not self.finetune_images:
            return
        image_path = self.finetune_images[self.finetune_index]
        self.finetune_current_image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if self.finetune_current_image is None:
            messagebox.showerror("Image Error", f"Could not read image:\n{image_path}")
            return
        self.finetune_count_var.set(
            f"{self.finetune_index + 1} / {len(self.finetune_images)} - {image_path.name}"
        )
        self._set_finetune_state(self.finetune_save_states.get(str(image_path), "Not Save"))
        self._render_finetune_current()

    def _commit_finetune_parameters(self, show_error: bool = True) -> bool:
        try:
            threshold = float(self.sobel_threshold_var.get())
            thickness = int(round(float(self.display_thickness_var.get())))
            blur_kernel = int(round(float(self.blur_kernel_var.get())))
            sobel_kernel = int(round(float(self.sobel_kernel_var.get())))
            cleanup_kernel = int(round(float(self.cleanup_kernel_var.get())))
        except (ValueError, tk.TclError):
            if show_error:
                messagebox.showerror(
                    "Invalid Fine-Tune Value",
                    "Please enter numeric values for all Fine-Tune parameters.",
                )
            return False

        threshold = min(max(threshold, 0.01), 0.50)
        thickness = min(max(thickness, 1), 8)
        blur_kernel = self._normalize_finetune_kernel(blur_kernel, 1, 15)
        sobel_kernel = self._normalize_finetune_kernel(sobel_kernel, 1, 7)
        cleanup_kernel = self._normalize_finetune_kernel(cleanup_kernel, 1, 9)
        self.sobel_threshold_var.set(f"{threshold:.2f}")
        self.display_thickness_var.set(str(thickness))
        self.blur_kernel_var.set(str(blur_kernel))
        self.sobel_kernel_var.set(str(sobel_kernel))
        self.cleanup_kernel_var.set(str(cleanup_kernel))
        if self.edge_mode_var.get() not in FINETUNE_OPTION_VALUES:
            self.edge_mode_var.set("Combined")
        self._render_finetune_current()
        return True

    def _render_finetune_current(self) -> None:
        if self.finetune_current_image is None:
            return
        try:
            threshold = float(self.sobel_threshold_var.get())
            thickness = int(float(self.display_thickness_var.get()))
            blur_kernel = int(float(self.blur_kernel_var.get()))
            sobel_kernel = int(float(self.sobel_kernel_var.get()))
            cleanup_kernel = int(float(self.cleanup_kernel_var.get()))
        except (ValueError, tk.TclError):
            return
        x_edge, y_edge, all_edges = create_sobel_edge_masks(
            self.finetune_current_image,
            threshold,
            blur_kernel_size=blur_kernel,
            sobel_kernel_size=sobel_kernel,
            cleanup_kernel_size=cleanup_kernel,
        )
        selected_edges = self._selected_finetune_edge_mask(x_edge, y_edge, all_edges)
        sobel_bgr = cv2.cvtColor(
            thicken_edge_for_display(selected_edges, thickness),
            cv2.COLOR_GRAY2BGR,
        )
        original_photo = tk_image_from_bgr_fixed(self.finetune_current_image, FINETUNE_PREVIEW_SIZE)
        sobel_photo = tk_image_from_bgr_fixed(sobel_bgr, FINETUNE_PREVIEW_SIZE)
        self.finetune_photos = [original_photo, sobel_photo]
        self.original_preview_label.configure(image=original_photo, text="")
        self.sobel_preview_label.configure(image=sobel_photo, text="")

    def _reset_finetune_parameters(self) -> None:
        self.sobel_threshold_var.set("0.12")
        self.display_thickness_var.set("2")
        self.blur_kernel_var.set("5")
        self.sobel_kernel_var.set("3")
        self.cleanup_kernel_var.set("3")
        self.edge_mode_var.set("Combined")
        self._render_finetune_current()

    def _step_finetune_parameter(self, variable: tk.StringVar, step: float) -> None:
        try:
            current_value = float(variable.get())
        except (ValueError, tk.TclError):
            current_value = 0.0
        next_value = current_value + step
        if abs(step) < 1:
            variable.set(f"{next_value:.2f}")
        else:
            variable.set(str(int(round(next_value))))
        self._commit_finetune_parameters(show_error=False)

    def _normalize_finetune_kernel(self, value: int, minimum: int, maximum: int) -> int:
        kernel_size = min(max(value, minimum), maximum)
        if kernel_size % 2 == 0:
            kernel_size += 1 if kernel_size < maximum else -1
        return kernel_size

    def _selected_finetune_edge_mask(self, x_edge, y_edge, all_edges):
        mode = self.edge_mode_var.get()
        if mode == "Sobel X":
            return x_edge
        if mode == "Sobel Y":
            return y_edge
        return all_edges

    def _previous_finetune_image(self) -> None:
        if not self.finetune_images:
            self._load_finetune_images()
            return
        self.finetune_index = (self.finetune_index - 1) % len(self.finetune_images)
        self._load_finetune_current()

    def _next_finetune_image(self) -> None:
        if not self.finetune_images:
            self._load_finetune_images()
            return
        self.finetune_index = (self.finetune_index + 1) % len(self.finetune_images)
        self._load_finetune_current()

    def _save_finetune_current(self) -> None:
        if self.finetune_current_image is None or not self.finetune_images:
            self._load_finetune_images()
            return
        if not self._commit_finetune_parameters():
            return
        image_path = self.finetune_images[self.finetune_index]
        threshold = float(self.sobel_threshold_var.get())
        thickness = int(float(self.display_thickness_var.get()))
        blur_kernel = int(float(self.blur_kernel_var.get()))
        sobel_kernel = int(float(self.sobel_kernel_var.get()))
        cleanup_kernel = int(float(self.cleanup_kernel_var.get()))
        x_edge, y_edge, all_edges = create_sobel_edge_masks(
            self.finetune_current_image,
            threshold,
            blur_kernel_size=blur_kernel,
            sobel_kernel_size=sobel_kernel,
            cleanup_kernel_size=cleanup_kernel,
        )
        selected_edges = self._selected_finetune_edge_mask(x_edge, y_edge, all_edges)
        sobel_bgr = cv2.cvtColor(
            thicken_edge_for_display(selected_edges, thickness),
            cv2.COLOR_GRAY2BGR,
        )
        output_dir = Path(self.output_var.get()).expanduser() / "finetune_model"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / f"{image_path.stem}_sobel_finetuned.png"), sobel_bgr)
        self.finetune_save_states[str(image_path)] = "Save"
        self._set_finetune_state("Save")

    def _not_save_finetune_current(self) -> None:
        if not self.finetune_images:
            self._load_finetune_images()
            return
        image_path = self.finetune_images[self.finetune_index]
        self.finetune_save_states[str(image_path)] = "Not Save"
        self._set_finetune_state("Not Save")

    def _set_finetune_state(self, state: str) -> None:
        normalized = "Save" if state == "Save" else "Not Save"
        self.finetune_state_var.set(normalized)
        color = ONLINE_GREEN if normalized == "Save" else RUNNING_ORANGE
        if hasattr(self, "finetune_state_canvas"):
            self.finetune_state_canvas.itemconfigure(self.finetune_state_dot, fill=color)

    def _clear_log(self) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _append_log(self, text: str) -> None:
        if not hasattr(self, "log_text"):
            return
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        if not text.endswith("\n"):
            self.log_text.insert("end", "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_status(self, message: str, color: str) -> None:
        self.status_var.set(message)
        if self.status_canvas is not None and self.status_dot is not None:
            self.status_canvas.itemconfigure(self.status_dot, fill=color)

    def _open_output_folder(self) -> None:
        output_path = Path(self.output_var.get()).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(output_path)])


def main() -> int:
    AurotekEdgeDashboard().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
