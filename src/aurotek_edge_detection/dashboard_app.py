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


FINETUNE_PREVIEW_SIZE = (700, 560)


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
        self.dashboard_total_var = tk.StringVar(value="0")
        self.dashboard_pass_var = tk.StringVar(value="0")
        self.dashboard_ng_var = tk.StringVar(value="0")
        self.dashboard_unknown_var = tk.StringVar(value="0")
        self.dashboard_update_var = tk.StringVar(value="Last update: -")

        self.is_running = False
        self.status_dot: int | None = None
        self.status_canvas: tk.Canvas | None = None
        self.result_card_photos: list[tuple[tk.PhotoImage, tk.PhotoImage]] = []
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
        tk.Label(
            status,
            textvariable=self.status_var,
            anchor="e",
            font=("Segoe UI", 10, "bold"),
            foreground="#ffffff",
            background=SIDEBAR,
            wraplength=440,
        ).pack(side="left")

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
        self.root.after(250, self._refresh_dashboard_on_startup)

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
            variable=self.input_var,
            button_text="Browse",
            command=lambda: self._browse_directory(self.input_var),
        )
        self._add_path_row(
            config_panel,
            row=2,
            label="Import .CSV",
            variable=self.csv_import_var,
            button_text="Browse",
            command=lambda: self._browse_directory(self.csv_import_var),
        )
        self._add_path_row(
            config_panel,
            row=3,
            label="Models Select",
            variable=self.model_var,
            button_text="Browse",
            command=lambda: self._browse_file(
                self.model_var,
                [("PyTorch model", "*.pt"), ("All files", "*.*")],
            ),
        )
        self._add_path_row(
            config_panel,
            row=4,
            label="Output",
            variable=self.output_var,
            button_text="Browse",
            command=lambda: self._browse_directory(self.output_var),
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
            pady=14,
        )
        summary_panel.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        summary_panel.columnconfigure(0, weight=1)

        title_block = tk.Frame(summary_panel, background=PANEL)
        title_block.grid(row=0, column=0, sticky="w")
        tk.Label(
            title_block,
            text="Production Inspection Overview",
            anchor="w",
            font=("Segoe UI Semibold", 18),
            foreground=TEXT,
            background=PANEL,
        ).pack(anchor="w")
        tk.Label(
            title_block,
            textvariable=self.dashboard_update_var,
            anchor="w",
            font=("Segoe UI", 10),
            foreground=MUTED,
            background=PANEL,
        ).pack(anchor="w", pady=(3, 0))

        kpi_row = tk.Frame(summary_panel, background=PANEL)
        kpi_row.grid(row=0, column=1, sticky="e")
        self._add_kpi_card(kpi_row, "TOTAL", self.dashboard_total_var, TEXT)
        self._add_kpi_card(kpi_row, "PASS", self.dashboard_pass_var, ONLINE_GREEN)
        self._add_kpi_card(kpi_row, "NG", self.dashboard_ng_var, ALERT_RED)
        self._add_kpi_card(kpi_row, "UNKNOWN", self.dashboard_unknown_var, RUNNING_ORANGE)

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
            text="Fine-Tune Model / Sobel Edge Detection",
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

        self.display_thickness_var = tk.StringVar(value="2")

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
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=10,
        )
        state_content.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        self.finetune_state_canvas = tk.Canvas(
            state_content, width=18, height=18, background=PANEL_2, highlightthickness=0
        )
        self.finetune_state_canvas.pack(side="left", padx=(0, 8))
        self.finetune_state_dot = self.finetune_state_canvas.create_oval(
            3, 3, 15, 15, fill=RUNNING_ORANGE, outline=BORDER
        )
        tk.Label(
            state_content,
            textvariable=self.finetune_state_var,
            foreground=TEXT,
            background=PANEL_2,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")
        tk.Label(
            state_content,
            textvariable=self.finetune_count_var,
            foreground=MUTED,
            background=PANEL_2,
            font=("Segoe UI", 10),
            anchor="e",
            justify="right",
            wraplength=230,
        ).pack(side="right", fill="x", expand=True)

        adjustment_panel = tk.Frame(
            parameter_panel,
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        adjustment_panel.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        adjustment_panel.columnconfigure(1, weight=1)
        tk.Label(
            adjustment_panel,
            text="Adjust",
            anchor="w",
            font=("Segoe UI Semibold", 12),
            foreground=ACCENT_DARK,
            background=PANEL_2,
        ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        self._add_finetune_slider_row(
            adjustment_panel,
            1,
            "Sobel Threshold",
            self.sobel_threshold_var,
            0.01,
            0.50,
            0.01,
        )
        self._add_finetune_slider_row(
            adjustment_panel,
            2,
            "Display Thickness",
            self.display_thickness_var,
            1,
            8,
            1,
        )

        image_actions = tk.Frame(
            parameter_panel,
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        image_actions.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        image_actions.columnconfigure(0, weight=1, uniform="finetune_actions")
        image_actions.columnconfigure(1, weight=1, uniform="finetune_actions")
        tk.Label(
            image_actions,
            text="Images",
            anchor="w",
            font=("Segoe UI Semibold", 12),
            foreground=ACCENT_DARK,
            background=PANEL_2,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
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
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        decision_actions.grid(row=4, column=0, sticky="ew")
        decision_actions.columnconfigure(0, weight=1, uniform="finetune_actions")
        decision_actions.columnconfigure(1, weight=1, uniform="finetune_actions")
        tk.Label(
            decision_actions,
            text="Decision",
            anchor="w",
            font=("Segoe UI Semibold", 12),
            foreground=ACCENT_DARK,
            background=PANEL_2,
        ).grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))
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
        panel = tk.Frame(parent, background="#111827", padx=10, pady=10)
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
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=HEADER_MUTED,
            background="#111827",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 8))
        image_slot = tk.Frame(
            panel,
            background="#0f172a",
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
            background="#0f172a",
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
            font=("Segoe UI", 11, "bold" if foreground == "#ffffff" else "normal"),
            pady=9,
        ).grid(
            row=row,
            column=column,
            columnspan=span,
            sticky="ew",
            padx=padx,
            pady=(0, 8),
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
    ) -> None:
        background = str(parent.cget("background"))
        tk.Label(
            parent,
            text=label,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=background,
        ).grid(row=row, column=0, sticky="w", pady=(0, 14))
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
        ).grid(row=row, column=1, sticky="ew", padx=(12, 10), pady=(0, 14))
        entry = tk.Entry(
            parent,
            textvariable=variable,
            background=PANEL,
            foreground=TEXT,
            insertbackground=TEXT,
            relief="solid",
            borderwidth=1,
            font=("Consolas", 11),
            width=8,
            justify="center",
        )
        entry.grid(row=row, column=2, sticky="ew", pady=(0, 14), ipady=7)
        entry.bind("<Return>", lambda _event: self._commit_finetune_parameters())
        entry.bind("<FocusOut>", lambda _event: self._commit_finetune_parameters(show_error=False))

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
            self.classifier.reset()

    def _save_configuration(self) -> None:
        self.classifier.reset()
        Path(self.output_var.get()).expanduser().mkdir(parents=True, exist_ok=True)
        self._set_status("Configuration saved.", ONLINE_GREEN)

    def _reset_configuration_defaults(self) -> None:
        self.input_var.set(str(DEFAULT_INPUT_DIR))
        self.csv_import_var.set(str(DEFAULT_CSV_DIR))
        self.model_var.set(str(DEFAULT_CLASSIFICATION_MODEL))
        self.output_var.set(str(DEFAULT_OUTPUT_DIR))
        self.classifier.reset()
        self._set_status("Configuration reset to default.", ONLINE_GREEN)

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

    def _load_point_cards(self) -> None:
        for child in self.cards_content.winfo_children():
            child.destroy()
        self.result_card_photos.clear()
        csv_path = Path(self.output_var.get()).expanduser() / "intrusion_measurements.csv"
        if not csv_path.exists():
            self._set_dashboard_counts(total=0, passed=0, failed=0, unknown=0)
            self._show_dashboard_placeholder(
                "No processed output found yet. Please check the Output path in Configuration."
            )
            self._set_status("Dashboard ready. No output found.", RUNNING_ORANGE)
            return
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        counts = {"PASS": 0, "NG": 0, "UNKNOWN": 0}
        rendered = 0
        for index, row in enumerate(rows, start=1):
            image_path = Path(row["image"])
            sobel_path = Path(self.output_var.get()).expanduser() / "sobel_edges" / (
                f"{image_path.stem}_sobel_edge.png"
            )
            classification = self._add_point_card(index, image_path, sobel_path, row)
            if classification is None:
                continue
            rendered += 1
            if classification in {"PASS", "NG"}:
                counts[classification] += 1
            else:
                counts["UNKNOWN"] += 1

        self._set_dashboard_counts(
            total=rendered,
            passed=counts["PASS"],
            failed=counts["NG"],
            unknown=counts["UNKNOWN"],
        )
        if rendered == 0:
            self._show_dashboard_placeholder(
                "Processed CSV was found, but no matching images/Sobel previews could be displayed."
            )
            self._set_status("Dashboard ready. No displayable images found.", RUNNING_ORANGE)
        else:
            self._set_status("Production dashboard ready.", ONLINE_GREEN)

    def _set_dashboard_counts(self, total: int, passed: int, failed: int, unknown: int) -> None:
        self.dashboard_total_var.set(str(total))
        self.dashboard_pass_var.set(str(passed))
        self.dashboard_ng_var.set(str(failed))
        self.dashboard_unknown_var.set(str(unknown))
        self.dashboard_update_var.set(
            f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    def _show_dashboard_placeholder(self, message: str) -> None:
        tk.Label(
            self.cards_content,
            text=message,
            anchor="center",
            justify="center",
            foreground=MUTED,
            background=PANEL,
            font=("Segoe UI", 15, "bold"),
            padx=30,
            pady=80,
        ).grid(row=0, column=0, sticky="nsew")

    def _add_point_card(
        self,
        point_number: int,
        image_path: Path,
        sobel_path: Path,
        row: dict[str, str],
    ) -> str | None:
        original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        sobel = cv2.imread(str(sobel_path), cv2.IMREAD_COLOR)
        if original is None or sobel is None:
            return None
        original_photo = tk_image_from_bgr(original, (430, 300))
        sobel_photo = tk_image_from_bgr(sobel, (430, 300))
        self.result_card_photos.append((original_photo, sobel_photo))
        classification_detail, classification = self._classification_detail(image_path, sobel_path)

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
            wraplength=820,
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

    def _classification_detail(self, image_path: Path, sobel_path: Path) -> tuple[str, str]:
        image_time = timestamp_from_name(image_path)
        csv_path, csv_rows = self._match_csv_for_image(image_path)
        csv_row = csv_rows[0] if csv_rows else None
        classification, confidence = self._predict_classification(sobel_path)
        csv_time = "-"
        if csv_row:
            csv_time = (
                csv_row.get("End_time")
                or csv_row.get("Start_time")
                or csv_row.get("Time")
                or "-"
            )
        image_time_text = image_time.strftime("%Y-%m-%d %H:%M:%S") if image_time else "-"
        csv_file_text = csv_path.name if csv_path else "-"
        confidence_text = f" ({confidence:.1%})" if confidence is not None else ""
        classification_group = classification if classification in {"PASS", "NG"} else "UNKNOWN"
        detail = (
            f"Classification: {classification}{confidence_text}\n"
            f"Image time: {image_time_text}\n"
            f"Matched CSV: {csv_file_text} ({csv_time})"
        )
        csv_detail = self._format_machine_csv_detail(csv_rows)
        if csv_detail:
            detail = f"{detail}\nProduct info: {csv_detail}"
        return detail, classification_group

    def _format_machine_csv_detail(self, csv_rows: list[dict[str, str]] | None) -> str:
        if not csv_rows:
            return ""

        csv_row = csv_rows[0]
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
            with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                return [
                    {str(key or ""): str(value or "") for key, value in row.items()}
                    for row in reader
                ]
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
        except (ValueError, tk.TclError):
            if show_error:
                messagebox.showerror(
                    "Invalid Fine-Tune Value",
                    "Please enter numeric values for Sobel Threshold and Display Thickness.",
                )
            return False

        threshold = min(max(threshold, 0.01), 0.50)
        thickness = min(max(thickness, 1), 8)
        self.sobel_threshold_var.set(f"{threshold:.2f}")
        self.display_thickness_var.set(str(thickness))
        self._render_finetune_current()
        return True

    def _render_finetune_current(self) -> None:
        if self.finetune_current_image is None:
            return
        try:
            threshold = float(self.sobel_threshold_var.get())
            thickness = int(float(self.display_thickness_var.get()))
        except (ValueError, tk.TclError):
            return
        _x_edge, _y_edge, all_edges = create_sobel_edge_masks(
            self.finetune_current_image,
            threshold,
        )
        sobel_bgr = cv2.cvtColor(thicken_edge_for_display(all_edges, thickness), cv2.COLOR_GRAY2BGR)
        original_photo = tk_image_from_bgr_fixed(self.finetune_current_image, FINETUNE_PREVIEW_SIZE)
        sobel_photo = tk_image_from_bgr_fixed(sobel_bgr, FINETUNE_PREVIEW_SIZE)
        self.finetune_photos = [original_photo, sobel_photo]
        self.original_preview_label.configure(image=original_photo, text="")
        self.sobel_preview_label.configure(image=sobel_photo, text="")

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
        _x_edge, _y_edge, all_edges = create_sobel_edge_masks(self.finetune_current_image, threshold)
        sobel_bgr = cv2.cvtColor(thicken_edge_for_display(all_edges, thickness), cv2.COLOR_GRAY2BGR)
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
