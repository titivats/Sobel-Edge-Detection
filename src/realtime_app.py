from __future__ import annotations

import argparse
import queue
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, ttk

from app_paths import (
    APP_ICON,
    DEFAULT_CSV_DIR,
    DEFAULT_IMAGE_DIR,
    PROJECT_ROOT,
)
from app_preferences import RealtimePreferences, save_realtime_preferences
from realtime_config import (
    ACCENT,
    ACCENT_DARK,
    BG,
    CARD_BORDER,
    DEFAULT_OUTPUT_DIR,
    DIVIDER,
    HEADER_MUTED,
    HEADER_TEXT,
    MUTED,
    NG_COLOR,
    PANEL,
    PANEL_2,
    PANEL_3,
    PASS_COLOR,
    SIDEBAR,
    SYSTEM_ONLINE,
    TEXT,
    UNKNOWN_COLOR,
)
from realtime_predictor import PredictionView, RealtimePredictor

MAX_RESULT_CARDS = 60
QUEUE_CAPACITY = 200
RESULT_CARD_WIDTH = 570
RESULT_CARD_GAP = 22
HEADER_HEIGHT = 116


@dataclass(frozen=True)
class RuntimeStatus:
    message: str


class RealtimePredictUi:
    """Tkinter app shell: config tab, result cards, and worker thread wiring."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("Realtime Edge Detection Monitor")
        self.root.geometry("1920x1080")
        self.root.minsize(1280, 720)
        self.root.configure(background=BG)
        if APP_ICON.exists():
            self.root.iconbitmap(str(APP_ICON))
        self.queue: queue.Queue[PredictionView | RuntimeStatus | Exception] = queue.Queue(
            maxsize=QUEUE_CAPACITY
        )
        self.result_cards: list[tuple[tk.Frame, ImageTk.PhotoImage]] = []
        self.result_columns = 3
        self.running = True
        self.stop_event = threading.Event()
        self.predictor_lock = threading.Lock()
        self.predictor: RealtimePredictor | None = None
        self.pending_config: tuple[Path, Path, Path] | None = None

        self.status_var = tk.StringVar(value="Starting...")
        self.count_var = tk.StringVar(value="PASS 0   NG 0   UNKNOWN 0")
        self.pass_count_var = tk.StringVar(value="0")
        self.ng_count_var = tk.StringVar(value="0")
        self.unknown_count_var = tk.StringVar(value="0")
        self.clock_var = tk.StringVar(value="")
        self.system_state_var = tk.StringVar(value="INITIALIZING")
        self.pass_count = 0
        self.ng_count = 0
        self.unknown_count = 0
        self.image_dir_var = tk.StringVar(value=str(args.image_dir))
        self.csv_dir_var = tk.StringVar(value=str(args.csv_dir))
        self.output_dir_var = tk.StringVar(value=str(args.output_dir))

        self.configure_style()
        self.build_tabs()
        self.build_result_tab()
        self.build_config_tab()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_clock()

    def configure_style(self) -> None:
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(16, 8, 0, 0))
        self.style.configure(
            "TNotebook.Tab",
            padding=(18, 8),
            font=("Segoe UI", 10, "bold"),
            background="#fee2e2",
            foreground="#7f1d1d",
            borderwidth=1,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", ACCENT_DARK), ("active", "#fecaca")],
            foreground=[("selected", HEADER_TEXT), ("active", "#7f1d1d")],
        )
        self.style.configure("TFrame", background=BG)

    def build_tabs(self) -> None:
        self.notebook = ttk.Notebook(self.root)
        self.result_tab = tk.Frame(self.notebook, background=BG)
        self.config_tab = tk.Frame(self.notebook, background=BG, padx=28, pady=26)
        self.notebook.add(self.result_tab, text="Result")
        self.notebook.add(self.config_tab, text="Config")
        self.notebook.pack(fill="both", expand=True)

    def build_result_tab(self) -> None:
        self.build_result_header()
        self.build_monitor_toolbar()
        viewport = tk.Frame(self.result_tab, background=BG)
        viewport.pack(fill="both", expand=True, padx=16, pady=(0, 14))
        self.canvas = tk.Canvas(
            viewport,
            background=BG,
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
        )
        self.vertical_scrollbar = tk.Scrollbar(
            viewport,
            orient="vertical",
            command=self.canvas.yview,
        )
        self.horizontal_scrollbar = tk.Scrollbar(
            viewport,
            orient="horizontal",
            command=self.canvas.xview,
        )
        self.content = tk.Frame(self.canvas, background=BG)
        self.content.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas_window = self.canvas.create_window(
            (0, 0),
            window=self.content,
            anchor="nw",
        )
        self.canvas.configure(
            yscrollcommand=self.vertical_scrollbar.set,
            xscrollcommand=self.horizontal_scrollbar.set,
        )
        self.canvas.bind("<Configure>", self.resize_content)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Shift-MouseWheel>", self.on_horizontal_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda _event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda _event: self.canvas.yview_scroll(3, "units"))
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.vertical_scrollbar.grid(row=0, column=1, sticky="ns")
        self.horizontal_scrollbar.grid(row=1, column=0, sticky="ew")
        viewport.rowconfigure(0, weight=1)
        viewport.columnconfigure(0, weight=1)

    def build_result_header(self) -> None:
        header = tk.Frame(
            self.result_tab,
            background=SIDEBAR,
            padx=22,
            pady=16,
            height=HEADER_HEIGHT,
        )
        header.pack(fill="x", padx=16, pady=(14, 10))
        header.pack_propagate(False)
        title_area = tk.Frame(header, background=PANEL)
        title_area.configure(background=SIDEBAR)
        title_area.pack(side="left", fill="x", expand=True)
        brand_line = tk.Frame(title_area, background=SIDEBAR)
        brand_line.pack(fill="x")
        tk.Label(
            brand_line,
            text="AUROTEK  |  MACHINE VISION",
            anchor="w",
            font=("Segoe UI", 9, "bold"),
            foreground=HEADER_MUTED,
            background=SIDEBAR,
        ).pack(side="left")
        tk.Label(
            brand_line,
            textvariable=self.clock_var,
            anchor="e",
            font=("Consolas", 10, "bold"),
            foreground=HEADER_TEXT,
            background=SIDEBAR,
        ).pack(side="right")
        tk.Label(
            title_area,
            text="Realtime Edge Detection Monitor",
            anchor="w",
            font=("Segoe UI Semibold", 23),
            foreground=HEADER_TEXT,
            background=SIDEBAR,
        ).pack(fill="x", pady=(5, 0))
        tk.Label(
            title_area,
            textvariable=self.status_var,
            anchor="w",
            font=("Segoe UI", 10),
            foreground=HEADER_MUTED,
            background=SIDEBAR,
        ).pack(fill="x", pady=(3, 0))
        metrics = tk.Frame(header, background=SIDEBAR)
        metrics.pack(side="right")
        self._add_metric(metrics, "PASS", self.pass_count_var, PASS_COLOR)
        self._add_metric(metrics, "NG", self.ng_count_var, NG_COLOR)
        self._add_metric(metrics, "UNKNOWN", self.unknown_count_var, UNKNOWN_COLOR)

    def build_monitor_toolbar(self) -> None:
        toolbar = tk.Frame(
            self.result_tab,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
            padx=16,
            pady=10,
        )
        toolbar.pack(fill="x", padx=16, pady=(0, 10))
        tk.Label(
            toolbar,
            text="●",
            foreground=SYSTEM_ONLINE,
            background=PANEL,
            font=("Segoe UI", 13, "bold"),
        ).pack(side="left")
        tk.Label(
            toolbar,
            textvariable=self.system_state_var,
            foreground="#374151",
            background=PANEL,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left", padx=(6, 18))
        tk.Frame(toolbar, background=DIVIDER, width=1, height=24).pack(side="left")
        tk.Label(
            toolbar,
            text="Automatic inspection  •  PASS / NG classification  •  GPU inference",
            foreground=MUTED,
            background=PANEL,
            font=("Segoe UI", 10),
        ).pack(side="left", padx=18)
        tk.Label(
            toolbar,
            text="LIVE PRODUCTION",
            foreground="#ffffff",
            background=ACCENT_DARK,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=5,
        ).pack(side="right")

    def _add_metric(
        self,
        parent: tk.Widget,
        label: str,
        variable: tk.StringVar,
        color: str,
    ) -> None:
        card = tk.Frame(
            parent,
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
            padx=16,
            pady=8,
        )
        card.pack(side="left", padx=(10, 0))
        tk.Label(
            card,
            textvariable=variable,
            foreground=color,
            background=PANEL_2,
            font=("Segoe UI Semibold", 20),
        ).pack()
        tk.Label(
            card,
            text=label,
            foreground="#7f1d1d",
            background=PANEL_2,
            font=("Segoe UI", 9, "bold"),
        ).pack()

    def build_config_tab(self) -> None:
        title = tk.Label(
            self.config_tab,
            text="MONITOR CONFIGURATION",
            anchor="w",
            font=("Segoe UI Semibold", 22),
            foreground="#111827",
            background=BG,
        )
        title.pack(fill="x", pady=(0, 14))
        tk.Label(
            self.config_tab,
            text="Configure production data sources and result storage. Changes apply without restarting.",
            anchor="w",
            font=("Segoe UI", 10),
            foreground=MUTED,
            background=BG,
        ).pack(fill="x", pady=(0, 18))
        panel = tk.Frame(
            self.config_tab,
            background=PANEL,
            padx=26,
            pady=24,
            highlightthickness=1,
            highlightbackground=CARD_BORDER,
        )
        panel.pack(fill="x")
        tk.Label(
            panel,
            text="Data Sources & Output",
            anchor="w",
            font=("Segoe UI", 18, "bold"),
            foreground="#111827",
            background=PANEL,
        ).pack(fill="x", pady=(0, 20))

        self.add_path_row(
            panel, "Input Image", self.image_dir_var, "Sobel edge detection image folder"
        )
        self.add_path_row(
            panel,
            ".CSV",
            self.csv_dir_var,
            "Product_Info CSV folder used to match image timestamp and naming",
        )
        self.add_path_row(
            panel,
            "Output",
            self.output_dir_var,
            "Folder for predicted images shown and saved by the UI",
        )

        button_row = tk.Frame(panel, background=PANEL)
        button_row.pack(fill="x", pady=(18, 0))
        tk.Button(
            button_row,
            text="Apply",
            command=self.apply_config,
            font=("Segoe UI", 11, "bold"),
            background=ACCENT_DARK,
            foreground="#ffffff",
            activebackground="#991b1b",
            activeforeground="#ffffff",
            relief="flat",
            padx=24,
            pady=8,
        ).pack(side="left")
        tk.Button(
            button_row,
            text="Reset Default",
            command=self.reset_config_defaults,
            font=("Segoe UI", 11),
            background=PANEL_3,
            foreground="#111827",
            activebackground="#fecaca",
            activeforeground="#7f1d1d",
            relief="flat",
            padx=18,
            pady=8,
        ).pack(side="left", padx=(10, 0))
        tk.Label(
            panel,
            text="Configuration is stored locally and restored at the next dashboard startup.",
            anchor="w",
            font=("Segoe UI", 9),
            foreground=MUTED,
            background=PANEL,
        ).pack(fill="x", pady=(18, 0))

    def add_path_row(
        self, parent: tk.Widget, label: str, variable: tk.StringVar, help_text: str
    ) -> None:
        row = tk.Frame(parent, background=PANEL)
        row.pack(fill="x", pady=10)
        tk.Label(
            row,
            text=label,
            width=14,
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            foreground="#111827",
            background=PANEL,
        ).pack(side="left")
        tk.Entry(
            row,
            textvariable=variable,
            font=("Consolas", 11),
            foreground="#111827",
            background=PANEL_2,
            insertbackground="#111827",
            relief="flat",
        ).pack(side="left", fill="x", expand=True, ipady=8)
        tk.Button(
            row,
            text="Browse",
            command=lambda: self.browse_directory(variable),
            font=("Segoe UI", 10),
            background=PANEL_3,
            foreground="#7f1d1d",
            activebackground="#fecaca",
            activeforeground="#7f1d1d",
            relief="flat",
            padx=14,
            pady=7,
        ).pack(side="left", padx=(10, 0))
        tk.Label(
            parent,
            text=help_text,
            anchor="w",
            font=("Segoe UI", 9),
            foreground=MUTED,
            background=PANEL,
        ).pack(fill="x", padx=(148, 0), pady=(0, 4))

    def create_predictor(self) -> RealtimePredictor:
        return RealtimePredictor(
            image_dir=self.args.image_dir,
            csv_dir=self.args.csv_dir,
            model_path=self.args.model,
            output_dir=self.args.output_dir,
            confidence=self.args.confidence,
            sobel_threshold_ratio=self.args.sobel_threshold_ratio,
        )

    def browse_directory(self, variable: tk.StringVar) -> None:
        selected = filedialog.askdirectory(initialdir=variable.get() or str(PROJECT_ROOT))
        if selected:
            variable.set(selected)

    def reset_config_defaults(self) -> None:
        self.image_dir_var.set(str(DEFAULT_IMAGE_DIR))
        self.csv_dir_var.set(str(DEFAULT_CSV_DIR))
        self.output_dir_var.set(str(DEFAULT_OUTPUT_DIR))

    def apply_config(self) -> None:
        image_dir = Path(self.image_dir_var.get()).expanduser()
        csv_dir = Path(self.csv_dir_var.get()).expanduser()
        output_dir = Path(self.output_dir_var.get()).expanduser()
        if not image_dir.exists():
            self.status_var.set(f"Error: Input Image path not found: {image_dir}")
            self.notebook.select(self.result_tab)
            return
        if not csv_dir.exists():
            self.status_var.set(f"Error: CSV path not found: {csv_dir}")
            self.notebook.select(self.result_tab)
            return
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.status_var.set(f"Error: Could not create output folder: {exc}")
            self.notebook.select(self.result_tab)
            return
        with self.predictor_lock:
            self.args.image_dir = image_dir
            self.args.csv_dir = csv_dir
            self.args.output_dir = output_dir
            self.pending_config = (image_dir, csv_dir, output_dir)
        try:
            save_realtime_preferences(
                RealtimePreferences(
                    image_dir=image_dir,
                    csv_dir=csv_dir,
                    output_dir=output_dir,
                )
            )
        except OSError as exc:
            self.status_var.set(f"Config applied but could not be saved: {exc}")
        self.clear_predictions()
        if not self.status_var.get().startswith("Config applied but"):
            self.status_var.set(f"Config queued. Watching: {image_dir}")
        self.notebook.select(self.result_tab)

    def clear_predictions(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        self.result_cards.clear()
        self.pass_count = 0
        self.ng_count = 0
        self.unknown_count = 0
        self.update_counts()
        self.canvas.yview_moveto(0)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize_content(self, event: tk.Event) -> None:
        minimum_dashboard_width = (RESULT_CARD_WIDTH + RESULT_CARD_GAP) * 3
        content_width = max(event.width, minimum_dashboard_width)
        self.canvas.itemconfigure(self.canvas_window, width=content_width)
        columns = max(1, content_width // (RESULT_CARD_WIDTH + RESULT_CARD_GAP))
        if columns != self.result_columns:
            self.result_columns = columns
            self._layout_result_cards()

    def on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120) * 3), "units")

    def on_horizontal_mousewheel(self, event: tk.Event) -> str:
        self.canvas.xview_scroll(int(-1 * (event.delta / 120) * 3), "units")
        return "break"

    def start(self) -> None:
        threading.Thread(target=self.worker_loop, daemon=True).start()
        self.root.after(200, self.consume_queue)
        self.root.mainloop()

    def update_clock(self) -> None:
        self.clock_var.set(datetime.now().strftime("%Y-%m-%d  %H:%M:%S"))
        if self.running:
            self.root.after(1000, self.update_clock)

    def close(self) -> None:
        self.running = False
        self.stop_event.set()
        self.root.destroy()

    def worker_loop(self) -> None:
        try:
            self._queue_item(RuntimeStatus("Loading YOLO classification model..."))
            predictor = self.create_predictor()
            with self.predictor_lock:
                self.predictor = predictor
                pending_config = self.pending_config
                self.pending_config = None
            if pending_config is not None:
                predictor.update_paths(*pending_config)
            self._queue_item(RuntimeStatus(f"Watching: {predictor.image_dir}"))
        except Exception as exc:  # noqa: BLE001 - surface startup errors in the UI.
            self._queue_item(exc)
            self.running = False
            return

        while not self.stop_event.is_set():
            try:
                processed_count = 0
                with self.predictor_lock:
                    predictor = self.predictor
                    pending_config = self.pending_config
                    self.pending_config = None
                if predictor is None:
                    self._queue_item(RuntimeStatus("Waiting for YOLO classification model..."))
                    self.stop_event.wait(max(float(self.args.poll_seconds), 0.5))
                    continue
                if pending_config is not None:
                    predictor.update_paths(*pending_config)
                    self._queue_item(RuntimeStatus(f"Watching: {predictor.image_dir}"))
                for view in predictor.scan_iter():
                    processed_count += 1
                    self._queue_item(view)
                if processed_count:
                    self._queue_item(RuntimeStatus(f"Processed {processed_count} new image(s)"))
            except Exception as exc:  # noqa: BLE001 - surface runtime errors in the UI.
                self._queue_item(exc)
            self.stop_event.wait(max(float(self.args.poll_seconds), 0.5))

    def _queue_item(self, item: PredictionView | RuntimeStatus | Exception) -> None:
        while not self.stop_event.is_set():
            try:
                self.queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def consume_queue(self) -> None:
        items_handled = 0
        while not self.queue.empty() and items_handled < 12:
            item = self.queue.get_nowait()
            items_handled += 1
            if isinstance(item, RuntimeStatus):
                self.status_var.set(item.message)
                self.system_state_var.set("SYSTEM ONLINE")
            elif isinstance(item, Exception):
                self.status_var.set(f"Error: {item}")
                self.system_state_var.set("SYSTEM ALERT")
            else:
                self.add_prediction(item)
        if self.running:
            self.root.after(50 if not self.queue.empty() else 200, self.consume_queue)

    def add_prediction(self, view: PredictionView) -> None:
        rgb = cv2.cvtColor(view.image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((560, 420))
        photo = ImageTk.PhotoImage(image)

        card_shell = tk.Frame(
            self.content,
            background=CARD_BORDER,
            padx=2,
            pady=2,
            width=RESULT_CARD_WIDTH,
            height=530,
        )
        card_shell.grid_propagate(False)
        card = tk.Frame(card_shell, background=PANEL)
        card.pack(fill="both", expand=True)
        card.grid_propagate(False)
        image_panel = tk.Frame(card, background="#111827", padx=8, pady=8)
        image_panel.pack(fill="x")
        tk.Label(image_panel, image=photo, background="#111827").pack()
        labels = (
            f"{view.class_names[0]} ({view.confidence:.1%})"
            if view.class_names
            else "No classification"
        )
        color = (
            PASS_COLOR
            if view.status == "PASS"
            else NG_COLOR if view.status == "NG" else UNKNOWN_COLOR
        )
        info_panel = tk.Frame(card, background=PANEL, padx=10, pady=9)
        info_panel.pack(fill="both", expand=True)
        top_line = tk.Frame(info_panel, background=PANEL)
        top_line.pack(fill="x")
        tk.Label(
            top_line,
            text=f"POINT {view.point_number:03d}",
            foreground="#7f1d1d",
            background=PANEL_3,
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=4,
        ).pack(side="left")
        tk.Label(
            top_line,
            text=view.status,
            foreground="#ffffff",
            background=color,
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=4,
        ).pack(side="right")
        detail = (
            f"{view.image_path.name}\n"
            f"Classification: {labels}\n"
            f"SN: {view.product_info.sn or '-'} | Table: {view.product_info.cuted_table or '-'}\n"
            f"Product: {view.product_info.product_id or '-'} | Recipe: {view.product_info.recipe_name or '-'}\n"
            f"{view.product_info.csv_path.name if view.product_info.csv_path else '-'}"
        )
        tk.Label(
            info_panel,
            text=detail,
            justify="left",
            anchor="nw",
            foreground="#111827",
            background=PANEL,
            font=("Segoe UI", 10),
            wraplength=560,
        ).pack(fill="both", expand=True, pady=(8, 0))
        self.result_cards.append((card_shell, photo))
        if len(self.result_cards) > MAX_RESULT_CARDS:
            oldest_card, _oldest_photo = self.result_cards.pop(0)
            oldest_card.destroy()
        self._layout_result_cards()
        self.increment_count(view.status)
        self.status_var.set(
            f"Latest: {view.image_path.name} -> Point {view.point_number} {view.status}. "
            f"Saved: {view.output_path}"
        )

    def _layout_result_cards(self) -> None:
        for index, (card, _photo) in enumerate(self.result_cards):
            card.grid(
                row=index // self.result_columns,
                column=index % self.result_columns,
                padx=14,
                pady=14,
                sticky="n",
            )

    def increment_count(self, status: str) -> None:
        if status == "PASS":
            self.pass_count += 1
        elif status == "NG":
            self.ng_count += 1
        else:
            self.unknown_count += 1
        self.update_counts()

    def update_counts(self) -> None:
        self.pass_count_var.set(str(self.pass_count))
        self.ng_count_var.set(str(self.ng_count))
        self.unknown_count_var.set(str(self.unknown_count))
        self.count_var.set(
            f"PASS {self.pass_count}   NG {self.ng_count}   UNKNOWN {self.unknown_count}"
        )
