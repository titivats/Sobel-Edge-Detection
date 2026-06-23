from __future__ import annotations

import argparse
import queue
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, ttk

from realtime_config import (
    ACCENT,
    APP_ICON,
    BG,
    DEFAULT_CSV_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_OUTPUT_DIR,
    MUTED,
    NG_COLOR,
    PANEL,
    PANEL_2,
    PANEL_3,
    PASS_COLOR,
    PROJECT_ROOT,
    TEXT,
    UNKNOWN_COLOR,
)
from realtime_predictor import PredictionView, RealtimePredictor


@dataclass(frozen=True)
class RuntimeStatus:
    message: str


class RealtimePredictUi:
    """Tkinter app shell: config tab, result cards, and worker thread wiring."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("Edge Detection Monitor")
        self.root.geometry("1920x1080")
        self.root.minsize(1280, 720)
        self.root.configure(background=BG)
        if APP_ICON.exists():
            self.root.iconbitmap(str(APP_ICON))
        self.queue: queue.Queue[PredictionView | RuntimeStatus | Exception] = queue.Queue()
        self.latest_images: list[ImageTk.PhotoImage] = []
        self.running = True
        self.predictor_lock = threading.Lock()

        self.status_var = tk.StringVar(value="Starting...")
        self.count_var = tk.StringVar(value="PASS 0   NG 0   UNKNOWN 0")
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
        self.create_predictor()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def configure_style(self) -> None:
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.style.configure("TNotebook", background=BG, borderwidth=0, tabmargins=(8, 5, 0, 0))
        self.style.configure(
            "TNotebook.Tab",
            padding=(14, 5),
            font=("Segoe UI", 10, "bold"),
            background="#e5e7eb",
            foreground="#111827",
            borderwidth=1,
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", PANEL), ("active", "#d1d5db")],
            foreground=[("selected", TEXT), ("active", "#111827")],
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
        self.canvas = tk.Canvas(self.result_tab, background=BG, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self.result_tab, orient="vertical", command=self.canvas.yview)
        self.content = tk.Frame(self.canvas, background=BG)
        self.content.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.content, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.bind("<Configure>", self.resize_content)
        self.canvas.bind_all("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind_all("<Button-4>", lambda _event: self.canvas.yview_scroll(-3, "units"))
        self.canvas.bind_all("<Button-5>", lambda _event: self.canvas.yview_scroll(3, "units"))
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

    def build_result_header(self) -> None:
        header = tk.Frame(self.result_tab, background=PANEL, padx=18, pady=14)
        header.pack(fill="x", padx=16, pady=(14, 8))
        title_area = tk.Frame(header, background=PANEL)
        title_area.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_area,
            text="Edge Detection Monitor",
            anchor="w",
            font=("Segoe UI", 19, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).pack(fill="x")
        tk.Label(
            title_area,
            textvariable=self.status_var,
            anchor="w",
            font=("Segoe UI", 10),
            foreground=MUTED,
            background=PANEL,
        ).pack(fill="x", pady=(3, 0))

        tk.Label(
            header,
            textvariable=self.count_var,
            font=("Segoe UI", 12, "bold"),
            foreground=ACCENT,
            background=PANEL_2,
            padx=18,
            pady=10,
        ).pack(side="right")

    def build_config_tab(self) -> None:
        panel = tk.Frame(self.config_tab, background=PANEL, padx=22, pady=20)
        panel.pack(fill="x")
        tk.Label(
            panel,
            text="Path Config",
            anchor="w",
            font=("Segoe UI", 18, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).pack(fill="x", pady=(0, 20))

        self.add_path_row(panel, "Input Image", self.image_dir_var, "Sobel edge detection image folder")
        self.add_path_row(panel, ".CSV", self.csv_dir_var, "Product_Info CSV folder used to match image timestamp and naming")
        self.add_path_row(panel, "Output", self.output_dir_var, "Folder for predicted images shown and saved by the UI")

        button_row = tk.Frame(panel, background=PANEL)
        button_row.pack(fill="x", pady=(18, 0))
        tk.Button(
            button_row,
            text="Apply",
            command=self.apply_config,
            font=("Segoe UI", 11, "bold"),
            background=ACCENT,
            foreground="#ffffff",
            activebackground="#60a5fa",
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
            foreground=TEXT,
            activebackground="#334155",
            activeforeground="#ffffff",
            relief="flat",
            padx=18,
            pady=8,
        ).pack(side="left", padx=(10, 0))

    def add_path_row(self, parent: tk.Widget, label: str, variable: tk.StringVar, help_text: str) -> None:
        row = tk.Frame(parent, background=PANEL)
        row.pack(fill="x", pady=10)
        tk.Label(
            row,
            text=label,
            width=14,
            anchor="w",
            font=("Segoe UI", 12, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).pack(side="left")
        tk.Entry(
            row,
            textvariable=variable,
            font=("Consolas", 11),
            foreground=TEXT,
            background=PANEL_2,
            insertbackground=TEXT,
            relief="flat",
        ).pack(side="left", fill="x", expand=True, ipady=8)
        tk.Button(
            row,
            text="Browse",
            command=lambda: self.browse_directory(variable),
            font=("Segoe UI", 10),
            background=PANEL_3,
            foreground=TEXT,
            activebackground="#334155",
            activeforeground="#ffffff",
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

    def create_predictor(self) -> None:
        self.predictor = RealtimePredictor(
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
        output_dir.mkdir(parents=True, exist_ok=True)
        with self.predictor_lock:
            self.predictor.update_paths(image_dir=image_dir, csv_dir=csv_dir, output_dir=output_dir)
        self.clear_predictions()
        self.status_var.set(f"Config applied. Watching: {image_dir}")
        self.notebook.select(self.result_tab)

    def clear_predictions(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        self.latest_images.clear()
        self.pass_count = 0
        self.ng_count = 0
        self.unknown_count = 0
        self.update_counts()
        self.canvas.yview_moveto(0)
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def resize_content(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self.canvas_window, width=event.width)

    def on_mousewheel(self, event: tk.Event) -> None:
        self.canvas.yview_scroll(int(-1 * (event.delta / 120) * 3), "units")

    def start(self) -> None:
        threading.Thread(target=self.worker_loop, daemon=True).start()
        self.root.after(200, self.consume_queue)
        self.root.mainloop()

    def close(self) -> None:
        self.running = False
        self.root.destroy()

    def worker_loop(self) -> None:
        while self.running:
            try:
                processed_count = 0
                with self.predictor_lock:
                    for view in self.predictor.scan_iter():
                        processed_count += 1
                        self.queue.put(view)
                if processed_count:
                    self.queue.put(RuntimeStatus(f"Processed {processed_count} new image(s)"))
            except Exception as exc:  # noqa: BLE001 - surface runtime errors in the UI.
                self.queue.put(exc)
            threading.Event().wait(max(float(self.args.poll_seconds), 0.5))

    def consume_queue(self) -> None:
        items_handled = 0
        while not self.queue.empty() and items_handled < 12:
            item = self.queue.get_nowait()
            items_handled += 1
            if isinstance(item, RuntimeStatus):
                self.status_var.set(item.message)
            elif isinstance(item, Exception):
                self.status_var.set(f"Error: {item}")
            else:
                self.add_prediction(item)
        if self.running:
            self.root.after(50 if not self.queue.empty() else 200, self.consume_queue)

    def add_prediction(self, view: PredictionView) -> None:
        rgb = cv2.cvtColor(view.image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((560, 420))
        photo = ImageTk.PhotoImage(image)
        self.latest_images.append(photo)

        row = len(self.latest_images) - 1
        card = tk.Frame(self.content, background=PANEL, padx=1, pady=1, width=600, height=520)
        card.grid(row=row // 3, column=row % 3, padx=14, pady=14, sticky="n")
        card.grid_propagate(False)
        image_panel = tk.Frame(card, background="#0f172a", padx=8, pady=8)
        image_panel.pack(fill="x")
        tk.Label(image_panel, image=photo, background="#0f172a").pack()
        labels = ", ".join(view.class_names) if view.class_names else "No detection"
        color = PASS_COLOR if view.status == "PASS" else NG_COLOR if view.status == "NG" else UNKNOWN_COLOR
        info_panel = tk.Frame(card, background=PANEL, padx=10, pady=9)
        info_panel.pack(fill="both", expand=True)
        top_line = tk.Frame(info_panel, background=PANEL)
        top_line.pack(fill="x")
        tk.Label(
            top_line,
            text=f"POINT {view.point_number:03d}",
            foreground=TEXT,
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
            f"{labels}\n"
            f"SN: {view.product_info.sn or '-'} | CutedTable: {view.product_info.cuted_table or '-'}\n"
            f"Product: {view.product_info.product_id or '-'} | Recipe: {view.product_info.recipe_name or '-'}\n"
            f"{view.product_info.csv_path.name if view.product_info.csv_path else '-'}"
        )
        tk.Label(
            info_panel,
            text=detail,
            justify="left",
            anchor="nw",
            foreground=TEXT,
            background=PANEL,
            font=("Segoe UI", 10),
            wraplength=560,
        ).pack(fill="both", expand=True, pady=(8, 0))
        self.increment_count(view.status)
        self.status_var.set(
            f"Latest: {view.image_path.name} -> Point {view.point_number} {view.status}. "
            f"Saved: {view.output_path}"
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
        self.count_var.set(f"PASS {self.pass_count}   NG {self.ng_count}   UNKNOWN {self.unknown_count}")
