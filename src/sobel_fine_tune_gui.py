from __future__ import annotations

import argparse
import base64
import json
import tkinter as tk
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

from app_paths import (
    DEFAULT_FINE_TUNE_IMAGE_DIR,
    DEFAULT_RECIPE_DIR,
    DEFAULT_TUNING_OUTPUT_DIR,
    PROJECT_ROOT,
    SOBEL_FINE_TUNE_ICON,
)
from edge_view import select_edge
from file_io import atomic_write_image, atomic_write_text, read_image
from image_file_discovery import find_images
from recipe_store import TuneSettings as RecipeTuneSettings
from recipe_store import safe_recipe_name, save_recipe
from sobel_edge_detection import create_sobel_edge_masks, thicken_edge_for_display

DISPLAY_MAX_WIDTH = 620
DISPLAY_MAX_HEIGHT = 560
WINDOW_BG = "#f3f4f6"
IMAGE_BG = "#111827"
PANEL = "#ffffff"
PANEL_2 = "#fff7f7"
RED = "#b91c1c"
RED_DARK = "#991b1b"
RED_LIGHT = "#fee2e2"
RED_TEXT = "#7f1d1d"
BORDER = "#d1d5db"
TEXT = "#111827"
SAVE_GREEN = "#22c55e"
NOT_SAVE_ORANGE = "#f59e0b"
BUTTON_WIDTH = 18
FORM_LABEL_WIDTH = 18


@dataclass
class GuiTuneSettings:
    sobel_threshold_ratio: float = 0.120
    black_threshold: int = 55
    display_thickness: int = 1
    edge_close_kernel: int = 1
    edge_close_iterations: int = 0
    edge_dilate_iterations: int = 0
    view_mode: int = 0

    def to_recipe_settings(self) -> RecipeTuneSettings:
        return RecipeTuneSettings(
            sobel_threshold_ratio=self.sobel_threshold_ratio,
            display_thickness=self.display_thickness,
            edge_close_kernel=self.edge_close_kernel,
            edge_close_iterations=self.edge_close_iterations,
            edge_dilate_iterations=self.edge_dilate_iterations,
            black_threshold=self.black_threshold,
            x_min_ratio=0.0,
            x_max_ratio=1.0,
            y_min_ratio=0.0,
            y_max_ratio=1.0,
            view_mode=self.view_mode,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sobel Fine Tune UI with image upload and side-by-side preview."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional image file or folder to load at startup.",
    )
    parser.add_argument(
        "--program-name",
        default="MANUAL_TUNED",
        help="Program/Product recipe name. Example: 160914002C01.rcp.",
    )
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=DEFAULT_RECIPE_DIR,
        help="Folder for saved tuned recipes. Default: settings.",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=DEFAULT_TUNING_OUTPUT_DIR,
        help="Folder for saved Sobel fine-tune images. Default: Output_files/tuning_saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = SobelFineTuneApp(args)
    app.run()
    return 0


class SobelFineTuneApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.root = tk.Tk()
        self.root.title("Fine Tune Image for Sobel Edge Detection")
        self.root.geometry("1500x860")
        self.root.minsize(1200, 760)
        self.root.rowconfigure(2, weight=1)
        self.root.configure(background=WINDOW_BG)
        if SOBEL_FINE_TUNE_ICON.exists():
            self.root.iconbitmap(str(SOBEL_FINE_TUNE_ICON))

        self.image_paths: list[Path] = []
        self.image_index = 0
        self.current_image: np.ndarray | None = None
        self.photo_refs: list[tk.PhotoImage] = []
        self.saved_settings: dict[str, GuiTuneSettings] = {}
        self.image_save_states: dict[str, str] = {}
        self.black_threshold = 55

        self.program_name_var = tk.StringVar(value=args.program_name)
        startup_input = args.input
        default_input = startup_input or DEFAULT_FINE_TUNE_IMAGE_DIR
        self.input_image_path_var = tk.StringVar(value=str(default_input))
        self.output_image_dir_var = tk.StringVar(value=str(args.output_image_dir))
        self.status_var = tk.StringVar(
            value="Select input image path to begin. Actual selected part: none."
        )
        self.save_state_var = tk.StringVar(value="Not Save")
        self.image_count_var = tk.StringVar(value="0 / 0")

        self.sobel_threshold_var = tk.DoubleVar(value=0.120)
        self.display_thickness_var = tk.IntVar(value=1)
        self.close_kernel_var = tk.IntVar(value=1)
        self.close_iter_var = tk.IntVar(value=0)
        self.dilate_iter_var = tk.IntVar(value=0)
        self.view_mode_var = tk.IntVar(value=0)

        self._configure_style()
        self._build_layout()
        self.root.bind("<Alt-Left>", lambda _event: self._previous_image())
        self.root.bind("<Alt-Right>", lambda _event: self._next_image())
        self.root.bind("<Control-s>", lambda _event: self._save_current())
        self.root.after(100, lambda: self._load_input_path(default_input))

    def run(self) -> None:
        self.root.mainloop()

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background=WINDOW_BG)
        style.configure("TLabel", background=WINDOW_BG, foreground=TEXT)
        style.configure(
            "TButton",
            padding=(10, 7),
            font=("Segoe UI", 10),
            background=PANEL,
            foreground=RED_TEXT,
            borderwidth=1,
            relief="solid",
        )
        style.map(
            "TButton",
            background=[("active", RED_LIGHT), ("pressed", "#fecaca")],
            foreground=[("active", RED_TEXT), ("pressed", RED_DARK)],
        )
        style.configure(
            "Primary.TButton",
            padding=(10, 7),
            font=("Segoe UI", 10, "bold"),
            background=RED,
            foreground="#ffffff",
            borderwidth=1,
            relief="solid",
        )
        style.map(
            "Primary.TButton",
            background=[("active", RED_DARK), ("pressed", "#7f1d1d")],
            foreground=[("disabled", "#d1d5db")],
        )
        style.configure(
            "TEntry",
            padding=5,
            fieldbackground=PANEL_2,
            foreground=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            borderwidth=1,
            relief="solid",
        )
        style.map(
            "TEntry",
            bordercolor=[("focus", RED), ("!focus", BORDER)],
            lightcolor=[("focus", RED), ("!focus", BORDER)],
            darkcolor=[("focus", RED), ("!focus", BORDER)],
        )
        style.configure("Image.TLabel", background=IMAGE_BG, foreground="#ffffff")
        style.configure(
            "FineTune.TRadiobutton",
            background=PANEL_2,
            foreground=TEXT,
            font=("Segoe UI", 10),
        )
        style.map(
            "FineTune.TRadiobutton",
            background=[("active", PANEL_2), ("selected", PANEL_2)],
            foreground=[("active", RED_TEXT), ("selected", TEXT)],
        )

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        header = tk.Frame(self.root, background=RED_DARK, padx=20, pady=14)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        tk.Label(
            header,
            text="Fine Tune Image for Sobel Edge Detection",
            anchor="w",
            font=("Segoe UI Semibold", 22),
            foreground="#ffffff",
            background=RED_DARK,
        ).grid(row=0, column=0, sticky="ew")
        status_area = tk.Frame(header, background=RED_DARK)
        status_area.grid(row=0, column=1, sticky="e", padx=(20, 0))
        self.save_state_canvas = tk.Canvas(
            status_area,
            width=16,
            height=16,
            background=RED_DARK,
            highlightthickness=0,
            bd=0,
        )
        self.save_state_canvas.grid(row=0, column=0, sticky="e", padx=(0, 8))
        self.save_state_dot = self.save_state_canvas.create_oval(
            3,
            3,
            13,
            13,
            fill=NOT_SAVE_ORANGE,
            outline="#ffffff",
        )
        tk.Label(
            status_area,
            textvariable=self.save_state_var,
            anchor="e",
            font=("Segoe UI", 10, "bold"),
            foreground="#ffffff",
            background=RED_DARK,
        ).grid(row=0, column=1, sticky="e", padx=(0, 14))
        tk.Label(
            status_area,
            textvariable=self.status_var,
            anchor="e",
            font=("Segoe UI", 10),
            foreground="#fecaca",
            background=RED_DARK,
            wraplength=720,
        ).grid(row=0, column=2, sticky="e")

        form_panel = tk.Frame(
            self.root,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=18,
            pady=14,
        )
        form_panel.grid(row=1, column=0, sticky="ew", padx=10, pady=(10, 10))
        form_panel.columnconfigure(1, weight=1)

        tk.Label(
            form_panel,
            text="Program name",
            width=FORM_LABEL_WIDTH,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(form_panel, textvariable=self.program_name_var, width=28).grid(
            row=0,
            column=1,
            sticky="w",
            padx=(8, 0),
            pady=(0, 8),
            ipady=2,
        )

        tk.Label(
            form_panel,
            text="Input image path",
            width=FORM_LABEL_WIDTH,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(form_panel, textvariable=self.input_image_path_var).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(8, 8),
            pady=(0, 8),
            ipady=2,
        )
        ttk.Button(
            form_panel,
            text="Image",
            command=self._select_input_image_file,
            width=10,
        ).grid(
            row=1,
            column=2,
            padx=(0, 6),
            pady=(0, 8),
        )
        ttk.Button(
            form_panel,
            text="Folder",
            command=self._select_input_image_path,
            width=10,
        ).grid(
            row=1,
            column=3,
            pady=(0, 8),
        )

        tk.Label(
            form_panel,
            text="Output image path",
            width=FORM_LABEL_WIDTH,
            anchor="w",
            font=("Segoe UI", 10, "bold"),
            foreground=TEXT,
            background=PANEL,
        ).grid(row=2, column=0, sticky="w")
        ttk.Entry(form_panel, textvariable=self.output_image_dir_var).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(8, 8),
            ipady=2,
        )
        ttk.Button(
            form_panel,
            text="Browse",
            command=self._select_output_image_dir,
            width=10,
        ).grid(
            row=2,
            column=2,
            columnspan=2,
            sticky="ew",
        )

        main = tk.Frame(
            self.root,
            background=PANEL,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        main.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 10))
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=0)
        main.rowconfigure(1, weight=1)

        tk.Label(
            main,
            text="Original Image",
            font=("Segoe UI", 12, "bold"),
            foreground=TEXT,
            background=PANEL,
            anchor="center",
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 6),
        )
        tk.Label(
            main,
            text="Sobel Edge Detection",
            font=("Segoe UI", 12, "bold"),
            foreground=TEXT,
            background=PANEL,
            anchor="center",
        ).grid(
            row=0,
            column=1,
            sticky="ew",
            pady=(0, 6),
        )
        self.original_label = ttk.Label(main, anchor="center", style="Image.TLabel")
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.sobel_label = ttk.Label(main, anchor="center", style="Image.TLabel")
        self.sobel_label.grid(row=1, column=1, sticky="nsew", padx=(0, 8))

        controls = tk.Frame(
            main,
            background=PANEL_2,
            highlightthickness=1,
            highlightbackground=BORDER,
            padx=12,
            pady=12,
        )
        controls.grid(row=1, column=2, sticky="ns")
        controls.columnconfigure(0, weight=1)
        tk.Label(
            controls,
            text="Fine Tune Controls",
            font=("Segoe UI", 12, "bold"),
            foreground=RED_TEXT,
            background=PANEL_2,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 10),
        )

        row = 1
        row = self._add_slider(
            controls,
            row,
            "Sobel Threshold",
            self.sobel_threshold_var,
            0.010,
            0.500,
        )
        row = self._add_slider(
            controls,
            row,
            "Display Thickness",
            self.display_thickness_var,
            1,
            12,
        )
        row = self._add_slider(
            controls,
            row,
            "Connect Gap Size",
            self.close_kernel_var,
            1,
            15,
        )
        row = self._add_slider(
            controls,
            row,
            "Connect Gap Iterations",
            self.close_iter_var,
            0,
            5,
        )
        row = self._add_slider(
            controls,
            row,
            "Expand Edge Iterations",
            self.dilate_iter_var,
            0,
            5,
        )

        view_frame = tk.Frame(controls, background=PANEL_2)
        view_frame.grid(row=row, column=0, sticky="ew", pady=(4, 10))
        tk.Label(
            view_frame,
            text="Edge View",
            anchor="w",
            foreground=TEXT,
            background=PANEL_2,
            font=("Segoe UI", 10, "bold"),
        ).grid(row=0, column=0, sticky="w")
        for value, text in ((0, "Auto"), (1, "X Edge"), (2, "Y Edge"), (3, "All Edges")):
            ttk.Radiobutton(
                view_frame,
                text=text,
                value=value,
                variable=self.view_mode_var,
                command=self._render_current,
                style="FineTune.TRadiobutton",
            ).grid(row=value + 1, column=0, sticky="w")
        row += 1

        button_frame = tk.Frame(controls, background=PANEL_2)
        button_frame.grid(row=row, column=0, sticky="ew", pady=(14, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(
            button_frame,
            text="Save This Image",
            command=self._save_current,
            style="Primary.TButton",
            width=BUTTON_WIDTH,
        ).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        ttk.Button(
            button_frame,
            text="Cancel This Image",
            command=self._cancel_current,
            width=BUTTON_WIDTH,
        ).grid(
            row=0,
            column=1,
            sticky="ew",
        )

        navigation_frame = tk.Frame(controls, background=PANEL_2)
        navigation_frame.grid(row=row + 1, column=0, sticky="ew", pady=(10, 0))
        navigation_frame.columnconfigure(0, weight=1)
        navigation_frame.columnconfigure(1, weight=1)
        ttk.Button(
            navigation_frame,
            text="Previous Image",
            command=self._previous_image,
            width=BUTTON_WIDTH,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(
            navigation_frame,
            text="Next Image",
            command=self._next_image,
            width=BUTTON_WIDTH,
        ).grid(row=0, column=1, sticky="ew")
        tk.Label(
            navigation_frame,
            textvariable=self.image_count_var,
            anchor="center",
            foreground=TEXT,
            background=PANEL_2,
            font=("Segoe UI", 10, "bold"),
        ).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(18, 0))

    def _add_slider(
        self,
        parent: tk.Widget,
        row: int,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
    ) -> int:
        frame = tk.Frame(parent, background=PANEL_2)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 9))
        frame.columnconfigure(0, weight=1)
        value_label = tk.Label(
            frame,
            width=10,
            anchor="e",
            foreground=TEXT,
            background=PANEL_2,
            font=("Segoe UI", 10),
        )
        value_label.grid(row=0, column=1, sticky="e")
        tk.Label(
            frame,
            text=label,
            foreground=TEXT,
            background=PANEL_2,
            font=("Segoe UI", 10),
        ).grid(row=0, column=0, sticky="w")
        scale = ttk.Scale(
            frame,
            variable=variable,
            from_=from_,
            to=to,
            command=lambda _value: self._on_slider_changed(value_label, variable),
        )
        scale.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self._on_slider_changed(value_label, variable, render=False)
        return row + 1

    def _on_slider_changed(
        self,
        value_label: ttk.Label,
        variable: tk.Variable,
        render: bool = True,
    ) -> None:
        value = variable.get()
        if isinstance(value, float):
            value_label.configure(text=f"{value:.3f}" if value < 1 else f"{value:.0f}")
        else:
            value_label.configure(text=str(value))
        if render:
            self._render_current()

    def _select_input_image_path(self) -> None:
        folder = filedialog.askdirectory(
            title="Select input image folder",
            initialdir=_dialog_initial_dir(self.input_image_path_var.get()),
        )
        if folder:
            self._load_input_path(Path(folder))

    def _select_input_image_file(self) -> None:
        image_path = filedialog.askopenfilename(
            title="Select input image",
            initialdir=_dialog_initial_dir(self.input_image_path_var.get()),
            filetypes=[
                ("Image files", "*.bmp *.png *.jpg *.jpeg *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if image_path:
            self._load_input_path(Path(image_path))

    def _select_output_image_dir(self) -> None:
        folder = filedialog.askdirectory(
            title="Select output image path",
            initialdir=_dialog_initial_dir(self.output_image_dir_var.get()),
        )
        if folder:
            self.output_image_dir_var.set(folder)

    def _load_input_path(self, input_path: Path) -> None:
        input_path = _resolve_user_path(input_path)
        self.input_image_path_var.set(str(input_path))
        self.status_var.set(f"Loading images from: {input_path}")
        self.root.update_idletasks()

        try:
            images = find_images(input_path)
        except (FileNotFoundError, OSError) as exc:
            self.status_var.set(f"Input error: {exc}")
            self._show_empty()
            return
        if not images:
            self.status_var.set(f"No supported images found in input image path: {input_path}")
            self._show_empty()
            return
        self._load_paths(images)

    def _load_paths(self, paths: list[Path]) -> None:
        if not paths:
            return
        self.image_paths = sorted(paths)
        self.image_index = 0
        self.status_var.set(f"Loaded {len(self.image_paths)} image(s).")
        self._load_current_image()

    def _previous_image(self) -> None:
        if not self.image_paths:
            return
        self.image_index = (self.image_index - 1) % len(self.image_paths)
        self._load_current_image()

    def _next_image(self) -> None:
        if not self.image_paths:
            return
        self.image_index = (self.image_index + 1) % len(self.image_paths)
        self._load_current_image()

    def _load_current_image(self) -> None:
        image_path = self._current_path()
        if image_path is None:
            return
        try:
            image = read_image(image_path)
        except OSError as exc:
            messagebox.showerror("Image Error", str(exc))
            return
        self.current_image = image
        self._apply_settings(self.saved_settings.get(str(image_path), GuiTuneSettings()))
        self.image_count_var.set(f"{self.image_index + 1} / {len(self.image_paths)}")
        self._set_status(
            f"Actual selected part: {self.image_index + 1} / {len(self.image_paths)} - {image_path.name}",
            self._save_state_for_image(image_path),
        )
        self._render_current()

    def _current_path(self) -> Path | None:
        if not self.image_paths:
            return None
        return self.image_paths[self.image_index]

    def _current_settings(self) -> GuiTuneSettings:
        return GuiTuneSettings(
            sobel_threshold_ratio=float(self.sobel_threshold_var.get()),
            black_threshold=self.black_threshold,
            display_thickness=int(round(self.display_thickness_var.get())),
            edge_close_kernel=int(round(self.close_kernel_var.get())),
            edge_close_iterations=int(round(self.close_iter_var.get())),
            edge_dilate_iterations=int(round(self.dilate_iter_var.get())),
            view_mode=int(self.view_mode_var.get()),
        )

    def _apply_settings(self, settings: GuiTuneSettings) -> None:
        self.sobel_threshold_var.set(settings.sobel_threshold_ratio)
        self.black_threshold = settings.black_threshold
        self.display_thickness_var.set(settings.display_thickness)
        self.close_kernel_var.set(settings.edge_close_kernel)
        self.close_iter_var.set(settings.edge_close_iterations)
        self.dilate_iter_var.set(settings.edge_dilate_iterations)
        self.view_mode_var.set(settings.view_mode)

    def _render_current(self) -> None:
        image_path = self._current_path()
        if self.current_image is None or image_path is None:
            self._show_empty()
            return

        settings = self._current_settings()
        sobel_preview = render_sobel_preview(self.current_image, settings)
        original_preview = self.current_image.copy()

        original_photo = _image_to_photo(_resize_for_display(original_preview))
        sobel_photo = _image_to_photo(_resize_for_display(sobel_preview))
        self.photo_refs = [original_photo, sobel_photo]
        self.original_label.configure(image=original_photo, text="")
        self.sobel_label.configure(image=sobel_photo, text="")

    def _show_empty(self) -> None:
        self.original_label.configure(text="No image loaded", image="")
        self.sobel_label.configure(text="No image loaded", image="")

    def _save_current(self) -> None:
        image_path = self._current_path()
        if image_path is None or self.current_image is None:
            return

        program_name = self.program_name_var.get().strip()
        if not program_name:
            messagebox.showerror("Save Error", "Program name is required.")
            return

        settings = self._current_settings()
        output_base = _resolve_user_path(Path(self.output_image_dir_var.get()))
        output_dir = output_base / safe_recipe_name(program_name)
        edge_path = output_dir / f"{image_path.stem}_sobel_edge.png"
        settings_path = output_dir / f"{image_path.stem}_settings.json"
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            sobel_preview = render_sobel_preview(self.current_image, settings)
            atomic_write_image(edge_path, sobel_preview)
            atomic_write_text(
                settings_path,
                json.dumps(
                    {
                        "program_name": program_name,
                        "image": str(image_path),
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "settings": asdict(settings),
                    },
                    indent=2,
                )
                + "\n",
            )
            save_recipe(
                settings=settings.to_recipe_settings(),
                recipe_dir=self.args.recipe_dir,
                program_name=program_name,
                sample_image=image_path,
            )
            self.saved_settings[str(image_path)] = settings
            self.image_save_states[str(image_path)] = "Save"
            self._write_session_index(program_name)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Save Error", str(exc))
            self.status_var.set(f"Save failed: {image_path.name}")
            return

        self._set_status(
            f"Saved actual selected part: {self.image_index + 1} / {len(self.image_paths)} - {image_path.name}",
            "Save",
        )

    def _cancel_current(self) -> None:
        image_path = self._current_path()
        if image_path is None:
            return
        self.image_save_states[str(image_path)] = "Not Save"
        settings = self.saved_settings.get(str(image_path), GuiTuneSettings())
        self._apply_settings(settings)
        self._render_current()
        self._set_status(
            f"Not Save actual selected part: {self.image_index + 1} / {len(self.image_paths)} - {image_path.name}",
            "Not Save",
        )

    def _save_state_for_image(self, image_path: Path) -> str:
        return self.image_save_states.get(
            str(image_path),
            "Save" if str(image_path) in self.saved_settings else "Not Save",
        )

    def _set_status(self, message: str, save_state: str | None = None) -> None:
        self.status_var.set(message)
        if save_state is None:
            return
        normalized = "Save" if save_state == "Save" else "Not Save"
        self.save_state_var.set(normalized)
        color = SAVE_GREEN if normalized == "Save" else NOT_SAVE_ORANGE
        if hasattr(self, "save_state_canvas"):
            self.save_state_canvas.itemconfigure(self.save_state_dot, fill=color)

    def _write_session_index(self, program_name: str) -> None:
        session_dir = self.args.recipe_dir
        session_dir.mkdir(parents=True, exist_ok=True)
        session_path = session_dir / f"{safe_recipe_name(program_name)}_images.json"
        rows = [
            {
                "image": image_path,
                "settings": asdict(settings),
            }
            for image_path, settings in sorted(self.saved_settings.items())
        ]
        atomic_write_text(
            session_path,
            json.dumps(
                {
                    "program_name": program_name,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "images": rows,
                },
                indent=2,
            )
            + "\n",
        )


def render_sobel_preview(image: np.ndarray, settings: GuiTuneSettings) -> np.ndarray:
    edge_x, edge_y, edge_all = create_sobel_edge_masks(
        image,
        settings.sobel_threshold_ratio,
        settings.edge_close_kernel,
        settings.edge_close_iterations,
        settings.edge_dilate_iterations,
    )
    selected_edge = select_edge(settings.view_mode, edge_x, edge_y, edge_all)
    display_edge = thicken_edge_for_display(selected_edge, settings.display_thickness)
    return cv2.cvtColor(display_edge, cv2.COLOR_GRAY2BGR)


def _resize_for_display(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / width, DISPLAY_MAX_HEIGHT / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _image_to_photo(image_bgr: np.ndarray) -> tk.PhotoImage:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise ValueError("Could not encode preview image")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=data)


def _resolve_user_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _dialog_initial_dir(value: str) -> str:
    if not value:
        return str(PROJECT_ROOT)
    path = _resolve_user_path(Path(value))
    return str(path if path.is_dir() else path.parent)


if __name__ == "__main__":
    raise SystemExit(main())
