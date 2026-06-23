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

from command_line_interface import DEFAULT_INPUT_PATH
from image_file_discovery import find_images
from sobel_edge_detection import create_sobel_edge_masks, thicken_edge_for_display
from sobel_tuning_ui_main import _safe_recipe_name, save_recipe
from sobel_tuning_ui_main import TuneSettings as RecipeTuneSettings


DISPLAY_MAX_WIDTH = 620
DISPLAY_MAX_HEIGHT = 560


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
        default=Path("configs") / "recipes",
        help="Folder for saved tuned recipes. Default: configs/recipes.",
    )
    parser.add_argument(
        "--output-image-dir",
        type=Path,
        default=Path("Output_files") / "tuning_saved",
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
        self.root.title("Sobel Fine Tune")
        self.root.geometry("1500x860")
        self.root.minsize(1200, 760)
        self.root.rowconfigure(3, weight=1)

        self.image_paths: list[Path] = []
        self.image_index = 0
        self.current_image: np.ndarray | None = None
        self.photo_refs: list[tk.PhotoImage] = []
        self.saved_settings: dict[str, GuiTuneSettings] = {}

        self.program_name_var = tk.StringVar(value=args.program_name)
        startup_input = args.input or (DEFAULT_INPUT_PATH if DEFAULT_INPUT_PATH.exists() else None)
        self.input_image_path_var = tk.StringVar(value=str(startup_input or Path("Input_files")))
        self.output_image_dir_var = tk.StringVar(value=str(args.output_image_dir))
        self.status_var = tk.StringVar(value="Select input image path to begin. Actual selected part: none.")
        self.image_count_var = tk.StringVar(value="0 / 0")

        self.sobel_threshold_var = tk.DoubleVar(value=0.120)
        self.black_threshold_var = tk.IntVar(value=55)
        self.display_thickness_var = tk.IntVar(value=1)
        self.close_kernel_var = tk.IntVar(value=1)
        self.close_iter_var = tk.IntVar(value=0)
        self.dilate_iter_var = tk.IntVar(value=0)
        self.view_mode_var = tk.IntVar(value=0)

        self._build_layout()
        if startup_input is not None:
            self._load_paths(find_images(startup_input))

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(6, weight=1)

        ttk.Label(toolbar, text="Program name").grid(row=0, column=0, sticky="w")
        ttk.Entry(toolbar, textvariable=self.program_name_var, width=28).grid(
            row=0,
            column=1,
            padx=(6, 14),
        )
        ttk.Button(toolbar, text="Previous Image", command=self._previous_image).grid(
            row=0,
            column=2,
            padx=(0, 6),
        )
        ttk.Button(toolbar, text="Next Image", command=self._next_image).grid(
            row=0,
            column=3,
            padx=(0, 14),
        )
        ttk.Label(toolbar, textvariable=self.image_count_var, width=14).grid(
            row=0,
            column=4,
            sticky="w",
        )
        ttk.Label(toolbar, textvariable=self.status_var).grid(row=0, column=6, sticky="e")

        input_bar = ttk.Frame(self.root, padding=(10, 0, 10, 6))
        input_bar.grid(row=1, column=0, sticky="ew")
        input_bar.columnconfigure(1, weight=1)
        ttk.Label(input_bar, text="Input image path").grid(row=0, column=0, sticky="w")
        ttk.Entry(input_bar, textvariable=self.input_image_path_var).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 6),
        )
        ttk.Button(input_bar, text="Browse", command=self._select_input_image_path).grid(
            row=0,
            column=2,
        )

        output_bar = ttk.Frame(self.root, padding=(10, 0, 10, 8))
        output_bar.grid(row=2, column=0, sticky="ew")
        output_bar.columnconfigure(1, weight=1)
        ttk.Label(output_bar, text="Output image path").grid(row=0, column=0, sticky="w")
        ttk.Entry(output_bar, textvariable=self.output_image_dir_var).grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(8, 6),
        )
        ttk.Button(output_bar, text="Browse", command=self._select_output_image_dir).grid(
            row=0,
            column=2,
        )

        main = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        main.grid(row=3, column=0, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=1)
        main.columnconfigure(2, weight=0)
        main.rowconfigure(1, weight=1)

        ttk.Label(main, text="Original Image", font=("Segoe UI", 12, "bold")).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 6),
        )
        ttk.Label(main, text="Sobel Edge Detection", font=("Segoe UI", 12, "bold")).grid(
            row=0,
            column=1,
            sticky="w",
            pady=(0, 6),
        )
        self.original_label = ttk.Label(main, anchor="center", relief="solid")
        self.original_label.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        self.sobel_label = ttk.Label(main, anchor="center", relief="solid")
        self.sobel_label.grid(row=1, column=1, sticky="nsew", padx=(0, 8))

        controls = ttk.Frame(main, padding=(12, 0, 0, 0))
        controls.grid(row=1, column=2, sticky="ns")
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, text="Fine Tune Controls", font=("Segoe UI", 12, "bold")).grid(
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
            0.005,
            "Lower shows more edges. Higher removes weak noise.",
        )
        row = self._add_slider(
            controls,
            row,
            "Black/White Threshold",
            self.black_threshold_var,
            0,
            255,
            1,
            "Reference threshold for dark router background.",
        )
        row = self._add_slider(
            controls,
            row,
            "Display Thickness",
            self.display_thickness_var,
            1,
            12,
            1,
            "Visual Sobel thickness only.",
        )
        row = self._add_slider(
            controls,
            row,
            "Connect Gap Size",
            self.close_kernel_var,
            1,
            15,
            1,
            "Morphology close kernel. Try 3 or 5 for broken lines.",
        )
        row = self._add_slider(
            controls,
            row,
            "Connect Gap Iterations",
            self.close_iter_var,
            0,
            5,
            1,
            "Number of close passes.",
        )
        row = self._add_slider(
            controls,
            row,
            "Expand Edge Iterations",
            self.dilate_iter_var,
            0,
            5,
            1,
            "Use carefully when Sobel line is too thin.",
        )

        view_frame = ttk.Frame(controls)
        view_frame.grid(row=row, column=0, sticky="ew", pady=(4, 10))
        ttk.Label(view_frame, text="Edge View").grid(row=0, column=0, sticky="w")
        for value, text in ((0, "Auto"), (1, "X Edge"), (2, "Y Edge"), (3, "All Edges")):
            ttk.Radiobutton(
                view_frame,
                text=text,
                value=value,
                variable=self.view_mode_var,
                command=self._render_current,
            ).grid(row=value + 1, column=0, sticky="w")
        row += 1

        button_frame = ttk.Frame(controls)
        button_frame.grid(row=row, column=0, sticky="ew", pady=(14, 0))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        ttk.Button(button_frame, text="Save This Image", command=self._save_current).grid(
            row=0,
            column=0,
            sticky="ew",
            padx=(0, 6),
        )
        ttk.Button(button_frame, text="Cancel This Image", command=self._cancel_current).grid(
            row=0,
            column=1,
            sticky="ew",
        )

    def _add_slider(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.Variable,
        from_: float,
        to: float,
        resolution: float,
        help_text: str,
    ) -> int:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=(0, 9))
        frame.columnconfigure(0, weight=1)
        value_label = ttk.Label(frame, width=10, anchor="e")
        value_label.grid(row=0, column=1, sticky="e")
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, text=help_text, foreground="#555555", wraplength=260).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="w",
        )
        scale = ttk.Scale(
            frame,
            variable=variable,
            from_=from_,
            to=to,
            command=lambda _value: self._on_slider_changed(value_label, variable),
        )
        scale.grid(row=2, column=0, columnspan=2, sticky="ew")
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
        folder = filedialog.askdirectory(title="Select input image path")
        if folder:
            input_path = Path(folder)
            self.input_image_path_var.set(str(input_path))
            images = find_images(input_path)
            if not images:
                self.status_var.set(f"No supported images found in input image path: {input_path}")
                return
            self._load_paths(images)

    def _select_output_image_dir(self) -> None:
        folder = filedialog.askdirectory(title="Select output image path")
        if folder:
            self.output_image_dir_var.set(folder)

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
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Image Error", f"Could not read image:\n{image_path}")
            return
        self.current_image = image
        self._apply_settings(self.saved_settings.get(str(image_path), GuiTuneSettings()))
        self.image_count_var.set(f"{self.image_index + 1} / {len(self.image_paths)}")
        self.status_var.set(
            f"Actual selected part: {self.image_index + 1} / {len(self.image_paths)} - {image_path.name}"
        )
        self._render_current()

    def _current_path(self) -> Path | None:
        if not self.image_paths:
            return None
        return self.image_paths[self.image_index]

    def _current_settings(self) -> GuiTuneSettings:
        return GuiTuneSettings(
            sobel_threshold_ratio=float(self.sobel_threshold_var.get()),
            black_threshold=int(round(self.black_threshold_var.get())),
            display_thickness=int(round(self.display_thickness_var.get())),
            edge_close_kernel=int(round(self.close_kernel_var.get())),
            edge_close_iterations=int(round(self.close_iter_var.get())),
            edge_dilate_iterations=int(round(self.dilate_iter_var.get())),
            view_mode=int(self.view_mode_var.get()),
        )

    def _apply_settings(self, settings: GuiTuneSettings) -> None:
        self.sobel_threshold_var.set(settings.sobel_threshold_ratio)
        self.black_threshold_var.set(settings.black_threshold)
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

        settings = self._current_settings()
        self.saved_settings[str(image_path)] = settings
        output_dir = Path(self.output_image_dir_var.get()) / _safe_recipe_name(self.program_name_var.get())
        output_dir.mkdir(parents=True, exist_ok=True)
        sobel_preview = render_sobel_preview(self.current_image, settings)
        edge_path = output_dir / f"{image_path.stem}_sobel_edge.png"
        cv2.imwrite(str(edge_path), sobel_preview)
        settings_path = output_dir / f"{image_path.stem}_settings.json"
        settings_path.write_text(
            json.dumps(
                {
                    "program_name": self.program_name_var.get(),
                    "image": str(image_path),
                    "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "settings": asdict(settings),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        save_recipe(
            settings=settings.to_recipe_settings(),
            recipe_dir=self.args.recipe_dir,
            program_name=self.program_name_var.get(),
            sample_image=image_path,
        )
        self._write_session_index()
        self.status_var.set(
            f"Saved actual selected part: {self.image_index + 1} / {len(self.image_paths)} - {image_path.name}"
        )

    def _cancel_current(self) -> None:
        image_path = self._current_path()
        if image_path is None:
            return
        settings = self.saved_settings.get(str(image_path), GuiTuneSettings())
        self._apply_settings(settings)
        self._render_current()
        self.status_var.set(
            f"Canceled actual selected part: {self.image_index + 1} / {len(self.image_paths)} - {image_path.name}"
        )

    def _write_session_index(self) -> None:
        session_dir = Path("configs") / "tuning_sessions"
        session_dir.mkdir(parents=True, exist_ok=True)
        program_name = self.program_name_var.get()
        session_path = session_dir / f"{_safe_recipe_name(program_name)}_image_settings.json"
        rows = [
            {
                "image": image_path,
                "settings": asdict(settings),
            }
            for image_path, settings in sorted(self.saved_settings.items())
        ]
        session_path.write_text(
            json.dumps(
                {
                    "program_name": program_name,
                    "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "images": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


def render_sobel_preview(image: np.ndarray, settings: GuiTuneSettings) -> np.ndarray:
    edge_x, edge_y, edge_all = create_sobel_edge_masks(
        image,
        settings.sobel_threshold_ratio,
        settings.edge_close_kernel,
        settings.edge_close_iterations,
        settings.edge_dilate_iterations,
    )
    selected_edge = _selected_edge(settings.view_mode, edge_x, edge_y, edge_all)
    display_edge = thicken_edge_for_display(selected_edge, settings.display_thickness)
    return cv2.cvtColor(display_edge, cv2.COLOR_GRAY2BGR)


def _selected_edge(
    view_mode: int,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    edge_all: np.ndarray,
) -> np.ndarray:
    if view_mode == 1:
        return edge_x
    if view_mode == 2:
        return edge_y
    if view_mode == 3:
        return edge_all
    x_count = int(np.count_nonzero(edge_x))
    y_count = int(np.count_nonzero(edge_y))
    return edge_x if x_count >= y_count else edge_y


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


if __name__ == "__main__":
    raise SystemExit(main())
