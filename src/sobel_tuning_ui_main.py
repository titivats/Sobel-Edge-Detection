from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from command_line_interface import DEFAULT_INPUT_PATH
from image_file_discovery import find_images
from inspection_zone import inspection_zone_from_ratios
from sobel_edge_detection import create_sobel_edge_masks, thicken_edge_for_display


WINDOW_NAME = "Sobel Fine Tune - Preview"
EDGE_WINDOW_NAME = "Sobel Fine Tune - Edge Mask"
CONTROL_WINDOW_NAME = "Sobel Fine Tune - Controls"
TRACKBAR_WIDTH = "Inspection width %"
TRACKBAR_HEIGHT = "Inspection height %"
TRACKBAR_X_CENTER = "Inspection X center %"
TRACKBAR_Y_CENTER = "Inspection Y center %"
TRACKBAR_SOBEL = "Sobel threshold x1000"
TRACKBAR_THICKNESS = "Display thickness"
TRACKBAR_CLOSE_KERNEL = "Close kernel"
TRACKBAR_CLOSE_ITER = "Close iter"
TRACKBAR_DILATE_ITER = "Dilate iter"
TRACKBAR_BLACK = "Black threshold"
TRACKBAR_VIEW = "View 0=Auto 1=X 2=Y 3=All"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual Sobel edge fine-tuning UI for inspection recipe values."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input image file or directory. Default: SepData camera folder.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Image index to open first when input is a folder. Default: 0.",
    )
    parser.add_argument(
        "--program-name",
        default="MANUAL_TUNED",
        help="Program/Product recipe name to save. Example: 160914002C01.",
    )
    parser.add_argument(
        "--recipe-dir",
        type=Path,
        default=Path("configs") / "recipes",
        help="Folder for saved tuned recipes. Default: configs/recipes.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    images = find_images(args.input)
    if not images:
        raise FileNotFoundError(f"No supported images found in: {args.input}")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(EDGE_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(CONTROL_WINDOW_NAME, cv2.WINDOW_NORMAL)
    _create_trackbars()

    image_index = min(max(args.start_index, 0), len(images) - 1)
    while True:
        image_path = images[image_index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        preview, edge_preview, command_text, settings = _render_preview(
            image,
            image_path,
            image_index,
            len(images),
        )
        cv2.imshow(WINDOW_NAME, preview)
        cv2.imshow(EDGE_WINDOW_NAME, edge_preview)

        key = cv2.waitKey(80) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("n"), ord("d")):
            image_index = (image_index + 1) % len(images)
        elif key in (ord("p"), ord("a")):
            image_index = (image_index - 1) % len(images)
        elif key == ord("c"):
            print(command_text)
        elif key == ord("s"):
            _save_preview(image_path, preview, edge_preview)
        elif key == ord("r"):
            recipe_path = save_recipe(
                settings=settings,
                recipe_dir=args.recipe_dir,
                program_name=args.program_name,
                sample_image=image_path,
            )
            print(f"Saved recipe: {recipe_path.resolve()}")

    cv2.destroyAllWindows()
    return 0


def _create_trackbars() -> None:
    cv2.createTrackbar(TRACKBAR_SOBEL, CONTROL_WINDOW_NAME, 120, 500, _noop)
    cv2.createTrackbar(TRACKBAR_THICKNESS, CONTROL_WINDOW_NAME, 1, 12, _noop)
    cv2.createTrackbar(TRACKBAR_CLOSE_KERNEL, CONTROL_WINDOW_NAME, 1, 15, _noop)
    cv2.createTrackbar(TRACKBAR_CLOSE_ITER, CONTROL_WINDOW_NAME, 0, 5, _noop)
    cv2.createTrackbar(TRACKBAR_DILATE_ITER, CONTROL_WINDOW_NAME, 0, 5, _noop)
    cv2.createTrackbar(TRACKBAR_BLACK, CONTROL_WINDOW_NAME, 55, 255, _noop)
    cv2.createTrackbar(TRACKBAR_WIDTH, CONTROL_WINDOW_NAME, 65, 100, _noop)
    cv2.createTrackbar(TRACKBAR_HEIGHT, CONTROL_WINDOW_NAME, 70, 100, _noop)
    cv2.createTrackbar(TRACKBAR_X_CENTER, CONTROL_WINDOW_NAME, 52, 100, _noop)
    cv2.createTrackbar(TRACKBAR_Y_CENTER, CONTROL_WINDOW_NAME, 50, 100, _noop)
    cv2.createTrackbar(TRACKBAR_VIEW, CONTROL_WINDOW_NAME, 0, 3, _noop)


def _render_preview(
    image: np.ndarray,
    image_path: Path,
    image_index: int,
    image_count: int,
) -> tuple[np.ndarray, np.ndarray, str, "TuneSettings"]:
    settings = _read_settings()
    edge_x, edge_y, edge_all = create_sobel_edge_masks(
        image,
        settings.sobel_threshold_ratio,
        settings.edge_close_kernel,
        settings.edge_close_iterations,
        settings.edge_dilate_iterations,
    )
    selected_edge = _selected_edge(settings.view_mode, edge_x, edge_y, edge_all)
    display_edge = thicken_edge_for_display(selected_edge, settings.display_thickness)
    focus_zone = inspection_zone_from_ratios(
        image.shape[:2],
        settings.x_min_ratio,
        settings.x_max_ratio,
        settings.y_min_ratio,
        settings.y_max_ratio,
    )

    preview = image.copy()
    _draw_edge_overlay(preview, display_edge, focus_zone)
    _draw_black_mask_overlay(preview, image, settings.black_threshold, focus_zone)
    _draw_text_panel(preview, image_path, image_index, image_count, settings)

    edge_preview = cv2.cvtColor(display_edge, cv2.COLOR_GRAY2BGR)
    command_text = _command_text(settings)
    return _fit_display(preview), _fit_display(edge_preview), command_text, settings


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


def _draw_edge_overlay(
    preview: np.ndarray,
    display_edge: np.ndarray,
    focus_zone,
) -> None:
    focus_mask = np.zeros(display_edge.shape, dtype=np.uint8)
    focus_mask[
        focus_zone.y_min : focus_zone.y_max + 1,
        focus_zone.x_min : focus_zone.x_max + 1,
    ] = 255
    selected_edge = cv2.bitwise_and(display_edge, focus_mask)
    preview[selected_edge > 0] = (255, 255, 255)


def _draw_black_mask_overlay(
    preview: np.ndarray,
    image: np.ndarray,
    black_threshold: int,
    focus_zone,
) -> None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    black_mask = gray <= black_threshold
    zone_mask = np.zeros(gray.shape, dtype=bool)
    zone_mask[
        focus_zone.y_min : focus_zone.y_max + 1,
        focus_zone.x_min : focus_zone.x_max + 1,
    ] = True
    overlay_mask = black_mask & zone_mask
    preview[overlay_mask] = (
        preview[overlay_mask].astype(np.float32) * 0.65 + np.array([120, 40, 0]) * 0.35
    ).astype(np.uint8)


def _draw_text_panel(
    preview: np.ndarray,
    image_path: Path,
    image_index: int,
    image_count: int,
    settings: "TuneSettings",
) -> None:
    panel_lines = [
        "Sobel Fine Tune",
        f"{image_index + 1}/{image_count}  {image_path.name}",
        f"Sobel threshold: {settings.sobel_threshold_ratio:.3f}",
        f"Black threshold: {settings.black_threshold}",
        f"Thickness: {settings.display_thickness}",
        (
            "Edge repair: "
            f"close={settings.edge_close_kernel}px x{settings.edge_close_iterations}, "
            f"dilate={settings.edge_dilate_iterations}"
        ),
        (
            "Inspection: "
            f"x={settings.x_min_ratio:.2f}..{settings.x_max_ratio:.2f}, "
            f"y={settings.y_min_ratio:.2f}..{settings.y_max_ratio:.2f}"
        ),
        "Keys: n/p image, c print command, s save preview, q quit",
        "Recipe: press r to save tuned recipe",
    ]
    width = min(preview.shape[1] - 20, 980)
    panel_height = 28 * len(panel_lines) + 16
    cv2.rectangle(preview, (10, 10), (10 + width, 10 + panel_height), (0, 0, 0), -1)
    for index, line in enumerate(panel_lines):
        cv2.putText(
            preview,
            line,
            (20, 40 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def _read_settings() -> "TuneSettings":
    sobel_threshold = max(cv2.getTrackbarPos(TRACKBAR_SOBEL, CONTROL_WINDOW_NAME), 1)
    display_thickness = max(cv2.getTrackbarPos(TRACKBAR_THICKNESS, CONTROL_WINDOW_NAME), 1)
    edge_close_kernel = max(cv2.getTrackbarPos(TRACKBAR_CLOSE_KERNEL, CONTROL_WINDOW_NAME), 1)
    edge_close_iterations = cv2.getTrackbarPos(TRACKBAR_CLOSE_ITER, CONTROL_WINDOW_NAME)
    edge_dilate_iterations = cv2.getTrackbarPos(TRACKBAR_DILATE_ITER, CONTROL_WINDOW_NAME)
    black_threshold = cv2.getTrackbarPos(TRACKBAR_BLACK, CONTROL_WINDOW_NAME)
    width_ratio = max(cv2.getTrackbarPos(TRACKBAR_WIDTH, CONTROL_WINDOW_NAME) / 100.0, 0.01)
    height_ratio = max(cv2.getTrackbarPos(TRACKBAR_HEIGHT, CONTROL_WINDOW_NAME) / 100.0, 0.01)
    x_center = cv2.getTrackbarPos(TRACKBAR_X_CENTER, CONTROL_WINDOW_NAME) / 100.0
    y_center = cv2.getTrackbarPos(TRACKBAR_Y_CENTER, CONTROL_WINDOW_NAME) / 100.0
    view_mode = cv2.getTrackbarPos(TRACKBAR_VIEW, CONTROL_WINDOW_NAME)

    x_min_ratio, x_max_ratio = _center_width_to_bounds(x_center, width_ratio)
    y_min_ratio, y_max_ratio = _center_width_to_bounds(y_center, height_ratio)
    return TuneSettings(
        sobel_threshold_ratio=sobel_threshold / 1000.0,
        display_thickness=display_thickness,
        edge_close_kernel=edge_close_kernel,
        edge_close_iterations=edge_close_iterations,
        edge_dilate_iterations=edge_dilate_iterations,
        black_threshold=black_threshold,
        x_min_ratio=x_min_ratio,
        x_max_ratio=x_max_ratio,
        y_min_ratio=y_min_ratio,
        y_max_ratio=y_max_ratio,
        view_mode=view_mode,
    )


def _center_width_to_bounds(center: float, width: float) -> tuple[float, float]:
    half_width = width / 2.0
    lower = center - half_width
    upper = center + half_width
    if lower < 0:
        upper -= lower
        lower = 0.0
    if upper > 1:
        lower -= upper - 1.0
        upper = 1.0
    return max(lower, 0.0), min(upper, 1.0)


def _command_text(settings: "TuneSettings") -> str:
    edge_view = _edge_view_arg(settings.view_mode)
    return (
        ".\\venv\\Scripts\\python.exe .\\sobel_edge_detect.py "
        f"--sobel-threshold-ratio {settings.sobel_threshold_ratio:.3f} "
        f"--edge-close-kernel {settings.edge_close_kernel} "
        f"--edge-close-iterations {settings.edge_close_iterations} "
        f"--edge-dilate-iterations {settings.edge_dilate_iterations} "
        f"--display-edge-thickness {settings.display_thickness} "
        f"--edge-view {edge_view}"
    )


def save_recipe(
    settings: "TuneSettings",
    recipe_dir: Path,
    program_name: str,
    sample_image: Path,
) -> Path:
    recipe_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_recipe_name(program_name)
    recipe_path = recipe_dir / f"{safe_name}.json"
    recipe = recipe_dict(settings, program_name, sample_image)
    recipe_path.write_text(
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    _upsert_recipe_index(recipe_dir / "recipe_index.csv", recipe)
    return recipe_path


def recipe_dict(
    settings: "TuneSettings",
    program_name: str,
    sample_image: Path,
) -> dict[str, object]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "recipe_name": program_name,
        "program_name": program_name,
        "created_at": now,
        "updated_at": now,
        "source": "sobel_tuning_ui",
        "sample_image": str(sample_image),
        "detection": {
            "black_threshold": settings.black_threshold,
            "sobel_threshold_ratio": settings.sobel_threshold_ratio,
            "edge_close_kernel": settings.edge_close_kernel,
            "edge_close_iterations": settings.edge_close_iterations,
            "edge_dilate_iterations": settings.edge_dilate_iterations,
            "display_edge_thickness": settings.display_thickness,
            "edge_view_mode": settings.view_mode,
            "edge_view": _edge_view_arg(settings.view_mode),
        },
        "inspection_zone": {
            "x_min_ratio": round(settings.x_min_ratio, 4),
            "x_max_ratio": round(settings.x_max_ratio, 4),
            "y_min_ratio": round(settings.y_min_ratio, 4),
            "y_max_ratio": round(settings.y_max_ratio, 4),
        },
    }


def _upsert_recipe_index(index_path: Path, recipe: dict[str, object]) -> None:
    fieldnames = [
        "program_name",
        "recipe_file",
        "updated_at",
        "black_threshold",
        "sobel_threshold_ratio",
        "edge_close_kernel",
        "edge_close_iterations",
        "edge_dilate_iterations",
        "display_edge_thickness",
        "inspection_x_min_ratio",
        "inspection_x_max_ratio",
        "inspection_y_min_ratio",
        "inspection_y_max_ratio",
        "sample_image",
    ]
    rows = []
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

    program_name = str(recipe["program_name"])
    recipe_file = f"{_safe_recipe_name(program_name)}.json"
    detection = recipe["detection"]
    inspection_zone = recipe["inspection_zone"]
    new_row = {
        "program_name": program_name,
        "recipe_file": recipe_file,
        "updated_at": str(recipe["updated_at"]),
        "black_threshold": str(detection["black_threshold"]),
        "sobel_threshold_ratio": f"{float(detection['sobel_threshold_ratio']):.3f}",
        "edge_close_kernel": str(detection["edge_close_kernel"]),
        "edge_close_iterations": str(detection["edge_close_iterations"]),
        "edge_dilate_iterations": str(detection["edge_dilate_iterations"]),
        "display_edge_thickness": str(detection["display_edge_thickness"]),
        "inspection_x_min_ratio": f"{float(inspection_zone['x_min_ratio']):.4f}",
        "inspection_x_max_ratio": f"{float(inspection_zone['x_max_ratio']):.4f}",
        "inspection_y_min_ratio": f"{float(inspection_zone['y_min_ratio']):.4f}",
        "inspection_y_max_ratio": f"{float(inspection_zone['y_max_ratio']):.4f}",
        "sample_image": str(recipe["sample_image"]),
    }
    rows = [row for row in rows if row.get("program_name") != program_name]
    rows.append(new_row)
    rows.sort(key=lambda row: row.get("program_name", ""))

    with index_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_recipe_name(program_name: str) -> str:
    safe_chars = []
    for char in program_name.strip():
        if char.isalnum() or char in ("-", "_", "."):
            safe_chars.append(char)
        else:
            safe_chars.append("_")
    safe_name = "".join(safe_chars).strip("._")
    return safe_name or "MANUAL_TUNED"


def _edge_view_arg(view_mode: int) -> str:
    if view_mode == 1:
        return "x"
    if view_mode == 2:
        return "y"
    return "all"


def _save_preview(image_path: Path, preview: np.ndarray, edge_preview: np.ndarray) -> None:
    output_dir = Path("Output_files") / "tuning_preview"
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"{image_path.stem}_preview.png"
    edge_path = output_dir / f"{image_path.stem}_edge.png"
    cv2.imwrite(str(preview_path), preview)
    cv2.imwrite(str(edge_path), edge_preview)
    print(f"Saved preview: {preview_path.resolve()}")
    print(f"Saved edge: {edge_path.resolve()}")


def _fit_display(image: np.ndarray, max_width: int = 1400, max_height: int = 900) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return image
    return cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def _noop(_value: int) -> None:
    return None


class TuneSettings:
    def __init__(
        self,
        sobel_threshold_ratio: float,
        display_thickness: int,
        edge_close_kernel: int,
        edge_close_iterations: int,
        edge_dilate_iterations: int,
        black_threshold: int,
        x_min_ratio: float,
        x_max_ratio: float,
        y_min_ratio: float,
        y_max_ratio: float,
        view_mode: int,
    ) -> None:
        self.sobel_threshold_ratio = sobel_threshold_ratio
        self.display_thickness = display_thickness
        self.edge_close_kernel = edge_close_kernel
        self.edge_close_iterations = edge_close_iterations
        self.edge_dilate_iterations = edge_dilate_iterations
        self.black_threshold = black_threshold
        self.x_min_ratio = x_min_ratio
        self.x_max_ratio = x_max_ratio
        self.y_min_ratio = y_min_ratio
        self.y_max_ratio = y_max_ratio
        self.view_mode = view_mode
