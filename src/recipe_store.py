from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from edge_view import edge_view_name
from file_io import atomic_write_text


@dataclass(frozen=True)
class TuneSettings:
    sobel_threshold_ratio: float
    display_thickness: int
    edge_close_kernel: int
    edge_close_iterations: int
    edge_dilate_iterations: int
    black_threshold: int
    x_min_ratio: float = 0.0
    x_max_ratio: float = 1.0
    y_min_ratio: float = 0.0
    y_max_ratio: float = 1.0
    view_mode: int = 0


def save_recipe(
    settings: TuneSettings,
    recipe_dir: Path,
    program_name: str,
    sample_image: Path,
) -> Path:
    recipe_dir.mkdir(parents=True, exist_ok=True)
    recipe_path = recipe_dir / f"{safe_recipe_name(program_name)}.json"
    recipe = recipe_dict(
        settings,
        program_name,
        sample_image,
        existing_recipe=_read_recipe(recipe_path),
    )
    atomic_write_text(
        recipe_path,
        json.dumps(recipe, indent=2, ensure_ascii=False) + "\n",
    )
    _upsert_recipe_index(recipe_dir / "recipe_index.csv", recipe)
    return recipe_path


def recipe_dict(
    settings: TuneSettings,
    program_name: str,
    sample_image: Path,
    existing_recipe: dict[str, object] | None = None,
) -> dict[str, object]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    created_at = str(existing_recipe.get("created_at", now)) if existing_recipe else now
    return {
        "recipe_name": program_name,
        "program_name": program_name,
        "created_at": created_at,
        "updated_at": now,
        "source": "sobel_fine_tune",
        "sample_image": str(sample_image),
        "detection": {
            "black_threshold": settings.black_threshold,
            "sobel_threshold_ratio": settings.sobel_threshold_ratio,
            "edge_close_kernel": settings.edge_close_kernel,
            "edge_close_iterations": settings.edge_close_iterations,
            "edge_dilate_iterations": settings.edge_dilate_iterations,
            "display_edge_thickness": settings.display_thickness,
            "edge_view_mode": settings.view_mode,
            "edge_view": edge_view_name(settings.view_mode),
        },
        "inspection_zone": {
            "x_min_ratio": round(settings.x_min_ratio, 4),
            "x_max_ratio": round(settings.x_max_ratio, 4),
            "y_min_ratio": round(settings.y_min_ratio, 4),
            "y_max_ratio": round(settings.y_max_ratio, 4),
        },
    }


def safe_recipe_name(program_name: str) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_" for char in program_name.strip()
    ).strip("._")
    return safe_name or "MANUAL_TUNED"


def _read_recipe(recipe_path: Path) -> dict[str, object] | None:
    if not recipe_path.exists():
        return None
    try:
        recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return recipe if isinstance(recipe, dict) else None


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
    rows: list[dict[str, str]] = []
    if index_path.exists():
        with index_path.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))

    program_name = str(recipe["program_name"])
    detection = recipe["detection"]
    inspection_zone = recipe["inspection_zone"]
    if not isinstance(detection, dict) or not isinstance(inspection_zone, dict):
        raise ValueError("Recipe detection and inspection_zone must be objects")

    new_row = {
        "program_name": program_name,
        "recipe_file": f"{safe_recipe_name(program_name)}.json",
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

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    atomic_write_text(index_path, output.getvalue())
