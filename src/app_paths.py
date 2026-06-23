from __future__ import annotations

import sys
from pathlib import Path


PROJECT_MARKERS = ("src", "configs", "assets")


def find_project_root(start: Path | None = None) -> Path:
    base = (start or _runtime_start()).resolve()
    for candidate in (base, *base.parents):
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return base


def _runtime_start() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = find_project_root()

ASSETS_DIR = PROJECT_ROOT / "assets"
CONFIG_DIR = PROJECT_ROOT / "configs"
DIST_DIR = PROJECT_ROOT / "dist"
OUTPUT_DIR = PROJECT_ROOT / "Output_files"
SEP_DATA_DIR = PROJECT_ROOT / "SepData"
YOLO_DIR = PROJECT_ROOT / "Yolo_train"

DEFAULT_IMAGE_DIR = SEP_DATA_DIR / "Picture"
DEFAULT_CSV_DIR = SEP_DATA_DIR / "Product_Info"
DEFAULT_MODEL = YOLO_DIR / "runs" / "sobel_yolo" / "weights" / "best.pt"
DEFAULT_RECIPE_DIR = CONFIG_DIR / "recipes"
DEFAULT_SOBEL_OUTPUT_DIR = OUTPUT_DIR
DEFAULT_TUNING_OUTPUT_DIR = OUTPUT_DIR / "tuning_saved"
DEFAULT_REALTIME_OUTPUT_DIR = OUTPUT_DIR / "Sobel_Image_OBB"
APP_ICON = ASSETS_DIR / "edge_detection_monitor.ico"
