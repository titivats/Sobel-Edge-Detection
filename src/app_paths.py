from __future__ import annotations

import sys
from pathlib import Path

PROJECT_MARKERS = ("assets", "settings")


def find_project_root(start: Path | None = None) -> Path:
    base = (start or _runtime_start()).resolve()
    if base.is_file():
        base = base.parent
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
SETTINGS_DIR = PROJECT_ROOT / "settings"
OUTPUT_DIR = PROJECT_ROOT / "Output_files"
SEP_DATA_DIR = PROJECT_ROOT / "SepData"
INPUT_FILES_DIR = PROJECT_ROOT / "Input_files"


def _preferred_path(primary: Path, fallback: Path) -> Path:
    return primary if primary.exists() else fallback


DEFAULT_IMAGE_DIR = _preferred_path(
    SEP_DATA_DIR / "Picture",
    INPUT_FILES_DIR / "Picture",
)
DEFAULT_CSV_DIR = _preferred_path(
    SEP_DATA_DIR / "Product_Info",
    INPUT_FILES_DIR / "Product Info",
)
DEFAULT_MODEL = PROJECT_ROOT / "best.pt"
DEFAULT_RECIPE_DIR = SETTINGS_DIR
DEFAULT_SOBEL_OUTPUT_DIR = OUTPUT_DIR
DEFAULT_TUNING_OUTPUT_DIR = OUTPUT_DIR / "tuning_saved"
DEFAULT_REALTIME_OUTPUT_DIR = OUTPUT_DIR / "Sobel_Image_OBB"
APP_ICON = ASSETS_DIR / "edge_detection_monitor.ico"
SOBEL_FINE_TUNE_ICON = ASSETS_DIR / "sobel_fine_tune_icon.ico"
