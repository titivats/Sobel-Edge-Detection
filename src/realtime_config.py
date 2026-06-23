from __future__ import annotations

import argparse
import sys
from pathlib import Path


def application_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


PROJECT_ROOT = application_root()
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "SepData" / "Picture"
DEFAULT_CSV_DIR = PROJECT_ROOT / "SepData" / "Product_Info"
DEFAULT_MODEL = PROJECT_ROOT / "Yolo_train" / "runs" / "sobel_yolo" / "weights" / "best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Output_files" / "Sobel_Image_OBB"
APP_ICON = PROJECT_ROOT / "assets" / "edge_detection_monitor.ico"

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PASS_CLASSES = {"In spec"}
NG_CLASSES = {"Out spec"}

BG = "#111827"
PANEL = "#182235"
PANEL_2 = "#202b40"
PANEL_3 = "#24324a"
TEXT = "#f8fafc"
MUTED = "#94a3b8"
ACCENT = "#38bdf8"
PASS_COLOR = "#22c55e"
NG_COLOR = "#ef4444"
UNKNOWN_COLOR = "#f59e0b"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time YOLO preview for Sobel images.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--sobel-threshold-ratio", type=float, default=0.12)
    return parser.parse_args()
