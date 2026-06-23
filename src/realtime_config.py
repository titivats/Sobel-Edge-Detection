from __future__ import annotations

import argparse
from pathlib import Path

from app_paths import (
    APP_ICON,
    DEFAULT_CSV_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_MODEL,
    DEFAULT_REALTIME_OUTPUT_DIR,
    PROJECT_ROOT,
)


DEFAULT_OUTPUT_DIR = DEFAULT_REALTIME_OUTPUT_DIR

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
