from __future__ import annotations

import argparse
from pathlib import Path

from app_paths import (
    DEFAULT_MODEL,
    DEFAULT_REALTIME_OUTPUT_DIR,
)
from app_preferences import load_realtime_preferences

DEFAULT_OUTPUT_DIR = DEFAULT_REALTIME_OUTPUT_DIR

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
PASS_CLASSES = {"in spec", "pass"}
NG_CLASSES = {"out spec", "ng"}

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
    preferences = load_realtime_preferences()
    parser = argparse.ArgumentParser(description="Real-time YOLO preview for Sobel images.")
    parser.add_argument("--image-dir", type=Path, default=preferences.image_dir)
    parser.add_argument("--csv-dir", type=Path, default=preferences.csv_dir)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=preferences.output_dir)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--sobel-threshold-ratio", type=float, default=0.12)
    return parser.parse_args()
