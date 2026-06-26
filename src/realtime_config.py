from __future__ import annotations

import argparse
from pathlib import Path

from app_paths import (
    DEFAULT_CLASSIFICATION_MODEL,
    DEFAULT_REALTIME_OUTPUT_DIR,
)
from app_preferences import load_realtime_preferences

DEFAULT_OUTPUT_DIR = DEFAULT_REALTIME_OUTPUT_DIR

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}

BG = "#f3f4f6"
SIDEBAR = "#991b1b"
PANEL = "#ffffff"
PANEL_2 = "#fff7f7"
PANEL_3 = "#fef2f2"
CARD_BORDER = "#e5e7eb"
DIVIDER = "#d1d5db"
TEXT = "#111827"
HEADER_TEXT = "#ffffff"
MUTED = "#6b7280"
HEADER_MUTED = "#fecaca"
ACCENT = "#dc2626"
ACCENT_DARK = "#b91c1c"
PASS_COLOR = "#16a34a"
NG_COLOR = "#dc2626"
UNKNOWN_COLOR = "#d97706"
SYSTEM_ONLINE = "#22c55e"


def parse_args() -> argparse.Namespace:
    preferences = load_realtime_preferences()
    parser = argparse.ArgumentParser(description="Real-time YOLO classification for Sobel images.")
    parser.add_argument("--image-dir", type=Path, default=preferences.image_dir)
    parser.add_argument("--csv-dir", type=Path, default=preferences.csv_dir)
    parser.add_argument("--model", type=Path, default=DEFAULT_CLASSIFICATION_MODEL)
    parser.add_argument("--output-dir", type=Path, default=preferences.output_dir)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--sobel-threshold-ratio", type=float, default=0.12)
    return parser.parse_args()
