from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app_paths import (
    DEFAULT_CSV_DIR,
    DEFAULT_IMAGE_DIR,
    DEFAULT_REALTIME_OUTPUT_DIR,
    SETTINGS_DIR,
)
from file_io import atomic_write_text

REALTIME_PREFERENCES_PATH = SETTINGS_DIR / "realtime.json"


@dataclass(frozen=True)
class RealtimePreferences:
    image_dir: Path = DEFAULT_IMAGE_DIR
    csv_dir: Path = DEFAULT_CSV_DIR
    output_dir: Path = DEFAULT_REALTIME_OUTPUT_DIR


def load_realtime_preferences(
    path: Path = REALTIME_PREFERENCES_PATH,
) -> RealtimePreferences:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Preferences must be a JSON object")
        return RealtimePreferences(
            image_dir=_path_value(payload, "image_dir", DEFAULT_IMAGE_DIR),
            csv_dir=_path_value(payload, "csv_dir", DEFAULT_CSV_DIR),
            output_dir=_path_value(payload, "output_dir", DEFAULT_REALTIME_OUTPUT_DIR),
        )
    except (OSError, ValueError, json.JSONDecodeError, TypeError):
        return RealtimePreferences()


def save_realtime_preferences(
    preferences: RealtimePreferences,
    path: Path = REALTIME_PREFERENCES_PATH,
) -> None:
    payload = {
        "image_dir": str(preferences.image_dir),
        "csv_dir": str(preferences.csv_dir),
        "output_dir": str(preferences.output_dir),
    }
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _path_value(payload: dict[str, object], key: str, default: Path) -> Path:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        return default
    return Path(value).expanduser()
