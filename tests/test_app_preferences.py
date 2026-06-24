from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app_preferences import (
    RealtimePreferences,
    load_realtime_preferences,
    save_realtime_preferences,
)


class AppPreferencesTest(unittest.TestCase):
    def test_realtime_preferences_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "realtime.json"
            expected = RealtimePreferences(
                image_dir=Path("images"),
                csv_dir=Path("csv"),
                output_dir=Path("output"),
            )

            save_realtime_preferences(expected, path)

            self.assertEqual(load_realtime_preferences(path), expected)

    def test_invalid_preferences_fall_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "realtime.json"
            path.write_text("{invalid", encoding="utf-8")

            preferences = load_realtime_preferences(path)

            self.assertTrue(preferences.image_dir)
            self.assertTrue(preferences.csv_dir)
            self.assertTrue(preferences.output_dir)


if __name__ == "__main__":
    unittest.main()
