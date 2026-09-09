from __future__ import annotations

import runpy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
LEGACY_SCRIPT = PROJECT_ROOT / "tools" / "sobel_measure.py"


if __name__ == "__main__":
    runpy.run_path(str(LEGACY_SCRIPT), run_name="__main__")
