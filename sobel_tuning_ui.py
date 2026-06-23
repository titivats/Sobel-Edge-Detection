from __future__ import annotations

import sys
import traceback
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from app_paths import PROJECT_ROOT as APP_PROJECT_ROOT
from sobel_fine_tune_gui import main


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception:  # noqa: BLE001 - keep double-click exe failures inspectable.
        log_dir = APP_PROJECT_ROOT / "Output_files" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / "Sobel_YOLO_Fine_Tune.log").write_text(
            traceback.format_exc(),
            encoding="utf-8",
        )
        traceback.print_exc(file=sys.stderr)
        raise SystemExit(1)
