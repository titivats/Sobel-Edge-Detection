from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from aurotek_edge_detection.router_intrusion_main import main


if __name__ == "__main__":
    raise SystemExit(main())
