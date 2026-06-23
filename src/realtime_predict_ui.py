from __future__ import annotations

import sys
import traceback

from realtime_app import RealtimePredictUi, RuntimeStatus
from realtime_config import DEFAULT_OUTPUT_DIR, parse_args
from realtime_overlay import status_from_classes
from realtime_predictor import PredictionView, RealtimePredictor
from realtime_product_info import (
    CsvPointCounter,
    ProductCsvIndex,
    ProductInfo,
    ProductInfoCache,
    find_product_csv,
    product_csv_files,
    read_product_info,
    timestamp_from_name,
)

__all__ = [
    "CsvPointCounter",
    "PredictionView",
    "ProductCsvIndex",
    "ProductInfo",
    "ProductInfoCache",
    "RealtimePredictUi",
    "RealtimePredictor",
    "RuntimeStatus",
    "find_product_csv",
    "main",
    "parse_args",
    "product_csv_files",
    "read_product_info",
    "status_from_classes",
    "timestamp_from_name",
]


def main() -> int:
    args = parse_args()
    try:
        ui = RealtimePredictUi(args)
        ui.start()
        return 0
    except Exception:  # noqa: BLE001 - log startup crashes for double-click launches.
        log_dir = DEFAULT_OUTPUT_DIR.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "Realtime_Sobel_YOLO.log"
        log_path.write_text(traceback.format_exc(), encoding="utf-8")
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
