from __future__ import annotations

from realtime_app import RealtimePredictUi, RuntimeStatus
from realtime_config import parse_args
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
    ui = RealtimePredictUi(args)
    ui.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
