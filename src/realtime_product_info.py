from __future__ import annotations

import bisect
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class ProductInfo:
    csv_path: Path | None
    values: dict[str, str]

    @property
    def sn(self) -> str:
        return self.values.get("SN", "")

    @property
    def product_id(self) -> str:
        return self.values.get("ProductId", "")

    @property
    def cuted_table(self) -> str:
        return self.values.get("CutedTable", "")

    @property
    def recipe_name(self) -> str:
        return self.values.get("Recipe_Name", "")

    @property
    def result(self) -> str:
        return self.values.get("Result", "")

    @property
    def end_time(self) -> str:
        return self.values.get("End_time", "")


def timestamp_from_name(path: Path) -> datetime | None:
    stem = path.stem.lstrip("_")
    for index in range(0, max(len(stem) - 14, 0) + 1):
        candidate = stem[index : index + 15]
        try:
            return datetime.strptime(candidate, "%Y%m%d_%H%M%S")
        except ValueError:
            continue
    return None


def product_csv_files(csv_dir: Path) -> list[Path]:
    if not csv_dir.exists():
        return []
    return sorted(csv_dir.glob("_*.csv"), key=lambda p: timestamp_from_name(p) or datetime.min)


def find_product_csv(image_path: Path, csv_dir: Path) -> Path | None:
    image_timestamp = timestamp_from_name(image_path)
    csv_files = product_csv_files(csv_dir)
    if image_timestamp is None or not csv_files:
        return csv_files[-1] if csv_files else None

    candidates = [
        csv_path
        for csv_path in csv_files
        if (timestamp_from_name(csv_path) or datetime.min) <= image_timestamp
    ]
    return candidates[-1] if candidates else csv_files[0]


def read_product_info(csv_path: Path | None) -> ProductInfo:
    if csv_path is None or not csv_path.exists():
        return ProductInfo(csv_path=None, values={})
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    first_row = next((row for row in rows if any(row.values())), {})
    return ProductInfo(
        csv_path=csv_path, values={key: value or "" for key, value in first_row.items()}
    )


class CsvPointCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}

    def next_point(self, csv_path: Path | None) -> int:
        key = csv_path.name if csv_path else "<no_csv>"
        point_number = self.counts.get(key, 0) + 1
        self.counts[key] = point_number
        return point_number


class ProductCsvIndex:
    """Cache CSV timestamps so each image can be matched without re-sorting files."""

    def __init__(self, csv_dir: Path) -> None:
        self.csv_dir = csv_dir
        self.signature: tuple[tuple[str, int, int], ...] | None = None
        self.entries: list[tuple[datetime, Path]] = []
        self.timestamps: list[datetime] = []

    def update_dir(self, csv_dir: Path) -> None:
        if self.csv_dir != csv_dir:
            self.csv_dir = csv_dir
            self.signature = None
            self.entries = []
            self.timestamps = []

    def match(self, image_path: Path) -> Path | None:
        self.refresh()
        if not self.entries:
            return None

        image_timestamp = timestamp_from_name(image_path)
        if image_timestamp is None:
            return self.entries[-1][1]

        index = bisect.bisect_right(self.timestamps, image_timestamp) - 1
        if index < 0:
            return self.entries[0][1]
        return self.entries[index][1]

    def refresh(self) -> None:
        if not self.csv_dir.exists():
            self.signature = None
            self.entries = []
            self.timestamps = []
            return

        csv_paths = list(self.csv_dir.glob("_*.csv"))
        signature_items = []
        for csv_path in csv_paths:
            try:
                stat = csv_path.stat()
                signature_items.append((csv_path.name, stat.st_mtime_ns, stat.st_size))
            except OSError:
                continue
        signature = tuple(sorted(signature_items))
        if signature == self.signature:
            return

        entries = []
        for csv_path in csv_paths:
            timestamp = timestamp_from_name(csv_path)
            if timestamp is not None:
                entries.append((timestamp, csv_path))
        self.entries = sorted(entries, key=lambda entry: entry[0])
        self.timestamps = [entry[0] for entry in self.entries]
        self.signature = signature


class ProductInfoCache:
    """Read each matched CSV once and refresh only when the file changes."""

    def __init__(self) -> None:
        self.cache: dict[Path, tuple[int, ProductInfo]] = {}

    def read(self, csv_path: Path | None) -> ProductInfo:
        if csv_path is None or not csv_path.exists():
            return ProductInfo(csv_path=None, values={})
        modified = csv_path.stat().st_mtime_ns
        cached = self.cache.get(csv_path)
        if cached and cached[0] == modified:
            return cached[1]
        product_info = read_product_info(csv_path)
        self.cache[csv_path] = (modified, product_info)
        return product_info

    def clear(self) -> None:
        self.cache.clear()
