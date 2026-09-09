"""Reading the machine's own data: run results and the images that belong to them.

Layout on disk
    Result\\_<yyyyMMdd>_<HHmmss>.csv   one file per production run; row 1 is the
                                      panel and carries Start_time / End_time
    Result\\<State>_<date>_<time>.csv  machine status log (ignored here)
    Picture\\<yyyyMMdd_HHmmss>.bmp     inspection images, one every ~5 s while
                                      the machine is Running

Pictures carry no run id, so they are matched to a run by falling inside that
run's [Start_time, End_time] window.  Run windows are sorted and do not overlap,
so a single sweep assigns every picture.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

# Column positions in the run result CSV
COL_SN, COL_ID, COL_BARCODE, COL_PRODUCT = 0, 1, 2, 3
COL_TABLE, COL_RECIPE, COL_RESULT = 4, 5, 6
COL_OFFSET_X, COL_OFFSET_Y, COL_ROTATE = 7, 8, 9
COL_START, COL_TACT, COL_OPERATOR = 10, 11, 12
COL_LENGTH, COL_WIDTH, COL_CONVEYOR, COL_CUTTING = 13, 14, 15, 16
COL_MESSAGE, COL_MACHINE, COL_SUBBOARDS, COL_END = 17, 18, 19, 20
MIN_COLUMNS = 22

PICTURE_STAMP = "%Y%m%d_%H%M%S"


@dataclass
class Run:
    result_file: str
    sn: str  # panel serial, from column 0 of the panel row
    run_id: str
    product_id: str
    table: str
    recipe: str
    passed: bool
    offset_x: float
    offset_y: float
    rotate_angle: float
    start: datetime
    end: datetime
    tact_time: str
    sub_boards: str
    message: str
    pictures: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Baselines, position names and models are per product and per table.

        The product is what the shop floor works in, and the two tables hold the
        panel differently, so their edges sit in different places even for the
        same product.
        """
        product = self.product_id.strip()
        table = self.table.strip()
        return f"{product}|{table}" if table else product


def _parse_dt(text: str) -> datetime | None:
    text = (text or "").strip().replace("/", "-")
    if not text:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(text, fmt)
        except ValueError:
            continue
        # aborted runs write 0001-01-01 into End_time
        return dt if dt.year >= 2000 else None
    return None


def _parse_float(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def index_pictures(picture_dir: str | Path, progress=None) -> dict[str, list[str]]:
    """Group every .bmp by its yyyyMMdd prefix. Sorted by name = sorted by time."""
    by_day: dict[str, list[str]] = {}
    count = 0
    with os.scandir(picture_dir) as it:
        for entry in it:
            if not entry.is_file() or not entry.name.lower().endswith(".bmp"):
                continue
            if len(entry.name) < 15:
                continue
            by_day.setdefault(entry.name[:8], []).append(entry.path)
            count += 1
            if progress and count % 20000 == 0:
                progress(count)
    for day in by_day:
        by_day[day].sort()
    return by_day


def available_days(result_dir: str | Path) -> list[str]:
    days: set[str] = set()
    with os.scandir(result_dir) as it:
        for entry in it:
            name = entry.name
            if name.startswith("_") and name.lower().endswith(".csv") and len(name) >= 9:
                days.add(name[1:9])
    return sorted(days)


def load_runs(result_dir: str | Path, day: str) -> list[Run]:
    """Read every run result for one day, in start order.

    AUO6000 exports exist in two formats.  Older files contain explicit
    ``Start_time``/``End_time`` fields, while newer 11-column files only carry
    a completion timestamp in the filename.  The latter uses ``CuttingTime``
    plus a small acquisition lead-in to form a conservative picture window.
    """
    runs: list[Run] = []
    prefix = f"_{day}_"
    with os.scandir(result_dir) as it:
        for entry in it:
            if not entry.name.startswith(prefix) or not entry.name.lower().endswith(".csv"):
                continue
            try:
                with open(
                    entry.path,
                    "r",
                    encoding="utf-8-sig",
                    errors="replace",
                    newline="",
                ) as fh:
                    rows = csv.DictReader(fh)
                    row = next(
                        (item for item in rows if (item.get("SN") or "").strip()),
                        None,
                    )
            except (OSError, csv.Error):
                continue
            if not row:
                continue

            product_id = (row.get("ProductId") or "").strip()
            if not product_id:
                continue

            start = _parse_dt(row.get("Start_time") or "")
            end = _parse_dt(row.get("End_time") or "")
            if start is None:
                try:
                    completed_at = datetime.strptime(
                        Path(entry.name).stem.lstrip("_"), PICTURE_STAMP
                    )
                except ValueError:
                    continue
                cutting_seconds = max(0.0, _parse_float(row.get("CuttingTime")))
                # New-format result files are written after the last picture.
                # A 30-second lead-in covers positioning/camera acquisition that
                # is not included in the router's CuttingTime field.
                end = completed_at
                start = completed_at - timedelta(seconds=max(120.0, cutting_seconds + 30.0))
            elif end is None:
                end = start + timedelta(seconds=120)

            raw_result = (row.get("Result") or "").strip().casefold()
            table = (row.get("CutedTable") or row.get("Table") or "").strip()
            runs.append(
                Run(
                    result_file=entry.name,
                    sn=(row.get("SN") or "").strip(),
                    run_id=(row.get("ID") or Path(entry.name).stem).strip(),
                    product_id=product_id,
                    table=table,
                    recipe=(row.get("Recipe_Name") or "").strip(),
                    passed=raw_result == "true",
                    offset_x=_parse_float(row.get("OffsetX")),
                    offset_y=_parse_float(row.get("OffsetY")),
                    rotate_angle=_parse_float(row.get("RotateAngle")),
                    start=start,
                    end=end,
                    tact_time=(row.get("Tact_time") or row.get("CuttingTime") or ""),
                    sub_boards=(row.get("SubBoardCount") or ""),
                    message=(row.get("Message") or "").strip(),
                )
            )
    runs.sort(key=lambda r: r.start)
    return runs


def attach_pictures(runs: list[Run], picture_paths: list[str]) -> int:
    """Assign each picture to the run whose time window contains it.

    Returns the number of pictures that did not land in any run.
    """
    stamped: list[tuple[datetime, str]] = []
    for path in picture_paths:
        stem = Path(path).stem
        try:
            stamped.append((datetime.strptime(stem, PICTURE_STAMP), path))
        except ValueError:
            continue
    stamped.sort(key=lambda t: t[0])

    unmatched = 0
    i = 0
    for when, path in stamped:
        while i < len(runs) and when > runs[i].end:
            i += 1
        if i >= len(runs):
            unmatched += len(stamped) - stamped.index((when, path))
            break
        if when >= runs[i].start:
            runs[i].pictures.append(path)
        else:
            unmatched += 1
    return unmatched


def load_day(result_dir: str | Path, day: str, picture_index: dict[str, list[str]]) -> list[Run]:
    runs = load_runs(result_dir, day)
    if runs:
        attach_pictures(runs, picture_index.get(day, []))
    return runs
