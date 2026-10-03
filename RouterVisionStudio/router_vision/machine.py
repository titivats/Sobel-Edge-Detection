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
import hashlib
import io
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

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
    source_path: str = ""
    source_sha256: str = ""

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


def read_run(path: str | Path, *, strict: bool = False) -> Run | None:
    """Parse one Result file; production callers reject malformed panel metadata."""
    path = Path(path)
    raw = path.read_bytes()
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    panels = [row for row in rows if (row.get("SN") or "").strip()]
    if not panels:
        if strict:
            raise ValueError(f"Result has no panel row: {path.name}")
        return None
    row = panels[0]
    product = (row.get("ProductId") or "").strip()
    recipe = (row.get("Recipe_Name") or "").strip()
    if not product:
        if strict:
            raise ValueError(f"Result ProductId is missing: {path.name}")
        return None
    raw_result = (row.get("Result") or "").strip().casefold()
    if strict:
        if len(panels) != 1 or not recipe or raw_result not in {"true", "false"}:
            raise ValueError(f"Result panel identity / recipe / result is ambiguous: {path.name}")
        for field_name in ("OffsetX", "OffsetY", "RotateAngle", "BitDiameter", "CuttingTime"):
            value = row.get(field_name)
            if value not in (None, ""):
                parsed = float(value)
                if not math.isfinite(parsed) or (
                    field_name in {"BitDiameter", "CuttingTime"} and parsed < 0
                ):
                    raise ValueError(f"Result {field_name} is invalid: {path.name}")
        for field_name in ("Start_time", "End_time"):
            if row.get(field_name) and _parse_dt(row[field_name]) is None:
                raise ValueError(f"Result {field_name} is invalid: {path.name}")
    start, end = _parse_dt(row.get("Start_time")), _parse_dt(row.get("End_time"))
    if start is None:
        try:
            end = datetime.strptime(path.stem.lstrip("_"), PICTURE_STAMP)
        except ValueError:
            if strict:
                raise ValueError(f"Result completion time is missing: {path.name}") from None
            return None
        cutting = max(0.0, _parse_float(row.get("CuttingTime")))
        start = end - timedelta(seconds=max(120.0, cutting + 30.0))
    elif end is None:
        if strict:
            raise ValueError(f"Result End_time is missing: {path.name}")
        end = start + timedelta(seconds=120)
    if strict and end < start:
        raise ValueError(f"Result ends before it starts: {path.name}")
    return Run(
        result_file=path.name,
        sn=(row.get("SN") or "").strip(),
        run_id=(row.get("ID") or path.stem).strip(),
        product_id=product,
        table=(row.get("CutedTable") or row.get("Table") or "").strip(),
        recipe=recipe,
        passed=raw_result == "true",
        offset_x=_parse_float(row.get("OffsetX")),
        offset_y=_parse_float(row.get("OffsetY")),
        rotate_angle=_parse_float(row.get("RotateAngle")),
        start=start,
        end=end,
        tact_time=row.get("Tact_time") or row.get("CuttingTime") or "",
        sub_boards=row.get("SubBoardCount") or "",
        message=(row.get("Message") or "").strip(),
        source_path=str(path.resolve()),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


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
                run = read_run(entry.path)
            except (OSError, ValueError, csv.Error):
                continue
            if run is not None:
                runs.append(run)
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
