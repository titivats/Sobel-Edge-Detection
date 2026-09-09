"""Discover Aurotek Router AUO6000 exports for the AVTR settings workflow.

AUO6000 stores pictures, result CSV files, logs, and recipes in sibling
directories.  Picture names contain capture timestamps but no panel serial
number, so this module groups pictures by acquisition gaps and pairs each group
with the next result file written by the router.
"""

from __future__ import annotations

import csv
import os
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

IMAGE_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg"}
PICTURE_STAMP = "%Y%m%d_%H%M%S"
RESULT_STAMP = "%Y%m%d_%H%M%S"
PANEL_GAP_SECONDS = 30
MAX_RESULT_LAG_SECONDS = 120


@dataclass(frozen=True)
class AUO6000Panel:
    sn: str
    result: str
    result_file: str
    result_at: datetime | None
    recipe_name: str = ""
    product_id: str = ""
    start_at: datetime | None = None
    end_at: datetime | None = None


@dataclass(frozen=True)
class AUO6000Image:
    path: str
    captured_at: datetime | None
    panel_sn: str = ""
    cut_point: int = 0
    cut_point_total: int = 0
    machine_result: str = "UNKNOWN"
    result_file: str = ""
    stored_format: str = "UNKNOWN"


@dataclass
class AUO6000Dataset:
    root: str
    picture_dir: str
    result_dir: str = ""
    recipe_dir: str = ""
    log_dir: str = ""
    images: list[AUO6000Image] = field(default_factory=list)
    panels: list[AUO6000Panel] = field(default_factory=list)
    recipe_files: list[str] = field(default_factory=list)
    log_files: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def expected_cut_points(self) -> int:
        # Count each inferred panel once. Counting ``cut_point_total`` on every
        # image weights a 30-point panel 2.5x more heavily than a 12-point panel
        # and can therefore select the wrong mode on a mixed production export.
        starts = [
            image for image in self.images if image.cut_point == 1 and image.cut_point_total > 0
        ]
        assigned = [image for image in starts if image.result_file]
        candidates = assigned or starts
        if not candidates:
            return 0

        product_by_result = {
            panel.result_file: panel.product_id.strip()
            for panel in self.panels
            if panel.result_file
        }
        per_product: dict[str, Counter[int]] = {}
        for image in candidates:
            product = product_by_result.get(image.result_file, "")
            per_product.setdefault(product, Counter())[image.cut_point_total] += 1

        # A single path may retain more than one recipe. If those products have
        # different normal cut counts, one global value would be unsafe and
        # misleading; the caller must select an unambiguous product instead.
        product_modes = {
            _unambiguous_mode(counts) for product, counts in per_product.items() if product
        }
        if len(product_modes) > 1 or 0 in product_modes:
            return 0

        return _unambiguous_mode(Counter(image.cut_point_total for image in candidates))

    @property
    def recipe_name(self) -> str:
        for panel in self.panels:
            if panel.recipe_name:
                return panel.recipe_name
        return Path(self.recipe_files[0]).name if self.recipe_files else ""

    @property
    def format_summary(self) -> str:
        counts = Counter(image.stored_format for image in self.images)
        return " / ".join(f"{name} {count}" for name, count in sorted(counts.items()))

    @property
    def image_index(self) -> dict[str, AUO6000Image]:
        return {_path_key(image.path): image for image in self.images}


def _path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(str(path)))


def _timestamp_from_name(path: Path, pattern: str) -> datetime | None:
    stem = path.stem.lstrip("_")
    try:
        return datetime.strptime(stem, pattern)
    except ValueError:
        return None


def _timestamp_from_csv(value: str | None) -> datetime | None:
    text = (value or "").strip().replace("/", "-")
    if not text:
        return None
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            parsed = datetime.strptime(text, pattern)
        except ValueError:
            continue
        return parsed if parsed.year >= 2000 else None
    return None


def _unambiguous_mode(counts: Counter[int]) -> int:
    if not counts:
        return 0
    highest = max(counts.values())
    winners = [value for value, frequency in counts.items() if frequency == highest]
    return winners[0] if len(winners) == 1 else 0


def _stored_image_format(path: Path) -> str:
    try:
        with path.open("rb") as fh:
            signature = fh.read(12)
    except OSError:
        return "UNREADABLE"
    if signature.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if signature.startswith(b"BM"):
        return "BMP"
    if signature.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    return "UNKNOWN"


def _router_root(selected: Path) -> tuple[Path, Path]:
    selected = selected.resolve()
    if selected.name.casefold() == "picture" and selected.parent.is_dir():
        return selected.parent, selected
    picture = selected / "Picture"
    if picture.is_dir():
        return selected, picture
    return selected, selected


def _read_panels(result_dir: Path) -> list[AUO6000Panel]:
    panels: list[AUO6000Panel] = []
    if not result_dir.is_dir():
        return panels
    for path in sorted(result_dir.glob("_*.csv"), key=lambda item: item.name.casefold()):
        try:
            with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as fh:
                rows = csv.DictReader(fh)
                row = next((item for item in rows if (item.get("SN") or "").strip()), None)
        except (OSError, csv.Error):
            continue
        if not row:
            continue
        raw_result = (row.get("Result") or "").strip().casefold()
        result = "GOOD" if raw_result == "true" else "NG" if raw_result == "false" else "UNKNOWN"
        panels.append(
            AUO6000Panel(
                sn=(row.get("SN") or "").strip(),
                result=result,
                result_file=str(path.resolve()),
                result_at=_timestamp_from_name(path, RESULT_STAMP),
                recipe_name=(row.get("Recipe_Name") or "").strip(),
                product_id=(row.get("ProductId") or "").strip(),
                start_at=_timestamp_from_csv(row.get("Start_time")),
                end_at=_timestamp_from_csv(row.get("End_time")),
            )
        )
    panels.sort(key=lambda panel: panel.start_at or panel.result_at or datetime.max)
    return panels


def _picture_groups(
    paths: list[Path],
    result_boundaries: list[datetime] | None = None,
) -> list[list[tuple[Path, datetime | None]]]:
    boundaries = sorted(result_boundaries or [])
    stamped = [(path, _timestamp_from_name(path, PICTURE_STAMP)) for path in paths]
    stamped.sort(key=lambda item: (item[1] or datetime.max, item[0].name.casefold()))
    groups: list[list[tuple[Path, datetime | None]]] = []
    for item in stamped:
        if not groups:
            groups.append([item])
            continue
        previous_time = groups[-1][-1][1]
        current_time = item[1]
        if (
            previous_time is not None
            and current_time is not None
            and (
                (current_time - previous_time).total_seconds() > PANEL_GAP_SECONDS
                or bisect_left(boundaries, current_time) != bisect_left(boundaries, previous_time)
            )
        ):
            groups.append([item])
        else:
            groups[-1].append(item)
    return groups


def _pair_groups_with_panels(
    groups: list[list[tuple[Path, datetime | None]]],
    panels: list[AUO6000Panel],
) -> list[AUO6000Panel | None]:
    available = list(panels)
    paired: list[AUO6000Panel | None] = []
    for group in groups:
        end = group[-1][1]
        candidate = None
        if end is not None:
            after = [
                panel
                for panel in available
                if panel.result_at
                and 0 <= (panel.result_at - end).total_seconds() <= MAX_RESULT_LAG_SECONDS
            ]
            if after:
                candidate = min(after, key=lambda panel: panel.result_at - end)
        paired.append(candidate)
        if candidate is not None:
            available.remove(candidate)
    return paired


def _assign_picture_groups(
    paths: list[Path],
    panels: list[AUO6000Panel],
) -> tuple[
    list[tuple[list[tuple[Path, datetime | None]], AUO6000Panel | None]],
    int,
]:
    """Assign pictures using exact legacy windows, then bounded modern pairing."""
    stamped = [(path, _timestamp_from_name(path, PICTURE_STAMP)) for path in paths]
    exact_panels = [
        panel
        for panel in panels
        if panel.start_at is not None
        and panel.end_at is not None
        and panel.end_at >= panel.start_at
    ]
    exact_files = {panel.result_file for panel in exact_panels}
    exact_matches: dict[str, list[tuple[Path, datetime | None]]] = {
        panel.result_file: [] for panel in exact_panels
    }
    remaining: list[Path] = []
    ambiguous_images: list[tuple[Path, datetime | None]] = []
    ambiguous = 0

    for item in stamped:
        path, captured_at = item
        matches = (
            [panel for panel in exact_panels if panel.start_at <= captured_at <= panel.end_at]
            if captured_at is not None
            else []
        )
        if len(matches) == 1:
            exact_matches[matches[0].result_file].append(item)
        else:
            if len(matches) > 1:
                ambiguous += 1
                ambiguous_images.append(item)
            else:
                remaining.append(path)

    assignments: list[tuple[list[tuple[Path, datetime | None]], AUO6000Panel | None]] = []
    assignments.extend(([item], None) for item in ambiguous_images)
    for panel in exact_panels:
        group = exact_matches[panel.result_file]
        if group:
            group.sort(key=lambda item: (item[1] or datetime.max, item[0].name.casefold()))
            assignments.append((group, panel))

    modern_panels = [panel for panel in panels if panel.result_file not in exact_files]
    modern_groups = _picture_groups(
        remaining, [panel.result_at for panel in modern_panels if panel.result_at is not None]
    )
    modern_pairings = _pair_groups_with_panels(modern_groups, modern_panels)
    assignments.extend(zip(modern_groups, modern_pairings))
    assignments.sort(
        key=lambda assignment: (
            assignment[0][0][1] or datetime.max,
            assignment[0][0][0].name.casefold(),
        )
    )
    return assignments, ambiguous


def scan_auo6000_dataset(directory: str | Path) -> AUO6000Dataset:
    """Scan an AUO6000 export root or its Picture directory without changing files."""
    selected = Path(directory)
    root, picture_dir = _router_root(selected)
    result_dir = root / "Result"
    recipe_dir = root / "Recipe"
    log_dir = root / "Log"

    if picture_dir.is_dir():
        picture_paths = sorted(
            (
                path
                for path in picture_dir.rglob("*")
                if path.is_file() and path.suffix.casefold() in IMAGE_EXTENSIONS
            ),
            key=lambda path: path.name.casefold(),
        )
    else:
        picture_paths = []

    panels = _read_panels(result_dir)
    assignments, ambiguous_windows = _assign_picture_groups(picture_paths, panels)
    images: list[AUO6000Image] = []
    for group, panel in assignments:
        total = len(group)
        for cut_point, (path, captured_at) in enumerate(group, 1):
            images.append(
                AUO6000Image(
                    path=str(path.resolve()),
                    captured_at=captured_at,
                    panel_sn=panel.sn if panel else "",
                    cut_point=cut_point,
                    cut_point_total=total,
                    machine_result=panel.result if panel else "UNKNOWN",
                    result_file=panel.result_file if panel else "",
                    stored_format=_stored_image_format(path),
                )
            )

    recipe_files = (
        [str(path.resolve()) for path in sorted(recipe_dir.glob("*.rcp"))]
        if recipe_dir.is_dir()
        else []
    )
    log_files = (
        [str(path.resolve()) for path in sorted(log_dir.glob("*.log"))] if log_dir.is_dir() else []
    )
    warnings: list[str] = []
    disguised = sum(
        1
        for image in images
        if Path(image.path).suffix.casefold() == ".bmp" and image.stored_format != "BMP"
    )
    if disguised:
        warnings.append(
            f"{disguised} image(s) use .bmp filenames but contain another image format. "
            "AVTR reads them by file content."
        )
    unassigned = sum(1 for image in images if not image.result_file)
    if unassigned:
        warnings.append(
            f"{unassigned} image(s) could not be linked to a panel result within "
            f"{MAX_RESULT_LAG_SECONDS} seconds."
        )
    if ambiguous_windows:
        warnings.append(
            f"{ambiguous_windows} image(s) matched overlapping panel time windows and "
            "were left unassigned."
        )
    matched_results = {image.result_file for image in images if image.result_file}
    unmatched_panels = sum(1 for panel in panels if panel.result_file not in matched_results)
    if unmatched_panels:
        warnings.append(f"{unmatched_panels} panel result(s) did not receive any images.")
    if assignments and len({len(group) for group, _panel in assignments}) > 1:
        warnings.append("Panel image groups do not contain the same number of cut points.")

    return AUO6000Dataset(
        root=str(root),
        picture_dir=str(picture_dir),
        result_dir=str(result_dir) if result_dir.is_dir() else "",
        recipe_dir=str(recipe_dir) if recipe_dir.is_dir() else "",
        log_dir=str(log_dir) if log_dir.is_dir() else "",
        images=images,
        panels=panels,
        recipe_files=recipe_files,
        log_files=log_files,
        warnings=warnings,
    )
