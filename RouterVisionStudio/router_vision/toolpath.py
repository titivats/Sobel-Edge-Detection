"""Cutting toolpath, read straight out of the machine's .rcp recipe.

The recipe is a .NET BinaryWriter stream. Geometry is stored as chains of
point records:

    [0x01][float X][float Y][float Z]      13 bytes each

preceded by a count, inside named elements ("Board", "SubBoard_01", "Line",
"FiducialMark").  Those names are UTF-16, but they do not always start on an
even byte offset, so searching for the text misses roughly half of them.
Scanning for the point pattern itself finds every segment regardless of
alignment, which is what this module does.

Only the header region is scanned - the tail of the file is a large embedded
image blob with no geometry in it.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path

# Geometry sits in the first tens of kB; the rest of the file is image data.
HEADER_SCAN_BYTES = 40_000
# Anything outside this is not a machine coordinate in millimetres.
COORD_LIMIT_MM = 5000.0
POINT_RECORD = 13


@dataclass
class Segment:
    """One closed or open polyline from the recipe."""

    offset: int
    points: list[tuple[float, float]]

    @property
    def is_slot(self) -> bool:
        """A 4-point segment is a routed slot / breakaway tab."""
        return len(self.points) == 4

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)

    @property
    def size_mm(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return x1 - x0, y1 - y0

    @property
    def centre(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0


@dataclass
class Toolpath:
    recipe: Path
    segments: list[Segment] = field(default_factory=list)

    @property
    def slots(self) -> list[Segment]:
        return [s for s in self.segments if s.is_slot]

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        if not self.segments:
            return 0.0, 0.0, 0.0, 0.0
        xs0, ys0, xs1, ys1 = zip(*(s.bounds for s in self.segments))
        return min(xs0), min(ys0), max(xs1), max(ys1)

    @property
    def span_mm(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.bounds
        return x1 - x0, y1 - y0

    @property
    def point_count(self) -> int:
        return sum(len(s.points) for s in self.segments)

    def summary(self) -> str:
        w, h = self.span_mm
        return (
            f"{len(self.segments)} segments, {len(self.slots)} slots, "
            f"{self.point_count} points, span {w:.2f} x {h:.2f} mm"
        )


def _read_chain(buf: bytes, start: int, end: int) -> tuple[list[tuple[float, float]], int]:
    points: list[tuple[float, float]] = []
    i = start
    while i + POINT_RECORD <= end:
        if buf[i] != 0x01:
            break
        x, y, z = struct.unpack_from("<fff", buf, i + 1)
        if not (isfinite(x) and isfinite(y) and isfinite(z)):
            break
        if abs(x) > COORD_LIMIT_MM or abs(y) > COORD_LIMIT_MM or abs(z) > COORD_LIMIT_MM:
            break
        points.append((x, y))
        i += POINT_RECORD
    return points, i


def load_toolpath(recipe_path: str | Path, scan_bytes: int = HEADER_SCAN_BYTES) -> Toolpath:
    """Parse the cutting geometry out of a recipe file."""
    path = Path(recipe_path)
    with open(path, "rb") as fh:
        buf = fh.read(scan_bytes)

    segments: list[Segment] = []
    end = len(buf)
    i = 0
    while i + POINT_RECORD <= end:
        if buf[i] == 0x01:
            points, nxt = _read_chain(buf, i, end)
            if len(points) >= 2:
                segments.append(Segment(offset=i, points=points))
                i = nxt
                continue
        i += 1
    return Toolpath(recipe=path, segments=segments)


def find_recipe(recipe_dir: str | Path, recipe_name: str) -> Path | None:
    """Recipes are filed under a customer sub-folder, so search one level down."""
    root = Path(recipe_dir)
    if not root.exists():
        return None
    name = recipe_name if recipe_name.lower().endswith(".rcp") else f"{recipe_name}.rcp"
    direct = root / name
    if direct.exists():
        return direct
    matches = list(root.glob(f"*/{name}"))
    if matches:
        return matches[0]
    stem = Path(name).stem.lower()
    for candidate in root.rglob("*.rcp"):
        if candidate.stem.lower() == stem:
            return candidate
    return None


def _quartiles(values: list[float]) -> tuple[float, float]:
    """Q1 and Q3 by nearest-rank, which needs no third-party maths."""
    ordered = sorted(values)
    n = len(ordered)
    return ordered[max(0, n // 4)], ordered[min(n - 1, (3 * n) // 4)]


def fit_bounds(
    segments: list[Segment], fence: float = 3.0
) -> tuple[tuple[float, float, float, float], list[int]]:
    """Bounds of the main body of the geometry, plus the outlying segments.

    The recipe is read by scanning bytes for point records, so a stray run of
    bytes can parse as a plausible but wildly misplaced segment. One of those is
    enough to blow up the drawing's bounding box and squash the real cut path
    into a corner, so the view is fitted to the cluster and told which segments
    were left out - they are still drawn, never silently dropped.

    Returns ((x0, y0, x1, y1), outlier_indices).
    """
    if not segments:
        return (0.0, 0.0, 0.0, 0.0), []

    def union(chosen: list[int]) -> tuple[float, float, float, float]:
        boxes = [segments[i].bounds for i in chosen]
        return (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )

    everything = list(range(len(segments)))
    if len(segments) < 4:
        return union(everything), []

    centres = [s.centre for s in segments]
    keep = everything
    for axis in (0, 1):
        values = [c[axis] for c in centres]
        q1, q3 = _quartiles(values)
        iqr = q3 - q1
        if iqr <= 0:
            continue  # every centre sits on one line: nothing to judge
        lo, hi = q1 - fence * iqr, q3 + fence * iqr
        keep = [i for i in keep if lo <= centres[i][axis] <= hi]

    # Refuse to "fit" if that means throwing away a large part of the drawing:
    # a genuinely spread-out panel must not be mistaken for noise.
    if len(keep) < max(2, int(0.6 * len(segments))):
        return union(everything), []
    return union(keep), [i for i in everything if i not in set(keep)]
