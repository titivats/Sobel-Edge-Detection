"""Slot measurement.

The inspection camera looks straight down at a routed channel: two strips of
green solder mask with the cut running between them.  For each scan row we walk
outward from the image centre until we hit solder mask on either side; the gap
between those two hits is the slot.  Taking the median over many rows makes the
result immune to the debris and burrs that sit in the channel.

This is plain thresholding plus a median - deterministic, no training, no model
file.  Calibration only records what "normal" looks like.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

# A side is only called rough when it wanders much more than the other one, so
# an ordinary straight slot with a speck of debris is never split up.
ROUGH_MIN_SPREAD_PX = 8.0  # below this the side counts as straight, full stop
ROUGH_RATIO = 4.0  # and it has to wander this many times the other side


def _spread(values: np.ndarray) -> float:
    """How far an edge wanders, ignoring the few worst rows.

    The 5th-to-95th percentile range answers 'is this edge a straight line?'
    without being fooled either way: a couple of debris hits on a straight edge
    barely move it, while a scalloped edge - a tab or a contour cut - blows it up.
    """
    if values.size == 0:
        return 0.0
    lo, hi = np.percentile(values, (5, 95))
    return float(hi - lo)


@dataclass
class CutMeasurement:
    valid: bool = False
    attempted: int = 0  # scan lines tried
    rows: int = 0  # scan lines that found both edges
    slot_left: int = 0  # median near edge, px (left for vertical, top for horizontal)
    slot_right: int = 0  # median far edge, px  (right / bottom)
    width: int = 0  # slot_right - slot_left, px - the gap across the slot
    width_sd: float = 0.0  # spread of per-line width, px
    orientation: str = "vertical"  # which way the channel runs
    left_spread: float = 0.0  # 5-95 percentile range of the near edge, px
    right_spread: float = 0.0  # same for the far edge
    rough_side: str = ""  # "left" / "right" when one side is not a straight edge
    row_y: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    row_l: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    row_r: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=int))
    error: str = ""

    @property
    def is_horizontal(self) -> bool:
        return self.orientation == "horizontal"

    @property
    def width_usable(self) -> bool:
        """A width needs two straight edges; one scalloped side makes it meaningless."""
        return self.valid and not self.rough_side

    @property
    def ref_side(self) -> str:
        """The side worth tracking: the straight one."""
        return "right" if self.rough_side == "left" else "left"

    @property
    def ref_edge(self) -> int:
        return self.slot_right if self.ref_side == "right" else self.slot_left


def _scan(mask: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Walk outward from the centre of each row of `mask` to the first set pixel.

    `mask` is (N, M): one scan line per row. Returns the lines that found solder
    mask on both sides, with the near and far edge positions.
    """
    centre = mask.shape[1] // 2
    near_part = mask[:, : centre + 1]
    far_part = mask[:, centre:]

    has_near = near_part.any(axis=1)
    has_far = far_part.any(axis=1)
    near = centre - np.argmax(near_part[:, ::-1], axis=1)
    far = centre + np.argmax(far_part, axis=1)

    ok = has_near & has_far & (far > near)
    return coords[ok], near[ok].astype(int), far[ok].astype(int)


def measure_image(
    path: str | Path,
    *,
    y0: int = 0,
    y1: int = 4000,  # clamped to the picture height: "down to the bottom"
    step: int = 1,
    green_delta: int = 5,
    green_min: int = 25,
    green_max: int = 140,
    min_valid_rows: int = 300,
    orientation: str = "auto",
) -> CutMeasurement:
    """Measure the routed slot in one inspection image.

    Some products present the channel running down the frame and others across
    it, so by default both directions are tried and the one that actually finds
    the slot wins.
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return CutMeasurement(error=f"cannot read image: {path}")

    height, width = img.shape[:2]
    step = max(1, step)

    g = img[:, :, 1].astype(np.int16)
    r = img[:, :, 2].astype(np.int16)
    mask_full = (g - r > green_delta) & (g > green_min) & (g < green_max)

    candidates = []

    if orientation in ("auto", "vertical"):
        # scan window limits which rows are used; each row is scanned left/right
        ry0, ry1 = max(0, y0), min(height, y1)
        if ry1 > ry0:
            ys = np.arange(ry0, ry1, step)
            band = mask_full[ry0:ry1:step]
            ys = ys[: band.shape[0]]
            candidates.append(("vertical",) + _scan(band, ys))

    if orientation in ("auto", "horizontal"):
        # mirror image: each column is scanned up/down, so transpose and reuse
        cx0, cx1 = 0, width
        xs = np.arange(cx0, cx1, step)
        band = mask_full[:, cx0:cx1:step].T  # (n_columns, height)
        xs = xs[: band.shape[0]]
        candidates.append(("horizontal",) + _scan(band, xs))

    if not candidates:
        return CutMeasurement(error="empty scan window")

    # pick whichever direction found the slot on more scan lines
    best = max(candidates, key=lambda c: len(c[1]))
    name, coords, near, far = best

    attempted = 0
    for cand in candidates:
        if cand[0] == name:
            attempted = max(attempted, len(cand[1]))
    if name == "vertical":
        attempted = len(np.arange(max(0, y0), min(height, y1), step))
    else:
        attempted = len(np.arange(0, width, step))

    out = CutMeasurement(attempted=int(attempted), rows=int(len(coords)), orientation=name)
    out.row_y = coords
    out.row_l = near
    out.row_r = far

    if out.rows >= min_valid_rows:
        widths = out.row_r - out.row_l
        out.slot_left = int(np.median(out.row_l))
        out.slot_right = int(np.median(out.row_r))
        out.width = out.slot_right - out.slot_left
        out.width_sd = float(np.std(widths))
        out.valid = True

        # Not every cut has two straight sides: a tab or a contour leaves one
        # edge scalloped, and a width measured against it is noise. Work out
        # which side is which and let the caller drop the bad one.
        out.left_spread = _spread(out.row_l)
        out.right_spread = _spread(out.row_r)
        left, right = out.left_spread, out.right_spread
        if left > ROUGH_MIN_SPREAD_PX and left > ROUGH_RATIO * max(right, 1e-6):
            out.rough_side = "left"
        elif right > ROUGH_MIN_SPREAD_PX and right > ROUGH_RATIO * max(left, 1e-6):
            out.rough_side = "right"
        # both sides equally busy: nothing to prefer, so keep the pair
    return out


OVERLAY_DOTS = 90  # how many scan rows get a yellow dot drawn


def render_overlay(path: str | Path, meas: CutMeasurement) -> np.ndarray:
    """Return a BGR image with the measurement drawn on top, for the viewer."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    return draw_overlay(img, meas)


def draw_overlay(img: np.ndarray, meas: CutMeasurement, scale: float = 1.0) -> np.ndarray:
    """Draw the measurement onto an image that is already loaded.

    `scale` says how much that image has been shrunk from the one that was
    measured, so thumbnails can be drawn on after resizing - marks drawn at full
    size and then shrunk would blur away to nothing.
    """
    height, width = img.shape[:2]
    thin = scale < 0.5
    weight = 1 if thin else 2
    dots = OVERLAY_DOTS // 2 if thin else OVERLAY_DOTS

    def at(value: float) -> int:
        return int(round(value * scale))

    horizontal = meas.is_horizontal

    # A scalloped side is left undrawn as well as unmeasured: showing marks that
    # nothing is judged against is what made the picture hard to read.
    draw_near = meas.rough_side != "left"
    draw_far = meas.rough_side != "right"

    if meas.valid:
        # measured edges - cyan, drawn across the slot
        edges = ([meas.slot_left] if draw_near else []) + ([meas.slot_right] if draw_far else [])
        for edge in edges:
            e = at(edge)
            if horizontal:
                cv2.line(img, (0, e), (width, e), (255, 255, 0), weight)
            else:
                cv2.line(img, (e, 0), (e, height), (255, 255, 0), weight)
        # Per-line hits - yellow. Every scan row is measured, but only a sample
        # of them is drawn: a dot for all thousand-odd rows would merge into a
        # solid bar and stop showing whether the edge is steady. The measurement
        # itself always uses every row.
        stride = max(1, len(meas.row_y) // dots)
        for coord, near, far in zip(
            meas.row_y[::stride], meas.row_l[::stride], meas.row_r[::stride]
        ):
            hits = ([near] if draw_near else []) + ([far] if draw_far else [])
            for edge in hits:
                point = (at(coord), at(edge)) if horizontal else (at(edge), at(coord))
                cv2.circle(img, point, weight, (0, 255, 255), -1)

    # The baseline is not drawn: it is a number to compare against, not a feature
    # of this picture, and a second pair of lines only made the cut harder to see.
    # dWidth and dShift carry that comparison instead.
    return img
