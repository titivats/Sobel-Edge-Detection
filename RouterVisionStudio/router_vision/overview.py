"""Render inspection evidence without depending on the Qt application."""

from __future__ import annotations

import math

import cv2
import numpy as np

from .production import PanelDecision


def letterbox(image: np.ndarray | None, width: int, height: int) -> np.ndarray:
    """Fit an image into a fixed dark canvas without changing its aspect ratio."""
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    if image is None or image.size == 0:
        cv2.putText(
            canvas,
            "UNREADABLE",
            (max(8, width // 2 - 54), height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )
        return canvas
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    source_h, source_w = image.shape[:2]
    scale = min(width / max(1, source_w), height / max(1, source_h))
    target_w = max(1, int(round(source_w * scale)))
    target_h = max(1, int(round(source_h * scale)))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
    x0 = (width - target_w) // 2
    y0 = (height - target_h) // 2
    canvas[y0 : y0 + target_h, x0 : x0 + target_w] = resized
    return canvas


def render_prediction_overview(
    classifier,
    result: PanelDecision,
    good_confidence_min: float = 0.95,
    *,
    cell_width: int = 360,
    cell_height: int = 220,
    tile_context: dict[str, str] | None = None,
    columns: int | None = None,
) -> np.ndarray:
    """Render every cut point into one Original/Sobel/prediction overview.

    Each cut-point tile always reserves the left pane for the cropped original
    image and the right pane for the exact Sobel edge map used by the model.
    The prediction is written beside the cut-point name in the top strip so it
    never covers evidence in either image pane.
    """
    details = sorted(result.details, key=lambda item: item.index)
    count = len(details)
    if not count:
        empty = np.full((240, 720, 3), 24, dtype=np.uint8)
        cv2.putText(
            empty,
            "NO CUT-POINT IMAGES AVAILABLE",
            (118, 128),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (210, 210, 210),
            2,
            cv2.LINE_AA,
        )
        return empty

    cell_width = max(260, int(cell_width))
    cell_height = max(180, int(cell_height))
    columns = (
        min(6, max(1, math.ceil(math.sqrt(count))))
        if columns is None
        else max(1, min(int(columns), count))
    )
    rows = math.ceil(count / columns)
    gap = 10
    overview_width = columns * cell_width + (columns + 1) * gap
    overview_height = rows * cell_height + (rows + 1) * gap
    overview = np.full((overview_height, overview_width, 3), 20, dtype=np.uint8)

    for slot, detail in enumerate(details):
        row, column = divmod(slot, columns)
        x0 = gap + column * (cell_width + gap)
        y0 = gap + row * (cell_height + gap)
        tile = np.full((cell_height, cell_width, 3), 32, dtype=np.uint8)

        original = cv2.imread(str(detail.path), cv2.IMREAD_COLOR)
        if original is not None:
            original = classifier.crop.apply(original)
        edge = classifier.extractor.edge_map(detail.path, classifier.crop)

        top = 31
        bottom = 24
        pane_gap = 4
        pane_width = (cell_width - pane_gap) // 2
        pane_height = cell_height - top - bottom
        tile[top : top + pane_height, :pane_width] = letterbox(original, pane_width, pane_height)
        tile[top : top + pane_height, pane_width + pane_gap :] = letterbox(
            edge, cell_width - pane_width - pane_gap, pane_height
        )

        label = detail.label or "UNREADABLE"
        confidence = float(detail.confidence)
        if label == "GOOD" and confidence >= good_confidence_min:
            colour = (45, 145, 55)
        elif label == "NG":
            colour = (40, 40, 215)
        else:
            colour = (0, 140, 230)

        point_text = (
            tile_context.get(detail.path, f"CUT POINT {detail.index + 1:02d} | ")
            if tile_context
            else f"CUT POINT {detail.index + 1:02d} | "
        )
        prediction_text = f"PREDICT: {label} {confidence:.1%}"
        if label == "GOOD" and confidence < good_confidence_min:
            prediction_text += " | BELOW THRESHOLD"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        while font_scale > 0.32:
            point_width = cv2.getTextSize(point_text, font, font_scale, 1)[0][0]
            prediction_width = cv2.getTextSize(prediction_text, font, font_scale, 1)[0][0]
            if point_width + prediction_width <= cell_width - 18:
                break
            font_scale -= 0.02
        point_width = cv2.getTextSize(point_text, font, font_scale, 1)[0][0]
        header_colour = (
            (80, 220, 95)
            if label == "GOOD" and confidence >= good_confidence_min
            else (70, 80, 245)
            if label == "NG"
            else (30, 175, 245)
        )
        cv2.putText(
            tile,
            point_text,
            (9, 22),
            font,
            font_scale,
            (238, 238, 238),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            prediction_text,
            (9 + point_width, 22),
            font,
            font_scale,
            header_colour,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            "ORIGINAL",
            (8, cell_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            tile,
            "SOBEL EDGE",
            (pane_width + pane_gap + 8, cell_height - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (205, 205, 205),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(tile, (0, 0), (cell_width - 1, cell_height - 1), colour, 3)
        overview[y0 : y0 + cell_height, x0 : x0 + cell_width] = tile

    return overview
