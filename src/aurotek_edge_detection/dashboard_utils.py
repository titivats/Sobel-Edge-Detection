from __future__ import annotations

import base64
import re
from datetime import datetime
from pathlib import Path
import tkinter as tk

import cv2


def timestamp_from_name(path: Path) -> datetime | None:
    match = re.search(r"(20\d{6})[_-](\d{6})", path.stem)
    if not match:
        return None
    try:
        return datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def tk_image_from_bgr(image_bgr, max_size: tuple[int, int]) -> tk.PhotoImage:
    image = image_bgr.copy()
    height, width = image.shape[:2]
    scale = min(max_size[0] / width, max_size[1] / height, 1.0)
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("Could not encode preview image")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=data)


def tk_image_from_bgr_fixed(
    image_bgr,
    size: tuple[int, int],
    background: tuple[int, int, int] = (0, 0, 0),
) -> tk.PhotoImage:
    width, height = size
    image_height, image_width = image_bgr.shape[:2]
    scale = min(width / image_width, height / image_height)
    resized_width = max(1, int(image_width * scale))
    resized_height = max(1, int(image_height * scale))
    resized = cv2.resize(
        image_bgr,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = cv2.copyMakeBorder(
        resized,
        (height - resized_height) // 2,
        height - resized_height - (height - resized_height) // 2,
        (width - resized_width) // 2,
        width - resized_width - (width - resized_width) // 2,
        cv2.BORDER_CONSTANT,
        value=background,
    )
    ok, encoded = cv2.imencode(".png", canvas)
    if not ok:
        raise ValueError("Could not encode preview image")
    data = base64.b64encode(encoded.tobytes()).decode("ascii")
    return tk.PhotoImage(data=data)
