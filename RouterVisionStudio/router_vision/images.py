"""Read images by content, including Unicode paths on Windows."""

from __future__ import annotations

import cv2
import numpy as np


def read_image(path, flags=cv2.IMREAD_COLOR):
    try:
        raw = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(raw, flags) if raw.size else None
    except (OSError, ValueError, cv2.error):
        return None
