from __future__ import annotations

import cv2
import numpy as np


def make_dark_mask(image: np.ndarray, black_threshold: int) -> np.ndarray:
    """Return a binary mask where the router background is black enough."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark_mask = np.where(gray_image <= black_threshold, 255, 0).astype(np.uint8)

    cleanup_kernel = np.ones((5, 5), dtype=np.uint8)
    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        cleanup_kernel,
        iterations=2,
    )
    return cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        cleanup_kernel,
        iterations=1,
    )
