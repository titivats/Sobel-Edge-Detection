from __future__ import annotations

import cv2
import numpy as np


def create_sobel_edge_masks(
    image: np.ndarray,
    threshold_ratio: float,
    blur_kernel_size: int = 5,
    sobel_kernel_size: int = 3,
    cleanup_kernel_size: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create Sobel X, Sobel Y, and combined Sobel edge masks."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_kernel_size = _normalize_odd_kernel(blur_kernel_size, minimum=1, maximum=31)
    sobel_kernel_size = _normalize_odd_kernel(sobel_kernel_size, minimum=1, maximum=31)
    cleanup_kernel_size = _normalize_odd_kernel(cleanup_kernel_size, minimum=1, maximum=31)

    if blur_kernel_size > 1:
        gray_image = cv2.GaussianBlur(gray_image, (blur_kernel_size, blur_kernel_size), 0)

    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=sobel_kernel_size)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=sobel_kernel_size)
    abs_gradient_x = np.abs(gradient_x)
    abs_gradient_y = np.abs(gradient_y)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    edge_x = _threshold_gradient(abs_gradient_x, threshold_ratio)
    edge_y = _threshold_gradient(abs_gradient_y, threshold_ratio)
    edge_all = _threshold_gradient(gradient_magnitude, threshold_ratio)

    if cleanup_kernel_size <= 1:
        return edge_x, edge_y, edge_all

    cleanup_kernel = np.ones((cleanup_kernel_size, cleanup_kernel_size), dtype=np.uint8)
    return (
        cv2.morphologyEx(edge_x, cv2.MORPH_CLOSE, cleanup_kernel),
        cv2.morphologyEx(edge_y, cv2.MORPH_CLOSE, cleanup_kernel),
        cv2.morphologyEx(edge_all, cv2.MORPH_CLOSE, cleanup_kernel),
    )


def thicken_edge_for_display(edge_mask: np.ndarray, thickness: int) -> np.ndarray:
    """Thicken saved edge images for readability without changing measurement."""
    if thickness <= 1:
        return edge_mask
    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    return cv2.dilate(edge_mask, kernel, iterations=1)


def _threshold_gradient(gradient: np.ndarray, threshold_ratio: float) -> np.ndarray:
    max_gradient = float(gradient.max())
    if max_gradient == 0:
        return np.zeros(gradient.shape, dtype=np.uint8)
    return np.where(gradient >= max_gradient * threshold_ratio, 255, 0).astype(np.uint8)


def _normalize_odd_kernel(value: int, minimum: int, maximum: int) -> int:
    kernel_size = int(round(value))
    kernel_size = min(max(kernel_size, minimum), maximum)
    if kernel_size % 2 == 0:
        kernel_size += 1 if kernel_size < maximum else -1
    return kernel_size
