from __future__ import annotations

import cv2
import numpy as np


def create_sobel_edge_masks(
    image: np.ndarray,
    threshold_ratio: float,
    close_kernel_size: int = 1,
    close_iterations: int = 0,
    dilate_iterations: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create thresholded Sobel X, Sobel Y, and combined edge masks."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    abs_gradient_x = np.abs(gradient_x)
    abs_gradient_y = np.abs(gradient_y)
    gradient_magnitude = cv2.magnitude(gradient_x, gradient_y)

    edge_x = _threshold_gradient(abs_gradient_x, threshold_ratio)
    edge_y = _threshold_gradient(abs_gradient_y, threshold_ratio)
    edge_all = _threshold_gradient(gradient_magnitude, threshold_ratio)

    return (
        repair_edge_mask(edge_x, close_kernel_size, close_iterations, dilate_iterations),
        repair_edge_mask(edge_y, close_kernel_size, close_iterations, dilate_iterations),
        repair_edge_mask(edge_all, close_kernel_size, close_iterations, dilate_iterations),
    )


def thicken_edge_for_display(edge_mask: np.ndarray, thickness: int) -> np.ndarray:
    """Optionally thicken saved edge images for readability."""
    if thickness <= 1:
        return edge_mask
    kernel = np.ones((thickness, thickness), dtype=np.uint8)
    return cv2.dilate(edge_mask, kernel, iterations=1)


def repair_edge_mask(
    edge_mask: np.ndarray,
    close_kernel_size: int,
    close_iterations: int,
    dilate_iterations: int,
) -> np.ndarray:
    """Connect small gaps in a Sobel mask before saving the edge image."""
    repaired = edge_mask
    close_kernel_size = _normalized_kernel_size(close_kernel_size)
    close_iterations = max(int(close_iterations), 0)
    dilate_iterations = max(int(dilate_iterations), 0)

    if close_kernel_size > 1 and close_iterations > 0:
        close_kernel = np.ones((close_kernel_size, close_kernel_size), dtype=np.uint8)
        repaired = cv2.morphologyEx(
            repaired,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=close_iterations,
        )
    if dilate_iterations > 0:
        dilate_kernel = np.ones((3, 3), dtype=np.uint8)
        repaired = cv2.dilate(repaired, dilate_kernel, iterations=dilate_iterations)
    return repaired


def _threshold_gradient(gradient: np.ndarray, threshold_ratio: float) -> np.ndarray:
    max_gradient = float(gradient.max())
    if max_gradient == 0:
        return np.zeros(gradient.shape, dtype=np.uint8)
    return np.where(gradient >= max_gradient * threshold_ratio, 255, 0).astype(np.uint8)


def _normalized_kernel_size(kernel_size: int) -> int:
    kernel_size = max(int(kernel_size), 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size
