from __future__ import annotations

from pathlib import Path

import numpy as np

from file_io import atomic_write_image


def save_sobel_edge_image(
    output_dir: Path,
    image_stem: str,
    display_edge: np.ndarray,
) -> Path:
    sobel_dir = output_dir / "sobel_edges"
    sobel_dir.mkdir(parents=True, exist_ok=True)
    output_path = sobel_dir / f"{image_stem}_sobel_edge.png"
    atomic_write_image(output_path, display_edge)
    return output_path
