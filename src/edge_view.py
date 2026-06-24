from __future__ import annotations

import numpy as np


def select_edge(
    view_mode: int,
    edge_x: np.ndarray,
    edge_y: np.ndarray,
    edge_all: np.ndarray,
) -> np.ndarray:
    if view_mode == 1:
        return edge_x
    if view_mode == 2:
        return edge_y
    if view_mode == 3:
        return edge_all
    return edge_x if np.count_nonzero(edge_x) >= np.count_nonzero(edge_y) else edge_y


def edge_view_name(view_mode: int) -> str:
    if view_mode == 1:
        return "x"
    if view_mode == 2:
        return "y"
    return "all"
