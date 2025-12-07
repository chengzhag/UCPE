#!/usr/bin/env python3

import numpy as np

__all__ = ["nearest"]


def nearest(img: np.ndarray, grid: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Nearest Neightbor Sampling"""

    b, _, h, w = img.shape

    round_grid = np.rint(grid).astype(np.int64)
    round_grid[:, 0, ...] %= h
    round_grid[:, 1, ...] %= w

    for i in range(b):
        y = round_grid[i, 0, ...]
        x = round_grid[i, 1, ...]
        out[i, ...] = img[i][:, y, x]

    return out
