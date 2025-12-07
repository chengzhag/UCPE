#!/usr/bin/env python3

import warnings

import numpy as np

from .bicubic import bicubic
from .bilinear import bilinear
from .nearest import nearest


def grid_sample(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray, mode: str = "bilinear"
) -> np.ndarray:
    """Numpy grid sampling algorithm

    params:
    - img (np.ndarray)
    - grid (np.ndarray)
    - out (np.ndarray)
    - mode (str): ('bilinear', 'bicubic', 'nearest')

    return:
    - out (np.ndarray)

    NOTE:
    - assumes that `img`, `grid`, and `out` have the same dimension of
      (batch, channel, height, width).
    - channel for `grid` should be 2 (yx)

    """

    if mode == "nearest":
        out = nearest(img, grid, out)
    elif mode == "bilinear":
        out = bilinear(img, grid, out)
    elif mode == "bicubic":
        # FIXME: bicubic algorithm is not perfect yet
        warnings.warn(
            "Bicubic interpolation is not perfect (especially when upsampling). Use with care!"
        )
        out = bicubic(img, grid, out)
    else:
        raise ValueError(f"ERR: {mode} is not supported")

    return out
