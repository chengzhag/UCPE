#!/usr/bin/env python3

"""Real and fake data for testing

Data to mimic
- RGB
- Grayscale
- Depth
"""

import numpy as np


def get_gradient_2d(start, stop, width, height, is_horizontal):
    if is_horizontal:
        return np.tile(np.linspace(start, stop, width), (height, 1))
    else:
        return np.tile(np.linspace(start, stop, height), (width, 1)).T


def get_gradient_3d(width, height, start_list, stop_list, is_horizontal_list):
    result = np.zeros((height, width, len(start_list)), dtype=np.float)

    # channel wise
    for i, (start, stop, is_horizontal) in enumerate(
        zip(start_list, stop_list, is_horizontal_list)
    ):
        result[:, :, i] = get_gradient_2d(
            start, stop, width, height, is_horizontal
        )

    return result


# Base Media Class


class BaseImage:
    ...


# Grayscale Image


class GrayScaleImage:
    ...


def grayscale_gradient(
    height: int,
    width: int,
    dtype: np.dtype = np.dtype(np.uint8),  # np.float32, np.uint8
) -> np.ndarray:
    result = np.zeros((height, width, 1), dtype=dtype)
    result[:, :, 0] = np.tile(np.linspace(0, 255, width), (height, 1))
    return result


# RGB Image


class RGB:
    ...


def rgb_gradient(
    height: int, width: int, dtype: np.dtype = np.dtype(np.uint8)
) -> np.ndarray:
    # NOTE: set parameters
    start_rgb = (0, 0, 192)
    stop_rgb = (255, 255, 64)
    horizontals = (True, False, False)

    result = np.zeros((height, width, 3), dtype=dtype)
    for i, (start, stop, is_horizontal) in enumerate(
        zip(start_rgb, stop_rgb, horizontals)
    ):
        result[:, :, i] = get_gradient_2d(
            start, stop, width, height, is_horizontal
        )
    return result


# Depth Image


def depth_grayscale_gradient(
    height: int, width: int, dtype: np.dtype = np.dtype(np.float64)
) -> np.ndarray:
    ...


if __name__ == "__main__":
    from PIL import Image

    hgs = grayscale_gradient(width=128, height=64, dtype=np.uint8)

    print(hgs.shape)

    # Image.fromarray(hgs.squeeze(-1)).save('tests/common/results/hgs.jpg', quality=100)

    color = rgb_gradient(128, 64)
    Image.fromarray(color).save("tests/common/results/color.jpg", quality=100)
