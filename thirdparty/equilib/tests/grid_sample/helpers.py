#!/usr/bin/env python3

"""Helper functions for grid_sample

- mainly tries to create a good dataset to test `grid_sample`
"""

from copy import deepcopy
from typing import Tuple

import numpy as np

from tests.helpers.sample_data import grayscale_gradient, rgb_gradient

np.random.seed(0)
print("DEBUG: random is set to 0")

NP_FLOATS = (np.float32, np.float64)


def make_copies(a: np.ndarray) -> np.ndarray:
    return deepcopy(a)


def create_single_data(
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    move_grid: bool = False,
    rand_img: bool = False,
    rand_grid: bool = False,
    dtype_img: np.dtype = np.dtype(np.uint8),
    dtype_grid: np.dtype = np.dtype(np.float32),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = create_single_img(c=c, h=h, w=w, rand=rand_img, dtype=dtype_img)

    grid = create_single_grid(
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        rand=rand_grid,
        move=move_grid,
        dtype=dtype_grid,
    )

    out = np.empty((c, h_grid, w_grid), dtype=dtype_img)

    return img, grid, out


def create_batch_data(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    move_grid: bool = False,
    rand_img: bool = False,
    rand_grid: bool = False,
    dtype_img: np.dtype = np.dtype(np.uint8),
    dtype_grid: np.dtype = np.dtype(np.float32),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img = create_batch_img(b=b, c=c, h=h, w=w, rand=rand_img, dtype=dtype_img)

    grid = create_batch_grid(
        b=b,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        rand=rand_grid,
        move=move_grid,
        dtype=dtype_grid,
    )

    out = np.empty((b, c, h_grid, w_grid), dtype=dtype_img)

    check_before_sampling(img, grid, out)

    # if b == 1:
    #     img = img.squeeze(0)
    #     grid = grid.squeeze(0)
    #     out = out.squeeze(0)

    return img, grid, out


def check_before_sampling(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> None:
    """Check image and grid shapes before sampling algorithm"""
    l_img = len(img.shape)
    l_grid = len(grid.shape)
    l_out = len(out.shape)

    assert 3 <= l_img <= 4 and 3 <= l_grid <= 4 and 3 <= l_out <= 4
    assert l_img == l_grid == l_out
    assert img.shape[0] == grid.shape[0] == out.shape[0]


def create_single_img(
    c: int,
    h: int,
    w: int,
    rand: bool = False,
    dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    if rand:
        img = (np.random.rand(c, h, w) * 255).astype(dtype)
    else:
        if c == 1:
            img = grayscale_gradient(height=h, width=w, dtype=dtype).transpose(
                2, 0, 1
            )
        elif c == 3:
            img = rgb_gradient(height=h, width=w, dtype=dtype).transpose(
                2, 0, 1
            )
        else:
            raise ValueError()

    if dtype in NP_FLOATS:
        img = (img / 255.0).astype(dtype)

    return img


def create_batch_img(
    b: int,
    c: int,
    h: int,
    w: int,
    rand: bool = False,
    dtype: np.dtype = np.dtype(np.uint8),
) -> np.ndarray:
    imgs = np.empty((b, c, h, w), dtype=dtype)
    for i in range(b):
        imgs[i, ...] = create_single_img(c, h, w, rand=rand, dtype=dtype)

    return imgs


def create_single_grid(
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    rand: bool = False,
    move: bool = False,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    assert h >= h_grid and w >= w_grid
    assert dtype in NP_FLOATS

    # create a 0 ~ 1 grid
    _xs = np.linspace(0, w_grid - 1, w_grid, dtype=np.float64) / (w_grid - 1)
    _ys = np.linspace(0, h_grid - 1, h_grid, dtype=np.float64) / (h_grid - 1)

    # hard code grid size respect to the image size (h, w)
    gw = w / 4
    gh = h / 4

    # convert grid to actual pixels in the image
    _xs *= gw - 1
    _ys *= gh - 1

    # center the grid in the image
    _xs += (w - gw) / 2
    _ys += (h - gh) / 2

    # add noise
    if rand:
        rx = 2 * np.random.rand(*_xs.shape) - 1
        ry = 2 * np.random.rand(*_ys.shape) - 1
        _xs += rx
        _ys += ry

    # move the grid
    if move:
        if rand:
            mx = w * (2 * np.random.rand(1) - 1)
            my = h * (2 * np.random.rand(1) - 1)
            _xs += mx
            _ys += my
        else:
            # move to the seam
            mx = w / 2
            my = h / 2
            _xs += mx
            _ys += my

    # wrap around
    _xs %= w
    _ys %= h

    # set type since random defaults to float64
    _xs = _xs.astype(dtype)
    _ys = _ys.astype(dtype)

    xs, ys = np.meshgrid(_xs, _ys)
    return np.stack((ys, xs), axis=0)


def create_batch_grid(
    b: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    rand: bool = False,
    move: bool = False,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    grids = np.empty((b, 2, h_grid, w_grid), dtype=dtype)
    for i in range(b):
        grids[i, ...] = create_single_grid(
            h=h,
            w=w,
            h_grid=h_grid,
            w_grid=w_grid,
            rand=rand,
            move=move,
            dtype=dtype,
        )

    return grids
