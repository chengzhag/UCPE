#!/usr/bin/env python3

import numpy as np

import torch

from tests.grid_sample.numpy.bilinear import (
    faster_bilinear as faster_bilinear_numpy,
)
from tests.grid_sample.torch.native import native_bilinear
from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer


def naive_bilinear(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    ...


def faster_bilinear(
    img: torch.Tensor, grid: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    b_in, _, h_in, w_in = img.shape

    min_grid = torch.floor(grid).type(torch.int64)
    max_grid = min_grid + 1
    d_grid = grid - min_grid

    # sometimes, min goes out-of-bounds...
    min_grid[:, 0, :, :] %= h_in
    min_grid[:, 1, :, :] %= w_in
    max_grid[:, 0, :, :] %= h_in
    max_grid[:, 1, :, :] %= w_in

    def linear_interp(v0, v1, d, L):
        return v0 * (1 - d) / L + v1 * d / L

    def interp2d(q00, q10, q01, q11, dy, dx):
        f0 = linear_interp(q00, q01, dx, 1)
        f1 = linear_interp(q10, q11, dx, 1)
        return linear_interp(f0, f1, dy, 1)

    # FIXME: looping batch... slow...
    for b in range(b_in):
        dy = d_grid[b, 0, ...]
        dx = d_grid[b, 1, ...]
        min_ys = min_grid[b, 0, ...]
        min_xs = min_grid[b, 1, ...]
        max_ys = max_grid[b, 0, ...]
        max_xs = max_grid[b, 1, ...]

        p00 = img[b][:, min_ys, min_xs]
        p10 = img[b][:, max_ys, min_xs]
        p01 = img[b][:, min_ys, max_xs]
        p11 = img[b][:, min_ys, max_xs]

        out[b, ...] = interp2d(p00, p10, p01, p11, dy, dx)

    return out


def compare_baseline():
    dtype_img = dtype_grid = np.dtype(np.float32)
    b = 16
    c = 3
    h = 2000
    w = 4000
    h_grid = 256
    w_grid = 512

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=True,
        rand_img=False,
        rand_grid=False,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    # initialize outputs:
    out_numpy = make_copies(out)
    out_native = make_copies(out)
    out_torch = torch.from_numpy(make_copies(out))

    out_numpy = func_timer(faster_bilinear_numpy)(img, grid, out_numpy)
    out_numpy = torch.from_numpy(out_numpy)

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)

    out_torch = func_timer(faster_bilinear)(img, grid, out_torch)

    print("\nChecking: numpy vs torch")
    print("close?", check_close(out_numpy, out_torch))
    print("MSE", mse(out_numpy, out_torch))
    print("MAE", mae(out_numpy, out_torch))

    out_native = func_timer(native_bilinear)(img.clone(), grid.clone())

    print("\nChecking: numpy vs native")
    print("close?", check_close(out_numpy, out_native))
    print("MSE", mse(out_numpy, out_native))
    print("MAE", mae(out_numpy, out_native))


if __name__ == "__main__":
    compare_baseline()
