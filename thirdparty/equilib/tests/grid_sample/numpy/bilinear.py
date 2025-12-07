#!/usr/bin/env python3

"""Bilinear Intopolation

"""

import numpy as np

from tests.grid_sample.numpy.baselines import (
    baseline_scipy_linear,
    baseline_cv2_linear,
)
from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer


def naive_bilinear(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """Simple bilinear interpolation

    NOTE: `img` and `grid` are 4 dim
    """

    _, _, h_in, w_in = img.shape
    b_out, _, h_out, w_out = out.shape

    def interp(v0, v1, d, L):
        return v0 * (1 - d) / L + v1 * d / L

    def interp2d(q00, q10, q01, q11, dy, dx):
        f0 = interp(q00, q01, dx, 1)
        f1 = interp(q10, q11, dx, 1)
        return interp(f0, f1, dy, 1)

    for b in range(b_out):
        for y_out in range(h_out):
            for x_out in range(w_out):
                y_in = grid[b, 0, y_out, x_out]
                x_in = grid[b, 1, y_out, x_out]
                y_min = np.floor(y_in).astype(np.int64)
                x_min = np.floor(x_in).astype(np.int64)
                y_max = y_min + 1
                x_max = x_min + 1
                dy = y_in - y_min
                dx = x_in - x_min

                # grid wrap
                y_min %= h_in
                x_min %= w_in
                y_max %= h_in
                x_max %= w_in

                p00 = img[b, :, y_min, x_min]
                p10 = img[b, :, y_max, x_min]
                p01 = img[b, :, y_min, x_max]
                p11 = img[b, :, y_max, x_max]

                out[b, :, y_out, x_out] = interp2d(p00, p10, p01, p11, dy, dx)

    return out


def faster_bilinear(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """Faster way of achieving bilinear without numba"""

    b_in, _, h_in, w_in = img.shape

    def interp(v0, v1, d, L):
        return v0 * (1 - d) / L + v1 * d / L

    def interp2d(q00, q10, q01, q11, dy, dx):
        f0 = interp(q00, q01, dx, 1)
        f1 = interp(q10, q11, dx, 1)
        return interp(f0, f1, dy, 1)

    min_grid = np.floor(grid).astype(np.int64)
    max_grid = min_grid + 1
    d_grid = grid - min_grid

    max_grid[:, 0, :, :] %= h_in
    max_grid[:, 1, :, :] %= w_in

    # FIXME: any way to do efficient batch?
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
        p11 = img[b][:, max_ys, max_xs]

        out[b, ...] = interp2d(p00, p10, p01, p11, dy, dx)

    return out


def bench_against_baselines():
    dtype_img = np.dtype(np.float64)
    dtype_grid = np.dtype(np.float32)
    b = 2
    c = 3
    h = 256
    w = 512
    h_grid = 64
    w_grid = 128

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=False,
        rand_img=False,
        rand_grid=False,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)
    out_naive = make_copies(out)
    out_faster = make_copies(out)

    out_scipy = func_timer(baseline_scipy_linear)(img, grid, out_scipy)
    out_cv2 = func_timer(baseline_cv2_linear)(img, grid, out_cv2)
    out_naive = func_timer(naive_bilinear)(img, grid, out_naive)
    out_faster = func_timer(faster_bilinear)(img, grid, out_faster)

    print("scipy vs naive")
    print("close?", check_close(out_naive, out_scipy))
    print("MSE", mse(out_naive, out_scipy))
    print("MAE", mae(out_naive, out_scipy))

    print("cv2 vs naive")
    print("close?", check_close(out_naive, out_cv2))
    print("MSE", mse(out_naive, out_cv2))
    print("MAE", mae(out_naive, out_cv2))

    print("faster vs naive")
    print("close?", check_close(out_naive, out_faster))
    print("MSE", mse(out_naive, out_faster))
    print("MAE", mae(out_naive, out_faster))


if __name__ == "__main__":
    np.random.seed(0)

    bench_against_baselines()
