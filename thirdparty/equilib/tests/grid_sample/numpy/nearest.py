#!/usr/bin/env python3

"""Nearest Neighbor

"""

from numba import njit

import numpy as np

from tests.grid_sample.numpy.baselines import (
    baseline_scipy_nearest,
    baseline_cv2_nearest,
)
from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer


def naive_nearest(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """Simple nearest neighbor interpolation

    NOTE: `img` and `grid` are 4 dims
    """

    _, _, h_in, w_in = img.shape
    bs, _, h_out, w_out = out.shape

    for b in range(bs):
        for y_out in range(h_out):
            for x_out in range(w_out):
                _y, _x = np.rint(grid[b, :, y_out, x_out]).astype(np.int64)
                _y %= h_in
                _x %= w_in

                out[b, :, y_out, x_out] = img[b, :, _y, _x]

    return out


def faster_nearest(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """Simple nearest neighbor interpolation

    Faster than scipy but slower than cv2
    - actually faster when the image size is larger
    """

    b_in, _, h_in, w_in = img.shape

    # FIXME: any method to run at once?
    round_grid = np.rint(grid).astype(np.int64)
    round_grid[:, 0, ...] %= h_in
    round_grid[:, 1, ...] %= w_in
    for b in range(b_in):
        y = round_grid[b, 0, ...]
        x = round_grid[b, 1, ...]
        out[b, ...] = img[b][:, y, x]

    return out


@njit
def run(img, grid, out, b, h, w):
    for i in range(b):
        for y_out in range(h):
            for x_out in range(w):
                _y, _x = grid[i, :, y_out, x_out]
                out[i, :, y_out, x_out] = img[i, :, _y, _x]
    return out


def numba_nearest(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """numba nearest neighbor interpolation

    so far, slower than "faster", but faster than scipy

    FIXME: better optimization for numba?
    """

    b_in, _, h_in, w_in = img.shape
    _, _, h_out, w_out = out.shape

    grid = np.rint(grid).astype(np.int64)
    grid[:, 0, ...] %= h_in
    grid[:, 1, ...] %= w_in

    out = run(img, grid, out, b_in, h_out, w_out)

    return out


def bench_against_baselines():
    dtype_img = np.dtype(np.float64)
    dtype_grid = np.dtype(np.float32)
    b = 2
    c = 3
    h = 64
    w = 128
    h_grid = 16
    w_grid = 32

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

    out_scipy = func_timer(baseline_scipy_nearest)(img, grid, out_scipy)
    out_cv2 = func_timer(baseline_cv2_nearest)(img, grid, out_cv2)
    out_naive = func_timer(naive_nearest)(img, grid, out_naive)
    out_faster = func_timer(faster_nearest)(img, grid, out_faster)

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


def bench_numba():
    dtype_img = np.dtype(np.float64)
    dtype_grid = np.dtype(np.float32)
    b = 32
    c = 3
    h = 2000
    w = 4000
    h_grid = 128
    w_grid = 256

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

    out_faster = make_copies(out)
    out_numba = make_copies(out)

    out_faster = func_timer(faster_nearest)(img, grid, out_faster)
    out_numba = func_timer(numba_nearest)(img, grid, out_numba)
    out_numba = func_timer(numba_nearest)(img, grid, out_numba)

    print("numba vs faster")
    print("close?", check_close(out_numba, out_faster))
    print("MSE", mse(out_numba, out_faster))
    print("MAE", mae(out_numba, out_faster))
    # faster than `faster` but at the cost of compiling the first time


if __name__ == "__main__":
    np.random.seed(0)

    bench_against_baselines()
    bench_numba()
