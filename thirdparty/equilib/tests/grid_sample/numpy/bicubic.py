#!/usr/bin/env python3

"""Bicubic interpolation

FIXME: very different outcome compared with scipy or cv2

NOTE: on PyTorch's implementation, it said it uses -0.75

- cv2 uses -0.75
- PIL uses -0.5

I wonder what scipy uses

"""

import numpy as np

from tests.grid_sample.numpy.baselines import (
    baseline_scipy_cubic,
    baseline_cv2_cubic,
)
from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer


def naive_bicubic(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """Bicubic single image

    Reference "Digital Image Processing p167-169
    """
    a: float = -0.75  # should be between -0.75 ~ -0.5

    b_in, c_in, h_in, w_in = img.shape
    b_out, _, h_out, w_out = grid.shape
    dtype = img.dtype

    def kernel(s, a):
        s = np.abs(s)
        if 0 <= s <= 1:
            return (a + 2) * (s**3) - (a + 3) * (s**2) + 1
        elif 1 < s <= 2:
            return a * (s**3) - (5 * a) * (s**2) + (8 * a) * abs(s) - 4 * a
        else:
            return 0

    def check_range(p, high):
        # FIXME: not really checking range
        return p % high

    for b in range(b_out):
        for y_out in range(h_out):
            for x_out in range(w_out):
                y_in = grid[b, 0, y_out, x_out]
                x_in = grid[b, 1, y_out, x_out]

                y_min = np.floor(y_in).astype(np.int64)
                x_min = np.floor(x_in).astype(np.int64)

                dy1 = 1 + (y_in - y_min)
                dy2 = y_in - y_min
                dy3 = y_min + 1 - y_in
                dy4 = y_min + 2 - y_in

                dx1 = 1 + (x_in - x_min)
                dx2 = x_in - x_min
                dx3 = x_min + 1 - x_in
                dx4 = x_min + 2 - x_in

                # calculate edge cases
                y1 = check_range(int(y_in - dy1), h_in)
                y2 = check_range(int(y_in - dy2), h_in)
                y3 = check_range(int(y_in + dy3), h_in)
                y4 = check_range(int(y_in + dy4), h_in)

                x1 = check_range(int(x_in - dx1), w_in)
                x2 = check_range(int(x_in - dx2), w_in)
                x3 = check_range(int(x_in + dx3), w_in)
                x4 = check_range(int(x_in + dx4), w_in)

                # creating left and right matrices
                mat_l = [
                    kernel(dx1, a),
                    kernel(dx2, a),
                    kernel(dx3, a),
                    kernel(dx4, a),
                ]
                mat_r = [
                    kernel(dy1, a),
                    kernel(dy2, a),
                    kernel(dy3, a),
                    kernel(dy4, a),
                ]
                mat_l = np.array(mat_l, dtype=dtype)
                mat_r = np.array(mat_r, dtype=dtype)
                mat_l = mat_l[np.newaxis, ...]
                mat_r = mat_r[..., np.newaxis]

                # sample pixel location from 16 locations
                # FIXME: find a better way of indexing
                mat_m = np.array(
                    [
                        [
                            img[b, :, y1, x1],
                            img[b, :, y2, x1],
                            img[b, :, y3, x1],
                            img[b, :, y4, x1],
                        ],
                        [
                            img[b, :, y1, x2],
                            img[b, :, y2, x2],
                            img[b, :, y3, x2],
                            img[b, :, y4, x2],
                        ],
                        [
                            img[b, :, y1, x3],
                            img[b, :, y2, x3],
                            img[b, :, y3, x3],
                            img[b, :, y4, x3],
                        ],
                        [
                            img[b, :, y1, x4],
                            img[b, :, y2, x4],
                            img[b, :, y3, x4],
                            img[b, :, y4, x4],
                        ],
                    ]
                )
                mat_m = mat_m.transpose((2, 0, 1))

                out[b, :, y_out, x_out] = (mat_l @ mat_m @ mat_r).squeeze()

    return out


def faster_bicubic(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray
) -> np.ndarray:
    """Faster way of achieving bicubic without numba

    slower than scipy by about 3 times
    """
    a = -0.75  # should be between -0.75 ~ -0.5

    b_in, c_in, h_in, w_in = img.shape
    b_out, _, h_out, w_out = grid.shape
    dtype = img.dtype

    def kernel(s, a, dtype):
        out = np.zeros_like(s, dtype=dtype)
        s = np.abs(s)
        mask1 = np.logical_and(0 <= s, s <= 1)
        mask2 = np.logical_and(1 < s, s <= 2)
        out[mask1] = (a + 2) * (s[mask1] ** 3) - (a + 3) * (s[mask1] ** 2) + 1
        out[mask2] = (
            a * (s[mask2] ** 3)
            - (5 * a) * (s[mask2] ** 2)
            + (8 * a) * s[mask2]
            - 4 * a
        )
        return out

    int_dtype = np.dtype(np.int64)

    min_grid = np.floor(grid).astype(int_dtype)

    d1 = 1 + (grid - min_grid)  # (b, 2, h, w)
    d2 = grid - min_grid
    d3 = min_grid + 1 - grid
    d4 = min_grid + 2 - grid

    c1 = (grid - d1).astype(int_dtype)  # (b, 2, h, w)
    c2 = (grid - d2).astype(int_dtype)
    c3 = (grid + d3).astype(int_dtype)
    c4 = (grid + d4).astype(int_dtype)

    c1[:, 0, ...] %= h_in
    c1[:, 1, ...] %= w_in
    c2[:, 0, ...] %= h_in
    c2[:, 1, ...] %= w_in
    c3[:, 0, ...] %= h_in
    c3[:, 1, ...] %= w_in
    c4[:, 0, ...] %= h_in
    c4[:, 1, ...] %= w_in

    # FIXME: this part is slow
    k1 = kernel(d1, a, dtype)  # (b, 2, h, w)
    k2 = kernel(d2, a, dtype)
    k3 = kernel(d3, a, dtype)
    k4 = kernel(d4, a, dtype)

    mat_l = np.stack(
        [k1[:, 1, ...], k2[:, 1, ...], k3[:, 1, ...], k4[:, 1, ...]], axis=-1
    )
    mat_r = np.stack(
        [k1[:, 0, ...], k2[:, 0, ...], k3[:, 0, ...], k4[:, 0, ...]], axis=-1
    )

    # FIXME: this part is slow
    mat_m = np.empty((b_out, c_in, h_out, w_out, 4, 4), dtype=dtype)
    for b in range(b_out):
        y1 = c1[b, 0, ...]  # (h, w)
        y2 = c2[b, 0, ...]
        y3 = c3[b, 0, ...]
        y4 = c4[b, 0, ...]

        x1 = c1[b, 1, ...]
        x2 = c2[b, 1, ...]
        x3 = c3[b, 1, ...]
        x4 = c4[b, 1, ...]

        mat_m_x1 = np.stack(
            [
                img[b][:, y1, x1],  # (c, h, w)
                img[b][:, y2, x1],
                img[b][:, y3, x1],
                img[b][:, y4, x1],
            ],
            axis=-1,
        )
        mat_m_x2 = np.stack(
            [
                img[b][:, y1, x2],
                img[b][:, y2, x2],
                img[b][:, y3, x2],
                img[b][:, y4, x2],
            ],
            axis=-1,
        )
        mat_m_x3 = np.stack(
            [
                img[b][:, y1, x3],
                img[b][:, y2, x3],
                img[b][:, y3, x3],
                img[b][:, y4, x3],
            ],
            axis=-1,
        )
        mat_m_x4 = np.stack(
            [
                img[b][:, y1, x4],
                img[b][:, y2, x4],
                img[b][:, y3, x4],
                img[b][:, y4, x4],
            ],
            axis=-1,
        )

        mat_m[b, ...] = np.stack(
            [mat_m_x1, mat_m_x2, mat_m_x3, mat_m_x4], axis=-2
        )

    mat_l = mat_l[:, np.newaxis, ..., np.newaxis, :]
    mat_r = mat_r[:, np.newaxis, ..., np.newaxis]
    out = (mat_l @ mat_m @ mat_r).squeeze()

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

    out_scipy = func_timer(baseline_scipy_cubic)(img, grid, out_scipy)
    out_cv2 = func_timer(baseline_cv2_cubic)(img, grid, out_cv2)
    out_naive = func_timer(naive_bicubic)(img, grid, out_naive)
    out_faster = func_timer(faster_bicubic)(img, grid, out_faster)

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
