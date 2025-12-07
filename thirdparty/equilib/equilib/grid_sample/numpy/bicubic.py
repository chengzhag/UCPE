#!/usr/bin/env python3

import numpy as np

__all__ = ["bicubic"]


def kernel(
    s: np.ndarray, a: float = -0.75, dtype: np.dtype = np.dtype(np.float32)
) -> np.ndarray:
    out = np.zeros_like(s, dtype)
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


def bicubic(img: np.ndarray, grid: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Bicubic Interpolation"""

    b_in, c_in, h_in, w_in = img.shape
    b_out, _, h_out, w_out = out.shape
    dtype = out.dtype
    # NOTE: this is hardcoded since pytorch is also -0.75
    a = -0.75

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
    out = (mat_l @ mat_m @ mat_r).squeeze(-1).squeeze(-1)

    return out
