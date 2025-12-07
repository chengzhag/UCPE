#!/usr/bin/env python3

import torch

__all__ = ["bilinear"]


def linear_interp(v0, v1, d, L):
    return v0 * (1 - d) / L + v1 * d / L


def interp2d(q00, q10, q01, q11, dy, dx):
    f0 = linear_interp(q00, q01, dx, 1)
    f1 = linear_interp(q10, q11, dx, 1)
    return linear_interp(f0, f1, dy, 1)


def bilinear(
    img: torch.Tensor, grid: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    b, _, h, w = img.shape

    min_grid = torch.floor(grid).type(torch.int64)
    max_grid = min_grid + 1
    d_grid = grid - min_grid

    min_grid[:, 0, :, :] %= h
    min_grid[:, 1, :, :] %= w
    max_grid[:, 0, :, :] %= h
    max_grid[:, 1, :, :] %= w

    # FIXME: anyway to do efficient batch?
    for i in range(b):
        dy = d_grid[i, 0, ...]
        dx = d_grid[i, 1, ...]
        min_ys = min_grid[i, 0, ...]
        min_xs = min_grid[i, 1, ...]
        max_ys = max_grid[i, 0, ...]
        max_xs = max_grid[i, 1, ...]

        min_ys %= h
        min_xs %= w

        p00 = img[i][:, min_ys, min_xs]
        p10 = img[i][:, max_ys, min_xs]
        p01 = img[i][:, min_ys, max_xs]
        p11 = img[i][:, min_ys, max_xs]

        out[i, ...] = interp2d(p00, p10, p01, p11, dy, dx)

    return out
