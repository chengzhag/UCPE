#!/usr/bin/env python3

import torch

__all__ = ["nearest"]


def nearest(
    img: torch.Tensor, grid: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """Nearest Neighbor Interpolation

    Merit of using this nearest instead is that the grid doesn't need to be a
    cuda tensor. Although it is a little bit slow since it is iterating batches
    """

    b, _, h, w = img.shape

    round_grid = torch.round(grid).type(torch.int64)
    round_grid[:, 0, ...] %= h
    round_grid[:, 1, ...] %= w

    # FIXME: find a better way of sampling batches
    for i in range(b):
        y = round_grid[i, 0, :, :]
        x = round_grid[i, 1, :, :]
        out[i, ...] = img[i][:, y, x]

    return out
