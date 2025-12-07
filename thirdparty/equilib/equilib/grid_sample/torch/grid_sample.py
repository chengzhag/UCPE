#!/usr/bin/env python3

from typing import Optional
import warnings

import torch

from .native import native_bicubic, native_bilinear, native_nearest
from .nearest import nearest
from .bilinear import bilinear
from .bicubic import bicubic

DTYPES = (torch.uint8, torch.float16, torch.float32, torch.float64)


def grid_sample(
    img: torch.Tensor,
    grid: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    mode: str = "bilinear",
    backend: str = "native",
) -> torch.Tensor:
    """Torch grid sampling algorithm

    params:
    - img (torch.Tensor)
    - grid (torch.Tensor)
    - out (Optional[torch.Tensor]): defaults to None
    - mode (str): ('bilinear', 'bicubic', 'nearest')
    - backend (str): ('native', 'pure')

    return:
    - img (torch.Tensor)

    NOTE: for `backend`, `pure` is relatively efficient since grid doesn't need
    to be in the same device as the `img`. However, `native` is faster.

    NOTE: for `pure` backends, we need to pass reference to `out`.

    NOTE: for `native` backends, we should pass anything for `out`

    """

    if backend == "native":
        if out is not None:
            # NOTE: out is created
            warnings.warn(
                "don't need to pass preallocated `out` to `grid_sample`"
            )
        assert img.device == grid.device, (
            f"ERR: when using {backend}, the devices of `img` and `grid` need"
            "to be on the same device"
        )
        if mode == "nearest":
            out = native_nearest(img, grid)
        elif mode == "bilinear":
            out = native_bilinear(img, grid)
        elif mode == "bicubic":
            out = native_bicubic(img, grid)
        else:
            raise ValueError(f"ERR: {mode} is not supported")
    elif backend == "pure":
        # NOTE: img and grid can be on different devices, but grid should be on the cpu
        # FIXME: since bilinear implementation depends on `grid` being on device, I'm removing
        # this warning and will put `grid` onto the same device until a fix is found
        # if grid.device.type == "cuda":
        #     warnings.warn("input `grid` should be on the cpu, but got a cuda tensor")
        assert (
            out is not None
        ), "ERR: need to pass reference to `out`, but got None"
        assert img.device == grid.device, (
            f"ERR: when using {backend}, the devices of `img` and `grid` need"
            "to be on the same device"
        )
        if mode == "nearest":
            out = nearest(img, grid, out)
        elif mode == "bilinear":
            out = bilinear(img, grid, out)
        elif mode == "bicubic":
            out = bicubic(img, grid, out)
        else:
            raise ValueError(f"ERR: {mode} is not supported")
    else:
        raise ValueError(f"ERR: {backend} is not supported")

    return out
