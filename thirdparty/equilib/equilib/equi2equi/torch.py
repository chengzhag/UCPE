#!/usr/bin/env python3

from typing import Dict, List, Optional

import torch

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    create_normalized_grid,
    create_rotation_matrices,
    get_device,
    pi,
)


def matmul(m: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    M = torch.matmul(R[:, None, None, ...], m)
    M = M.squeeze(-1)

    return M


def convert_grid(
    M: torch.Tensor, h_equi: int, w_equi: int, method: str = "robust"
) -> torch.Tensor:
    # convert to rotation
    phi = torch.asin(M[..., 2] / torch.norm(M, dim=-1))
    theta = torch.atan2(M[..., 1], M[..., 0])

    if method == "robust":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui = torch.where(ui < 0, ui + w_equi, ui)
        ui = torch.where(ui >= w_equi, ui - w_equi, ui)
        uj = torch.where(uj < 0, uj + h_equi, uj)
        uj = torch.where(uj >= h_equi, uj - h_equi, uj)
    else:
        raise ValueError(f"ERR: {method} is not supported")

    # stack the pixel maps into a grid
    grid = torch.stack((uj, ui), dim=-3)

    return grid


def run(
    src: torch.Tensor,
    rots: List[Dict[str, float]],
    z_down: bool,
    mode: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    clip_output: bool = True,
    backend: str = "native",
) -> torch.Tensor:
    """Run Equi2Equi

    params:
    - src (torch.Tensor): 4 dims (b, c, h, w)
    - rot (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - height, width (Optional[int]): height and width of the target
    - backend (str): backend of torch `grid_sample` (default: `native`)

    return:
    - out (torch.Tensor)

    NOTE: acceptable dtypes for `src` are currently uint8, float32, and float64.
    Floats are prefered since numpy calculations are optimized for floats.

    NOTE: output array has the same dtype as `src`

    NOTE: you can override `equilib`'s grid_sample with over grid sampling methods
    using `override_func`. The input to this function have to match `grid_sample`.

    """

    assert (
        len(src.shape) == 4
    ), f"ERR: input `src` should be 4-dim (b, c, h, w), but got {len(src.shape)}"
    assert len(src) == len(
        rots
    ), f"ERR: length of `src` and `rot` differs: {len(src)} vs {len(rots)}"

    src_dtype = src.dtype
    assert src_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {src_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if src.device.type == "cuda":
        dtype = torch.float32 if src_dtype == torch.uint8 else src_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if src_dtype == torch.uint8 else src_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and src_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        src = src.type(torch.float32)

    bs, c, h_equi, w_equi = src.shape
    src_device = get_device(src)

    assert type(height) == type(
        width
    ), "ERR: `height` and `width` does not match types (maybe it was set separately?)"
    if height is None and width is None:
        height = h_equi
        width = w_equi
    else:
        assert isinstance(height, int) and isinstance(width, int)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initialize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, height, width),  # type: ignore
            dtype=dtype,
            device=src_device,
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if src.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    m = create_normalized_grid(
        height=height, width=width, batch=bs, dtype=tmp_dtype, device=tmp_device
    )
    m = m.unsqueeze(-1)

    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device
    )

    # rotate the grid
    M = matmul(m, R)

    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
    # FIXME: better way of forcing `grid` to be the same dtype?
    if src.dtype != grid.dtype:
        grid = grid.type(src.dtype)
    if src.device != grid.device:
        grid = grid.to(src.device)

    # grid sample
    out = torch_grid_sample(
        img=src,
        grid=grid,
        out=out,  # FIXME: is this necessary?
        mode=mode,
        backend=backend,
    )

    # NOTE: we assume that `out` keeps it's dtype

    # clip by input
    out = (
        out.type(src_dtype)
        if src_dtype == torch.uint8 or not clip_output
        else torch.clip(out, torch.min(src), torch.max(src))
    )

    return out
