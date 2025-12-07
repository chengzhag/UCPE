#!/usr/bin/env python3

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    create_global2camera_rotation_matrix,
    create_normalized_grid,
    create_intrinsic_matrix,
    create_rotation_matrices,
    get_device,
    pi,
)


@lru_cache(maxsize=128)
def create_global2cam_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    K = create_intrinsic_matrix(
        height=height,
        width=width,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )
    g2c_rot = create_global2camera_rotation_matrix(dtype=dtype, device=device)

    return K @ g2c_rot.T


def prep_matrices(
    height: int,
    width: int,
    h_pers: int,
    w_pers: int,
    batch: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = create_normalized_grid(
        height=height, width=width, batch=batch, dtype=dtype, device=device
    )
    m = m.unsqueeze(-1)
    G = create_global2cam_matrix(
        height=h_pers,
        width=w_pers,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )

    return m, G


def matmul(m: torch.Tensor, G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    M = torch.matmul(torch.matmul(G, R)[:, None, None, ...], m)
    M = M.squeeze(-1)

    return M


def convert_grid(
    M: torch.Tensor, h_pers: int, w_pers: int,
) -> torch.Tensor:
    # calculate image coordinates
    ui = M[..., 0] / M[..., 2]
    uj = M[..., 1] / M[..., 2]
    ui[M[..., 2] < 0] = -1
    uj[M[..., 2] < 0] = -1
    ui += 0.5
    uj += 0.5
    ui = torch.where(ui < 0, -1, ui)
    ui = torch.where(ui >= w_pers, -1, ui)
    uj = torch.where(uj < 0, -1, uj)
    uj = torch.where(uj >= h_pers, -1, uj)

    # stack the pixel maps into a grid
    grid = torch.stack((uj, ui), dim=-3)
    return grid


def run(
    pers: torch.Tensor,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
    mode: str,
    clip_output: bool = True,
    backend: str = "native",
) -> torch.Tensor:
    """Run Pers2Equi

    params:
    - pers (torch.Tensor): 4 dims (b, c, h, w)
    - rots (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - height, width (int): height and width of equirectangular image
    - fov_x (float): fov of horizontal axis in degrees of the perspective image
    - skew (float): skew of the perspective image
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - backend (str): backend of torch `grid_sample` (default: `native`)

    returns:
    - out (torch.Tensor)

    NOTE: `backend` can be either `native` or `pure`

    """

    assert (
        len(pers.shape) == 4
    ), f"ERR: input `pers` should be 4-dim (b, c, h, w), but got {len(pers.shape)}"
    assert len(pers) == len(
        rots
    ), f"ERR: length of pers and rot differs: {len(pers)} vs {len(rots)}"

    pers_dtype = pers.dtype
    assert pers_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input perspective image has dtype of {pers_dtype} which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as pers
    if pers.device.type == "cuda":
        dtype = torch.float32 if pers_dtype == torch.uint8 else pers_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if pers_dtype == torch.uint8 else pers_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and pers_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        pers = pers.type(torch.float32)

    bs, c, h_pers, w_pers = pers.shape
    img_device = get_device(pers)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initialize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, height, width), dtype=dtype, device=img_device
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if pers.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # create grid and transform matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        h_pers=h_pers,
        w_pers=w_pers,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=tmp_dtype,
        device=tmp_device,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device
    )

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_pers=h_pers, w_pers=w_pers)

    # if backend == "native":
    #     grid = grid.to(img_device)
    # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
    # FIXME: better way of forcing `grid` to be the same dtype?
    if pers.dtype != grid.dtype:
        grid = grid.type(pers.dtype)
    if pers.device != grid.device:
        grid = grid.to(pers.device)

    mask = torch.logical_or(grid[:, 0] < 0, grid[:, 1] < 0)
    mask = mask[:, None].repeat_interleave(pers.shape[1], dim=1)
    
    # grid sample
    out = torch_grid_sample(
        img=pers,
        grid=grid,
        out=out,  # FIXME: is this necessary?
        mode=mode,
        backend=backend,
    )

    # NOTE: we assume that `out` keeps it's dtype

    out[mask] = 0
    out = (
        out.type(pers_dtype)
        if pers_dtype == torch.uint8 or not clip_output
        else torch.clip(out, torch.min(pers), torch.max(pers))
    )

    return out

