#!/usr/bin/env python3

from typing import Dict, List, Union

import torch

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    create_rotation_matrices,
    create_xyz_grid,
    get_device,
    pi,
)


def cube_hsplits(cube_h: torch.Tensor) -> List[torch.Tensor]:
    """Returns list of horizontal splits (doesn't split batch)"""
    assert cube_h.shape[-2] * 6 == cube_h.shape[-1]
    # order ["F", "R", "B", "L", "U", "D"]
    splits = torch.split(
        cube_h, split_size_or_sections=cube_h.shape[-2], dim=-1
    )
    assert len(splits) == 6
    assert splits[0].shape == (*cube_h.shape[0:3], cube_h.shape[-2])
    return splits


def cube_h2list(cube_h: torch.Tensor) -> List[List[torch.Tensor]]:
    bs = cube_h.shape[0]
    cube_lists = []
    for b in range(bs):
        cube_lists.append(
            list(
                torch.split(
                    cube_h[b], split_size_or_sections=cube_h.shape[-2], dim=-1
                )
            )
        )
    return cube_lists


def cube_h2dict(cube_h: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
    bs = cube_h.shape[0]
    cube_list = cube_hsplits(cube_h)

    cube_dicts = []
    for b in range(bs):
        cube_dicts.append(
            {
                k: cube_list[i][b].clone()
                for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
            }
        )
    return cube_dicts


def cube_h2dice(cube_h: torch.Tensor) -> torch.Tensor:
    bs = cube_h.shape[0]
    cube_list = cube_hsplits(cube_h)

    w = cube_h.shape[-2]
    cube_dice = torch.zeros(
        (bs, cube_h.shape[-3], w * 3, w * 4),
        dtype=cube_h.dtype,
        device=cube_h.device,
    )
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for b in range(bs):
        for i, (sx, sy) in enumerate(sxy):
            cube_dice[
                b, :, sy * w : (sy + 1) * w, sx * w : (sx + 1) * w
            ] = cube_list[i][b, ...].clone()

    return cube_dice


def matmul(m: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    M = torch.matmul(R[:, None, None, ...], m)
    M = M.squeeze(-1)

    return M


def convert_grid(
    xyz: torch.Tensor, h_equi: int, w_equi: int, method: str = "robust"
) -> torch.Tensor:
    # convert to rotation
    phi = torch.asin(xyz[..., 2] / torch.norm(xyz, dim=-1))
    theta = torch.atan2(xyz[..., 1], xyz[..., 0])

    if method == "robust":
        ui = (theta / (2 * pi) - 0.5) * w_equi - 0.5
        uj = (0.5 - phi / pi) * h_equi - 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        ui = (theta / (2 * pi) - 0.5) * w_equi - 0.5
        uj = (0.5 - phi / pi) * h_equi - 0.5
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
    equi: torch.Tensor,
    rots: List[Dict[str, float]],
    w_face: int,
    cube_format: str,
    z_down: bool,
    mode: str,
    clip_output: bool = True,
    backend: str = "native",
) -> Union[
    torch.Tensor, List[List[torch.Tensor]], List[Dict[str, torch.Tensor]]
]:
    """Run Equi2Cube

    params:
    - equi (torch.Tensor): 4 dims (b, c, h, w)
    - rots (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - w_face (int): width of the cube face
    - cube_format (str): ('horizon', 'list', 'dict', 'dice')
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - backend (str): backend of torch `grid_sample` (default: `native`)

    returns:
    - cubemaps

    NOTE: `backend` can be either `native` or `pure`

    """

    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        rots
    ), f"ERR: length of equi and rot differs: {len(equi)} vs {len(rots)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if equi.device.type == "cuda":
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and equi_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        equi = equi.type(torch.float32)

    bs, c, h_equi, w_equi = equi.shape
    img_device = get_device(equi)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initialize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, w_face, w_face * 6), dtype=dtype, device=img_device
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if equi.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # create grid
    xyz = create_xyz_grid(
        w_face=w_face, batch=bs, dtype=tmp_dtype, device=tmp_device
    )
    xyz = xyz.unsqueeze(-1)

    # FIXME: not sure why, but z-axis is facing the opposite
    # probably I need to change the way I choose the xyz coordinates
    # this is a temporary fix for now
    z_down = not z_down
    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device
    )

    # rotate grid
    xyz = matmul(xyz, R)

    # create a pixel map grid
    grid = convert_grid(xyz=xyz, h_equi=h_equi, w_equi=w_equi, method="robust")

    # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
    # FIXME: better way of forcing `grid` to be the same dtype?
    if equi.dtype != grid.dtype:
        grid = grid.type(equi.dtype)
    if equi.device != grid.device:
        grid = grid.to(equi.device)

    # grid sample
    out = torch_grid_sample(
        img=equi, grid=grid, out=out, mode=mode, backend=backend
    )

    # clip by input
    out = (
        out.type(equi_dtype)
        if equi_dtype == torch.uint8 or not clip_output
        else torch.clip(out, torch.min(equi), torch.max(equi))
    )

    # reformat the output
    # FIXME: needs to test this
    if cube_format == "horizon":
        pass
    elif cube_format == "list":
        out = cube_h2list(out)  # type: ignore
    elif cube_format == "dict":
        out = cube_h2dict(out)  # type: ignore
    elif cube_format == "dice":
        out = cube_h2dice(out)
    else:
        raise NotImplementedError("{} is not supported".format(cube_format))

    return out
