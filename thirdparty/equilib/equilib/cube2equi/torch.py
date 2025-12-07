#!/usr/bin/env python3

import math

from typing import Dict, List, Union

import torch

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import get_device

__all__ = ["convert2horizon", "run"]


def single_list2horizon(cube: List[torch.Tensor]) -> torch.Tensor:
    _, _, w = cube[0].shape
    assert len(cube) == 6
    assert sum(face.shape[-1] == w for face in cube) == 6
    return torch.cat(cube, dim=-1)


def dice2horizon(dices: torch.Tensor) -> torch.Tensor:
    assert len(dices.shape) == 4
    w = dices.shape[-2] // 3
    assert dices.shape[-2] == w * 3 and dices.shape[-1] == w * 4

    # create a (b, c, h, w) horizon array
    device = get_device(dices)
    horizons = torch.empty(
        (*dices.shape[0:2], w, w * 6), dtype=dices.dtype, device=device
    )

    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        horizons[..., i * w : (i + 1) * w] = dices[
            ..., sy * w : (sy + 1) * w, sx * w : (sx + 1) * w
        ]
    return horizons


def dict2horizon(dicts: List[Dict[str, torch.Tensor]]) -> torch.Tensor:
    face_key = ("F", "R", "B", "L", "U", "D")
    c, _, w = dicts[0][face_key[0]].shape
    dtype = dicts[0][face_key[0]].dtype
    device = get_device(dicts[0][face_key[0]])
    horizons = torch.empty(
        (len(dicts), c, w, w * 6), dtype=dtype, device=device
    )
    for b, cube in enumerate(dicts):
        horizons[b, ...] = single_list2horizon([cube[k] for k in face_key])
    return horizons


def list2horizon(lists: List[List[torch.Tensor]]) -> torch.Tensor:
    assert len(lists[0][0].shape) == 3
    c, w, _ = lists[0][0].shape
    dtype = lists[0][0].dtype
    device = get_device(lists[0][0])
    horizons = torch.empty(
        (len(lists), c, w, w * 6), dtype=dtype, device=device
    )
    for b, cube in enumerate(lists):
        horizons[b, ...] = single_list2horizon(cube)
    return horizons


def convert2horizon(
    cubemap: Union[
        torch.Tensor,
        List[torch.Tensor],
        List[List[torch.Tensor]],
        Dict[str, torch.Tensor],
        List[Dict[str, torch.Tensor]],
    ],
    cube_format: str,
) -> torch.Tensor:
    """Converts supported cubemap formats to horizon

    params:
    - cubemap
    - cube_format (str): ('horizon', 'dice', 'dict', 'list')

    return:
    - horizon (torch.Tensor)

    """

    # FIXME: better typing for mypy...

    if cube_format in ("horizon", "dice"):
        assert isinstance(
            cubemap, torch.Tensor
        ), f"ERR: cubemap {cube_format} needs to be a torch.Tensor"
        if len(cubemap.shape) == 2:
            # single grayscale
            # NOTE: this rarely happens since we assume grayscales are (1, h, w)
            cubemap = cubemap[None, None, ...]
        elif len(cubemap.shape) == 3:
            # batched grayscale
            # single rgb
            # FIXME: how to tell apart batched grayscale and rgb?
            # Assume that grayscale images are also 3 dim (from loading images)
            cubemap = cubemap[None, ...]

        if cube_format == "dice":
            cubemap = dice2horizon(cubemap)
    elif cube_format == "list":
        assert isinstance(
            cubemap, list
        ), f"ERR: cubemap {cube_format} needs to be a list"
        if isinstance(cubemap[0], torch.Tensor):
            # single
            cubemap = list2horizon([cubemap])  # type: ignore
        else:
            cubemap = list2horizon(cubemap)  # type: ignore
    elif cube_format == "dict":
        if isinstance(cubemap, dict):
            cubemap = [cubemap]
        assert isinstance(cubemap, list)
        assert isinstance(
            cubemap[0], dict
        ), f"ERR: cubemap {cube_format} needs to have dict inside the list"
        cubemap = dict2horizon(cubemap)  # type: ignore
    else:
        raise ValueError(f"ERR: {cube_format} is not supported")

    assert (
        len(cubemap.shape) == 4
    ), f"ERR: cubemap needs to be 4 dim, but got {cubemap.shape}"

    return cubemap


def _equirect_facetype(h: int, w: int) -> torch.Tensor:
    """0F 1R 2B 3L 4U 5D"""

    int_dtype = torch.int64

    w_ratio = (w - 1) / w
    h_ratio = (h - 1) / h

    tp = torch.roll(
        torch.arange(4)  # 1
        .repeat_interleave(w // 4)  # 2 same as np.repeat
        .unsqueeze(0)
        .transpose(0, 1)  # 3
        .repeat(1, h)  # 4
        .view(-1, h)  # 5
        .transpose(0, 1),  # 6
        shifts=3 * w // 8,
        dims=1,
    )

    # Prepare ceil mask
    mask = torch.zeros((h, w // 4), dtype=torch.bool)
    idx = torch.linspace(-(math.pi * w_ratio), math.pi * w_ratio, w // 4) / 4
    idx = h // 2 - torch.round(
        torch.atan(torch.cos(idx)) * h / (math.pi * h_ratio)
    )
    idx = idx.type(int_dtype)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = torch.roll(torch.cat([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[torch.flip(mask, dims=(0,))] = 5

    return tp.type(int_dtype)


def create_equi_grid(
    h_out: int,
    w_out: int,
    w_face: int,
    batch: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    w_ratio = (w_out - 1) / w_out
    h_ratio = (h_out - 1) / h_out
    theta = torch.linspace(
        -(math.pi * w_ratio),
        math.pi * w_ratio,
        steps=w_out,
        dtype=dtype,
        device=device,
    )
    phi = torch.linspace(
        (math.pi * h_ratio) / 2,
        -(math.pi * h_ratio) / 2,
        steps=h_out,
        dtype=dtype,
        device=device,
    )
    phi, theta = torch.meshgrid([phi, theta], indexing="ij")

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = _equirect_facetype(h_out, w_out)

    # xy coordinate map
    coor_x = torch.zeros((h_out, w_out), dtype=dtype, device=device)
    coor_y = torch.zeros((h_out, w_out), dtype=dtype, device=device)

    # FIXME: there's a bug where left section (3L) has artifacts
    # on top and bottom
    # It might have to do with 4U or 5D
    for i in range(6):
        mask = tp == i

        if i < 4:
            coor_x[mask] = 0.5 * torch.tan(theta[mask] - math.pi * i / 2)
            coor_y[mask] = (
                -0.5
                * torch.tan(phi[mask])
                / torch.cos(theta[mask] - math.pi * i / 2)
            )
        elif i == 4:
            c = 0.5 * torch.tan(math.pi / 2 - phi[mask])
            coor_x[mask] = c * torch.sin(theta[mask])
            coor_y[mask] = c * torch.cos(theta[mask])
        elif i == 5:
            c = 0.5 * torch.tan(math.pi / 2 - torch.abs(phi[mask]))
            coor_x[mask] = c * torch.sin(theta[mask])
            coor_y[mask] = -c * torch.cos(theta[mask])

    # Final renormalize
    coor_x = torch.clamp(
        torch.clamp(coor_x + 0.5, 0, 1) * w_face, 0, w_face - 1
    )
    coor_y = torch.clamp(
        torch.clamp(coor_y + 0.5, 0, 1) * w_face, 0, w_face - 1
    )

    # change x axis of the x coordinate map
    for i in range(6):
        mask = tp == i
        coor_x[mask] = coor_x[mask] + w_face * i

    # repeat batch
    coor_x = coor_x.repeat(batch, 1, 1) - 0.5
    coor_y = coor_y.repeat(batch, 1, 1) - 0.5

    grid = torch.stack((coor_y, coor_x), dim=-3).to(device)
    return grid


def run(
    horizon: torch.Tensor,
    height: int,
    width: int,
    mode: str,
    clip_output: bool = True,
    backend: str = "native",
) -> torch.Tensor:
    """Run Cube2Equi

    params:
    - horizon (torch.Tensor)
    - height, widht (int): output equirectangular image's size
    - mode (str)
    - backend (str): backend of torch `grid_sample` (default: `native`)

    return:
    - equi (torch.Tensor)

    NOTE: we assume that the input `horizon` is a 4 dim array

    """

    assert (
        len(horizon.shape) == 4
    ), f"ERR: `horizon` should be 4-dim (b, c, h, w), but got {horizon.shape}"

    horizon_dtype = horizon.dtype
    assert horizon_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input horizon has dtype of {horizon_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as horizon
    if horizon.device.type == "cuda":
        dtype = torch.float32 if horizon_dtype == torch.uint8 else horizon_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if horizon_dtype == torch.uint8 else horizon_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and horizon_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        horizon = horizon.type(torch.float32)

    bs, c, w_face, _ = horizon.shape
    horizon_device = get_device(horizon)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initilaize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, height, width), dtype=dtype, device=horizon_device
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if horizon.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # create sampling grid
    grid = create_equi_grid(
        h_out=height,
        w_out=width,
        w_face=w_face,
        batch=bs,
        dtype=tmp_dtype,
        device=tmp_device,
    )

    # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
    # FIXME: better way of forcing `grid` to be the same dtype?
    if horizon.dtype != grid.dtype:
        grid = grid.type(horizon.dtype)
    if horizon.device != grid.device:
        grid = grid.to(horizon.device)

    # grid sample
    out = torch_grid_sample(
        img=horizon, grid=grid, out=out, mode=mode, backend=backend
    )

    out = (
        out.type(horizon_dtype)
        if horizon_dtype == torch.uint8 or not clip_output
        else torch.clip(out, torch.min(horizon), torch.max(horizon))
    )

    return out
