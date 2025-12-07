#!/usr/bin/env python3

from typing import Optional

import torch

from equilib.torch_utils.intrinsic import pi
from einops import repeat


def create_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create coordinate grid with height and width

    `z-axis` scale is `1`

    params:
    - height (int)
    - width (int)
    - batch (Optional[int])
    - dtype (torch.dtype)
    - device (torch.device)

    return:
    - grid (torch.Tensor)
    """

    # NOTE: RuntimeError: "linspace_cpu" not implemented for Half
    if device.type == "cpu":
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: {dtype} is not supported by {device.type}\n"
            "If device is `cpu`, use float32 or float64"
        )

    _xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    _ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy
    ys, xs = torch.meshgrid([_ys, _xs], indexing="ij")
    zs = torch.ones_like(xs, dtype=dtype, device=device)
    grid = torch.stack((xs, ys, zs), dim=2)
    # grid shape (h, w, 3)

    # batched (stacked copies)
    if batch is not None:
        assert isinstance(
            batch, int
        ), f"ERR: batch needs to be integer: batch={batch}"
        assert (
            batch > 0
        ), f"ERR: batch size needs to be larger than 0: batch={batch}"
        # FIXME: faster way of copying?
        # grid = torch.cat([grid.unsqueeze(0)] * batch)
        grid = repeat(grid, "... -> b ...", b=batch)
        
        # grid shape is (b, h, w, 3)

    return grid


def create_normalized_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create coordinate grid with height and width

    NOTE: primarly used for equi2equi

    params:
    - height (int)
    - width (int)
    - batch (Optional[int])
    - dtype (torch.dtype)

    return:
    - grid (torch.Tensor)

    """

    # NOTE: RuntimeError: "linspace_cpu" not implemented for Half
    if device.type == "cpu":
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: {dtype} is not supported by {device.type}\n"
            "If device is `cpu`, use float32 or float64"
        )

    xs = torch.linspace(0, width - 1, width, dtype=dtype, device=device)
    ys = torch.linspace(0, height - 1, height, dtype=dtype, device=device)
    theta = xs * 2 * pi / width - pi
    phi = ys * pi / height - pi / 2
    phi, theta = torch.meshgrid([phi, theta], indexing="ij")
    a = torch.stack((theta, phi), dim=-1)
    norm_A = 1
    x = norm_A * torch.cos(a[..., 1]) * torch.cos(a[..., 0])
    y = norm_A * torch.cos(a[..., 1]) * torch.sin(a[..., 0])
    z = norm_A * torch.sin(a[..., 1])
    grid = torch.stack((x, y, z), dim=-1)

    # batched (stacked copies)
    if batch is not None:
        assert isinstance(
            batch, int
        ), f"ERR: batch needs to be integer: batch={batch}"
        assert (
            batch > 0
        ), f"ERR: batch size needs to be larger than 0: batch={batch}"
        # FIXME: faster way of copying?
        grid = torch.cat([grid.unsqueeze(0)] * batch)
        # grid shape is (b, h, w, 3)

    return grid


def create_xyz_grid(
    w_face: int,
    batch: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """xyz coordinates of the faces of the cube"""

    ratio = (w_face - 1) / w_face

    out = torch.zeros((w_face, w_face * 6, 3), dtype=dtype, device=device)
    rng = torch.linspace(
        -0.5 * ratio, 0.5 * ratio, w_face, dtype=dtype, device=device
    )

    # NOTE: https://github.com/pytorch/pytorch/issues/15301
    # Torch meshgrid behaves differently than numpy

    # Front face (x = 0.5)
    out[:, 0 * w_face : 1 * w_face, [2, 1]] = torch.stack(
        torch.meshgrid([-rng, rng], indexing="ij"), -1
    )
    out[:, 0 * w_face : 1 * w_face, 0] = 0.5

    # Right face (y = -0.5)
    out[:, 1 * w_face : 2 * w_face, [2, 0]] = torch.stack(
        torch.meshgrid([-rng, -rng], indexing="ij"), -1
    )
    out[:, 1 * w_face : 2 * w_face, 1] = 0.5

    # Back face (x = -0.5)
    out[:, 2 * w_face : 3 * w_face, [2, 1]] = torch.stack(
        torch.meshgrid([-rng, -rng], indexing="ij"), -1
    )
    out[:, 2 * w_face : 3 * w_face, 0] = -0.5

    # Left face (y = 0.5)
    out[:, 3 * w_face : 4 * w_face, [2, 0]] = torch.stack(
        torch.meshgrid([-rng, rng], indexing="ij"), -1
    )
    out[:, 3 * w_face : 4 * w_face, 1] = -0.5

    # Up face (z = 0.5)
    out[:, 4 * w_face : 5 * w_face, [0, 1]] = torch.stack(
        torch.meshgrid([rng, rng], indexing="ij"), -1
    )
    out[:, 4 * w_face : 5 * w_face, 2] = 0.5

    # Down face (z = -0.5)
    out[:, 5 * w_face : 6 * w_face, [0, 1]] = torch.stack(
        torch.meshgrid([-rng, rng], indexing="ij"), -1
    )
    out[:, 5 * w_face : 6 * w_face, 2] = -0.5

    if batch is not None:
        assert isinstance(
            batch, int
        ), f"ERR: batch needs to be integer: batch={batch}"
        assert (
            batch > 0
        ), f"ERR: batch size needs to be larger than 0: batch={batch}"
        # FIXME: faster way of copying?
        out = torch.cat([out.unsqueeze(0)] * batch)
        # grid shape is (b, h, w, 3)

    return out
