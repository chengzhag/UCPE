#!/usr/bin/env python3

from typing import Optional

import numpy as np


def create_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Create coordinate grid with height and width

    NOTE: primarly used for equi2pers

    `z-axis` scale is `1`

    params:
    - height (int)
    - width (int)
    - batch (Optional[int])
    - dtype (np.dtype)

    return:
    - grid (np.ndarray)

    """

    _xs = np.linspace(0, width - 1, width, dtype=dtype)
    _ys = np.linspace(0, height - 1, height, dtype=dtype)
    xs, ys = np.meshgrid(_xs, _ys)
    zs = np.ones_like(xs, dtype=dtype)
    grid = np.stack((xs, ys, zs), axis=-1)
    # grid shape is (h, w, 3)

    # batched (stacked copies)
    if batch is not None:
        assert isinstance(
            batch, int
        ), f"ERR: batch needs to be integer: batch={batch}"
        assert (
            batch > 0
        ), f"ERR: batch size needs to be larger than 0: batch={batch}"
        # FIXME: faster way of copying?
        grid = np.concatenate([grid[np.newaxis, ...]] * batch)
        # grid shape is (b, h, w, 3)

    return grid


def create_normalized_grid(
    height: int,
    width: int,
    batch: Optional[int] = None,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Create coordinate grid with height and width

    NOTE: primarly used for equi2equi

    params:
    - height (int)
    - width (int)
    - batch (Optional[int])
    - dtype (np.dtype)

    return:
    - grid (np.ndarray)

    """

    xs = np.linspace(0, width - 1, width, dtype=dtype)
    ys = np.linspace(0, height - 1, height, dtype=dtype)
    theta = xs * 2 * np.pi / width - np.pi
    phi = ys * np.pi / height - np.pi / 2
    theta, phi = np.meshgrid(theta, phi)
    a = np.stack((theta, phi), axis=-1)
    norm_A = 1
    x = norm_A * np.cos(a[..., 1]) * np.cos(a[..., 0])
    y = norm_A * np.cos(a[..., 1]) * np.sin(a[..., 0])
    z = norm_A * np.sin(a[..., 1])
    grid = np.stack((x, y, z), axis=-1)

    if batch is not None:
        assert isinstance(
            batch, int
        ), f"ERR: batch needs to be integer: batch={batch}"
        assert (
            batch > 0
        ), f"ERR: batch size needs to be larger than 0: batch={batch}"
        # FIXME: faster way of copying?
        grid = np.concatenate([grid[np.newaxis, ...]] * batch)
        # grid shape is (b, h, w, 3)

    return grid


def create_xyz_grid(
    w_face: int,
    batch: Optional[int] = None,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """xyz coordinates of the faces of the cube"""

    ratio = (w_face - 1) / w_face

    out = np.zeros((w_face, w_face * 6, 3), dtype=dtype)
    rng = np.linspace(-0.5 * ratio, 0.5 * ratio, num=w_face, dtype=dtype)

    # Front face (x = 0.5)
    out[:, 0 * w_face : 1 * w_face, [1, 2]] = np.stack(
        np.meshgrid(rng, -rng), -1
    )
    out[:, 0 * w_face : 1 * w_face, 0] = 0.5

    # Right face (y = -0.5)
    out[:, 1 * w_face : 2 * w_face, [0, 2]] = np.stack(
        np.meshgrid(-rng, -rng), -1
    )
    out[:, 1 * w_face : 2 * w_face, 1] = 0.5

    # Back face (x = -0.5)
    out[:, 2 * w_face : 3 * w_face, [1, 2]] = np.stack(
        np.meshgrid(-rng, -rng), -1
    )
    out[:, 2 * w_face : 3 * w_face, 0] = -0.5

    # Left face (y = 0.5)
    out[:, 3 * w_face : 4 * w_face, [0, 2]] = np.stack(
        np.meshgrid(rng, -rng), -1
    )
    out[:, 3 * w_face : 4 * w_face, 1] = -0.5

    # Up face (z = 0.5)
    out[:, 4 * w_face : 5 * w_face, [1, 0]] = np.stack(
        np.meshgrid(rng, rng), -1
    )
    out[:, 4 * w_face : 5 * w_face, 2] = 0.5

    # Down face (z = -0.5)
    out[:, 5 * w_face : 6 * w_face, [1, 0]] = np.stack(
        np.meshgrid(rng, -rng), -1
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
        out = np.concatenate([out[np.newaxis, ...]] * batch)

    return out
