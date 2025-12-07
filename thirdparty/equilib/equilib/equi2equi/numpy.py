#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from equilib.grid_sample import numpy_grid_sample
from equilib.numpy_utils import create_normalized_grid, create_rotation_matrices


def matmul(m: np.ndarray, R: np.ndarray, method: str = "faster") -> np.ndarray:
    if method == "robust":
        # When target image size is smaller, it might be faster with `matmul`
        # but not by much
        M = np.matmul(R[:, np.newaxis, np.newaxis, ...], m)
    elif method == "faster":
        # `einsum` is probably fastest, but it might not be accurate
        # I've tested it, and it's really close when it is float64,
        # but loses precision for float32
        # trade off between precision and speed i guess
        # around x3 ~ x10 faster (faster when batch size is high)
        batch_size = m.shape[0]
        M = np.empty_like(m)
        for b in range(batch_size):
            M[b, ...] = np.einsum(
                "ik,...kj->...ij", R[b, ...], m[b, ...], optimize=True
            )
    else:
        raise ValueError(f"ERR: {method} is not supported")

    M = M.squeeze(-1)
    return M


def convert_grid(
    M: np.ndarray, h_equi: int, w_equi: int, method: str = "robust"
) -> np.ndarray:
    # convert to rotation
    phi = np.arcsin(M[..., 2] / np.linalg.norm(M, axis=-1))
    theta = np.arctan2(M[..., 1], M[..., 0])

    if method == "robust":
        # convert to pixel
        # I thought it would be faster if it was done all at once,
        # but it was faster separately
        ui = (theta - np.pi) * w_equi / (2 * np.pi)
        uj = (phi - np.pi / 2) * h_equi / np.pi
        ui += 0.5
        uj += 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        # NOTE: this asserts that theta and phi are in range
        # the range of theta is -pi ~ pi
        # the range of phi is -pi/2 ~ pi/2
        # this means that if the input `rots` have rotations that are
        # out of range, it will not work with `faster`
        ui = (theta - np.pi) * w_equi / (2 * np.pi)
        uj = (phi - np.pi / 2) * h_equi / np.pi
        ui += 0.5
        uj += 0.5
        ui = np.where(ui < 0, ui + w_equi, ui)
        ui = np.where(ui >= w_equi, ui - w_equi, ui)
        uj = np.where(uj < 0, uj + h_equi, uj)
        uj = np.where(uj >= h_equi, uj - h_equi, uj)
    else:
        raise ValueError(f"ERR: {method} is not supported")

    # stack the pixel maps into a grid
    grid = np.stack((uj, ui), axis=-3)

    return grid


def run(
    src: np.ndarray,
    rots: List[Dict[str, float]],
    z_down: bool,
    mode: str,
    height: Optional[int] = None,
    width: Optional[int] = None,
    clip_output: bool = True,
    override_func: Optional[Callable[[], Any]] = None,
) -> np.ndarray:
    """Run Equi2Equi

    params:
    - src (np.ndarray): 4 dims (b, c, h, w)
    - rot (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - height, width (Optional[int]): height and width of the target
    - override_func (Callable): function for overriding `grid_sample`

    return:
    - out (np.ndarray)

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
    ), f"ERR: batch size of `src` and `rot` differs: {len(src)} vs {len(rots)}"

    src_dtype = src.dtype
    assert src_dtype in (np.uint8, np.float32, np.float64), (
        f"ERR: input equirectangular image has dtype of {src_dtype}\n"
        f"which is incompatible: try {(np.uint8, np.float32, np.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    dtype = (
        np.dtype(np.float32) if src_dtype == np.dtype(np.uint8) else src_dtype
    )
    assert dtype in (np.float32, np.float64), (
        f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
        f"try {(np.float32, np.float64)}"
    )

    bs, c, h_equi, w_equi = src.shape

    assert type(height) == type(
        width
    ), "ERR: `height` and `width` does not match types (maybe it was set separately?)"
    if height is None and width is None:
        height = h_equi
        width = w_equi
    else:
        assert isinstance(height, int) and isinstance(width, int)

    # initialize output array
    out = np.empty((bs, c, height, width), dtype=dtype)

    # create grid and transfrom matrix
    m = create_normalized_grid(
        height=height, width=width, batch=bs, dtype=dtype
    )
    m = m[..., np.newaxis]

    # create batched rotation matrices
    R = create_rotation_matrices(rots=rots, z_down=z_down, dtype=dtype)

    # rotate the grid
    M = matmul(m, R, method="faster")

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    # grid sample
    if override_func is not None:
        out = override_func(  # type: ignore
            img=src, grid=grid, out=out, mode=mode
        )
    else:
        out = numpy_grid_sample(img=src, grid=grid, out=out, mode=mode)

    # clip by input
    out = (
        out.astype(src_dtype)
        if src_dtype == np.dtype(np.uint8) or not clip_output
        else np.clip(out, np.min(src), np.max(src))
    )

    return out
