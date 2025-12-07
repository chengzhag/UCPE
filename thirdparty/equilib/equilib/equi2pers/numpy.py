#!/usr/bin/env python3

from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np

from equilib.grid_sample import numpy_grid_sample
from equilib.numpy_utils import (
    create_global2camera_rotation_matrix,
    create_grid,
    create_intrinsic_matrix,
    create_rotation_matrices,
)


@lru_cache(maxsize=128)
def create_cam2global_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    K = create_intrinsic_matrix(
        height=height, width=width, fov_x=fov_x, skew=skew, dtype=dtype
    )
    g2c_rot = create_global2camera_rotation_matrix(dtype=dtype)

    # FIXME: change to faster inverse
    K_inv = np.linalg.inv(K)

    return g2c_rot @ K_inv


def prep_matrices(
    height: int,
    width: int,
    batch: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: np.dtype = np.dtype(np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    m = create_grid(height=height, width=width, batch=batch, dtype=dtype)
    m = m[..., np.newaxis]
    G = create_cam2global_matrix(
        height=height, width=width, fov_x=fov_x, skew=skew, dtype=dtype
    )

    return m, G


def matmul(
    m: np.ndarray, G: np.ndarray, R: np.ndarray, method: str = "faster"
) -> np.ndarray:
    if method == "robust":
        # When target image size is smaller, it might be faster with `matmul`
        # but not by much
        M = np.matmul(np.matmul(R, G)[:, np.newaxis, np.newaxis, ...], m)
    elif method == "faster":
        # `einsum` is probably fastest, but it might not be accurate
        # I've tested it, and it's really close when it is float64,
        # but loses precision for float32
        # trade off between precision and speed i guess
        # around x3 ~ x10 faster (faster when batch size is high)
        batch_size = m.shape[0]
        M = np.empty_like(m)
        C = np.einsum("bik,kj->bij", R, G, optimize=True)
        for b in range(batch_size):
            M[b, ...] = np.einsum(
                "ik,...kj->...ij", C[b, ...], m[b, ...], optimize=True
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
    equi: np.ndarray,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
    mode: str,
    clip_output: bool = True,
    override_func: Optional[Callable[[], Any]] = None,
) -> np.ndarray:
    """Run Equi2Pers

    params:
    - equi (np.ndarray): 4 dims (b, c, h, w)
    - rot (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - height, width (int): height and width of perspective view
    - fov_x (float): fov of horizontal axis in degrees
    - skew (float): skew of the camera
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - override_func (Callable): function for overriding `grid_sample`

    return:
    - out (np.ndarray)

    NOTE: acceptable dtypes for `equi` are currently uint8, float32, and float64.
    Floats are prefered since numpy calculations are optimized for floats.

    NOTE: output array has the same dtype as `equi`

    NOTE: you can override `equilib`'s grid_sample with over grid sampling methods
    using `override_func`. The input to this function have to match `grid_sample`.

    """

    # NOTE: Assume that the inputs `equi` and `rots` are already batched up
    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        rots
    ), f"ERR: batch size of equi and rot differs: {len(equi)} vs {len(rots)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (np.uint8, np.float32, np.float64), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}\n"
        f"which is incompatible: try {(np.uint8, np.float32, np.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    dtype = (
        np.dtype(np.float32) if equi_dtype == np.dtype(np.uint8) else equi_dtype
    )
    assert dtype in (np.float32, np.float64), (
        f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
        f"try {(np.float32, np.float64)}"
    )

    bs, c, h_equi, w_equi = equi.shape

    # initialize output array
    out = np.empty((bs, c, height, width), dtype=dtype)

    # create grid and transfrom matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(rots=rots, z_down=z_down, dtype=dtype)

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    # grid sample
    if override_func is not None:
        # NOTE: override func is used for test purposes
        out = override_func(  # type: ignore
            img=equi, grid=grid, out=out, mode=mode
        )
    else:
        out = numpy_grid_sample(
            img=equi,
            grid=grid,
            out=out,  # FIXME: pass-by-reference confusing?
            mode=mode,
        )

    out = (
        out.astype(equi_dtype)
        if equi_dtype == np.dtype(np.uint8) or not clip_output
        else np.clip(out, np.min(equi), np.max(equi))
    )

    return out


def get_bounding_fov(
    equi: np.ndarray,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
) -> np.ndarray:
    # NOTE: Assume that the inputs `equi` and `rots` are already batched up
    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        rots
    ), f"ERR: batch size of equi and rot differs: {len(equi)} vs {len(rots)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (np.uint8, np.float32, np.float64), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}\n"
        f"which is incompatible: try {(np.uint8, np.float32, np.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    dtype = (
        np.dtype(np.float32) if equi_dtype == np.dtype(np.uint8) else equi_dtype
    )
    assert dtype in (np.float32, np.float64), (
        f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
        f"try {(np.float32, np.float64)}"
    )

    bs, c, h_equi, w_equi = equi.shape

    # create grid and transfrom matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(rots=rots, z_down=z_down, dtype=dtype)

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    bboxs = []

    # top row
    for out_x in range(width):
        bboxs.append(grid[:, :, 0, out_x])

    # right column
    for out_y in range(height):
        if out_y > 0:  # exclude first
            bboxs.append(grid[:, :, out_y, width - 1])

    # bottom row
    for out_x in range(width - 2, 0, -1):
        bboxs.append(grid[:, :, height - 1, out_x])

    # left column
    for out_y in range(height - 1, 0, -1):
        bboxs.append(grid[:, :, out_y, 0])

    assert len(bboxs) == width * 2 + (height - 2) * 2

    bboxs = np.stack(bboxs, axis=1)
    bboxs = np.rint(bboxs).astype(np.int64)

    return bboxs
