#!/usr/bin/env python3

from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple, Optional

import numpy as np

from equilib.grid_sample import numpy_grid_sample
from equilib.numpy_utils import (
    create_global2camera_rotation_matrix,
    create_normalized_grid,
    create_intrinsic_matrix,
    create_rotation_matrices,
)


@lru_cache(maxsize=128)
def create_global2cam_matrix(
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
    # K_inv = np.linalg.inv(K)

    return K @ g2c_rot.T


def prep_matrices(
    height: int,
    width: int,
    h_pers: int,
    w_pers: int,
    batch: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: np.dtype = np.dtype(np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    m = create_normalized_grid(height=height, width=width, batch=batch, dtype=dtype)
    m = m[..., np.newaxis]

    G = create_global2cam_matrix(
        height=h_pers, width=w_pers, fov_x=fov_x, skew=skew, dtype=dtype
    )

    return m, G


def matmul(
    m: np.ndarray, G: np.ndarray, R: np.ndarray, method: str = "faster"
) -> np.ndarray:
    if method == "robust":
        # When target image size is smaller, it might be faster with `matmul`
        # but not by much
        M = np.matmul(np.matmul(G[np.newaxis, ...], R)[:, np.newaxis, np.newaxis, ...], m)
    elif method == "faster":
        # `einsum` is probably fastest, but it might not be accurate
        # I've tested it, and it's really close when it is float64,
        # but loses precision for float32
        # trade off between precision and speed i guess
        # around x3 ~ x10 faster (faster when batch size is high)
        batch_size = m.shape[0]
        M = np.empty_like(m)
        C = np.einsum("ik,bkj->bij", G, R, optimize=True)
        for b in range(batch_size):
            M[b, ...] = np.einsum(
                "ik,...kj->...ij", C[b, ...], m[b, ...], optimize=True
            )
    else:
        raise ValueError(f"ERR: {method} is not supported")
    M = M.squeeze(-1)
    return M


def convert_grid(
    M: np.ndarray, h_pers: int, w_pers: int
) -> np.ndarray:
    # calculate image coordinates
    ui = M[..., 0] / M[..., 2]
    uj = M[..., 1] / M[..., 2]
    ui[M[..., 2] < 0] = -1
    uj[M[..., 2] < 0] = -1
    ui += 0.5
    uj += 0.5
    ui = np.where(ui < 0, -1, ui)
    ui = np.where(ui >= w_pers, -1, ui)
    uj = np.where(uj < 0, -1, uj)
    uj = np.where(uj >= h_pers, -1, uj)

    # stack the pixel maps into a grid
    grid = np.stack((uj, ui), axis=-3)

    return grid


def run(
    pers: np.ndarray,
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
    """Run Pers2Equi

    params:
    - pers (np.ndarray): 4 dims (b, c, h, w)
    - rot (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - height, width (int): height and width of equirectangular image
    - fov_x (float): fov of horizontal axis in degrees of the perspective image
    - skew (float): skew of the perspective image
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - override_func (Callable): function for overriding `grid_sample`

    return:
    - out (np.ndarray)

    NOTE: acceptable dtypes for `pers` are currently uint8, float32, and float64.
    Floats are prefered since numpy calculations are optimized for floats.

    NOTE: output array has the same dtype as `pers`

    NOTE: you can override `equilib`'s grid_sample with over grid sampling methods
    using `override_func`. The input to this function have to match `grid_sample`.

    """

    # NOTE: Assume that the inputs `pers` and `rots` are already batched up
    assert (
        len(pers.shape) == 4
    ), f"ERR: input `pers` should be 4-dim (b, c, h, w), but got {len(pers.shape)}"
    assert len(pers) == len(
        rots
    ), f"ERR: batch size of pers and rot differs: {len(pers)} vs {len(rots)}"

    pers_dtype = pers.dtype
    assert pers_dtype in (np.uint8, np.float32, np.float64), (
        f"ERR: input perspective image has dtype of {pers_dtype}\n"
        f"which is incompatible: try {(np.uint8, np.float32, np.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as pers
    dtype = (
        np.dtype(np.float32) if pers_dtype == np.dtype(np.uint8) else pers_dtype
    )
    assert dtype in (np.float32, np.float64), (
        f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
        f"try {(np.float32, np.float64)}"
    )

    bs, c, h_pers, w_pers = pers.shape

    # initialize output array
    out = np.empty((bs, c, height, width), dtype=dtype)

    # create grid and transfrom matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        h_pers=h_pers,
        w_pers=w_pers,
        fov_x=fov_x,
        skew=skew,
        batch=bs,
        dtype=dtype,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(rots=rots, z_down=z_down, dtype=dtype)

    # rotate and transform the grid
    M = matmul(m, G, R, method="robust")

    # create a pixel map grid
    grid = convert_grid(M=M, h_pers=h_pers, w_pers=w_pers)

    # grid sample
    if override_func is not None:
        # NOTE: override func is used for test purposes
        out = override_func(  # type: ignore
            img=pers, grid=grid, out=out, mode=mode
        )
    else:
        out = numpy_grid_sample(
            img=pers,
            grid=grid,
            out=out,  # FIXME: pass-by-reference confusing?
            mode=mode,
        )

    mask = (grid[:, 0] < 0) | (grid[:, 1] < 0)
    mask = mask[:, None].repeat(pers.shape[1], axis=1)
    out[mask] = 0

    out = (
        out.astype(pers_dtype)
        if pers_dtype == np.dtype(np.uint8) or not clip_output
        else np.clip(out, np.min(pers), np.max(pers))
    )

    return out
