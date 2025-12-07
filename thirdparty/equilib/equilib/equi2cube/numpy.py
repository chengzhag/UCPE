#!/usr/bin/env python3

from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np

from equilib.grid_sample import numpy_grid_sample
from equilib.numpy_utils import create_xyz_grid, create_rotation_matrices


def cube_hsplits(cube_h: np.ndarray) -> List[np.ndarray]:
    """Returns list of horizontal splits (doesn't split batch)"""
    assert cube_h.shape[-2] * 6 == cube_h.shape[-1]
    # order ["F", "R", "B", "L", "U", "D"]
    splits = np.split(cube_h, 6, axis=-1)  # works batched
    assert len(splits) == 6
    assert splits[0].shape == (*cube_h.shape[0:3], cube_h.shape[-2])
    return splits


def cube_h2list(cube_h: np.ndarray) -> List[List[np.ndarray]]:
    bs = cube_h.shape[0]
    cube_lists = []
    for b in range(bs):
        cube_lists.append(np.split(cube_h[b], 6, axis=-1))
    return cube_lists


def cube_h2dict(cube_h: np.ndarray) -> List[Dict[str, np.ndarray]]:
    bs = cube_h.shape[0]
    cube_list = cube_hsplits(cube_h)

    cube_dicts = []
    for b in range(bs):
        cube_dicts.append(
            {
                k: deepcopy(cube_list[i][b])
                for i, k in enumerate(["F", "R", "B", "L", "U", "D"])
            }
        )
    return cube_dicts


def cube_h2dice(cube_h: np.ndarray) -> np.ndarray:
    bs = cube_h.shape[0]
    cube_list = cube_hsplits(cube_h)

    w = cube_h.shape[-2]
    cube_dice = np.zeros(
        (bs, cube_h.shape[-3], w * 3, w * 4), dtype=cube_h.dtype
    )
    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for b in range(bs):
        for i, (sx, sy) in enumerate(sxy):
            cube_dice[
                b, :, sy * w : (sy + 1) * w, sx * w : (sx + 1) * w
            ] = deepcopy(cube_list[i][b, ...])

    return cube_dice


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
    xyz: np.ndarray, h_equi: int, w_equi: int, method: str = "robust"
) -> np.ndarray:
    # convert to rotation
    phi = np.arcsin(xyz[..., 2] / np.linalg.norm(xyz, axis=-1))
    theta = np.arctan2(xyz[..., 1], xyz[..., 0])

    if method == "robust":
        # convert to pixel
        # I thought it would be faster if it was done all at once,
        # but it was faster separately
        ui = (theta / (2 * np.pi) - 0.5) * w_equi - 0.5
        uj = (0.5 - phi / np.pi) * h_equi - 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        # NOTE: this asserts that theta and phi are in range
        # the range of theta is -pi ~ pi
        # the range of phi is -pi/2 ~ pi/2
        # this means that if the input `rots` have rotations that are
        # out of range, it will not work with `faster`
        ui = (theta / (2 * np.pi) - 0.5) * w_equi - 0.5
        uj = (0.5 - phi / np.pi) * h_equi - 0.5
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
    w_face: int,
    cube_format: str,
    z_down: bool,
    mode: str,
    clip_output: bool = True,
    override_func: Optional[Callable[[], Any]] = None,
) -> Union[np.ndarray, List[List[np.ndarray]], List[Dict[str, np.ndarray]]]:
    """Call Equi2Cube

    params:
    - equi (np.ndarray)
    - rots (List[Dict[str, float]])
    - w_face (int)
    - cube_format (str): ('horizon', 'list', 'dict', 'dice')
    - z_down (str)
    - mode (str)
    - override_func (Callable): function for overriding `grid_sample`

    returns:
    - cubemaps

    """

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

    # initialize output array (horizon)
    out = np.empty((bs, c, w_face, w_face * 6), dtype=dtype)

    # create grid
    xyz = create_xyz_grid(w_face=w_face, batch=bs, dtype=dtype)
    xyz = xyz[..., np.newaxis]

    # FIXME: not sure why, but z-axis is facing the opposite
    # probably I need to change the way I choose the xyz coordinates
    # this is a temporary fix for now
    z_down = not z_down
    # create batched rotation matrices
    R = create_rotation_matrices(rots=rots, z_down=z_down, dtype=dtype)

    # rotate grid
    xyz = matmul(xyz, R, method="faster")

    # create a pixel map grid
    grid = convert_grid(xyz=xyz, h_equi=h_equi, w_equi=w_equi, method="robust")

    # grid sample
    if override_func is not None:
        out = override_func(  # type: ignore
            img=equi, grid=grid, out=out, mode=mode
        )
    else:
        out = numpy_grid_sample(img=equi, grid=grid, out=out, mode=mode)

    # clip by input
    out = (
        out.astype(equi_dtype)
        if equi_dtype == np.dtype(np.uint8) or not clip_output
        else np.clip(out, np.min(equi), np.max(equi))
    )

    # reformat the output
    # FIXME: needs to test this
    if cube_format == "horizon":
        pass
    elif cube_format == "list":
        out = cube_h2list(out)
    elif cube_format == "dict":
        out = cube_h2dict(out)
    elif cube_format == "dice":
        out = cube_h2dice(out)
    else:
        raise NotImplementedError("{} is not supported".format(cube_format))

    return out
