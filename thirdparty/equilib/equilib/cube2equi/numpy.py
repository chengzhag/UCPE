#!/usr/bin/env python3

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# from equilib.grid_sample import numpy_grid_sample
from equilib.grid_sample.numpy.bilinear import interp2d

__all__ = ["convert2horizon", "run"]


def single_list2horizon(cube: List[np.ndarray]) -> np.ndarray:
    _, _, w = cube[0].shape
    assert len(cube) == 6
    assert sum(face.shape[-1] == w for face in cube) == 6
    return np.concatenate(cube, axis=-1)


def dice2horizon(dices: np.ndarray) -> np.ndarray:
    assert len(dices.shape) == 4
    w = dices.shape[-2] // 3
    assert dices.shape[-2] == w * 3 and dices.shape[-1] == w * 4

    # create a (b, c, h, w) horizon array
    horizons = np.empty((*dices.shape[0:2], w, w * 6), dtype=dices.dtype)

    # Order: F R B L U D
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        horizons[..., i * w : (i + 1) * w] = dices[
            ..., sy * w : (sy + 1) * w, sx * w : (sx + 1) * w
        ]
    return horizons


def dict2horizon(dicts: List[Dict[str, np.ndarray]]) -> np.ndarray:
    face_key = ("F", "R", "B", "L", "U", "D")
    c, _, w = dicts[0][face_key[0]].shape
    dtype = dicts[0][face_key[0]].dtype
    horizons = np.empty((len(dicts), c, w, w * 6), dtype=dtype)
    for b, cube in enumerate(dicts):
        horizons[b, ...] = single_list2horizon([cube[k] for k in face_key])
    return horizons


def list2horizon(lists: List[List[np.ndarray]]) -> np.ndarray:
    assert len(lists[0][0].shape) == 3
    c, w, _ = lists[0][0].shape
    dtype = lists[0][0].dtype
    horizons = np.empty((len(lists), c, w, w * 6), dtype=dtype)
    for b, cube in enumerate(lists):
        horizons[b, ...] = single_list2horizon(cube)
    return horizons


def convert2horizon(
    cubemap: Union[
        np.ndarray,
        List[np.ndarray],
        List[List[np.ndarray]],
        Dict[str, np.ndarray],
        List[Dict[str, np.ndarray]],
    ],
    cube_format: str,
) -> np.ndarray:
    """Converts supported cubemap formats to horizon

    params:
    - cubemap
    - cube_format (str): ('horizon', 'dice', 'dict', 'list')

    return:
    - horizon (np.ndarray)

    """

    # FIXME: better typing for mypy...

    if cube_format in ("horizon", "dice"):
        assert isinstance(
            cubemap, np.ndarray
        ), f"ERR: cubemap {cube_format} needs to be a np.ndarray"
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
        if isinstance(cubemap[0], np.ndarray):
            # single
            cubemap = [cubemap]
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


def _equirect_facetype(h: int, w: int) -> np.ndarray:
    """0F 1R 2B 3L 4U 5D"""

    int_dtype = np.dtype(np.int64)

    w_ratio = (w - 1) / w
    h_ratio = (h - 1) / h

    tp = np.roll(
        np.arange(4).repeat(w // 4)[None, :].repeat(h, 0), 3 * w // 8, 1
    )

    # Prepare ceil mask
    mask = np.zeros((h, w // 4), bool)
    idx = np.linspace(-(np.pi * w_ratio), np.pi * w_ratio, w // 4) / 4
    idx = h // 2 - np.around(np.arctan(np.cos(idx)) * h / (np.pi * h_ratio))
    idx = idx.astype(int_dtype)
    for i, j in enumerate(idx):
        mask[:j, i] = 1
    mask = np.roll(np.concatenate([mask] * 4, 1), 3 * w // 8, 1)

    tp[mask] = 4
    tp[np.flip(mask, 0)] = 5

    return tp.astype(int_dtype)


def create_equi_grid(
    h_out: int,
    w_out: int,
    w_face: int,
    batch: int,
    dtype: np.dtype = np.dtype(np.float32),
) -> Tuple[np.ndarray, np.ndarray]:
    w_ratio = (w_out - 1) / w_out
    h_ratio = (h_out - 1) / h_out
    theta = np.linspace(
        -(np.pi * w_ratio), np.pi * w_ratio, num=w_out, dtype=dtype
    )
    phi = np.linspace(
        np.pi * h_ratio / 2, -(np.pi * h_ratio) / 2, num=h_out, dtype=dtype
    )
    theta, phi = np.meshgrid(theta, phi)

    # Get face id to each pixel: 0F 1R 2B 3L 4U 5D
    tp = _equirect_facetype(h_out, w_out)

    # xy coordinate map
    coor_x = np.zeros((h_out, w_out), dtype=dtype)
    coor_y = np.zeros((h_out, w_out), dtype=dtype)

    for i in range(6):
        mask = tp == i

        if i < 4:
            coor_x[mask] = 0.5 * np.tan(theta[mask] - np.pi * i / 2)
            coor_y[mask] = (
                -0.5 * np.tan(phi[mask]) / np.cos(theta[mask] - np.pi * i / 2)
            )
        elif i == 4:
            c = 0.5 * np.tan(np.pi / 2 - phi[mask])
            coor_x[mask] = c * np.sin(theta[mask])
            coor_y[mask] = c * np.cos(theta[mask])
        elif i == 5:
            c = 0.5 * np.tan(np.pi / 2 - np.abs(phi[mask]))
            coor_x[mask] = c * np.sin(theta[mask])
            coor_y[mask] = -c * np.cos(theta[mask])

    # Final renormalize
    # coor_x = np.clip(np.clip(coor_x + 0.5, 0, 1) * w_face, 0, w_face - 1)
    # coor_y = np.clip(np.clip(coor_y + 0.5, 0, 1) * w_face, 0, w_face - 1)

    coor_x = (np.clip(coor_x, -0.5, 0.5) + 0.5) * w_face
    coor_y = (np.clip(coor_y, -0.5, 0.5) + 0.5) * w_face

    # change x axis of the x coordinate map
    for i in range(6):
        mask = tp == i
        coor_x[mask] = coor_x[mask] + w_face * i

    grid = np.stack((coor_y, coor_x), axis=0) - 0.5
    grid = np.concatenate([grid[np.newaxis, ...]] * batch)
    return grid, tp


def numpy_grid_sample(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray, cube_face_id: np.ndarray
):
    b, _, h, w = img.shape

    min_grid = np.floor(grid).astype(np.int64)
    max_grid = min_grid + 1
    d_grid = grid - min_grid

    _, _, grid_h, grid_w = grid.shape
    cube_face_min_grid = min_grid // h
    cube_face_max_grid = max_grid // h

    min_grid[:, 0, :, :] = np.clip(min_grid[:, 0, :, :], 0, None)
    max_grid[:, 0, :, :] = np.clip(max_grid[:, 0, :, :], None, h - 1)

    # FIXME: any way to do efficient batch?
    for i in range(b):
        for y in range(grid_h):
            for x in range(grid_w):
                if (
                    cube_face_max_grid[i, 1, y, x]
                    != cube_face_min_grid[i, 1, y, x]
                ):
                    if cube_face_max_grid[i, 1, y, x] != cube_face_id[y, x]:
                        max_grid[i, 1, y, x] -= 1
                    else:
                        min_grid[i, 1, y, x] += 1

        dy = d_grid[i, 0, ...]
        dx = d_grid[i, 1, ...]
        min_ys = min_grid[i, 0, ...]
        min_xs = min_grid[i, 1, ...]
        max_ys = max_grid[i, 0, ...]
        max_xs = max_grid[i, 1, ...]

        p00 = img[i][:, min_ys, min_xs]
        p10 = img[i][:, max_ys, min_xs]
        p01 = img[i][:, min_ys, max_xs]
        p11 = img[i][:, max_ys, max_xs]

        out[i, ...] = interp2d(p00, p10, p01, p11, dy, dx)

    return out


def run(
    horizon: np.ndarray,
    height: int,
    width: int,
    mode: str,
    clip_output: bool = True,
    override_func: Optional[Callable[[], Any]] = None,
) -> np.ndarray:
    """Run Cube2Equi

    params:
    - horizon (np.ndarray)
    - height, widht (int): output equirectangular image's size
    - mode (str)

    return:
    - equi (np.ndarray)

    NOTE: we assume that the input `horizon` is a 4 dim array

    """

    assert (
        len(horizon.shape) == 4
    ), f"ERR: `horizon` should be 4-dim (b, c, h, w), but got {horizon.shape}"

    horizon_dtype = horizon.dtype
    assert horizon_dtype in (np.uint8, np.float32, np.float64), (
        f"ERR: input horizon has dtype of {horizon_dtype}\n"
        f"which is incompatible: try {(np.uint8, np.float32, np.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as horizon
    dtype = (
        np.dtype(np.float32)
        if horizon_dtype == np.dtype(np.uint8)
        else horizon_dtype
    )
    assert dtype in (np.float32, np.float64), (
        f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
        f"try {(np.float32, np.float64)}"
    )

    bs, c, w_face, _ = horizon.shape

    # initialize output equi
    out = np.empty((bs, c, height, width), dtype=dtype)

    # create sampling grid
    grid, tp = create_equi_grid(
        h_out=height, w_out=width, w_face=w_face, batch=bs, dtype=dtype
    )

    # grid sample
    if override_func is not None:
        out = override_func(  # type: ignore
            img=horizon, grid=grid, out=out, mode=mode
        )
    else:
        # out = numpy_grid_sample(img=horizon, grid=grid, out=out, mode=mode)
        out = numpy_grid_sample(
            img=horizon, grid=grid, out=out, cube_face_id=tp
        )

    # clip by the input
    out = (
        out.astype(horizon_dtype)
        if horizon_dtype == np.dtype(np.uint8) or not clip_output
        else np.clip(out, np.min(horizon), np.max(horizon))
    )

    return out
