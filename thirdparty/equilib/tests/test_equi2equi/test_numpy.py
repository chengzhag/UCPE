#!/usr/bin/env python3

"""Mainly tests against naive method or baselines (cv2 and scipy)

FIXME: better assertions
"""

from copy import deepcopy
import os
from typing import Optional

import numpy as np

import pytest

from equilib.equi2equi.numpy import run

from tests.grid_sample.numpy.baselines import (
    cv2,
    grid_sample_cv2,
    grid_sample_scipy,
    map_coordinates,
)
from tests.helpers.benchmarking import check_close, how_many_closes, mae, mse
from tests.helpers.image_io import load2numpy, save
from tests.helpers.timer import func_timer, wrapped_partial
from tests.helpers.rot_path import (
    create_rots,
    create_rots_pitch,
    create_rots_yaw,
)

run_cv2 = wrapped_partial(run, override_func=grid_sample_cv2)
run_scipy = wrapped_partial(run, override_func=grid_sample_scipy)

SAVE_ROOT = "tests/test_equi2equi/results"
DATA_ROOT = "tests/data"
IMG_NAME = "test.jpg"


def get_img(dtype: np.dtype = np.float32) -> np.ndarray:
    # path to equirectangular image
    path = os.path.join(DATA_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def make_batch(img: np.ndarray, bs: int = 1) -> np.ndarray:
    imgs = np.empty((bs, *img.shape), dtype=img.dtype)
    for b in range(bs):
        imgs[b, ...] = deepcopy(img)
    return imgs


def get_metrics(o1: np.ndarray, o2: np.ndarray, rtol: float, atol: float):
    is_close = check_close(o1, o2, rtol=rtol, atol=atol)
    r_close = how_many_closes(o1, o2, rtol=rtol, atol=atol)
    err_mse = mse(o1, o2)
    err_mae = mae(o1, o2)
    return is_close, r_close, err_mse, err_mae


def bench_baselines(
    bs: int,
    z_down: bool,
    mode: str,
    height: Optional[int],
    width: Optional[int],
    dtype: np.dtype = np.dtype(np.float32),
    rotation: str = "forward",
    save_outputs: bool = False,
) -> None:
    # print parameters for debugging
    print()
    print("bs, grid(height, width):", bs, (height, width))
    print("dtype:", dtype)
    print("z_down:", z_down)
    print("mode:", mode)
    print("rotation:", rotation)

    if dtype == np.float32:
        rtol = 1e-03
        atol = 1e-05
    elif dtype == np.float64:
        rtol = 1e-05
        atol = 1e-08
    else:
        rtol = 1e-01
        atol = 1e-03

    # obtaining single equirectangular image
    img = get_img(dtype)

    # making batch
    imgs = make_batch(img, bs=bs)
    print(imgs.shape, imgs.dtype)

    # generate rotation parameters
    if rotation == "forward":
        rots = create_rots(bs=bs)
    elif rotation == "pitch":
        rots = create_rots_pitch(bs=bs)
    elif rotation == "yaw":
        rots = create_rots_yaw(bs=bs)
    else:
        raise ValueError

    print("scipy:")
    out_scipy = func_timer(run_scipy)(
        src=imgs,
        rots=rots,
        z_down=z_down,
        mode=mode,
        height=height,
        width=width,
    )
    print("cv2")
    out_cv2 = func_timer(run_cv2)(
        src=imgs,
        rots=rots,
        z_down=z_down,
        mode=mode,
        height=height,
        width=width,
    )
    print("numpy")
    out = func_timer(run)(
        src=imgs,
        rots=rots,
        z_down=z_down,
        mode=mode,
        height=height,
        width=width,
    )

    assert (
        out.dtype == out_scipy.dtype == out_cv2.dtype == dtype
    ), "output dims should match"

    # FIXME: add valid assertions

    # quantitative
    print()
    print(">>> compare against scipy")
    is_close, r_close, err_mse, err_mae = get_metrics(
        out, out_scipy, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-03
    assert err_mae < 1e-01

    print()
    print(">>> compare against cv2")
    is_close, r_close, err_mse, err_mae = get_metrics(
        out, out_cv2, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-03
    assert err_mae < 1e-01

    print()
    print(">>> compare scipy and cv2")
    is_close, r_close, err_mse, err_mae = get_metrics(
        out_cv2, out_scipy, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-03
    assert err_mae < 1e-01

    if save_outputs:
        # qualitative
        # save the outputs and see the images
        for b in range(bs):
            save(out[b], os.path.join(SAVE_ROOT, f"out_{b}.jpg"))
            save(out_cv2[b], os.path.join(SAVE_ROOT, f"out_cv2_{b}.jpg"))
            save(out_scipy[b], os.path.join(SAVE_ROOT, f"out_scipy_{b}.jpg"))


@pytest.mark.skipif(cv2 is None, reason="cv2 is None; not installed")
@pytest.mark.skipif(
    map_coordinates is None,
    reason="scipy.map_coordinate is None; not installed",
)
@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("z_down", [False])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rotation", ["pitch"])
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64])
def test_equi2equi_against_baselines(
    bs: int,
    z_down: bool,
    mode: str,
    dtype: np.dtype,
    rotation: str,
    height: int,
    width: int,
) -> None:
    bench_baselines(
        bs=bs,
        z_down=z_down,
        mode=mode,
        height=height,
        width=width,
        dtype=dtype,
        rotation=rotation,
    )
