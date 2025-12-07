#!/usr/bin/env python3

"""Create baseline methods using cv2 and scipy for comparison

- Since the sampling grid is distorted, there are inconsistency between
  each of the methods
- cv2 seems the fastest
- scipy is faster than numpy when the sampling grid is large

- nearest: numpy == cv2
- bilinear: numpy == scipy

What metrics should I use to calculate sampling accuracy?

"""

import os
from copy import deepcopy

import numpy as np

from equilib.equi2pers.numpy import run

from tests.grid_sample.numpy.baselines import grid_sample_cv2, grid_sample_scipy
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

SAVE_ROOT = "tests/equi2pers/results"
DATA_ROOT = "tests/data"
IMG_NAME = "test.jpg"


def get_img(dtype: np.dtype = np.dtype(np.float32)):
    path = os.path.join(DATA_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def make_batch(img: np.ndarray, bs: int = 1):
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
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: np.dtype = np.dtype(np.float32),
    rotation: str = "forward",
    save_outputs: bool = False,
) -> None:
    # print parameters for debugging
    print()
    print("bs, grid(height, width, fov_x):", bs, (height, width, fov_x))
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
        equi=imgs,
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )
    print("cv2")
    out_cv2 = func_timer(run_cv2)(
        equi=imgs,
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )
    print("numpy")
    out = func_timer(run)(
        equi=imgs,
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )

    assert (
        out.dtype == out_scipy.dtype == out_cv2.dtype == dtype
    ), "output dtypes should match"

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

    assert err_mse < 1e-05
    assert err_mae < 1e-03

    print()
    print(">>> compare against cv2")
    is_close, r_close, err_mse, err_mae = get_metrics(
        out, out_cv2, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-05
    assert err_mae < 1e-03

    print()
    print(">>> compare scipy and cv2")
    is_close, r_close, err_mse, err_mae = get_metrics(
        out_cv2, out_scipy, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-05
    assert err_mae < 1e-03

    if save_outputs:
        # qualitative
        # save the outputs and see the images
        for b in range(bs):
            save(out[b], os.path.join(SAVE_ROOT, f"out_{b}.jpg"))
            save(out_cv2[b], os.path.join(SAVE_ROOT, f"out_cv2_{b}.jpg"))
            save(out_scipy[b], os.path.join(SAVE_ROOT, f"out_scipy_{b}.jpg"))


if __name__ == "__main__":
    # parameters
    save_outputs = True
    rotation = "pitch"  # ('forward', 'pitch', 'yaw')

    # variables
    bs = 8
    height = 256
    width = 512
    fov_x = 90.0
    dtype = np.dtype(np.float32)
    z_down = True
    mode = "bilinear"

    bench_baselines(
        bs=bs,
        height=height,
        width=width,
        fov_x=fov_x,
        z_down=z_down,
        mode=mode,
        dtype=dtype,
        rotation=rotation,
        save_outputs=save_outputs,
    )
