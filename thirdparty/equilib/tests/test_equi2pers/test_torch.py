#!/usr/bin/env python3

"""Mainly tests against numpy method

FIXME: better evaluation (fixing values for errors)

"""

import os
from copy import deepcopy

import numpy as np

import pytest

import torch

from equilib.equi2pers.numpy import run as run_numpy
from equilib.equi2pers.torch import run as run

from tests.helpers.benchmarking import check_close, how_many_closes, mae, mse
from tests.helpers.image_io import load2numpy, load2torch, save
from tests.helpers.timer import func_timer, wrapped_partial
from tests.helpers.rot_path import (
    create_rots,
    create_rots_pitch,
    create_rots_yaw,
)

run_native = wrapped_partial(run, backend="native")
run_pure = wrapped_partial(run, backend="pure")

IMG_ROOT = "tests/data"
SAVE_ROOT = "tests/test_equi2pers/results"
IMG_NAME = "test.jpg"


def get_numpy_img(dtype: np.dtype = np.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def get_torch_img(dtype: torch.dtype = torch.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2torch(path, dtype=dtype, is_cv2=False)
    return img


def make_batch(img, bs: int = 1):
    if isinstance(img, np.ndarray):
        imgs = np.empty((bs, *img.shape), dtype=img.dtype)
        for b in range(bs):
            imgs[b, ...] = deepcopy(img)
    elif torch.is_tensor(img):
        imgs = torch.empty((bs, *img.shape), dtype=img.dtype)
        for b in range(bs):
            imgs[b, ...] = img.clone()
    else:
        raise ValueError()
    return imgs


def get_metrics(o1, o2, rtol: float, atol: float):
    is_close = check_close(o1, o2, rtol=rtol, atol=atol)
    r_close = how_many_closes(o1, o2, rtol=rtol, atol=atol)
    err_mse = mse(o1, o2)
    err_mae = mae(o1, o2)
    return is_close, r_close, err_mse, err_mae


def bench_cpu(
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
        torch_dtype = torch.float32
        rtol = 1e-03
        atol = 1e-05
    elif dtype == np.float64:
        torch_dtype = torch.float64
        rtol = 1e-05
        atol = 1e-08
    else:
        torch_dtype = torch.uint8
        rtol = 1e-01
        atol = 1e-03

    numpy_img = get_numpy_img(dtype=dtype)
    torch_img = get_torch_img(dtype=torch_dtype)

    numpy_imgs = make_batch(numpy_img, bs=bs)
    torch_imgs = make_batch(torch_img, bs=bs)

    # generate rotation parameters
    if rotation == "forward":
        rots = create_rots(bs=bs)
    elif rotation == "pitch":
        rots = create_rots_pitch(bs=bs)
    elif rotation == "yaw":
        rots = create_rots_yaw(bs=bs)
    else:
        raise ValueError

    print("numpy")
    numpy_out = func_timer(run_numpy)(
        equi=numpy_imgs,
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )

    print("native")
    native_out = func_timer(run_native)(
        equi=torch_imgs.clone(),
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )

    print("pure")
    pure_out = func_timer(run_pure)(
        equi=torch_imgs.clone(),
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )

    numpy_out = torch.from_numpy(numpy_out)

    assert (
        numpy_out.dtype == native_out.dtype == pure_out.dtype == torch_dtype
    ), "output dtypes should match"
    assert (
        numpy_out.shape == native_out.shape == pure_out.shape
    ), "output dims should match"

    # quantitative
    print()
    print(">>> compare native and numpy")
    is_close, r_close, err_mse, err_mae = get_metrics(
        native_out, numpy_out, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-03
    assert err_mae < 1e-02

    print()
    print(">>> compare native and pure")
    is_close, r_close, err_mse, err_mae = get_metrics(
        native_out, pure_out, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-03
    assert err_mae < 1e-02

    print()
    print(">>> compare pure and numpy")
    is_close, r_close, err_mse, err_mae = get_metrics(
        pure_out, numpy_out, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert err_mse < 1e-03
    assert err_mae < 1e-02

    if save_outputs:
        # qualitative
        # save the outputs and see the images
        for b in range(bs):
            save(
                numpy_out[b], os.path.join(SAVE_ROOT, f"out_cpu_numpy_{b}.jpg")
            )
            save(
                native_out[b],
                os.path.join(SAVE_ROOT, f"out_cpu_native_{b}.jpg"),
            )
            save(pure_out[b], os.path.join(SAVE_ROOT, f"out_cpu_pure_{b}.jpg"))


def bench_gpu(
    bs: int,
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: np.dtype = np.float32,
    torch_dtype: torch.dtype = torch.float32,
    rotation: str = "forward",
    save_outputs: bool = False,
) -> None:
    device = torch.device("cuda")
    assert torch_dtype in (torch.float16, torch.float32, torch.float64)

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

    numpy_img = get_numpy_img(dtype=dtype)
    torch_img = get_torch_img(dtype=torch_dtype)

    numpy_imgs = make_batch(numpy_img, bs=bs)
    torch_imgs = make_batch(torch_img, bs=bs)

    # generate rotation parameters
    if rotation == "forward":
        rots = create_rots(bs=bs)
    elif rotation == "pitch":
        rots = create_rots_pitch(bs=bs)
    elif rotation == "yaw":
        rots = create_rots_yaw(bs=bs)
    else:
        raise ValueError

    print("numpy")
    numpy_out = func_timer(run_numpy)(
        equi=numpy_imgs,
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )

    print("native")
    native_out = func_timer(run_native)(
        equi=torch_imgs.clone().to(device),
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )

    print("pure")
    pure_out = func_timer(run_pure)(
        equi=torch_imgs.clone().to(device),
        rots=rots,
        height=height,
        width=width,
        fov_x=fov_x,
        skew=0.0,
        z_down=z_down,
        mode=mode,
    )

    numpy_out = torch.from_numpy(numpy_out)
    numpy_out = numpy_out.type(torch_dtype)
    print(numpy_out.shape, numpy_out.dtype)
    numpy_out = numpy_out.to(device)

    assert (
        numpy_out.dtype == native_out.dtype == pure_out.dtype == torch_dtype
    ), "output dtypes should match"
    assert (
        numpy_out.shape == native_out.shape == pure_out.shape
    ), "output dims should match"

    # quantitative
    print()
    print(">>> compare native and numpy")
    is_close, r_close, err_mse, err_mae = get_metrics(
        native_out, numpy_out, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    if torch_dtype == torch.float16:
        assert err_mse < 1
        assert err_mae < 1
    else:
        assert err_mse < 1e-03
        assert err_mae < 1e-02

    print()
    print(">>> compare native and pure")
    is_close, r_close, err_mse, err_mae = get_metrics(
        native_out, pure_out, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    if torch_dtype == torch.float16:
        assert err_mse < 1
        assert err_mse < 1
    else:
        assert err_mse < 1e-03
        assert err_mae < 1e-02

    print()
    print(">>> compare pure and numpy")
    is_close, r_close, err_mse, err_mae = get_metrics(
        pure_out, numpy_out, rtol=rtol, atol=atol
    )
    print("close?", is_close)
    print("how many closes?", r_close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    if torch_dtype == torch.float16:
        assert err_mse < 1
        assert err_mae < 1
    else:
        assert err_mse < 1e-03
        assert err_mae < 1e-02

    numpy_out = numpy_out.to("cpu")
    native_out = native_out.to("cpu")
    pure_out = pure_out.to("cpu")

    if save_outputs:
        # qualitative
        # save the outputs and see the images
        for b in range(bs):
            save(
                numpy_out[b], os.path.join(SAVE_ROOT, f"out_gpu_numpy_{b}.jpg")
            )
            save(
                native_out[b],
                os.path.join(SAVE_ROOT, f"out_gpu_native_{b}.jpg"),
            )
            save(pure_out[b], os.path.join(SAVE_ROOT, f"out_gpu_pure_{b}.jpg"))


@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("fov_x", [90.0])
@pytest.mark.parametrize("z_down", [False])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rotation", ["pitch"])
def test_equi2pers_cpu(
    bs: int,
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: np.dtype,
    rotation: str,
) -> None:
    bench_cpu(
        bs=bs,
        height=height,
        width=width,
        fov_x=fov_x,
        z_down=z_down,
        mode=mode,
        dtype=dtype,
        rotation=rotation,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("fov_x", [90.0])
@pytest.mark.parametrize("z_down", [False])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [np.float32])
@pytest.mark.parametrize("torch_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rotation", ["pitch"])
def test_equi2pers_gpu(
    bs: int,
    height: int,
    width: int,
    fov_x: float,
    z_down: bool,
    mode: str,
    dtype: np.dtype,
    torch_dtype: torch.dtype,
    rotation: str,
) -> None:
    bench_gpu(
        bs=bs,
        height=height,
        width=width,
        fov_x=fov_x,
        z_down=z_down,
        mode=mode,
        dtype=dtype,
        torch_dtype=torch_dtype,
        rotation=rotation,
    )
