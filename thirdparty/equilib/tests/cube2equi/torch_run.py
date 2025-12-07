#!/usr/bin/env python3

"""Run torch cube2equi along with comparisons with numpy based method

"""

from copy import deepcopy
import os

import numpy as np

import torch

from equilib.cube2equi.numpy import run as run_numpy
from equilib.cube2equi.torch import run as run

from tests.helpers.benchmarking import check_close, how_many_closes, mae, mse
from tests.helpers.image_io import load2numpy, load2torch, save
from tests.helpers.timer import func_timer, wrapped_partial

run_native = wrapped_partial(run, backend="native")
run_pure = wrapped_partial(run, backend="pure")

IMG_ROOT = "tests/data"
SAVE_ROOT = "tests/cube2equi/results"
IMG_NAME = "test_horizon.jpg"


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
    mode: str,
    dtype: np.dtype = np.dtype(np.float32),
    save_outputs: bool = False,
) -> None:
    # print parameters for debugging
    print()
    print("bs, grid(height, width):", bs, (height, width))
    print("dtype:", dtype)
    print("mode:", mode)

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

    print("numpy")
    numpy_out = func_timer(run_numpy)(
        horizon=numpy_imgs, height=height, width=width, mode=mode
    )

    print("native")
    native_out = func_timer(run_native)(
        horizon=torch_imgs.clone(), height=height, width=width, mode=mode
    )

    print("pure")
    pure_out = func_timer(run_pure)(
        horizon=torch_imgs.clone(), height=height, width=width, mode=mode
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

    assert err_mse < 1e-04
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
                numpy_out[b],
                os.path.join(SAVE_ROOT, f"out_torch_numpy_{b}.jpg"),
            )
            save(
                native_out[b],
                os.path.join(SAVE_ROOT, f"out_torch_native_{b}.jpg"),
            )
            save(
                pure_out[b], os.path.join(SAVE_ROOT, f"out_torch_pure_{b}.jpg")
            )


def bench_gpu(
    bs: int,
    height: int,
    width: int,
    mode: str,
    dtype: np.dtype = np.dtype(np.float32),
    torch_dtype: torch.dtype = torch.float32,
    save_outputs: bool = False,
) -> None:
    device = torch.device("cuda")
    assert torch_dtype in (torch.float16, torch.float32, torch.float64)

    # print parameters for debugging
    print()
    print("bs, grid(height, width):", bs, (height, width))
    print("dtype:", dtype)
    print("mode:", mode)

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

    print("numpy")
    numpy_out = func_timer(run_numpy)(
        horizon=numpy_imgs, height=height, width=width, mode=mode
    )

    print("native")
    native_out = func_timer(run_native)(
        horizon=torch_imgs.clone().to(device),
        height=height,
        width=width,
        mode=mode,
    )

    print("pure")
    pure_out = func_timer(run_pure)(
        horizon=torch_imgs.clone().to(device),
        height=height,
        width=width,
        mode=mode,
    )

    numpy_out = torch.from_numpy(numpy_out)
    numpy_out = numpy_out.type(torch_dtype)
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

    assert err_mse < 1e-04
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
                numpy_out[b], os.path.join(SAVE_ROOT, f"out_gpu_numpy_{b}.jpg")
            )
            save(
                native_out[b],
                os.path.join(SAVE_ROOT, f"out_gpu_native_{b}.jpg"),
            )
            save(pure_out[b], os.path.join(SAVE_ROOT, f"out_gpu_pure_{b}.jpg"))


if __name__ == "__main__":
    # parameters
    save_outputs = True

    # variables
    bs = 8
    height = 512
    width = 1024
    dtype = np.dtype(np.float32)
    mode = "bilinear"

    torch_dtype = torch.float32

    bench_cpu(
        bs=bs,
        height=height,
        width=width,
        mode=mode,
        dtype=dtype,
        save_outputs=save_outputs,
    )
    bench_gpu(
        bs=bs,
        height=height,
        width=width,
        mode=mode,
        dtype=dtype,
        torch_dtype=torch_dtype,
        save_outputs=save_outputs,
    )
