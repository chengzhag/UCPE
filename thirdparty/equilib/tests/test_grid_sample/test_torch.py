#!/usr/bin/env python3

import numpy as np

import torch

import pytest

from equilib.grid_sample.torch import grid_sample

from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.grid_sample.torch import (
    faster_bicubic,
    faster_bilinear,
    faster_nearest,
)
from tests.grid_sample.numpy.baselines import (
    baseline_cv2_cubic,
    baseline_cv2_linear,
    baseline_cv2_nearest,
    baseline_scipy_cubic,
    baseline_scipy_linear,
    baseline_scipy_nearest,
    cv2,
    map_coordinates,
)
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer, wrapped_partial

native_nearest = wrapped_partial(grid_sample, mode="nearest", backend="native")
native_bilinear = wrapped_partial(
    grid_sample, mode="bilinear", backend="native"
)
native_bicubic = wrapped_partial(grid_sample, mode="bicubic", backend="native")
pure_nearest = wrapped_partial(grid_sample, mode="nearest", backend="pure")
pure_bilinear = wrapped_partial(grid_sample, mode="bilinear", backend="pure")
pure_bicubic = wrapped_partial(grid_sample, mode="bicubic", backend="pure")

"""Test for torch `grid_sample`
"""


@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float64])
@pytest.mark.parametrize("dtype_grid", [np.float64])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float32, torch.float64])
def test_faster_vs_pure_cpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-6
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    img = torch.from_numpy(img).type(dtype_tensor)
    grid = torch.from_numpy(grid).type(dtype_tensor)

    print("\nNEAREST: faster vs pure")
    out_faster = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_faster = func_timer(faster_nearest)(img, grid, out_faster)
    out_pure = func_timer(pure_nearest)(img, grid, out_pure)

    close = check_close(out_faster, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_faster, out_pure)
    err_mae = mae(out_faster, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: faster vs pure")
    out_faster = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_faster = func_timer(faster_bilinear)(img, grid, out_faster)
    out_pure = func_timer(pure_bilinear)(img, grid, out_pure)

    close = check_close(out_faster, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_faster, out_pure)
    err_mae = mae(out_faster, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: faster vs pure")
    out_faster = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_faster = func_timer(faster_bicubic)(img, grid, out_faster)
    out_pure = func_timer(pure_bicubic)(img, grid, out_pure)

    close = check_close(out_faster, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_faster, out_pure)
    err_mae = mae(out_faster, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float32])
@pytest.mark.parametrize("dtype_grid", [np.float32])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float16, torch.float32])
def test_faster_vs_pure_gpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-6
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    device = torch.device("cuda")

    img = torch.from_numpy(img).type(dtype_tensor).to(device)
    grid = torch.from_numpy(grid).type(dtype_tensor).to(device)
    # grid = torch.from_numpy(grid).type(torch.float32)

    print("\nNEAREST: faster vs pure")
    out_faster = (
        torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    )
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    out_faster = func_timer(faster_nearest)(img, grid, out_faster)
    out_pure = func_timer(pure_nearest)(img, grid, out_pure)

    close = check_close(out_faster, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_faster, out_pure)
    err_mae = mae(out_faster, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: faster vs pure")
    out_faster = (
        torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    )
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    out_faster = func_timer(faster_bilinear)(img, grid, out_faster)
    out_pure = func_timer(pure_bilinear)(img, grid, out_pure)

    close = check_close(out_faster, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_faster, out_pure)
    err_mae = mae(out_faster, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: faster vs pure")
    out_faster = (
        torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    )
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    out_faster = func_timer(faster_bicubic)(img, grid, out_faster)
    out_pure = func_timer(pure_bicubic)(img, grid, out_pure)

    close = check_close(out_faster, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_faster, out_pure)
    err_mae = mae(out_faster, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol


@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float64])
@pytest.mark.parametrize("dtype_grid", [np.float64])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float32, torch.float64])
def test_native_vs_pure_cpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-6
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    img = torch.from_numpy(img).type(dtype_tensor)
    grid = torch.from_numpy(grid).type(dtype_tensor)

    print("\nNEAREST: native vs pure")
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_native = func_timer(native_nearest)(img.clone(), grid.clone())
    out_pure = func_timer(pure_nearest)(img, grid, out_pure)

    close = check_close(out_native, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_pure)
    err_mae = mae(out_native, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: faster vs pure")
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_native = func_timer(native_bilinear)(img.clone(), grid.clone())
    out_pure = func_timer(pure_bilinear)(img, grid, out_pure)

    close = check_close(out_native, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_pure)
    err_mae = mae(out_native, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: faster vs pure")
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor)
    out_native = func_timer(native_bicubic)(img.clone(), grid.clone())
    out_pure = func_timer(pure_bicubic)(img, grid, out_pure)

    close = check_close(out_native, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_pure)
    err_mae = mae(out_native, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float32])
@pytest.mark.parametrize("dtype_grid", [np.float32])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float16, torch.float32])
def test_native_vs_pure_gpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("dtype_tensor:", dtype_tensor)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-5
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    device = torch.device("cuda")

    img = torch.from_numpy(img).type(dtype_tensor).to(device)
    grid = torch.from_numpy(grid).type(dtype_tensor).to(device)

    print("\nNEAREST: native vs pure")
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    out_native = func_timer(native_nearest)(img.clone(), grid.clone())
    out_pure = func_timer(pure_nearest)(img, grid, out_pure)

    close = check_close(out_native, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_pure)
    err_mae = mae(out_native, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: faster vs pure")
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    out_native = func_timer(native_bilinear)(img.clone(), grid.clone())
    out_pure = func_timer(pure_bilinear)(img, grid, out_pure)

    close = check_close(out_native, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_pure)
    err_mae = mae(out_native, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: faster vs pure")
    out_pure = torch.from_numpy(make_copies(out)).type(dtype_tensor).to(device)
    out_native = func_timer(native_bicubic)(img.clone(), grid.clone())
    out_pure = func_timer(pure_bicubic)(img, grid, out_pure)

    close = check_close(out_native, out_pure, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_pure)
    err_mae = mae(out_native, out_pure)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol


@pytest.mark.skipif(cv2 is None, reason="cv2 is None; not installed")
@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float64])
@pytest.mark.parametrize("dtype_grid", [np.float64])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float32, torch.float64])
def test_native_vs_cv2_cpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("dtype_tensor:", dtype_tensor)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-6
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    img_th = torch.from_numpy(make_copies(img)).type(dtype_tensor)
    grid_th = torch.from_numpy(make_copies(grid)).type(dtype_tensor)

    print("\nNEAREST: native vs cv2")
    out_cv2 = make_copies(out)
    out_native = func_timer(native_nearest)(img_th.clone(), grid_th.clone())
    out_cv2 = func_timer(baseline_cv2_nearest)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2).type(dtype_tensor)

    close = check_close(out_native, out_cv2, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_cv2)
    err_mae = mae(out_native, out_cv2)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: native vs cv2")
    out_cv2 = make_copies(out)
    out_native = func_timer(native_bilinear)(img_th.clone(), grid_th.clone())
    out_cv2 = func_timer(baseline_cv2_linear)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2).type(dtype_tensor)

    close = check_close(out_native, out_cv2, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_cv2)
    err_mae = mae(out_native, out_cv2)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: native vs cv2")
    out_cv2 = make_copies(out)
    out_native = func_timer(native_bicubic)(img_th.clone(), grid_th.clone())
    out_cv2 = func_timer(baseline_cv2_cubic)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2).type(dtype_tensor)

    close = check_close(out_native, out_cv2, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_cv2)
    err_mae = mae(out_native, out_cv2)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.skipif(cv2 is None, reason="cv2 is None; not installed")
@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float32])
@pytest.mark.parametrize("dtype_grid", [np.float32])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float16, torch.float32])
def test_native_vs_cv2_gpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("dtype_tensor:", dtype_tensor)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-5
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    device = torch.device("cuda")

    img_th = torch.from_numpy(make_copies(img)).type(dtype_tensor).to(device)
    grid_th = torch.from_numpy(make_copies(grid)).type(dtype_tensor).to(device)

    print("\nNEAREST: native vs cv2")
    out_cv2 = make_copies(out)
    out_native = func_timer(native_nearest)(img_th.clone(), grid_th.clone())
    out_cv2 = func_timer(baseline_cv2_nearest)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2).type(dtype_tensor).to(device)

    close = check_close(out_native, out_cv2, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_cv2)
    err_mae = mae(out_native, out_cv2)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: native vs cv2")
    out_cv2 = make_copies(out)
    out_native = func_timer(native_bilinear)(img_th.clone(), grid_th.clone())
    out_cv2 = func_timer(baseline_cv2_linear)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2).type(dtype_tensor).to(device)

    close = check_close(out_native, out_cv2, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_cv2)
    err_mae = mae(out_native, out_cv2)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: native vs cv2")
    out_cv2 = make_copies(out)
    out_native = func_timer(native_bicubic)(img_th.clone(), grid_th.clone())
    out_cv2 = func_timer(baseline_cv2_cubic)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2).type(dtype_tensor).to(device)

    close = check_close(out_native, out_cv2, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_cv2)
    err_mae = mae(out_native, out_cv2)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol


@pytest.mark.skipif(
    map_coordinates is None,
    reason="scipy.map_coordinate is None; not installed",
)
@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float64])
@pytest.mark.parametrize("dtype_grid", [np.float64])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float32, torch.float64])
def test_native_vs_scipy_cpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("dtype_tensor:", dtype_tensor)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-6
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    img_th = torch.from_numpy(make_copies(img)).type(dtype_tensor)
    grid_th = torch.from_numpy(make_copies(grid)).type(dtype_tensor)

    print("\nNEAREST: native vs scipy")
    out_scipy = make_copies(out)
    out_native = func_timer(native_nearest)(img_th.clone(), grid_th.clone())
    out_scipy = func_timer(baseline_scipy_nearest)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy).type(dtype_tensor)

    close = check_close(out_native, out_scipy, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_scipy)
    err_mae = mae(out_native, out_scipy)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: native vs scipy")
    out_scipy = make_copies(out)
    out_native = func_timer(native_bilinear)(img_th.clone(), grid_th.clone())
    out_scipy = func_timer(baseline_scipy_linear)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy).type(dtype_tensor)

    close = check_close(out_native, out_scipy, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_scipy)
    err_mae = mae(out_native, out_scipy)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: native vs scipy")
    out_scipy = make_copies(out)
    out_native = func_timer(native_bicubic)(img_th.clone(), grid_th.clone())
    out_scipy = func_timer(baseline_scipy_cubic)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy).type(dtype_tensor)

    close = check_close(out_native, out_scipy, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_scipy)
    err_mae = mae(out_native, out_scipy)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.skipif(
    map_coordinates is None,
    reason="scipy.map_coordinate is None; not installed",
)
@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float32])
@pytest.mark.parametrize("dtype_grid", [np.float32])
@pytest.mark.parametrize("move_grid", [False])
@pytest.mark.parametrize("rand_img", [False])
@pytest.mark.parametrize("rand_grid", [False])  # set to True
@pytest.mark.parametrize("dtype_tensor", [torch.float16, torch.float32])
def test_native_vs_scipy_gpu(
    b: int,
    c: int,
    h: int,
    w: int,
    h_grid: int,
    w_grid: int,
    dtype_img: np.dtype,
    dtype_grid: np.dtype,
    move_grid: bool,
    rand_img: bool,
    rand_grid: bool,
    dtype_tensor: torch.dtype,
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("dtype_tensor:", dtype_tensor)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=move_grid,
        rand_img=rand_img,
        rand_grid=rand_grid,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    if dtype_tensor == torch.float32:
        # single
        rtol = 1e-3
        atol = 1e-5
    elif dtype_tensor == torch.float64:
        # double
        rtol = 1e-5
        atol = 1e-8
    else:
        # half
        # NOTE: only works for gpu
        rtol = 1e-2
        atol = 1e-4

    device = torch.device("cuda")

    img_th = torch.from_numpy(make_copies(img)).type(dtype_tensor).to(device)
    grid_th = torch.from_numpy(make_copies(grid)).type(dtype_tensor).to(device)

    print("\nNEAREST: native vs scipy")
    out_scipy = make_copies(out)
    out_native = func_timer(native_nearest)(img_th.clone(), grid_th.clone())
    out_scipy = func_timer(baseline_scipy_nearest)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy).type(dtype_tensor).to(device)

    close = check_close(out_native, out_scipy, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_scipy)
    err_mae = mae(out_native, out_scipy)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: native vs scipy")
    out_scipy = make_copies(out)
    out_native = func_timer(native_bilinear)(img_th.clone(), grid_th.clone())
    out_scipy = func_timer(baseline_scipy_linear)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy).type(dtype_tensor).to(device)

    close = check_close(out_native, out_scipy, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_scipy)
    err_mae = mae(out_native, out_scipy)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: native vs scipy")
    out_scipy = make_copies(out)
    out_native = func_timer(native_bicubic)(img_th.clone(), grid_th.clone())
    out_scipy = func_timer(baseline_scipy_cubic)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy).type(dtype_tensor).to(device)

    close = check_close(out_native, out_scipy, rtol=rtol, atol=atol)
    err_mse = mse(out_native, out_scipy)
    err_mae = mae(out_native, out_scipy)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol
