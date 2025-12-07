#!/usr/bin/env python3

import numpy as np

import pytest

from equilib.grid_sample.numpy import grid_sample

from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.grid_sample.numpy import (
    faster_bicubic,
    faster_bilinear,
    faster_nearest,
    naive_bicubic,
    naive_bilinear,
    naive_nearest,
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

nearest = wrapped_partial(grid_sample, mode="nearest")
bilinear = wrapped_partial(grid_sample, mode="bilinear")
bicubic = wrapped_partial(grid_sample, mode="bicubic")

"""Test for numpy `grid_sample`

When testing against `cv2` or `scipy`, the results should match except when
`rand_grid` parameter is set to `True`

"""


@pytest.mark.parametrize("b", [2])
@pytest.mark.parametrize("c", [3])
@pytest.mark.parametrize("h", [128])
@pytest.mark.parametrize("w", [256])
@pytest.mark.parametrize("h_grid", [32])
@pytest.mark.parametrize("w_grid", [64])
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("dtype_grid", [np.float32, np.float64])
@pytest.mark.parametrize("move_grid", [True])
@pytest.mark.parametrize("rand_img", [True])
@pytest.mark.parametrize("rand_grid", [True])  # set to True
def test_against_bench(
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
) -> None:
    print("\n")
    print("dtype_img:", dtype_img, "dtype_grid:", dtype_grid)
    print("move:", move_grid)
    print("rand_img:", rand_img)
    print("rand_grid", rand_grid)

    fast_test = False

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

    if dtype_grid == np.dtype(np.float32) or dtype_img == np.dtype(np.float32):
        rtol = 1e-3
        atol = 1e-6
    else:
        rtol = 1e-5
        atol = 1e-8

    if fast_test:
        print("\nNEAREST: test vs sampler")
        out_faster = make_copies(out)
        out_sampler = make_copies(out)
        out_faster = func_timer(faster_nearest)(img, grid, out_faster)
        out_sampler = func_timer(nearest)(img, grid, out_sampler)

        close = check_close(out_faster, out_sampler, rtol=rtol, atol=atol)
        err_mse = mse(out_faster, out_sampler)
        err_mae = mae(out_faster, out_sampler)

        print("close?", close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        assert close, "ERR: two arrays were not close"
        assert err_mse < atol
        assert err_mae < atol

        print("\nBILINEAR: test vs sampler")
        out_faster = make_copies(out)
        out_sampler = make_copies(out)
        out_faster = func_timer(faster_bilinear)(img, grid, out_faster)
        out_sampler = func_timer(bilinear)(img, grid, out_sampler)

        close = check_close(out_faster, out_sampler, rtol=rtol, atol=atol)
        err_mse = mse(out_faster, out_sampler)
        err_mae = mae(out_faster, out_sampler)

        print("close?", close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        assert close, "ERR: two arrays were not close"
        assert err_mse < atol
        assert err_mae < atol

        print("\nBICUBIC: test vs sampler")
        out_faster = make_copies(out)
        out_sampler = make_copies(out)
        out_faster = func_timer(faster_bicubic)(img, grid, out_faster)
        out_sampler = func_timer(bicubic)(img, grid, out_sampler)

        close = check_close(out_faster, out_sampler, rtol=rtol, atol=atol)
        err_mse = mse(out_faster, out_sampler)
        err_mae = mae(out_faster, out_sampler)

        print("close?", close)
        print("MSE", err_mse)
        print("MAE", err_mae)
        assert close, "ERR: two arrays were not close"
        assert err_mse < atol
        assert err_mae < atol

    else:
        print("\nNEAREST: test vs sampler")
        out_naive = make_copies(out)
        out_sampler = make_copies(out)
        out_naive = func_timer(naive_nearest)(img, grid, out_naive)
        out_sampler = func_timer(nearest)(img, grid, out_sampler)

        close = check_close(out_naive, out_sampler, rtol=rtol, atol=atol)
        err_mse = mse(out_naive, out_sampler)
        err_mae = mae(out_naive, out_sampler)

        print("close?", close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        assert close, "ERR: two arrays were not close"
        assert err_mse < atol
        assert err_mae < atol

        print("\nBILINEAR: test vs sampler")
        out_naive = make_copies(out)
        out_sampler = make_copies(out)
        out_naive = func_timer(naive_bilinear)(img, grid, out_naive)
        out_sampler = func_timer(bilinear)(img, grid, out_sampler)

        close = check_close(out_naive, out_sampler, rtol=rtol, atol=atol)
        err_mse = mse(out_naive, out_sampler)
        err_mae = mae(out_naive, out_sampler)

        print("close?", close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        assert close, "ERR: two arrays were not close"
        assert err_mse < atol
        assert err_mae < atol

        print("\nBICUBIC: test vs sampler")
        out_naive = make_copies(out)
        out_sampler = make_copies(out)
        out_naive = func_timer(naive_bicubic)(img, grid, out_naive)
        out_sampler = func_timer(bicubic)(img, grid, out_sampler)

        close = check_close(out_naive, out_sampler, rtol=rtol, atol=atol)
        err_mse = mse(out_naive, out_sampler)
        err_mae = mae(out_naive, out_sampler)

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
@pytest.mark.parametrize("dtype_img", [np.uint8, np.float32, np.float64])
@pytest.mark.parametrize("dtype_grid", [np.float32, np.float64])
@pytest.mark.parametrize("move_grid", [True])
@pytest.mark.parametrize("rand_img", [True])
@pytest.mark.parametrize(
    "rand_grid", [False]
)  # results differ when set to True
def test_against_cv2(
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

    if dtype_grid == np.dtype(np.float32):
        rtol = 1e-3
        atol = 1e-6
    else:
        rtol = 1e-5
        atol = 1e-8

    print("\nNEAREST: cv2 vs sampler")
    out_cv2 = make_copies(out)
    out_sampler = make_copies(out)
    out_cv2 = func_timer(baseline_cv2_nearest)(img, grid, out_cv2)
    out_sampler = func_timer(nearest)(img, grid, out_sampler)

    close = check_close(out_cv2, out_sampler, rtol=rtol, atol=atol)
    err_mse = mse(out_cv2, out_sampler)
    err_mae = mae(out_cv2, out_sampler)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: cv2 vs sampler")
    out_cv2 = make_copies(out)
    out_sampler = make_copies(out)
    out_cv2 = func_timer(baseline_cv2_linear)(img, grid, out_cv2)
    out_sampler = func_timer(bilinear)(img, grid, out_sampler)

    close = check_close(out_cv2, out_sampler, rtol=rtol, atol=atol)
    err_mse = mse(out_cv2, out_sampler)
    err_mae = mae(out_cv2, out_sampler)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: cv2 vs sampler")
    out_cv2 = make_copies(out)
    out_sampler = make_copies(out)
    out_cv2 = func_timer(baseline_cv2_cubic)(img, grid, out_cv2)
    out_sampler = func_timer(bicubic)(img, grid, out_sampler)

    close = check_close(out_cv2, out_sampler, rtol=rtol, atol=atol)
    err_mse = mse(out_cv2, out_sampler)
    err_mae = mae(out_cv2, out_sampler)

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
@pytest.mark.parametrize("move_grid", [True])
@pytest.mark.parametrize("rand_img", [True])
@pytest.mark.parametrize("rand_grid", [False])
def test_against_scipy(
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

    if dtype_grid == np.dtype(np.float32):
        rtol = 1e-3
        atol = 1e-6
    else:
        rtol = 1e-5
        atol = 1e-8

    print("\nNEAREST: scipy vs sampler")
    out_scipy = make_copies(out)
    out_sampler = make_copies(out)
    out_scipy = func_timer(baseline_scipy_nearest)(img, grid, out_scipy)
    out_sampler = func_timer(nearest)(img, grid, out_sampler)

    close = check_close(out_scipy, out_sampler, rtol=rtol, atol=atol)
    err_mse = mse(out_scipy, out_sampler)
    err_mae = mae(out_scipy, out_sampler)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBILINEAR: scipy vs sampler")
    out_scipy = make_copies(out)
    out_sampler = make_copies(out)
    out_scipy = func_timer(baseline_scipy_linear)(img, grid, out_scipy)
    out_sampler = func_timer(bilinear)(img, grid, out_sampler)

    close = check_close(out_scipy, out_sampler, rtol=rtol, atol=atol)
    err_mse = mse(out_scipy, out_sampler)
    err_mae = mae(out_scipy, out_sampler)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol

    print("\nBICUBIC: scipy vs sampler")
    out_scipy = make_copies(out)
    out_sampler = make_copies(out)
    out_scipy = func_timer(baseline_scipy_cubic)(img, grid, out_scipy)
    out_sampler = func_timer(bicubic)(img, grid, out_sampler)

    close = check_close(out_scipy, out_sampler, rtol=rtol, atol=atol)
    err_mse = mse(out_scipy, out_sampler)
    err_mae = mae(out_scipy, out_sampler)

    print("close?", close)
    print("MSE", err_mse)
    print("MAE", err_mae)

    assert close, "ERR: two arrays were not close"
    assert err_mse < atol
    assert err_mae < atol
