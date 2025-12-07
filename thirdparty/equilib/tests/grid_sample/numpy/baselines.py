#!/usr/bin/env python3

"""Baselines and references

Baseline method to compare precision against

Baseline methods can be slow, but precise so the output can be treated as GT

Maybe use other libraries like `scipy` or `cv2` for reference.

`scipy`:
    - `scipy.ndimage.map_coordinates`
    - https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    - limited to single channel
    - "spline filters" - change `order` for interpolation
    - very robust but at the cost of speed

`cv2`:
    - `cv2.remap`
    - https://docs.opencv.org/3.4/da/d54/group__imgproc__transform.html
    - only runs on float32 grids
    - bilinear interpolation is not accurate
    - Fastest (around x4 ~ x7, even up to x10 faster than scipy.map_coordinates)
    - at the cost of loss in precison
"""

import numpy as np

try:
    import cv2
except ImportError:
    print("cv2 is not installed")
    cv2 = None

try:
    from scipy.ndimage import map_coordinates
except ImportError:
    print("scipy is not installed")
    map_coordinates = None

from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer, time_func_loop, wrapped_partial


def baseline_scipy(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray, order: int = 1
) -> np.ndarray:
    """Slower than cv2, but *accurate*"""

    assert 0 <= order <= 5
    """n-th order spline
    order = 0  # nearest
    order = 1  # linear
    order = 2  # quadratic
    order = 3  # cubic
    order = 4
    order = 5
    """

    b_in, c_in, _, _ = img.shape

    for j in range(b_in):
        # has to loop for each channels
        for i in range(c_in):
            out[j, i, ...] = map_coordinates(
                img[j, i, ...],
                grid[j, ...],
                order=order,
                mode="grid-wrap",  # 'wrap'
            )

    return out


def old_scipy(img: np.ndarray, grid: np.ndarray, order: int = 1) -> np.ndarray:
    """Old version of grid_sample code

    `out` is created in the function, so there might be allocation
    costs inside the function which is bad (I think, not sure)
    """

    assert 0 <= order <= 5
    """n-th order spline
    order = 0  # nearest
    order = 1  # linear
    order = 2  # quadratic
    order = 3  # cubic
    order = 4
    order = 5
    """

    b_in, c_in, _, _ = img.shape
    b_out, _, h_out, w_out = grid.shape
    out = np.empty((b_out, c_in, h_out, w_out), dtype=img.dtype)
    for j in range(b_in):
        # has to loop for each channels
        for i in range(c_in):
            out[j, i, ...] = map_coordinates(
                img[j, i, ...],
                grid[j, ...],
                order=order,
                mode="grid-wrap",  # 'wrap'
            )

    return out


baseline_scipy_nearest = wrapped_partial(baseline_scipy, order=0)
baseline_scipy_linear = wrapped_partial(baseline_scipy, order=1)
baseline_scipy_quadratic = wrapped_partial(baseline_scipy, order=2)
baseline_scipy_cubic = wrapped_partial(baseline_scipy, order=3)


def grid_sample_scipy(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray, mode: str = "bilinear"
) -> np.ndarray:
    if mode == "nearest":
        out = baseline_scipy_nearest(img, grid, out)
    elif mode == "bilinear":
        out = baseline_scipy_linear(img, grid, out)
    elif mode == "bicubic":
        out = baseline_scipy_cubic(img, grid, out)
    else:
        raise ValueError(f"ERR: {mode} is not supported")

    return out


def baseline_cv2(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray, mode: str = "linear"
) -> np.ndarray:
    """Fastest at the cost of low precision"""

    interp_methods = {
        "linear": cv2.INTER_LINEAR,
        "nearest": cv2.INTER_NEAREST,
        "cubic": cv2.INTER_CUBIC,
        "lanczos4": cv2.INTER_LANCZOS4,
    }
    interp = interp_methods.get(mode, None)
    assert interp is not None, f"ERR: {mode} interpolation not supported"

    if grid.dtype != np.dtype(np.float32):
        # NOTE: input type must be float32
        grid = grid.astype(np.float32)

    b, c, _, _ = img.shape

    img = img.transpose(0, 2, 3, 1)
    if c == 1:
        # single channel
        img = img.squeeze(-1)
        for i in range(b):
            out[i, ...] = cv2.remap(
                img[i, ...],
                grid[i, 1, ...],
                grid[i, 0, ...],
                interpolation=interp,
                borderMode=cv2.BORDER_WRAP,
            )[..., np.newaxis].transpose((2, 0, 1))
    else:
        # 3 channels
        for i in range(b):
            out[i, ...] = cv2.remap(
                img[i, ...],
                grid[i, 1, ...],
                grid[i, 0, ...],
                interpolation=interp,
                borderMode=cv2.BORDER_WRAP,
            ).transpose((2, 0, 1))

    return out


baseline_cv2_nearest = wrapped_partial(baseline_cv2, mode="nearest")
baseline_cv2_linear = wrapped_partial(baseline_cv2, mode="linear")
baseline_cv2_cubic = wrapped_partial(baseline_cv2, mode="cubic")


def grid_sample_cv2(
    img: np.ndarray, grid: np.ndarray, out: np.ndarray, mode: str = "bilinear"
) -> np.ndarray:
    if mode == "nearest":
        out = baseline_cv2_nearest(img, grid, out)
    elif mode == "bilinear":
        out = baseline_cv2_linear(img, grid, out)
    elif mode == "bicubic":
        out = baseline_cv2_cubic(img, grid, out)
    else:
        raise ValueError(f"ERR: {mode} is not supported")

    return out


def bench_pbr():
    dtype_img = np.dtype(np.float64)
    dtype_grid = np.dtype(np.float32)
    b = 16
    c = 3
    h = 2000
    w = 4000
    h_grid = 200
    w_grid = 400

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=False,
        rand_img=False,
        rand_grid=False,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    time_func_loop(old_scipy, {"img": img, "grid": grid, "order": 0}, num=100)
    time_func_loop(
        baseline_scipy_nearest, {"img": img, "grid": grid, "out": out}, num=100
    )
    # initializing `out` is a tiny bit faster (sometimes?)


def bench_batches():
    dtype_img = np.dtype(np.float64)
    dtype_grid = np.dtype(np.float32)
    b = 4
    c = 3
    h = 256
    w = 512
    h_grid = 64
    w_grid = 128

    img, grid, out = create_batch_data(
        b=b,
        c=c,
        h=h,
        w=w,
        h_grid=h_grid,
        w_grid=w_grid,
        move_grid=False,
        rand_img=False,
        rand_grid=False,
        dtype_img=dtype_img,
        dtype_grid=dtype_grid,
    )

    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)

    # Some patterns I noticed:
    # - cv2 == scipy when coordinates are either integers or in increments of 1/32
    #   - which also means that it operates in 32 bit floats
    #   - 1/64 increments would result in errors to be pretty large

    print("TEST NEAREST")
    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)
    out_scipy = func_timer(baseline_scipy_nearest)(img, grid, out_scipy)
    out_cv2 = func_timer(baseline_cv2_nearest)(img, grid, out_cv2)

    print("close?", check_close(out_scipy, out_cv2))
    print("MSE", mse(out_scipy, out_cv2))
    print("MAE", mae(out_scipy, out_cv2))

    print()
    print("TEST LINEAR")
    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)
    out_scipy = func_timer(baseline_scipy_linear)(img, grid, out_scipy)
    out_cv2 = func_timer(baseline_cv2_linear)(img, grid, out_cv2)

    print("close?", check_close(out_scipy, out_cv2))
    print("MSE", mse(out_scipy, out_cv2))
    print("MAE", mae(out_scipy, out_cv2))

    print()
    print("TEST CUBIC")
    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)
    out_scipy = func_timer(baseline_scipy_cubic)(img, grid, out_scipy)
    out_cv2 = func_timer(baseline_cv2_cubic)(img, grid, out_cv2)

    print("close?", check_close(out_scipy, out_cv2))
    print("MSE", mse(out_scipy, out_cv2))
    print("MAE", mae(out_scipy, out_cv2))


if __name__ == "__main__":
    np.random.seed(0)

    bench_batches()
    # bench_pbr()
