#!/usr/bin/env python3

import numpy as np

import torch

from equilib.grid_sample.torch.native import native

from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.grid_sample.numpy.baselines import (
    baseline_cv2_cubic,
    baseline_cv2_linear,
    baseline_cv2_nearest,
    baseline_scipy_cubic,
    baseline_scipy_linear,
    baseline_scipy_nearest,
)
from tests.grid_sample.numpy.nearest import naive_nearest
from tests.grid_sample.numpy.bilinear import naive_bilinear
from tests.grid_sample.numpy.bicubic import naive_bicubic
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer, wrapped_partial

# wrapped with custom wrapper so the `__name__` doesn't get lost
native_nearest = wrapped_partial(native, mode="nearest")
native_bilinear = wrapped_partial(native, mode="bilinear")
native_bicubic = wrapped_partial(native, mode="bicubic")

"""
Benchmark the performance of pytorch's native grid_sample
against some baseline methods (scipy and cv2)

Initial outlook seems that bilinear and bicubic has major errors.
However, this is for float32 images and if the output image is uint8 images,
the precision wouldn't matter too much (I hope)
One major upside (or downside in some cases) is that torch implementation of
grid sampling on CPU is much faster than scipy or cv2. This means that
grid sampling for scipy and cv2 is more robust, but slower. torch implementation
of grid sampling loses precision at the cost of speed.

Comparing it to naive methods, I believe that the errors are accumulated through
the boundaries of the image. In the torch implementation of grid sample, there
are no wrapping mode for padding. The other methods (scipy, cv2, naive numpy)
implements wrapping mode for padding that wraps so that when the sampler wants to sample
a pixel outside the right boarder, it will use the values on the other side which is the
left. The default padding method for torch was `zero`, but when I changed the padding mode
to `reflection` the error reduce drastically.

TODO: do some tests for uint8
"""


def compare_nearest():
    dtype_img = dtype_grid = np.dtype(np.float32)
    b = 2
    c = 3
    h = 128
    w = 256
    h_grid = 32
    w_grid = 64

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

    # initialize outputs:
    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)
    out_naive = make_copies(out)
    out_native = make_copies(out)

    out_scipy = func_timer(baseline_scipy_nearest)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy)
    # print("scipy:", out_scipy.shape, out_scipy.dtype)
    # print(out_scipy)

    out_cv2 = func_timer(baseline_cv2_nearest)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2)
    # print("cv2:", out_cv2.shape, out_cv2.dtype)
    # print(out_cv2)

    out_naive = func_timer(naive_nearest)(img, grid, out_naive)
    out_naive = torch.from_numpy(out_naive)

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)
    # print("img:", img.shape, img.dtype)
    # print("coords:", coords.shape, coords.dtype)

    out_native = func_timer(native_nearest)(img, grid)
    # print("native:", out.shape, out.dtype)

    print("\nChecking: scipy vs naive")
    print("close?", check_close(out_scipy, out_naive))
    print("MSE", mse(out_scipy, out_naive))
    print("MAE", mae(out_scipy, out_naive))

    print("\nChecking: scipy vs cv2")
    print("close?", check_close(out_scipy, out_cv2))
    print("MSE", mse(out_scipy, out_cv2))
    print("MAE", mae(out_scipy, out_cv2))

    print("\nChecking: native vs scipy")
    print("close?", check_close(out_native, out_scipy))
    print("MSE", mse(out_native, out_scipy))
    print("MAE", mae(out_native, out_scipy))

    print("\nChecking: native vs cv2")
    print("close?", check_close(out_native, out_cv2))
    print("MSE", mse(out_native, out_cv2))
    print("MAE", mae(out_native, out_cv2))

    print("\nChecking: native vs naive")
    print("close?", check_close(out_native, out_naive))
    print("MSE", mse(out_native, out_naive))
    print("MAE", mae(out_native, out_naive))


def compare_bilinear():
    dtype_img = dtype_grid = np.dtype(np.float32)
    b = 2
    c = 3
    h = 128
    w = 256
    h_grid = 32
    w_grid = 64

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

    # initialize outputs:
    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)
    out_naive = make_copies(out)

    out_scipy = func_timer(baseline_scipy_linear)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy)
    # print("scipy:", out_scipy.shape, out_scipy.dtype)
    # print(out_scipy)

    out_cv2 = func_timer(baseline_cv2_linear)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2)
    # print("cv2:", out_cv2.shape, out_cv2.dtype)
    # print(out_cv2)

    out_naive = func_timer(naive_bilinear)(img, grid, out_naive)
    out_naive = torch.from_numpy(out_naive)

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)
    # print("img:", img.shape, img.dtype)
    # print("coords:", coords.shape, coords.dtype)

    out_native = func_timer(native_bilinear)(img, grid)
    # print("native:", out.shape, out.dtype)

    print("\nChecking: scipy vs naive")
    print("close?", check_close(out_scipy, out_naive))
    print("MSE", mse(out_scipy, out_naive))
    print("MAE", mae(out_scipy, out_naive))

    print("\nChecking: scipy vs cv2")
    print("close?", check_close(out_scipy, out_cv2))
    print("MSE", mse(out_scipy, out_cv2))
    print("MAE", mae(out_scipy, out_cv2))

    print("\nChecking: native vs scipy")
    print("close?", check_close(out_native, out_scipy))
    print("MSE", mse(out_native, out_scipy))
    print("MAE", mae(out_native, out_scipy))

    print("\nChecking: native vs cv2")
    print("close?", check_close(out_native, out_cv2))
    print("MSE", mse(out_native, out_cv2))
    print("MAE", mae(out_native, out_cv2))

    print("\nChecking: native vs naive")
    print("close?", check_close(out_native, out_naive))
    print("MSE", mse(out_native, out_naive))
    print("MAE", mae(out_native, out_naive))


def compare_bicubic():
    dtype_img = dtype_grid = np.dtype(np.float64)
    b = 2
    c = 3
    h = 128
    w = 256
    h_grid = 32
    w_grid = 64

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

    # initialize outputs:
    out_scipy = make_copies(out)
    out_cv2 = make_copies(out)
    out_naive = make_copies(out)
    out_native = make_copies(out)

    out_scipy = func_timer(baseline_scipy_cubic)(img, grid, out_scipy)
    out_scipy = torch.from_numpy(out_scipy)
    # print("scipy:", out_scipy.shape, out_scipy.dtype)
    # print(out_scipy)

    out_cv2 = func_timer(baseline_cv2_cubic)(img, grid, out_cv2)
    out_cv2 = torch.from_numpy(out_cv2)
    # print("cv2:", out_cv2.shape, out_cv2.dtype)
    # print(out_cv2)

    out_naive = func_timer(naive_bicubic)(img, grid, out_naive)
    out_naive = torch.from_numpy(out_naive)

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)
    # print("img:", img.shape, img.dtype)
    # print("coords:", coords.shape, coords.dtype)

    out_native = func_timer(native_bicubic)(img, grid)
    # print("native:", out.shape, out.dtype)

    print("\nChecking: scipy vs naive")
    print("close?", check_close(out_scipy, out_naive))
    print("MSE", mse(out_scipy, out_naive))
    print("MAE", mae(out_scipy, out_naive))

    print("\nChecking: scipy vs cv2")
    print("close?", check_close(out_scipy, out_cv2))
    print("MSE", mse(out_scipy, out_cv2))
    print("MAE", mae(out_scipy, out_cv2))

    print("\nChecking: native vs scipy")
    print("close?", check_close(out_native, out_scipy))
    print("MSE", mse(out_native, out_scipy))
    print("MAE", mae(out_native, out_scipy))

    print("\nChecking: native vs cv2")
    print("close?", check_close(out_native, out_cv2))
    print("MSE", mse(out_native, out_cv2))
    print("MAE", mae(out_native, out_cv2))

    print("\nChecking: native vs naive")
    print("close?", check_close(out_native, out_naive))
    print("MSE", mse(out_native, out_naive))
    print("MAE", mae(out_native, out_naive))


def check_gpu():
    device = torch.device("cuda")

    dtype_img = np.dtype(np.float32)
    dtype_grid = np.dtype(np.float32)
    b = 2
    c = 3
    h = 128
    w = 256
    h_grid = 32
    w_grid = 64

    img, grid, _ = create_batch_data(
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

    img = torch.from_numpy(img)
    coords = torch.from_numpy(grid)

    # ~move img to device but not~
    # NOTE: THEY BOTH NEED TO BE ON THE SAME DEVICE
    img = img.to(device)
    coords = coords.to(device)

    _ = native_nearest(img, coords)


if __name__ == "__main__":
    np.random.seed(0)

    print(">>> TEST NEAREST")
    compare_nearest()
    print()
    print(">>> TEST BILINEAR")
    compare_bilinear()
    print()
    print(">>> TEST BICUBIC")
    compare_bicubic()

    # check_gpu()
