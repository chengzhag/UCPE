#!/usr/bin/env python3

import numpy as np

import torch

from equilib.torch_utils import get_device

from tests.grid_sample.numpy.nearest import (
    faster_nearest as faster_nearest_numpy,
)
from tests.grid_sample.torch.native import native_nearest
from tests.grid_sample.helpers import create_batch_data, make_copies
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer


def old_naive_nearest(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    b_in, c_in, h_in, w_in = img.shape
    b_out, _, h_out, w_out = grid.shape
    dtype = img.dtype
    device = get_device(img)

    out = torch.empty((b_in, c_in, h_out, w_out), dtype=dtype, device=device)

    for b in range(b_out):
        for y_out in range(h_out):
            for x_out in range(w_out):
                _y, _x = torch.round(grid[b, :, y_out, x_out]).type(torch.int64)
                _y %= h_in
                _x %= w_in
                # can index with cuda variable
                out[b, :, y_out, x_out] = img[b, :, _y, _x]

    return out


def naive_nearest(
    img: torch.Tensor, grid: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    b_in, c_in, h_in, w_in = img.shape
    b_out, _, h_out, w_out = out.shape

    for b in range(b_out):
        for y_out in range(h_out):
            for x_out in range(w_out):
                _y, _x = torch.round(grid[b, :, y_out, x_out]).type(torch.int64)
                _y %= h_in
                _x %= w_in
                # can index with cuda variable
                out[b, :, y_out, x_out] = img[b, :, _y, _x]

    return out


def old_faster_nearest(img: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """

    NOTE:
    - grid shouldn't be in device (might be slow and will use a lot of VRAM)?
    - need to test this out
    """

    b_in, c_in, h_in, w_in = img.shape
    b_out, _, h_out, w_out = grid.shape
    dtype = grid.dtype
    device = get_device(img)

    if grid.device.type == "cuda":
        import warnings

        warnings.warn(
            "input `grid` should be on the cpu, but got a cuda tensor"
        )

    out = torch.zeros((b_in, c_in, h_out, w_out), dtype=dtype, device=device)

    round_grid = torch.round(grid).type(torch.int64)
    round_grid[:, 0, ...] %= h_in
    round_grid[:, 1, ...] %= w_in

    for b in range(b_out):
        y = round_grid[b, 0, :, :]
        x = round_grid[b, 1, :, :]
        out[b, ...] = img[b][:, y, x]

    return out


def faster_nearest(
    img: torch.Tensor, grid: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    """ """

    b_in, c_in, h_in, w_in = img.shape

    round_grid = torch.round(grid).type(torch.int64)
    round_grid[:, 0, ...] %= h_in
    round_grid[:, 1, ...] %= w_in

    for b in range(b_in):
        y = round_grid[b, 0, :, :]
        x = round_grid[b, 1, :, :]
        out[b, ...] = img[b][:, y, x]

    return out


def check_gpu():
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

    # NOTE: grid can be on cpu!
    device = torch.device("cuda")

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)
    out = torch.from_numpy(out)
    print(img.device == grid.device)

    img = img.to(device)
    # grid = grid.to(device)
    out = out.to(device)

    print(img.device == grid.device)

    out_old = func_timer(old_faster_nearest)(img, grid)

    out_new = func_timer(faster_nearest)(img, grid, out)

    print(check_close(out_old, out_new))


def compare_baseline():
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
    out_numpy = make_copies(out)
    out_native = make_copies(out)
    out_torch = torch.from_numpy(make_copies(out))

    out_numpy = func_timer(faster_nearest_numpy)(img, grid, out_numpy)
    out_numpy = torch.from_numpy(out_numpy)

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)

    out_torch = func_timer(faster_nearest)(img, grid, out_torch)

    print("\nChecking: numpy vs torch")
    print("close?", check_close(out_numpy, out_torch))
    print("MSE", mse(out_numpy, out_torch))
    print("MAE", mae(out_numpy, out_torch))

    out_native = func_timer(native_nearest)(img, grid)

    print("\nChecking: numpy vs native")
    print("close?", check_close(out_numpy, out_native))
    print("MSE", mse(out_numpy, out_native))
    print("MAE", mae(out_numpy, out_native))


def compare_native():
    # NOTE: might not be useful since torch implementation follows
    # numpy implementation. Just curious about the speed

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
    out_naive = torch.from_numpy(make_copies(out))
    out_faster = torch.from_numpy(make_copies(out))

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)

    # NOTE: native method overwrites `img` and `coords` when run
    # so you have to run it last
    # use `.clone()` when you want to reuse the same variables
    out_native = func_timer(native_nearest)(img.clone(), grid.clone())
    out_naive = func_timer(naive_nearest)(img, grid, out_naive)
    out_faster = func_timer(faster_nearest)(img, grid, out_faster)

    print("\nChecking: naive vs native")
    print("close?", check_close(out_native, out_naive))
    print("MSE", mse(out_native, out_naive))
    print("MAE", mae(out_native, out_naive))

    print("\nChecking: naive vs faster")
    print("close?", check_close(out_faster, out_naive))
    print("MSE", mse(out_faster, out_naive))
    print("MAE", mae(out_faster, out_naive))

    print("\nChecking: native vs faster")
    print("close?", check_close(out_faster, out_native))
    print("MSE", mse(out_faster, out_native))
    print("MAE", mae(out_faster, out_native))


def compare_old():
    # NOTE: might not be useful since torch implementation follows
    # numpy implementation. Just curious about the speed

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
    out_new = torch.from_numpy(make_copies(out))

    img = torch.from_numpy(img)
    grid = torch.from_numpy(grid)

    out_old = func_timer(old_faster_nearest)(img, grid)
    out_new = func_timer(faster_nearest)(img, grid, out_new)

    print("\nChecking: old vs new")
    print("close?", check_close(out_old, out_new))
    print("MSE", mse(out_old, out_new))
    print("MAE", mae(out_old, out_new))


if __name__ == "__main__":
    check_gpu()
    compare_old()
    # compare_baseline()
    # compare_native()
