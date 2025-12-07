#!/usr/bin/env python3

import os

import numpy as np

import pytest

import torch

from equilib.cube2equi.base import cube2equi

from tests.helpers.image_io import (
    load2numpy,
    load2torch,
    # save,
)
from tests.helpers.timer import func_timer

IMG_ROOT = "tests/data"
SAVE_ROOT = "tests/test_cube2equi/results"


def create_single_numpy_input(cube_format: str, dtype: np.dtype = np.float32):
    if cube_format in ("horizon", "dice"):
        img_path = os.path.join(IMG_ROOT, f"test_{cube_format}.jpg")
        img = load2numpy(img_path, dtype=dtype, is_cv2=False)
    elif cube_format in ("dict", "list"):
        img_paths = os.path.join(IMG_ROOT, "test_dict_{k}.jpg")
        img = {}
        for k in ("F", "R", "B", "L", "U", "D"):
            img_path = img_paths.format(k=k)
            img[k] = load2numpy(img_path, dtype=dtype, is_cv2=False)
        if cube_format == "list":
            img = list(img.values())
    else:
        raise ValueError
    return img


def create_batch_numpy_input(
    cube_format: str,
    batch_size: int,
    dtype: np.dtype = np.float32,
):
    imgs = []
    for i in range(batch_size):
        img = create_single_numpy_input(cube_format, dtype=dtype)
        imgs.append(img)

    if cube_format in ("horizon", "dice"):
        imgs = np.empty((len(imgs), *imgs[0].shape), dtype=imgs[0].dtype)
        for i, img in enumerate(imgs):
            imgs[i, ...] = img
    return imgs


def create_single_torch_input(
    cube_format: str, dtype: torch.dtype = torch.float32
):
    if cube_format in ("horizon", "dice"):
        img_path = os.path.join(IMG_ROOT, f"test_{cube_format}.jpg")
        img = load2torch(img_path, dtype=dtype, is_cv2=False)
    elif cube_format in ("dict", "list"):
        img_paths = os.path.join(IMG_ROOT, "test_dict_{k}.jpg")
        img = {}
        for k in ("F", "R", "B", "L", "U", "D"):
            img_path = img_paths.format(k=k)
            img[k] = load2torch(img_path, dtype=dtype, is_cv2=False)
        if cube_format == "list":
            img = list(img.values())
    else:
        raise ValueError
    return img


def create_batch_torch_input(
    cube_format: str,
    batch_size: int,
    dtype: torch.dtype = torch.float32,
):
    imgs = []
    for i in range(batch_size):
        img = create_single_torch_input(cube_format, dtype=dtype)
        imgs.append(img)

    if cube_format in ("horizon", "dice"):
        imgs = torch.empty((len(imgs), *imgs[0].shape), dtype=imgs[0].dtype)
        for i, img in enumerate(imgs):
            imgs[i, ...] = img
    return imgs


def numpy_single(
    height: int,
    width: int,
    cube_format: str,
    mode: str,
    dtype: np.dtype,
) -> None:
    # print parameters for debugging
    print()
    print("grid(height, width):", (height, width))
    print("cube_format:", cube_format)
    print("dtype:", dtype)
    print("mode:", mode)

    cubemap = create_single_numpy_input(cube_format, dtype=dtype)

    out = func_timer(cube2equi)(
        cubemap=cubemap,
        cube_format=cube_format,
        height=height,
        width=width,
        mode=mode,
    )

    assert out.shape == (3, height, width)
    assert out.dtype == dtype


def torch_single(
    height: int,
    width: int,
    cube_format: str,
    mode: str,
    dtype: torch.dtype,
) -> None:
    # print parameters for debugging
    print()
    print("grid(height, width):", (height, width))
    print("cube_format", cube_format)
    print("dtype:", dtype)
    print("mode:", mode)

    cubemap = create_single_torch_input(cube_format, dtype=dtype)

    out = func_timer(cube2equi)(
        cubemap=cubemap,
        cube_format=cube_format,
        height=height,
        width=width,
        mode=mode,
    )

    assert out.shape == (3, height, width)
    assert out.dtype == dtype


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("cube_format", ["dice", "dict", "list", "horizon"])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_single(
    height: int,
    width: int,
    cube_format: str,
    mode: str,
    dtype: np.dtype,
) -> None:
    numpy_single(
        height=height,
        width=width,
        cube_format=cube_format,
        mode=mode,
        dtype=dtype,
    )


@pytest.mark.parametrize("height", [32])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("cube_format", ["dice", "dict", "list", "horizon"])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torch_single(
    height: int,
    width: int,
    cube_format: str,
    mode: str,
    dtype: torch.dtype,
) -> None:
    torch_single(
        height=height,
        width=width,
        cube_format=cube_format,
        mode=mode,
        dtype=dtype,
    )
