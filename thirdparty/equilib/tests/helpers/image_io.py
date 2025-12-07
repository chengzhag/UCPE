#!/usr/bin/env python3

"""IO for images (numpy and torch)"""

import os
from typing import Union
import warnings

try:
    import cv2
except ImportError:
    print("cv2 is not installed")
    cv2 = None

try:
    from PIL import Image
except ImportError:
    print("PIL is not installed")
    Image = None

import numpy as np

import torch

from torchvision import transforms

to_tensor = transforms.Compose([transforms.ToTensor()])

to_PIL = transforms.Compose([transforms.ToPILImage()])


def _open_as_PIL(img_path: str) -> Image.Image:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    img = Image.open(img_path)
    assert img is not None
    if img.getbands() == tuple("RGBA"):
        # NOTE: Sometimes images are RGBA
        img = img.convert("RGB")
    return img


def _open_as_cv2(img_path: str) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    # FIXME: shouldn't use `imread` since it won't auto detect color space
    warnings.warn("Cannot handle color spaces other than RGB")
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
    assert img is not None
    return img


def load2numpy(
    img_path: str, dtype: np.dtype, is_cv2: bool = False
) -> np.ndarray:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)
        img = np.asarray(img)

    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img = np.transpose(img, (2, 0, 1))

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0
    dist_dtype = np.dtype(dtype)
    if dist_dtype in (np.float32, np.float64):
        img = img / 255.0
    img = img.astype(dist_dtype)

    return img


def load2torch(
    img_path: str, dtype: torch.dtype, is_cv2: bool = False
) -> torch.Tensor:
    assert os.path.exists(img_path), f"{img_path} doesn't exist"
    if is_cv2:
        # FIXME: currently only supports RGB
        img = _open_as_cv2(img_path)
    else:
        img = _open_as_PIL(img_path)

    # NOTE: Convert dtypes
    # if uint8, keep 0-255
    # if float, convert to 0.0-1.0 (ToTensor)
    if dtype in (torch.float16, torch.float32, torch.float64):
        img = to_tensor(img)
        # FIXME: force typing since I have no idea how to change types in
        # PIL; also it's easier to change type using `type`; might be slower
        img = img.type(dtype)
        # NOTE: automatically adds channel for grayscale
    elif dtype == torch.uint8:
        img = torch.from_numpy(np.array(img, dtype=np.uint8, copy=True))
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        else:
            img = img.permute((2, 0, 1)).contiguous()
        assert img.dtype == torch.uint8

    return img


def _numpy2PIL(img: np.ndarray) -> Image.Image:
    """Supports RGB and grayscale"""

    # FIXME: need to test fro depth image
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes

    if len(img.shape) == 3:
        if img.shape[0] == 3:
            # RGB
            # move the channel to last dim
            img = np.transpose(img, (1, 2, 0))
        elif img.shape[0] == 1:
            # grayscale
            # FIXME: is this necessary?
            img = img.squeeze(0)

    # convert float to uint8
    if img.dtype in (np.float32, np.float64):
        img *= 255
        img = img.astype(np.uint8)

    assert img.dtype == np.uint8
    return Image.fromarray(img)  # if depth, we need to change this to 'F'


def _torch2PIL(img: torch.Tensor) -> Image.Image:
    if img.device == "cuda":
        # move to cpu
        img = img.to("cpu")

    if img.dtype == torch.uint8:
        # if the img is uint8, it is already 0-255
        # FIXME: hacky
        img = img.numpy()
        assert img.dtype == np.uint8
        img = Image.fromarray(img)
    else:
        img = to_PIL(img)
    return img


def save(img: Union[np.ndarray, torch.Tensor], path: str) -> None:
    assert len(img.shape) == 3, f"{img.shape} is not a valid input format"
    if isinstance(img, np.ndarray):
        img = _numpy2PIL(img)
        img.save(path)
    elif torch.is_tensor(img):
        img = _torch2PIL(img)
        img.save(path)
    else:
        raise ValueError()


def _test_numpy():
    rgb_jpg_path = "data/earthmap4k.jpg"  # "data/equi.jpg"
    rgb_png_path = "data/equi2.png"
    gray_jpg_path = "data/earthspec4k.jpg"

    dtype = np.float32

    # load rgb jpg
    rgb_jpg = load2numpy(img_path=rgb_jpg_path, dtype=dtype, is_cv2=False)

    assert len(rgb_jpg.shape) == 3
    assert rgb_jpg.shape[0] == 3
    assert rgb_jpg.dtype == dtype

    # load rgb png (rgba)
    rgb_png = load2numpy(img_path=rgb_png_path, dtype=dtype, is_cv2=False)

    assert len(rgb_png.shape) == 3
    assert rgb_png.shape[0] == 3
    assert rgb_png.dtype == dtype

    # load grayscale jpg
    gray_jpg = load2numpy(img_path=gray_jpg_path, dtype=dtype, is_cv2=False)

    assert len(gray_jpg.shape) == 3
    assert gray_jpg.shape[0] == 1
    assert gray_jpg.dtype == dtype

    print("Passed")


def _test_torch():
    rgb_jpg_path = "data/earthmap4k.jpg"  # "data/equi.jpg"
    rgb_png_path = "data/equi2.png"
    gray_jpg_path = "data/earthspec4k.jpg"

    dtype = torch.float32

    # load rgb jpg
    rgb_jpg = load2torch(img_path=rgb_jpg_path, dtype=dtype, is_cv2=False)

    assert len(rgb_jpg.shape) == 3
    assert rgb_jpg.shape[0] == 3
    assert rgb_jpg.dtype == dtype

    # load rgb png (rgba)
    rgb_png = load2torch(img_path=rgb_png_path, dtype=dtype, is_cv2=False)

    assert len(rgb_png.shape) == 3
    assert rgb_png.shape[0] == 3
    assert rgb_png.dtype == dtype

    # load grayscale jpg
    gray_jpg = load2torch(img_path=gray_jpg_path, dtype=dtype, is_cv2=False)

    assert len(gray_jpg.shape) == 3
    assert gray_jpg.shape[0] == 1
    assert gray_jpg.dtype == dtype

    print("Passed")


if __name__ == "__main__":
    # running tests for different types of images
    _test_numpy()
    _test_torch()
