#!/usr/bin/env python3

"""Profilers for memory usage"""

import math
import sys
from typing import Any

import numpy as np
import torch


class MemSize:
    def __init__(self, size: int) -> None:
        ...


def convert_size_decimal(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0b"
    size_name = ("b", "kb", "mb", "gb", "tb", "pb", "eb", "zb", "yb")
    i = int(math.floor(math.log(size_bytes, 1000)))
    p = math.pow(1000, i)
    s = round(size_bytes / p, 3)
    return "{} {}".format(s, size_name[i])


def convert_size_binary(size_bytes: int) -> str:
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 3)
    return "{} {}".format(s, size_name[i])


def get_obj_size(obj: Any) -> str:
    size_in_bytes = sys.getsizeof(obj)
    size = convert_size_decimal(size_in_bytes)
    return size


def get_np_size(tensor: np.ndarray, pretty: bool = True) -> str:
    size_in_bytes = tensor.nbytes  # sys.getsizeof(tensor)
    if pretty:
        size = convert_size_decimal(size_bytes=size_in_bytes)
    else:
        size = str(size_in_bytes)
    return size


def get_torch_size(tensor: torch.Tensor, pretty: bool = True) -> str:
    size_in_bytes = tensor.element_size() * tensor.nelement()
    if pretty:
        size = convert_size_decimal(size_bytes=size_in_bytes)
    else:
        size = str(size_in_bytes)
    return size


if __name__ == "__main__":
    arr32 = np.zeros((512, 256, 3), dtype=np.dtype(np.float32))
    arr64 = np.zeros((512, 256, 3), dtype=np.dtype(np.float))

    print("arr32", get_np_size(arr32))
    print("arr64", get_np_size(arr64))

    tensor16 = torch.zeros((512, 256, 3), dtype=torch.float16)
    tensor32 = torch.zeros((512, 256, 3), dtype=torch.float32)
    tensor64 = torch.zeros((512, 256, 3), dtype=torch.float64)

    print("tensor16", get_torch_size(tensor16))
    print("tensor32", get_torch_size(tensor32))
    print("tensor64", get_torch_size(tensor64))
