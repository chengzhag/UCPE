#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .numpy import convert2horizon as convert2horizon_numpy, run as run_numpy
from .torch import convert2horizon as convert2horizon_torch, run as run_torch

__all__ = ["Cube2Equi", "cube2equi"]

ArrayLike = Union[np.ndarray, torch.Tensor]
CubeMaps = Union[
    # single/batch 'horizon' or 'dice'
    np.ndarray,
    torch.Tensor,
    # single 'list'
    List[np.ndarray],
    List[torch.Tensor],
    # batch 'list'
    List[List[np.ndarray]],
    List[List[torch.Tensor]],
    # single 'dict'
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    # batch 'dict'
    List[Dict[str, np.ndarray]],
    List[Dict[str, np.ndarray]],
]


class Cube2Equi(object):
    """
    params:
    - w_out, h_out (int): equirectangular image size
    - cube_format (str): input cube format("dice", "horizon", "dict", "list")
    - mode (str): interpolation mode, defaults to "bilinear"

    inputs:
    - cubemap (np.ndarray, torch.Tensor, dict, list)

    returns:
    - equi (np.ndarray, torch.Tensor)
    """

    def __init__(
        self,
        height: int,
        width: int,
        cube_format: str,
        clip_output: bool = True,
        mode: str = "bilinear",
    ) -> None:
        self.height = height
        self.width = width
        self.cube_format = cube_format
        self.clip_output = clip_output
        self.mode = mode

    def __call__(self, cubemap: CubeMaps, **kwargs) -> ArrayLike:
        return cube2equi(
            cubemap=cubemap,
            cube_format=self.cube_format,
            width=self.width,
            height=self.height,
            clip_output=self.clip_output,
            mode=self.mode,
            **kwargs,
        )


def cube2equi(
    cubemap: CubeMaps,
    cube_format: str,
    height: int,
    width: int,
    clip_output: bool = True,
    mode: str = "bilinear",
    **kwargs,
) -> ArrayLike:
    """
    params:
    - cubemap
    - cube_format (str): ("dice", "horizon", "dict", "list")
    - height, width (int): output size
    - mode (str): "bilinear"

    return:
    - equi (np.ndarray, torch.Tensor)
    """

    assert width % 8 == 0 and height % 8 == 0

    # Try and detect which type it is ("numpy" or "torch")
    # FIXME: any cleaner way of detecting?
    _type = None
    if cube_format in ("dice", "horizon"):
        if isinstance(cubemap, np.ndarray):
            _type = "numpy"
        elif torch.is_tensor(cubemap):
            _type = "torch"
    elif cube_format == "dict":
        if isinstance(cubemap, dict):
            if isinstance(cubemap["F"], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap["F"], torch.Tensor):
                _type = "torch"
        elif isinstance(cubemap, list):
            assert isinstance(cubemap[0], dict)
            if isinstance(cubemap[0]["F"], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap[0]["F"], torch.Tensor):
                _type = "torch"
    elif cube_format == "list":
        assert isinstance(cubemap, list)
        if isinstance(cubemap[0], list):
            if isinstance(cubemap[0][0], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap[0][0], torch.Tensor):
                _type = "torch"
        else:
            if isinstance(cubemap[0], np.ndarray):
                _type = "numpy"
            elif isinstance(cubemap[0], torch.Tensor):
                _type = "torch"
    assert _type is not None, "ERR: input type is not numpy or torch"

    if _type == "numpy":
        horizon = convert2horizon_numpy(
            cubemap=cubemap, cube_format=cube_format
        )
        out = run_numpy(
            horizon=horizon,
            height=height,
            width=width,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    elif _type == "torch":
        horizon = convert2horizon_torch(
            cubemap=cubemap, cube_format=cube_format
        )
        out = run_torch(
            horizon=horizon,
            height=height,
            width=width,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    else:
        raise ValueError("Oops something went wrong here")

    if out.shape[0] == 1:
        out = out.squeeze(0)

    return out
