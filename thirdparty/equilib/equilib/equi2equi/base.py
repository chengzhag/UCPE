#!/usr/bin/env python3

from typing import Dict, List, Optional, Union

import numpy as np

import torch

from .numpy import run as run_numpy
from .torch import run as run_torch

__all__ = ["Equi2Equi", "equi2equi"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Rot = Union[Dict[str, float], List[Dict[str, float]]]


class Equi2Equi(object):
    """
    params:
    - w_out, h_out (optional int): equi image size
    - clip_output (bool) whether to clip values in the range of input
    - mode (str): interpolation mode, defaults to "bilinear"
    - z_down (bool)

    input params:
    - src (np.ndarray, torch.Tensor)
    - rots (dict, list[dict])

    return:
    - equi (np.ndarray, torch.Tensor)
    """

    def __init__(
        self,
        height: Optional[int] = None,
        width: Optional[int] = None,
        clip_output: bool = True,
        mode: str = "bilinear",
        z_down: bool = False,
    ) -> None:
        self.height = height
        self.width = width
        self.clip_output = clip_output
        self.mode = mode
        self.z_down = z_down

    def __call__(self, src: ArrayLike, rots: Rot, **kwargs) -> ArrayLike:
        return equi2equi(
            src=src,
            rots=rots,
            clip_output=self.clip_output,
            mode=self.mode,
            z_down=self.z_down,
            **kwargs,
        )


def equi2equi(
    src: ArrayLike,
    rots: Rot,
    clip_output: bool = True,
    mode: str = "bilinear",
    z_down: bool = False,
    height: Optional[int] = None,
    width: Optional[int] = None,
    **kwargs,
) -> ArrayLike:
    """
    params:
    - src
    - rots
    - mode (str): interpolation mode, defaults to "bilinear"
    - z_down (bool)
    - height, width (optional int): output image size

    returns:
    - out

    """

    _type = None
    if isinstance(src, np.ndarray):
        _type = "numpy"
    elif torch.is_tensor(src):
        _type = "torch"
    else:
        raise ValueError

    is_single = False
    if len(src.shape) == 3 and isinstance(rots, dict):
        # probably the input was a single image
        src = src[None, ...]
        rots = [rots]
        is_single = True
    elif len(src.shape) == 3:
        # probably a grayscale image
        src = src[:, None, ...]

    assert isinstance(rots, list), "ERR: rots is not a list"
    if _type == "numpy":
        out = run_numpy(
            src=src,
            rots=rots,
            mode=mode,
            z_down=z_down,
            height=height,
            width=width,
            clip_output=clip_output,
            **kwargs,
        )
    elif _type == "torch":
        out = run_torch(
            src=src,
            rots=rots,
            mode=mode,
            z_down=z_down,
            height=height,
            width=width,
            clip_output=clip_output,
            **kwargs,
        )
    else:
        raise ValueError

    # make sure that the output batch dim is removed if it's only a single image
    if is_single:
        out = out.squeeze(0)

    return out
