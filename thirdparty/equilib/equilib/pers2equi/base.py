#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .numpy import run as run_numpy
from .torch import run as run_torch

__all__ = ["Pers2Equi", "pers2equi"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Rot = Union[Dict[str, float], List[Dict[str, float]]]


class Pers2Equi(object):
    """
    params:
    - height, width (int): equirectangular size
    - z_down (bool)
    - mode (str)
    - clip_output (bool)

    inputs:
    - pers (np.ndarray, torch.Tensor)
    - rot (dict, list): Dict[str, float]

    returns:
    - equi (np.ndarray, torch.Tensor)

    """

    def __init__(
        self,
        height: int,
        width: int,
        z_down: bool = False,
        mode: str = "bilinear",
        clip_output: bool = True,
    ) -> None:
        self.height = height
        self.width = width
        self.mode = mode
        self.z_down = z_down
        self.clip_output = clip_output
        # FIXME: maybe do useful stuff like precalculating the grid or something

    def __call__(self, pers: ArrayLike, rots: Rot, fov_x: float, **kwargs) -> ArrayLike:
        # FIXME: should optimize since some parts of the code can be calculated
        # before hand.
        # 1. calculate grid
        # 2. grid sample
        return pers2equi(
            pers=pers,
            rots=rots,
            height=self.height,
            width=self.width,
            fov_x=fov_x,
            z_down=self.z_down,
            mode=self.mode,
            clip_output=self.clip_output,
            **kwargs,
        )

    # def get_bounding_fov(self, equi: ArrayLike, rots: Rot) -> ArrayLike:
    #     return get_bounding_fov(
    #         equi=equi,
    #         rots=rots,
    #         height=self.height,
    #         width=self.width,
    #         fov_x=self.fov_x,
    #         skew=self.skew,
    #         z_down=self.z_down,
    #     )


def pers2equi(
    pers: ArrayLike,
    rots: Rot,
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
    mode: str = "bilinear",
    z_down: bool = False,
    clip_output: bool = True,
    **kwargs,
) -> ArrayLike:
    """
    params:
    - equi
    - rots
    - height, width (int): perspective size
    - fov_x (float): perspective image fov of x-axis
    - z_down (bool)
    - skew (float): skew intrinsic parameter

    returns:
    - equi (np.ndarray, torch.Tensor)

    """

    _type = None
    if isinstance(pers, np.ndarray):
        _type = "numpy"
    elif torch.is_tensor(pers):
        _type = "torch"
    else:
        raise ValueError

    is_single = False
    if len(pers.shape) == 3 and isinstance(rots, dict):
        # probably the input was a single image
        pers = pers[None, ...]
        rots = [rots]
        is_single = True
    elif len(pers.shape) == 3:
        # probably a grayscale image
        pers = pers[:, None, ...]

    assert isinstance(rots, list), "ERR: rots is not a list"
    if _type == "numpy":
        out = run_numpy(
            pers=pers,
            rots=rots,
            height=height,
            width=width,
            fov_x=fov_x,
            skew=skew,
            z_down=z_down,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    elif _type == "torch":
        out = run_torch(
            pers=pers,
            rots=rots,
            height=height,
            width=width,
            fov_x=fov_x,
            skew=skew,
            z_down=z_down,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    else:
        raise ValueError

    # make sure that the output batch dim is removed if it's only a single image
    if is_single:
        out = out.squeeze(0)

    return out
