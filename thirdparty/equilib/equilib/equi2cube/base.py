#!/usr/bin/env python3

from typing import Dict, List, Union

import numpy as np

import torch

from .numpy import run as run_numpy
from .torch import run as run_torch

__all__ = ["Equi2Cube", "equi2cube"]

ArrayLike = Union[np.ndarray, torch.Tensor]
Rot = Union[Dict[str, float], List[Dict[str, float]]]
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


class Equi2Cube(object):
    """
    params:
    - w_face (int): cube face width
    - cube_format (str): ("dice", "horizon", "dict", "list")
    - mode (str)
    - z_down (bool)

    inputs:
    - equi (np.ndarray, torch.Tensor)
    - rots (dict, list[dict]): {"roll", "pitch", "yaw"}

    returns:
    - cube (np.ndarray, torch.Tensor, list, dict)
    """

    def __init__(
        self,
        w_face: int,
        cube_format: str,
        z_down: bool = False,
        clip_output: bool = True,
        mode: str = "bilinear",
    ) -> None:
        self.w_face = w_face
        self.cube_format = cube_format
        self.z_down = z_down
        self.clip_output = clip_output
        self.mode = mode

    def __call__(self, equi: ArrayLike, rots: Rot, **kwargs) -> CubeMaps:
        return equi2cube(
            equi=equi,
            rots=rots,
            w_face=self.w_face,
            cube_format=self.cube_format,
            z_down=self.z_down,
            clip_output=self.clip_output,
            mode=self.mode,
            **kwargs,
        )


def equi2cube(
    equi: ArrayLike,
    rots: Rot,
    w_face: int,
    cube_format: str,
    z_down: bool = False,
    clip_output: bool = True,
    mode: str = "bilinear",
    **kwargs,
) -> CubeMaps:
    """
    params:
    - equi (np.ndarray, torch.Tensor)
    - rot (dict, list[dict]): {"roll", "pitch", "yaw"}
    - w_face (int): cube face width
    - cube_format (str): ("dice", "horizon", "dict", "list")
    - z_down (bool)
    - mode (str)

    returns:
    - cube (np.ndarray, torch.Tensor, dict, list)

    """

    _type = None
    if isinstance(equi, np.ndarray):
        _type = "numpy"
    elif torch.is_tensor(equi):
        _type = "torch"
    else:
        raise ValueError

    is_single = False
    if len(equi.shape) == 3 and isinstance(rots, dict):
        # probably the input was a single image
        equi = equi[None, ...]
        rots = [rots]
        is_single = True
    elif len(equi.shape) == 3:
        # probably a grayscale image
        equi = equi[:, None, ...]

    assert isinstance(rots, list), "ERR: rots is not a list"
    if _type == "numpy":
        out = run_numpy(
            equi=equi,
            rots=rots,
            w_face=w_face,
            cube_format=cube_format,
            z_down=z_down,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    elif _type == "torch":
        out = run_torch(
            equi=equi,
            rots=rots,
            w_face=w_face,
            cube_format=cube_format,
            z_down=z_down,
            clip_output=clip_output,
            mode=mode,
            **kwargs,
        )
    else:
        raise ValueError

    # make sure that the output batch dim is removed if it's only a single cubemap
    if is_single:
        out = out[0]

    return out
