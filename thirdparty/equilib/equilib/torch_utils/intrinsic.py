#!/usr/bin/env python3

import torch

pi = torch.Tensor([3.14159265358979323846])


def deg2rad(tensor: torch.Tensor) -> torch.Tensor:
    """Function that converts angles from degrees to radians"""
    return tensor * pi.to(tensor.device).type(tensor.dtype) / 180.0


def create_intrinsic_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create intrinsic matrix

    params:
    - height, width (int)
    - fov_x (float): make sure it's in degrees
    - skew (float): 0.0
    - dtype (torch.dtype): torch.float32
    - device (torch.device): torch.device("cpu")

    returns:
    - K (torch.tensor): 3x3 intrinsic matrix
    """
    f = width / (2 * torch.tan(deg2rad(torch.tensor(fov_x, dtype=dtype)) / 2))
    K = torch.tensor(
        [[f, skew, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    return K
