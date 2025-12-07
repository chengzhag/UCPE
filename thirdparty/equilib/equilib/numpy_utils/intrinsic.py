#!/usr/bin/env python3

import numpy as np


def create_intrinsic_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Create intrinsic matrix

    params:
    - height, width (int)
    - fov_x (float): make sure it's in degrees
    - skew (float): 0.0
    - dtype (np.dtype): np.float32

    returns:
    - K (np.ndarray): 3x3 intrinsic matrix
    """

    # perspective projection (focal length)
    f = width / (2.0 * np.tan(np.radians(fov_x).astype(dtype) / 2.0))
    # transform between camera frame and pixel coordinates
    K = np.array(
        [[f, skew, width / 2], [0.0, f, height / 2], [0.0, 0.0, 1.0]],
        dtype=dtype,
    )

    return K
