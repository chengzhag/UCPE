#!/usr/bin/env python3

"""Preparations

What is it?
- mainly preprocess the input arguments so it is cleaned up
  for the later functions

Input params:
- pers_h
- pers_w
- batch
- dtype
- rot / rots

Outputs:
- m: pixel grid
- G: "cam-to-world" rotation matrix
- R: batched rotation matrices as a tensor

"""

from functools import lru_cache

import numpy as np

from equilib.numpy_utils import (
    create_global2camera_rotation_matrix,
    create_grid,
    create_intrinsic_matrix,
    create_rotation_matrices,
)

from tests.helpers.timer import func_timer


@lru_cache(maxsize=128)
def prepare_transform_matrix(
    height: int,
    width: int,
    fov_x: float = 90.0,
    skew: float = 0.0,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Handy function that takes care of intrinsics and camera transforms

    probably should cache this since this is the same if the arguments are
    the same:
    - results -> speed up of around x30

    """

    K = create_intrinsic_matrix(
        height=height, width=width, fov_x=fov_x, skew=skew, dtype=dtype
    )

    g2c_mat = create_global2camera_rotation_matrix(dtype=dtype)

    # NOTE: this function is slow; use mkl when possible
    K_inv = np.linalg.inv(K)

    return g2c_mat @ K_inv


def example(
    rots,
    height: int,
    width: int,
    batch: int,
    dtype: np.dtype = np.dtype(np.float32),
    fov_x: float = 90.0,
    skew: float = 0.0,
):
    """An example of the `prep` function"""
    m = create_grid(height=height, width=width, batch=batch, dtype=dtype)
    m = m[..., np.newaxis]

    G = prepare_transform_matrix(
        height=height, width=width, fov_x=fov_x, skew=skew, dtype=dtype
    )

    R = create_rotation_matrices(rots=rots, z_down=True, dtype=dtype)

    return m, G, R


def check_cache():
    data = [
        {
            "rots": [
                {"roll": 0.0, "pitch": np.pi / 2 + 0.1, "yaw": np.pi + 0.1}
            ]
            * 32,
            "height": 256,
            "width": 512,
            "batch": 32,
            "dtype": np.dtype(np.float32),
        },
        {
            "rots": [
                {"roll": 0.0, "pitch": np.pi / 2 + 0.1, "yaw": np.pi + 0.1}
            ]
            * 32,
            "height": 256,
            "width": 512,
            "batch": 32,
            "dtype": np.dtype(np.float32),
        },
        {
            "rots": [
                {"roll": 0.0, "pitch": np.pi / 2 + 0.1, "yaw": np.pi + 0.1}
            ]
            * 64,
            "height": 256,
            "width": 512,
            "batch": 64,
            "dtype": np.dtype(np.float32),
        },
        {
            "rots": [
                {"roll": 0.0, "pitch": np.pi / 2 + 0.1, "yaw": np.pi + 0.1}
            ]
            * 32,
            "height": 128,
            "width": 256,
            "batch": 32,
            "dtype": np.dtype(np.float32),
        },
    ]

    func = func_timer(example)

    for d in data:
        m, G, R = func(**d)


if __name__ == "__main__":
    check_cache()


# if __name__ == "__main__":
#     height = 256
#     width = 512
#     batch = 4
#     dtype = np.dtype(np.float64)

#     rot = {
#         "roll": 0.,
#         "pitch": np.pi / 4,
#         "yaw": np.pi / 4,
#     }
#     rots = [rot] * batch

#     m, G, R = example(
#         rots=rots,
#         pers_h=height,
#         pers_w=width,
#         batch=batch,
#         dtype=dtype,
#     )

#     # END

#     C = R @ G
#     print(C.shape)

#     # calc sampling grid because I want to
#     M = C[:, np.newaxis, np.newaxis, ...] @ m
#     print(M.shape)
