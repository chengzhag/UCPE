#!/usr/bin/env python3

from typing import Dict, List

import numpy as np


"""Rotations

- http://planning.cs.uiuc.edu/node102.html
- https://en.wikipedia.org/wiki/Rotation_matrix
    - Use Rodrigues' rotation formula?
"""


def create_global2camera_rotation_matrix(
    dtype=np.dtype(np.float32),
) -> np.ndarray:
    """Rotation from global (world) to camera coordinates

    - camera coordinate has the z-axis as forward and x- and y-axis for image.
    - when we rotate the mapping grid, we want to convert the camera coordinate
      to the world coordinate where x-axis is forward.
    """

    R_XY = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # X <-> Y
        dtype=dtype,
    )
    R_YZ = np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],  # Y <-> Z
        dtype=dtype,
    )

    return R_XY @ R_YZ


def create_rotation_matrix(
    roll: float,
    pitch: float,
    yaw: float,
    z_down: bool = True,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Create Rotation Matrix

    params:
    - roll, pitch, yaw (float): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (np.dtype): data types

    returns:
    - R (np.ndarray): 3x3 rotation matrix
    """

    # calculate rotation about the x-axis (roll)
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(roll), -np.sin(roll)],
            [0.0, np.sin(roll), np.cos(roll)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the y-axis (pitch)
    if not z_down:
        pitch = -pitch
    R_y = np.array(
        [
            [np.cos(pitch), 0.0, np.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-np.sin(pitch), 0.0, np.cos(pitch)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the z-axis (yaw)
    if not z_down:
        yaw = -yaw
    R_z = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )

    return R_z @ R_y @ R_x


def create_rotation_matrix_at_once(
    roll: float,
    pitch: float,
    yaw: float,
    z_down: bool = True,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Create rotation matrix at once"

    params:
    - roll, pitch, yaw (float): in radians
    - z_down (bool): flips pitch and yaw directions
    - dtype (np.dtype): data types

    returns:
    - R (np.ndarray): 3x3 rotation matrix

    NOTE: same results as `create_rotation_matrix` but a little bit faster
    """

    if not z_down:
        pitch = -pitch
        yaw = -yaw

    return np.array(
        [
            [
                np.cos(yaw) * np.cos(pitch),
                np.cos(yaw) * np.sin(pitch) * np.sin(roll)
                - np.sin(yaw) * np.cos(roll),
                np.cos(yaw) * np.sin(pitch) * np.cos(roll)
                + np.sin(yaw) * np.sin(roll),
            ],
            [
                np.sin(yaw) * np.cos(pitch),
                np.sin(yaw) * np.sin(yaw) * np.sin(pitch) * np.sin(roll)
                + np.cos(yaw) * np.cos(roll),
                np.sin(yaw) * np.sin(pitch) * np.cos(roll)
                - np.cos(yaw) * np.sin(roll),
            ],
            [
                -np.sin(pitch),
                np.cos(pitch) * np.sin(roll),
                np.cos(pitch) * np.cos(roll),
            ],
        ],
        dtype=dtype,
    )


def create_rotation_matrices(
    rots: List[Dict[str, float]],
    z_down: bool = True,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Create rotation matrices from batch of rotations

    This methods creates a bx3x3 np.ndarray where `b` referes to the number
    of rotations (rots) given in the input
    """

    R = np.empty((len(rots), 3, 3), dtype=dtype)
    for i, rot in enumerate(rots):
        # FIXME: maybe default to `create_rotation_matrix_at_once`?
        R[i, ...] = create_rotation_matrix(**rot, z_down=z_down, dtype=dtype)

    return R


def create_rotation_matrix_dep(
    x: float,
    y: float,
    z: float,
    z_down: bool = True,
    dtype: np.dtype = np.dtype(np.float32),
) -> np.ndarray:
    """Create rotation matrix

    NOTE: DEPRECATED
    """

    # calculate rotation about the x-axis
    R_x = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(x), -np.sin(x)],
            [0.0, np.sin(x), np.cos(x)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the y-axis
    if not z_down:
        y = -y
    R_y = np.array(
        [
            [np.cos(y), 0.0, -np.sin(y)],
            [0.0, 1.0, 0.0],
            [np.sin(y), 0.0, np.cos(y)],
        ],
        dtype=dtype,
    )
    # calculate rotation about the z-axis
    if not z_down:
        z = -z
    R_z = np.array(
        [
            [np.cos(z), np.sin(z), 0.0],
            [-np.sin(z), np.cos(z), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=dtype,
    )

    return R_z @ R_y @ R_x
