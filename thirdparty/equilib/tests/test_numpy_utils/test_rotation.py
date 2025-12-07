#!/usr/bin/env python3

import copy

import numpy as np

import pytest

from equilib.numpy_utils.rotation import (
    create_rotation_matrices,
    create_rotation_matrix,
    create_rotation_matrix_at_once,
)

from tests.helpers.timer import func_timer


# GT made with https://keisan.casio.com/exec/system/15362817755710
# TODO: more test cases
GT = [
    {
        "rot": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
        "in_coord": [1.0, 0.0, 1.0],
        "out_coord": [1.0, 0.0, 1.0],
    },
    {
        "rot": {"roll": 0.0, "pitch": np.pi / 4, "yaw": np.pi / 4},
        "in_coord": [1.0, 0.0, 0.0],
        "out_coord": [0.5, 0.5, -0.707106781186547524400844362105],
    },
    {
        "rot": {"roll": 0.0, "pitch": np.pi / 3, "yaw": -np.pi / 3},
        "in_coord": [3.0, 2.0, 1.0],
        "out_coord": [
            2.91506350946109661690930792688,
            -1.04903810567665797014558475613,
            -2.09807621135331594029116951226,
        ],
    },
]


@pytest.mark.parametrize("dtype", [np.dtype(np.float32), np.dtype(np.float64)])
def test_rotation_matrix(dtype):
    for gt in GT:
        in_coord = np.array(gt["in_coord"], dtype=dtype)
        gt_coord = np.array(gt["out_coord"], dtype=dtype)

        R = create_rotation_matrix(**gt["rot"], z_down=True, dtype=dtype)
        R_ = create_rotation_matrix_at_once(
            **gt["rot"], z_down=True, dtype=dtype
        )

        coord = R @ in_coord
        coord_ = R_ @ in_coord

        assert np.allclose(coord, gt_coord)
        assert np.allclose(coord_, gt_coord)


def test_rotation_matrix_down():
    dtype = np.dtype(np.float64)

    for gt in GT:
        # NOTE: make sure to not change the GT!
        rot = copy.deepcopy(gt["rot"])
        rot["pitch"] = -rot["pitch"]
        rot["yaw"] = -rot["yaw"]

        in_coord = np.array(gt["in_coord"], dtype=dtype)
        gt_coord = np.array(gt["out_coord"], dtype=dtype)

        R = create_rotation_matrix(**rot, z_down=False, dtype=dtype)
        R_ = create_rotation_matrix_at_once(**rot, z_down=False, dtype=dtype)

        coord = R @ in_coord
        coord_ = R_ @ in_coord

        assert np.allclose(coord, gt_coord)
        assert np.allclose(coord_, gt_coord)


@pytest.mark.parametrize("dtype", [np.dtype(np.float32), np.dtype(np.float64)])
def test_rotation_matrices(dtype):
    rots = [gt["rot"] for gt in GT]
    R = create_rotation_matrices(rots=rots, z_down=True, dtype=dtype)
    in_coords = np.array([gt["in_coord"] for gt in GT], dtype=dtype)[
        ..., np.newaxis
    ]
    gt_coords = np.array([gt["out_coord"] for gt in GT], dtype=dtype)[
        ..., np.newaxis
    ]

    coords = R @ in_coords

    assert np.allclose(coords, gt_coords)


def bench_rotation_matrix():
    rot_mat = func_timer(create_rotation_matrix)
    rot_mat_at_once = func_timer(create_rotation_matrix_at_once)
    dtype = np.dtype(np.float64)

    for gt in GT:
        _ = rot_mat(**gt["rot"], z_down=True, dtype=dtype)
        _ = rot_mat_at_once(**gt["rot"], z_down=True, dtype=dtype)


if __name__ == "__main__":
    bench_rotation_matrix()
