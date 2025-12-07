#!/usr/bin/env python3

import copy
from itertools import product

import numpy as np

import torch

import pytest

from equilib.torch_utils.rotation import (
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation_matrix_cpu(dtype):
    device = torch.device("cpu")
    for gt in GT:
        in_coord = torch.tensor(gt["in_coord"], dtype=dtype, device=device)
        gt_coord = torch.tensor(gt["out_coord"], dtype=dtype, device=device)

        R = create_rotation_matrix(
            **gt["rot"], z_down=True, dtype=dtype, device=device
        )
        R_ = create_rotation_matrix_at_once(
            **gt["rot"], z_down=True, dtype=dtype, device=device
        )

        coord = R @ in_coord
        coord_ = R_ @ in_coord

        assert torch.allclose(coord, gt_coord)
        assert torch.allclose(coord_, gt_coord)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_rotation_matrix_gpu(dtype):
    device = torch.device("cuda")
    for gt in GT:
        in_coord = torch.tensor(gt["in_coord"], dtype=dtype, device=device)
        gt_coord = torch.tensor(gt["out_coord"], dtype=dtype, device=device)

        R = create_rotation_matrix(
            **gt["rot"], z_down=True, dtype=dtype, device=device
        )
        R_ = create_rotation_matrix_at_once(
            **gt["rot"], z_down=True, dtype=dtype, device=device
        )

        coord = R @ in_coord
        coord_ = R_ @ in_coord

        if dtype == torch.float16:
            assert torch.allclose(coord, gt_coord, rtol=1e-3, atol=1e-5)
            assert torch.allclose(coord_, gt_coord, rtol=1e-3, atol=1e-5)
        else:
            assert torch.allclose(coord, gt_coord)
            assert torch.allclose(coord_, gt_coord)


@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        # torch.device("cuda"),
    ],
)
def test_rotation_matrix_down(device):
    dtype = torch.float32

    for gt in GT:
        # NOTE: make sure to not change the GT!
        rot = copy.deepcopy(gt["rot"])
        rot["pitch"] = -rot["pitch"]
        rot["yaw"] = -rot["yaw"]

        in_coord = torch.tensor(gt["in_coord"], dtype=dtype, device=device)
        gt_coord = torch.tensor(gt["out_coord"], dtype=dtype, device=device)

        R = create_rotation_matrix(
            **rot, z_down=False, dtype=dtype, device=device
        )
        R_ = create_rotation_matrix_at_once(
            **rot, z_down=False, dtype=dtype, device=device
        )

        coord = R @ in_coord
        coord_ = R_ @ in_coord

        assert torch.allclose(coord, gt_coord)
        assert torch.allclose(coord_, gt_coord)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_rotation_matrices_cpu(dtype):
    device = torch.device("cpu")

    rots = [gt["rot"] for gt in GT]
    R = create_rotation_matrices(
        rots=rots, z_down=True, dtype=dtype, device=device
    )
    in_coords = torch.tensor(
        [gt["in_coord"] for gt in GT], dtype=dtype, device=device
    ).unsqueeze(-1)
    gt_coords = torch.tensor(
        [gt["out_coord"] for gt in GT], dtype=dtype, device=device
    ).unsqueeze(-1)

    coords = R @ in_coords

    assert torch.allclose(coords, gt_coords)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_rotation_matrices_gpu(dtype):
    device = torch.device("cuda")

    rots = [gt["rot"] for gt in GT]
    R = create_rotation_matrices(
        rots=rots, z_down=True, dtype=dtype, device=device
    )
    in_coords = torch.tensor(
        [gt["in_coord"] for gt in GT], dtype=dtype, device=device
    ).unsqueeze(-1)
    gt_coords = torch.tensor(
        [gt["out_coord"] for gt in GT], dtype=dtype, device=device
    ).unsqueeze(-1)

    coords = R @ in_coords

    if dtype == torch.float16:
        assert torch.allclose(coords, gt_coords, rtol=1e-3, atol=1e-5)
    else:
        assert torch.allclose(coords, gt_coords)


def bench_rotation_matrix(dtype, device):
    import gc

    # wrapping functions
    rot_mat = func_timer(create_rotation_matrix)
    rot_mat_at_once = func_timer(create_rotation_matrix_at_once)

    # FIXME: can't get the variables released somehow
    for gt in GT:
        a = rot_mat(**gt["rot"], z_down=True, dtype=dtype, device=device)
        del a
        gc.collect()
        torch.cuda.empty_cache()
        b = rot_mat_at_once(
            **gt["rot"], z_down=True, dtype=dtype, device=device
        )
        del b
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    dtypes = [torch.float16, torch.float32, torch.float64]
    devices = [torch.device("cpu"), torch.device("cuda")]

    for dtype, device in product(dtypes, devices):
        print(f"\ntesting for {dtype}/{device}")
        bench_rotation_matrix(dtype=dtype, device=device)
