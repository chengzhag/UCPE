#!/usr/bin/env python3

from typing import Optional
import torch

import pytest

from equilib.torch_utils.grid import create_grid


@pytest.mark.parametrize("height", [48, 64])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("batch", [None, 4])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grid_cpu(
    height: int, width: int, batch: Optional[int], dtype: torch.dtype
):
    device = torch.device("cpu")

    grid = create_grid(
        height=height, width=width, batch=batch, dtype=dtype, device=device
    )

    assert grid.dtype == dtype
    if batch is not None:
        assert grid.shape == (batch, height, width, 3)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    assert torch.allclose(
                        grid[b, h, w],
                        torch.tensor([w, h, 1], dtype=dtype, device=device),
                    )
    else:
        assert grid.shape == (height, width, 3)
        for h in range(height):
            for w in range(width):
                assert torch.allclose(
                    grid[h, w],
                    torch.tensor([w, h, 1], dtype=dtype, device=device),
                )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda device is not available"
)
@pytest.mark.parametrize("height", [4])
@pytest.mark.parametrize("width", [8])
@pytest.mark.parametrize("batch", [None, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_grid_gpu(
    height: int, width: int, batch: Optional[int], dtype: torch.dtype
):
    device = torch.device("cuda")

    grid = create_grid(
        height=height, width=width, batch=batch, dtype=dtype, device=device
    )

    assert grid.dtype == dtype
    if batch is not None:
        assert grid.shape == (batch, height, width, 3)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    assert torch.allclose(
                        grid[b, h, w],
                        torch.tensor([w, h, 1], dtype=dtype, device=device),
                    )
    else:
        assert grid.shape == (height, width, 3)
        for h in range(height):
            for w in range(width):
                assert torch.allclose(
                    grid[h, w],
                    torch.tensor([w, h, 1], dtype=dtype, device=device),
                )
