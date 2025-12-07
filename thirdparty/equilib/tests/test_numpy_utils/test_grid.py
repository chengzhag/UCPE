#!/usr/bin/env python3

from typing import Optional
import numpy as np

import pytest

from equilib.numpy_utils.grid import create_grid


@pytest.mark.parametrize("height", [48, 64])
@pytest.mark.parametrize("width", [64])
@pytest.mark.parametrize("batch", [None, 4])
@pytest.mark.parametrize("dtype", [np.dtype(np.float32), np.dtype(np.float64)])
def test_grid(height: int, width: int, batch: Optional[int], dtype: np.dtype):
    grid = create_grid(height=height, width=width, batch=batch, dtype=dtype)

    assert grid.dtype == dtype
    if batch is not None:
        assert grid.shape == (batch, height, width, 3)
        for b in range(batch):
            for h in range(height):
                for w in range(width):
                    assert np.allclose(
                        grid[b, h, w], np.array([w, h, 1], dtype=dtype)
                    )
    else:
        assert grid.shape == (height, width, 3)
        for h in range(height):
            for w in range(width):
                assert np.allclose(grid[h, w], np.array([w, h, 1], dtype=dtype))
