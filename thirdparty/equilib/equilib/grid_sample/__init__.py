#!/usr/bin/env python3

from .numpy import grid_sample as numpy_grid_sample
from .torch import grid_sample as torch_grid_sample

__all__ = ["numpy_grid_sample", "torch_grid_sample"]
