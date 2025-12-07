#!/usr/bin/env python3

import sys  # noqa

import numpy as np  # noqa
import torch  # noqa

try:
    from PIL import Image
except ImportError:
    print("PIL is not installed")
    Image = None


if __name__ == "__main__":
    print("loaded imports")
