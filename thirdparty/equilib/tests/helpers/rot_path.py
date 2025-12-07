#!/usr/bin/env python3

"""Module for managing paths for batched rots
"""

from typing import Optional
import numpy as np


def create_rots_pitch(bs: int = 1, m: Optional[int] = None):
    rots = []
    if m is None:
        m = bs
    for b in range(bs):
        rot = {"roll": 0.0, "pitch": b * np.pi / m - np.pi / 2, "yaw": 0.0}
        rots.append(rot)
    return rots


def create_rots_yaw(bs: int = 1, m: Optional[int] = None):
    rots = []
    if m is None:
        m = bs
    for b in range(bs):
        rot = {"roll": 0.0, "pitch": 0.0, "yaw": b * np.pi / (m / 2) - np.pi}
        rots.append(rot)
    return rots


def create_rots(bs: int = 1):
    rots = []
    for _ in range(bs):
        rot = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        rots.append(rot)
    return rots
