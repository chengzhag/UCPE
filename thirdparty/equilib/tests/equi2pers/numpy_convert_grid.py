#!/usr/bin/env python3

"""Convert grid M into mapping grid of pixel in the equirectangular image to target image

Input:
- M: sampling grid
- h_equi, w_equi: height and width of equirectangular image

Output:
- grid: mapping grid

"""

from copy import deepcopy
from itertools import combinations

import numpy as np

from tests.equi2pers.numpy_prep import example
from tests.equi2pers.numpy_matmul import DATA, baseline_v2
from tests.helpers.benchmarking import check_close, mae, mse
from tests.helpers.timer import func_timer


def calc_phi(M):
    """This is very expensive

    FIXME: find faster alternatives
    """
    batch, height, width, _ = M.shape
    dtype = M.dtype
    phi = np.empty((batch, height, width), dtype=dtype)
    phi = np.arcsin(M[..., 2] / np.linalg.norm(M, axis=-1))
    return phi


def calc_theta(M):
    """This is also expensive

    FIXME: find faster alternatives
    """
    batch, height, width, _ = M.shape
    dtype = M.dtype
    theta = np.empty((batch, height, width), dtype=dtype)
    theta = np.arctan2(M[..., 1], M[..., 0])
    return theta


def rot2pix(phi, theta, h_equi, w_equi):
    """rotation (sphere) to cylinder pixels of equirectangular image

    Assumptions:
    - x-axis faces center of equirectangular image
    """
    ui = (theta - np.pi) * w_equi / (2 * np.pi)
    uj = (phi - np.pi / 2) * h_equi / np.pi

    # this method might be faster?
    # ui = (theta / (2 * np.pi) - 0.5) * w_equi
    # uj = (phi / np.pi - 0.5) * h_equi

    # FIXME: would we need to center it when sampling?
    # Assuming that the cylinder maps onto the sphere
    # given theta and phi, it is more sound to add 0.5
    # so that in the grid sampling algorithm, it will use
    # the center of the pixel instead of top left.
    # Does cv2.remap automatically do this?
    ui += 0.5
    uj += 0.5
    return ui, uj


def refine_v1(phi, theta, h_equi, w_equi):
    """Fastest but might not be robust when phi and theta are out of range"""
    ui, uj = rot2pix(phi, theta, h_equi, w_equi)

    # not as robust when theta and phi is not correct
    # when phi/theta assumptions hold, it's decent speed
    # for example, when ui + w_equi is still negative (e.g. ui > -w_equi)
    # modulo will be more robust since it can handle those values
    # e.g.:
    # 180 % 360 => 180
    # (180 * 2.5) % 360 => 90  # 180 * 2.5 = 450
    # (-180 * 2.5) % 360 => 270  # -180 * 2.5 = -450
    ui = np.where(ui < 0, ui + w_equi, ui)
    ui = np.where(ui >= w_equi, ui - w_equi, ui)
    uj = np.where(uj < 0, uj + h_equi, uj)
    uj = np.where(uj >= h_equi, uj - h_equi, uj)

    grid = np.stack((uj, ui), axis=-3)
    return grid


def refine_v2(phi, theta, h_equi, w_equi):
    """Slower than v1 but faster than v3"""
    ui, uj = rot2pix(phi, theta, h_equi, w_equi)
    ui %= w_equi
    uj %= h_equi

    grid = np.stack((uj, ui), axis=-3)
    return grid


def refine_v3(phi, theta, h_equi, w_equi):
    """A little bit slower than v2"""
    ui = (((theta - np.pi) * w_equi / (2 * np.pi)) + 0.5) % w_equi
    uj = (((phi - np.pi / 2) * h_equi / np.pi) + 0.5) % h_equi
    # this method might be faster?
    # ui = (theta / (2 * np.pi) - 0.5) * w_equi + 0.5
    # uj = (phi / np.pi - 0.5) * h_equi + 0.5

    grid = np.stack((uj, ui), axis=-3)
    return grid


def refine_v4(phi, theta, h_equi, w_equi):
    """Slowest

    modulo is expensive
    """
    ui, uj = rot2pix(phi, theta, h_equi, w_equi)

    # not as robust when theta and phi is not correct
    # when phi/theta assumptions hold, it's decent speed
    # for example, when ui + w_equi is still negative (e.g. ui > -w_equi)
    # modulo will be more robust since it can handle those values
    # e.g.:
    # 180 % 360 => 180
    # (180 * 2.5) % 360 => 90  # 180 * 2.5 = 450
    # (-180 * 2.5) % 360 => 270  # -180 * 2.5 = -450
    ui = np.where(ui < 0, ui % w_equi, ui)
    ui = np.where(ui >= w_equi, ui % w_equi, ui)
    uj = np.where(uj < 0, uj % h_equi, uj)
    uj = np.where(uj >= h_equi, uj % h_equi, uj)

    grid = np.stack((uj, ui), axis=-3)
    return grid


"""Benchmarking

"""


def bench():
    h_equi = 2000
    w_equi = 4000

    funcs = [
        refine_v1,
        refine_v2,
        refine_v3,
        # refine_v4,
    ]

    for i, data in enumerate(DATA):
        print()
        print(f"TEST {i+1}")
        print(f"batch size: {data['batch']}")
        print(
            f"height/width/type: {data['height']}/{data['width']}/{data['dtype']}"
        )

        m, G, R = example(**data)
        M = baseline_v2(R, G, m)
        M = M.squeeze(-1)

        phi = func_timer(calc_phi)(M)
        theta = func_timer(calc_theta)(M)

        # how about when phi and theta is not between range?
        # theta should be in the range of 0 ~ 2pi
        # phi should be in the range of 0 ~ pi
        phi += 2.1 * np.pi  # this will fail

        grids = {}
        for func in funcs:
            func_name = func.__name__
            func = func_timer(func)
            grid = func(phi, theta, h_equi, w_equi)
            grids[func_name] = deepcopy(grid)

        for m1, m2 in combinations(grids, 2):
            g1 = grids[m1]
            g2 = grids[m2]
            print(f"{m1} vs {m2}")
            print("Close?", check_close(g1, g2))
            print("MSE:", mse(g1, g2))
            print("MAE:", mae(g1, g2))


if __name__ == "__main__":
    bench()
