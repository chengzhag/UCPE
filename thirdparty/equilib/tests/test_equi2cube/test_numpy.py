#!/usr/bin/env python3

"""Mainly tests against naive method or baselines (cv2 and scipy)

FIXME: better assertions
"""

from copy import deepcopy
import os

import numpy as np

import pytest

from equilib.equi2cube.numpy import run

from tests.grid_sample.numpy.baselines import (
    cv2,
    grid_sample_cv2,
    grid_sample_scipy,
    map_coordinates,
)
from tests.helpers.benchmarking import check_close, how_many_closes, mae, mse
from tests.helpers.image_io import load2numpy, save
from tests.helpers.timer import func_timer, wrapped_partial
from tests.helpers.rot_path import (
    create_rots,
    create_rots_pitch,
    create_rots_yaw,
)

run_cv2 = wrapped_partial(run, override_func=grid_sample_cv2)
run_scipy = wrapped_partial(run, override_func=grid_sample_scipy)

SAVE_ROOT = "tests/test_equi2cube/results"
DATA_ROOT = "tests/data"
IMG_NAME = "test.jpg"


def get_img(dtype: np.dtype = np.float32) -> np.ndarray:
    # path to equirectangular image
    path = os.path.join(DATA_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def make_batch(img: np.ndarray, bs: int = 1) -> np.ndarray:
    imgs = np.empty((bs, *img.shape), dtype=img.dtype)
    for b in range(bs):
        imgs[b, ...] = deepcopy(img)
    return imgs


def get_metrics(o1: np.ndarray, o2: np.ndarray, rtol: float, atol: float):
    is_close = check_close(o1, o2, rtol=rtol, atol=atol)
    r_close = how_many_closes(o1, o2, rtol=rtol, atol=atol)
    err_mse = mse(o1, o2)
    err_mae = mae(o1, o2)
    return is_close, r_close, err_mse, err_mae


def bench_baselines(
    bs: int,
    z_down: bool,
    mode: str,
    w_face: int,
    cube_format: str,
    dtype: np.dtype = np.dtype(np.float32),
    rotation: str = "forward",
    save_outputs: bool = False,
) -> None:
    # print parameters for debugging
    print()
    print("bs, w_face:", bs, w_face)
    print("cube format:", cube_format)
    print("dtype:", dtype)
    print("z_down:", z_down)
    print("mode:", mode)
    print("rotation:", rotation)

    if dtype == np.float32:
        rtol = 1e-03
        atol = 1e-05
    elif dtype == np.float64:
        rtol = 1e-05
        atol = 1e-08
    else:
        rtol = 1e-01
        atol = 1e-03

    # obtaining single equirectangular image
    img = get_img(dtype)

    # making batch
    imgs = make_batch(img, bs=bs)
    print(imgs.shape, imgs.dtype)

    # generate rotation parameters
    if rotation == "forward":
        rots = create_rots(bs=bs)
    elif rotation == "pitch":
        rots = create_rots_pitch(bs=bs)
    elif rotation == "yaw":
        rots = create_rots_yaw(bs=bs)
    else:
        raise ValueError

    print("scipy:")
    out_scipy = func_timer(run_scipy)(
        equi=imgs,
        rots=rots,
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
    )
    print("cv2")
    out_cv2 = func_timer(run_cv2)(
        equi=imgs,
        rots=rots,
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
    )
    print("numpy")
    out = func_timer(run)(
        equi=imgs,
        rots=rots,
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
    )

    # evaluation depends on the output format
    # - reformat to np.ndarray?
    if cube_format in ("horizon", "dice"):
        # can be tested as the same before since `out` is np.ndarray

        assert out.shape == out_scipy.shape == out_cv2.shape
        assert out.dtype == out_scipy.dtype == out_cv2.dtype == dtype

        # quantitative
        print()
        print(">>> compare against scipy")
        is_close, r_close, err_mse, err_mae = get_metrics(
            out, out_scipy, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        assert err_mse < 1e-04
        assert err_mae < 1e-02

        print()
        print(">>> compare against cv2")
        is_close, r_close, err_mse, err_mae = get_metrics(
            out, out_cv2, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        assert err_mse < 1e-04
        assert err_mae < 1e-02

        print()
        print(">>> compare scipy and cv2")
        is_close, r_close, err_mse, err_mae = get_metrics(
            out_cv2, out_scipy, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        assert err_mse < 1e-04
        assert err_mae < 1e-02

        if save_outputs:
            # qualitative
            # save the outputs and see the images
            for b in range(bs):
                save(
                    out[b],
                    os.path.join(SAVE_ROOT, f"out_{cube_format}_{b}.jpg"),
                )
                save(
                    out_cv2[b],
                    os.path.join(SAVE_ROOT, f"out_cv2_{cube_format}_{b}.jpg"),
                )
                save(
                    out_scipy[b],
                    os.path.join(SAVE_ROOT, f"out_scipy_{cube_format}_{b}.jpg"),
                )

    elif cube_format == "list":
        # output format is List[List[np.ndarray]]
        # order ["F", "R", "B", "L", "U", "D"]
        assert isinstance(out, list)
        for b in range(bs):
            assert isinstance(out[b], list)
            for i, face in enumerate(["F", "R", "B", "L", "U", "D"]):
                print()
                print(f">>> Testing batch: {b}, face: {face}")
                _out = out[b][i]
                _out_cv2 = out_cv2[b][i]
                _out_scipy = out_scipy[b][i]

                assert isinstance(_out, np.ndarray)
                assert isinstance(_out_cv2, np.ndarray)
                assert isinstance(_out_scipy, np.ndarray)

                assert _out.shape == _out_cv2.shape == _out_scipy.shape
                assert _out.dtype == _out_cv2.dtype == _out_scipy.dtype == dtype

                # quantitative
                print()
                print(">>> compare against scipy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _out, _out_scipy, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                assert err_mse < 1e-04
                assert err_mae < 1e-02

                print()
                print(">>> compare against cv2")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _out, _out_cv2, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                assert err_mse < 1e-04
                assert err_mae < 1e-02

                print()
                print(">>> compare scipy and cv2")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _out_cv2, _out_scipy, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                save(
                    _out,
                    os.path.join(
                        SAVE_ROOT, f"out_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _out_cv2,
                    os.path.join(
                        SAVE_ROOT, f"out_cv2_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _out_scipy,
                    os.path.join(
                        SAVE_ROOT, f"out_scipy_{cube_format}_{b}_{face}.jpg"
                    ),
                )

    elif cube_format == "dict":
        # output format is List[Dict[str, np.ndarray]]
        # order ["F", "R", "B", "L", "U", "D"]
        assert isinstance(out, list)

        for b in range(bs):
            assert isinstance(out[b], dict)
            for (face,) in ["F", "R", "B", "L", "U", "D"]:
                print()
                print(f">>> Testing batch: {b}, face: {face}")
                _out = out[b][face]
                _out_cv2 = out_cv2[b][face]
                _out_scipy = out_scipy[b][face]

                assert isinstance(_out, np.ndarray)
                assert isinstance(_out_cv2, np.ndarray)
                assert isinstance(_out_scipy, np.ndarray)

                assert _out.shape == _out_cv2.shape == _out_scipy.shape
                assert _out.dtype == _out_cv2.dtype == _out_scipy.dtype == dtype

                # quantitative
                print()
                print(">>> compare against scipy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _out, _out_scipy, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                assert err_mse < 1e-04
                assert err_mae < 1e-02

                print()
                print(">>> compare against cv2")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _out, _out_cv2, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                assert err_mse < 1e-04
                assert err_mae < 1e-02

                print()
                print(">>> compare scipy and cv2")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _out_cv2, _out_scipy, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                save(
                    _out,
                    os.path.join(
                        SAVE_ROOT, f"out_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _out_cv2,
                    os.path.join(
                        SAVE_ROOT, f"out_cv2_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _out_scipy,
                    os.path.join(
                        SAVE_ROOT, f"out_scipy_{cube_format}_{b}_{face}.jpg"
                    ),
                )
    else:
        raise ValueError


@pytest.mark.skipif(cv2 is None, reason="cv2 is None; not installed")
@pytest.mark.skipif(
    map_coordinates is None,
    reason="scipy.map_coordinate is None; not installed",
)
@pytest.mark.parametrize("bs", [1, 4])
@pytest.mark.parametrize("z_down", [False])
@pytest.mark.parametrize("mode", ["nearest", "bilinear", "bicubic"])
@pytest.mark.parametrize("w_face", [32])
@pytest.mark.parametrize("cube_format", ["dice", "dict", "list"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rotation", ["pitch"])
def test_equi2cube_against_baselines(
    bs: int,
    z_down: bool,
    mode: str,
    w_face: int,
    cube_format: str,
    dtype: np.dtype,
    rotation: str,
) -> None:
    bench_baselines(
        bs=bs,
        z_down=z_down,
        mode=mode,
        w_face=w_face,
        cube_format=cube_format,
        dtype=dtype,
        rotation=rotation,
    )
