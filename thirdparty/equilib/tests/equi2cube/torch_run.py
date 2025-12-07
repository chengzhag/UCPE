#!/usr/bin/env python3

from copy import deepcopy
import os

import numpy as np

import torch

from equilib.equi2cube.numpy import run as run_numpy
from equilib.equi2cube.torch import run as run

from tests.helpers.benchmarking import check_close, how_many_closes, mae, mse
from tests.helpers.image_io import load2numpy, load2torch, save
from tests.helpers.timer import func_timer, wrapped_partial
from tests.helpers.rot_path import (
    create_rots,
    create_rots_pitch,
    create_rots_yaw,
)

run_native = wrapped_partial(run, backend="native")
run_pure = wrapped_partial(run, backend="pure")

IMG_ROOT = "tests/data"
SAVE_ROOT = "tests/equi2cube/results"
IMG_NAME = "test.jpg"


def get_numpy_img(dtype: np.dtype = np.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2numpy(path, dtype=dtype, is_cv2=False)
    return img


def get_torch_img(dtype: torch.dtype = torch.float32):
    path = os.path.join(IMG_ROOT, IMG_NAME)
    img = load2torch(path, dtype=dtype, is_cv2=False)
    return img


def make_batch(img, bs: int = 1):
    if isinstance(img, np.ndarray):
        imgs = np.empty((bs, *img.shape), dtype=img.dtype)
        for b in range(bs):
            imgs[b, ...] = deepcopy(img)
    elif torch.is_tensor(img):
        imgs = torch.empty((bs, *img.shape), dtype=img.dtype)
        for b in range(bs):
            imgs[b, ...] = img.clone()
    else:
        raise ValueError()
    return imgs


def get_metrics(o1, o2, rtol: float, atol: float):
    is_close = check_close(o1, o2, rtol=rtol, atol=atol)
    r_close = how_many_closes(o1, o2, rtol=rtol, atol=atol)
    err_mse = mse(o1, o2)
    err_mae = mae(o1, o2)
    return is_close, r_close, err_mse, err_mae


def bench_cpu(
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
        torch_dtype = torch.float32
        rtol = 1e-03
        atol = 1e-05
    elif dtype == np.float64:
        torch_dtype = torch.float64
        rtol = 1e-05
        atol = 1e-08
    else:
        torch_dtype = torch.uint8
        rtol = 1e-01
        atol = 1e-03

    numpy_img = get_numpy_img(dtype=dtype)
    torch_img = get_torch_img(dtype=torch_dtype)

    numpy_imgs = make_batch(numpy_img, bs=bs)
    torch_imgs = make_batch(torch_img, bs=bs)

    # generate rotation parameters
    if rotation == "forward":
        rots = create_rots(bs=bs)
    elif rotation == "pitch":
        rots = create_rots_pitch(bs=bs)
    elif rotation == "yaw":
        rots = create_rots_yaw(bs=bs)
    else:
        raise ValueError

    print("numpy:")
    numpy_out = func_timer(run_numpy)(
        equi=numpy_imgs,
        rots=rots,
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
    )
    print("native")
    native_out = func_timer(run_native)(
        equi=torch_imgs.clone(),
        rots=rots,
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
    )
    print("pure")
    pure_out = func_timer(run_pure)(
        equi=torch_imgs.clone(),
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
        numpy_out = torch.from_numpy(numpy_out)

        assert numpy_out.shape == pure_out.shape == native_out.shape
        assert (
            numpy_out.dtype == pure_out.dtype == native_out.dtype == torch_dtype
        )

        # quantitative
        print()
        print(">>> compare native and numpy")
        is_close, r_close, err_mse, err_mae = get_metrics(
            native_out, numpy_out, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        print()
        print(">>> compare native and pure")
        is_close, r_close, err_mse, err_mae = get_metrics(
            native_out, pure_out, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        print()
        print(">>> compare pure and numpy")
        is_close, r_close, err_mse, err_mae = get_metrics(
            pure_out, numpy_out, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        if save_outputs:
            # qualitative
            # save the outputs and see the images
            for b in range(bs):
                save(
                    numpy_out[b],
                    os.path.join(
                        SAVE_ROOT, f"out_cpu_numpy_{cube_format}_{b}.jpg"
                    ),
                )
                save(
                    pure_out[b],
                    os.path.join(
                        SAVE_ROOT, f"out_cpu_pure_{cube_format}_{b}.jpg"
                    ),
                )
                save(
                    native_out[b],
                    os.path.join(
                        SAVE_ROOT, f"out_cpu_native_{cube_format}_{b}.jpg"
                    ),
                )

    elif cube_format == "list":
        # output format is List[List[np.ndarray]]
        # order ["F", "R", "B", "L", "U", "D"]
        assert isinstance(native_out, list)
        for b in range(bs):
            assert isinstance(native_out[b], list)
            for i, face in enumerate(["F", "R", "B", "L", "U", "D"]):
                print()
                print(f">>> Testing batch: {b}, face: {face}")
                _numpy_out = torch.from_numpy(numpy_out[b][i])
                _native_out = native_out[b][i]
                _pure_out = pure_out[b][i]

                assert torch.is_tensor(_numpy_out)
                assert torch.is_tensor(_native_out)
                assert torch.is_tensor(_pure_out)

                assert _numpy_out.shape == _native_out.shape == _pure_out.shape
                assert (
                    _numpy_out.dtype
                    == _native_out.dtype
                    == _pure_out.dtype
                    == torch_dtype
                )

                # quantitative
                print()
                print(">>> compare native and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare native and pure")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _pure_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare pure and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _pure_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                save(
                    _numpy_out,
                    os.path.join(
                        SAVE_ROOT, f"out_cpu_numpy_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _native_out,
                    os.path.join(
                        SAVE_ROOT,
                        f"out_cpu_native_{cube_format}_{b}_{face}.jpg",
                    ),
                )
                save(
                    _pure_out,
                    os.path.join(
                        SAVE_ROOT, f"out_cpu_pure_{cube_format}_{b}_{face}.jpg"
                    ),
                )

    elif cube_format == "dict":
        # output format is List[Dict[str, np.ndarray]]
        # order ["F", "R", "B", "L", "U", "D"]
        assert isinstance(native_out, list)
        for b in range(bs):
            assert isinstance(native_out[b], dict)
            for (face,) in ["F", "R", "B", "L", "U", "D"]:
                print()
                print(f">>> Testing batch: {b}, face: {face}")
                _numpy_out = torch.from_numpy(numpy_out[b][face])
                _native_out = native_out[b][face]
                _pure_out = pure_out[b][face]

                assert torch.is_tensor(_numpy_out)
                assert torch.is_tensor(_native_out)
                assert torch.is_tensor(_pure_out)

                assert _numpy_out.shape == _native_out.shape == _pure_out.shape
                assert (
                    _numpy_out.dtype
                    == _native_out.dtype
                    == _pure_out.dtype
                    == torch_dtype
                )

                # quantitative
                print()
                print(">>> compare native and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare native and pure")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _pure_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare pure and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _pure_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                save(
                    _numpy_out,
                    os.path.join(
                        SAVE_ROOT, f"out_cpu_numpy_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _native_out,
                    os.path.join(
                        SAVE_ROOT,
                        f"out_cpu_native_{cube_format}_{b}_{face}.jpg",
                    ),
                )
                save(
                    _pure_out,
                    os.path.join(
                        SAVE_ROOT, f"out_cpu_pure_{cube_format}_{b}_{face}.jpg"
                    ),
                )

    else:
        raise ValueError


def bench_gpu(
    bs: int,
    z_down: bool,
    mode: str,
    w_face: int,
    cube_format: str,
    dtype: np.dtype = np.dtype(np.float32),
    torch_dtype: torch.dtype = torch.float32,
    rotation: str = "forward",
    save_outputs: bool = False,
) -> None:
    device = torch.device("cuda")
    assert torch_dtype in (torch.float16, torch.float32, torch.float64)

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

    numpy_img = get_numpy_img(dtype=dtype)
    torch_img = get_torch_img(dtype=torch_dtype)

    numpy_imgs = make_batch(numpy_img, bs=bs)
    torch_imgs = make_batch(torch_img, bs=bs)

    # generate rotation parameters
    if rotation == "forward":
        rots = create_rots(bs=bs)
    elif rotation == "pitch":
        rots = create_rots_pitch(bs=bs)
    elif rotation == "yaw":
        rots = create_rots_yaw(bs=bs)
    else:
        raise ValueError

    print("numpy:")
    numpy_out = func_timer(run_numpy)(
        equi=numpy_imgs,
        rots=rots,
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
    )
    print("native")
    native_out = func_timer(run_native)(
        equi=torch_imgs.clone().to(device),
        rots=rots,
        w_face=w_face,
        cube_format=cube_format,
        z_down=z_down,
        mode=mode,
    )
    print("pure")
    pure_out = func_timer(run_pure)(
        equi=torch_imgs.clone().to(device),
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
        numpy_out = torch.from_numpy(numpy_out).type(torch_dtype).to(device)

        assert numpy_out.shape == pure_out.shape == native_out.shape
        assert (
            numpy_out.dtype == pure_out.dtype == native_out.dtype == torch_dtype
        )

        # quantitative
        print()
        print(">>> compare native and numpy")
        is_close, r_close, err_mse, err_mae = get_metrics(
            native_out, numpy_out, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        print()
        print(">>> compare native and pure")
        is_close, r_close, err_mse, err_mae = get_metrics(
            native_out, pure_out, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        print()
        print(">>> compare pure and numpy")
        is_close, r_close, err_mse, err_mae = get_metrics(
            pure_out, numpy_out, rtol=rtol, atol=atol
        )
        print("close?", is_close)
        print("how many closes?", r_close)
        print("MSE", err_mse)
        print("MAE", err_mae)

        if save_outputs:
            # qualitative
            # save the outputs and see the images
            for b in range(bs):
                save(
                    numpy_out[b],
                    os.path.join(
                        SAVE_ROOT, f"out_gpu_numpy_{cube_format}_{b}.jpg"
                    ),
                )
                save(
                    pure_out[b],
                    os.path.join(
                        SAVE_ROOT, f"out_gpu_pure_{cube_format}_{b}.jpg"
                    ),
                )
                save(
                    native_out[b],
                    os.path.join(
                        SAVE_ROOT, f"out_gpu_native_{cube_format}_{b}.jpg"
                    ),
                )

    elif cube_format == "list":
        # output format is List[List[np.ndarray]]
        # order ["F", "R", "B", "L", "U", "D"]
        assert isinstance(native_out, list)
        for b in range(bs):
            assert isinstance(native_out[b], list)
            for i, face in enumerate(["F", "R", "B", "L", "U", "D"]):
                print()
                print(f">>> Testing batch: {b}, face: {face}")
                _numpy_out = (
                    torch.from_numpy(numpy_out[b][i])
                    .type(torch_dtype)
                    .to(device)
                )
                _native_out = native_out[b][i]
                _pure_out = pure_out[b][i]

                assert torch.is_tensor(_numpy_out)
                assert torch.is_tensor(_native_out)
                assert torch.is_tensor(_pure_out)

                assert _numpy_out.shape == _native_out.shape == _pure_out.shape
                assert (
                    _numpy_out.dtype
                    == _native_out.dtype
                    == _pure_out.dtype
                    == torch_dtype
                )

                # quantitative
                print()
                print(">>> compare native and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare native and pure")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _pure_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare pure and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _pure_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                save(
                    _numpy_out,
                    os.path.join(
                        SAVE_ROOT, f"out_gpu_numpy_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _native_out,
                    os.path.join(
                        SAVE_ROOT,
                        f"out_gpu_native_{cube_format}_{b}_{face}.jpg",
                    ),
                )
                save(
                    _pure_out,
                    os.path.join(
                        SAVE_ROOT, f"out_gpu_pure_{cube_format}_{b}_{face}.jpg"
                    ),
                )

    elif cube_format == "dict":
        # output format is List[Dict[str, np.ndarray]]
        # order ["F", "R", "B", "L", "U", "D"]
        assert isinstance(native_out, list)
        for b in range(bs):
            assert isinstance(native_out[b], dict)
            for (face,) in ["F", "R", "B", "L", "U", "D"]:
                print()
                print(f">>> Testing batch: {b}, face: {face}")
                _numpy_out = (
                    torch.from_numpy(numpy_out[b][face])
                    .type(torch_dtype)
                    .to(device)
                )
                _native_out = native_out[b][face]
                _pure_out = pure_out[b][face]

                assert torch.is_tensor(_numpy_out)
                assert torch.is_tensor(_native_out)
                assert torch.is_tensor(_pure_out)

                assert _numpy_out.shape == _native_out.shape == _pure_out.shape
                assert (
                    _numpy_out.dtype
                    == _native_out.dtype
                    == _pure_out.dtype
                    == torch_dtype
                )

                # quantitative
                print()
                print(">>> compare native and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare native and pure")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _native_out, _pure_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                print()
                print(">>> compare pure and numpy")
                is_close, r_close, err_mse, err_mae = get_metrics(
                    _pure_out, _numpy_out, rtol=rtol, atol=atol
                )
                print("close?", is_close)
                print("how many closes?", r_close)
                print("MSE", err_mse)
                print("MAE", err_mae)

                save(
                    _numpy_out,
                    os.path.join(
                        SAVE_ROOT, f"out_gpu_numpy_{cube_format}_{b}_{face}.jpg"
                    ),
                )
                save(
                    _native_out,
                    os.path.join(
                        SAVE_ROOT,
                        f"out_gpu_native_{cube_format}_{b}_{face}.jpg",
                    ),
                )
                save(
                    _pure_out,
                    os.path.join(
                        SAVE_ROOT, f"out_gpu_pure_{cube_format}_{b}_{face}.jpg"
                    ),
                )

    else:
        raise ValueError


if __name__ == "__main__":
    # parameters
    save_outputs = True
    rotation = "pitch"  # ('forward', 'pitch', 'yaw')

    # variables
    bs = 8
    w_face = 256
    cube_format = "dice"  # ('horizon', 'list', 'dict', 'dice')
    dtype = np.dtype(np.float32)
    z_down = True
    mode = "bilinear"

    torch_dtype = torch.float32

    bench_cpu(
        bs=bs,
        z_down=z_down,
        mode=mode,
        w_face=w_face,
        cube_format=cube_format,
        dtype=dtype,
        rotation=rotation,
        save_outputs=save_outputs,
    )

    bench_gpu(
        bs=bs,
        z_down=z_down,
        mode=mode,
        w_face=w_face,
        cube_format=cube_format,
        dtype=dtype,
        rotation=rotation,
        save_outputs=save_outputs,
    )
