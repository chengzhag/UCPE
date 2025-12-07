#!/usr/bin/env python3

import argparse
import os.path as osp
import time
from typing import Union

import cv2

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from PIL import Image

from equilib import Equi2Equi

matplotlib.use("Agg")

RESULT_PATH = "./results"
DATA_PATH = "./data"


def preprocess(
    img: Union[np.ndarray, Image.Image], is_cv2: bool = False
) -> np.ndarray:
    """Preprocesses image"""
    if isinstance(img, np.ndarray) and is_cv2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(img, Image.Image):
        # Sometimes images are RGBA
        img = img.convert("RGB")
        img = np.asarray(img)
    assert len(img.shape) == 3, "input must be dim=3"
    assert img.shape[-1] == 3, "input must be HWC"
    img = np.transpose(img, (2, 0, 1))
    return img


def test_video(path: str) -> None:
    """Test video"""
    # Rotation:
    pi = np.pi
    inc = pi / 180
    roll = 0  # -pi/2 < a < pi/2
    pitch = 0  # -pi < b < pi
    yaw = 0

    # Initialize equi2equi
    equi2equi = Equi2Equi(height=320, width=640, mode="bilinear")

    times = []
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()

        rot = {"roll": roll, "pitch": pitch, "yaw": yaw}

        if not ret:
            break

        s = time.time()
        src_img = preprocess(frame, is_cv2=True)
        out_img = equi2equi(src=src_img, rots=rot)
        out_img = np.transpose(out_img, (1, 2, 0))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
        e = time.time()
        times.append(e - s)

        # cv2.imshow("video", out_img)

        # change direction `wasd` or exit with `q`
        k = cv2.waitKey(1)
        if k == ord("q"):
            break
        if k == ord("w"):
            roll -= inc
        if k == ord("s"):
            roll += inc
        if k == ord("a"):
            pitch += inc
        if k == ord("d"):
            pitch -= inc

    cap.release()
    cv2.destroyAllWindows()

    print(sum(times) / len(times))
    x_axis = list(range(len(times)))
    plt.plot(x_axis, times)
    save_path = osp.join(RESULT_PATH, "times_equi2equi_numpy_video.png")
    plt.savefig(save_path)


def test_image(path: str) -> None:
    """Test single image"""
    # Rotation:
    rot = {"roll": 0, "pitch": 0, "yaw": np.pi / 4}  # vertical, horizontal

    # Initialize equi2equi
    equi2equi = Equi2Equi(height=320, width=640, mode="bilinear")

    # Open Image
    src_img = Image.open(path)
    src_img = preprocess(src_img)

    out_img = equi2equi(src=src_img, rots=rot)

    out_img = np.transpose(out_img, (1, 2, 0))
    out_img = Image.fromarray(out_img)

    out_path = osp.join(RESULT_PATH, "output_equi2equi_numpy_image.jpg")
    out_img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--data", nargs="?", default=None, type=str)
    args = parser.parse_args()

    data_path = args.data
    if args.video:
        if data_path is None:
            data_path = osp.join(DATA_PATH, "R0010028_er_30.MP4")
        assert osp.exists(data_path)
        test_video(data_path)
    else:
        if data_path is None:
            data_path = osp.join(DATA_PATH, "equi.jpg")
        assert osp.exists(data_path)
        test_image(data_path)


if __name__ == "__main__":
    main()
