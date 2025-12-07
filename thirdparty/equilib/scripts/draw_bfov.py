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

from equilib import Equi2Pers

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


def draw_lines(
    equi: np.ndarray, points: np.ndarray, to_cv2: bool = False
) -> np.ndarray:
    if to_cv2:
        equi = cv2.cvtColor(equi, cv2.COLOR_RGB2BGR)

    points = points.tolist()
    points = [(x, y) for y, x in points]

    for index, point in enumerate(points):
        if index == len(points) - 1:
            next_point = points[0]
        else:
            next_point = points[index + 1]

        if (
            abs(point[0] - next_point[0]) < 100
            and abs(point[1] - next_point[1]) < 100
        ):
            cv2.line(equi, point, next_point, (0, 255, 0), thickness=2)

    return equi


def test_video(
    path: str, h_pers: int = 480, w_pers: int = 640, fov_x: float = 90.0
) -> None:
    """Test video"""
    # Rotation:
    pi = np.pi
    inc = pi / 180
    roll = 0  # -pi/2 < a < pi/2
    pitch = 0  # -pi < b < pi
    yaw = 0

    # Initialize equi2pers
    equi2pers = Equi2Pers(
        height=h_pers, width=w_pers, fov_x=fov_x, mode="bilinear"
    )

    times = []
    cap = cv2.VideoCapture(path)

    while cap.isOpened():
        ret, frame = cap.read()

        rot = {"roll": roll, "pitch": pitch, "yaw": yaw}

        if not ret:
            break

        s = time.time()
        equi_img = preprocess(frame, is_cv2=True)
        pers_img = equi2pers(equi=equi_img, rots=rot)
        pers_img = np.transpose(pers_img, (1, 2, 0))
        pers_img = cv2.cvtColor(pers_img, cv2.COLOR_RGB2BGR)
        e = time.time()
        times.append(e - s)

        # cv2.imshow("video", pers)

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
    save_path = osp.join(RESULT_PATH, "times_equi2pers_numpy_video.png")
    plt.savefig(save_path)


def test_image(
    path: str, h_pers: int = 480, w_pers: int = 640, fov_x: float = 90.0
) -> None:
    """Test single image"""
    # Rotation:
    rot = {
        "roll": 0,  #
        "pitch": np.pi / 2,  # vertical
        "yaw": np.pi,  # horizontal
    }

    # Initialize equi2pers
    equi2pers = Equi2Pers(
        height=h_pers, width=w_pers, fov_x=fov_x, mode="bilinear"
    )

    # Open Image
    equi = Image.open(path)
    equi = preprocess(equi)

    points = equi2pers.get_bounding_fov(equi=equi, rots=rot)
    equi = np.transpose(equi, (1, 2, 0))
    out_equi = draw_lines(equi, points)

    out_equi = Image.fromarray(out_equi)

    save_path = osp.join(RESULT_PATH, "test_bounding_box.jpg")
    out_equi.save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--data", nargs="?", default=None, type=str)
    args = parser.parse_args()

    # Variables:
    h_pers = 480
    w_pers = 640
    fov_x = 90

    data_path = args.data
    if args.video:
        if data_path is None:
            data_path = osp.join(DATA_PATH, "R0010028_er_30.MP4")
        assert osp.exists(data_path)
        test_video(data_path, h_pers, w_pers, fov_x)
    else:
        if data_path is None:
            data_path = osp.join(DATA_PATH, "equi.jpg")
        assert osp.exists(data_path)
        test_image(data_path, h_pers, w_pers, fov_x)


if __name__ == "__main__":
    main()
