#!/usr/bin/env python3

from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np
import torch

from equilib.grid_sample import torch_grid_sample
from equilib.torch_utils import (
    create_global2camera_rotation_matrix,
    create_grid,
    create_intrinsic_matrix,
    create_rotation_matrices,
    get_device,
    pi,
)
from einops import repeat


@lru_cache(maxsize=128)
def ucm_unproject_grid(
    height: int,
    width: int,
    fx: float | torch.Tensor,
    fy: float | torch.Tensor,
    cx: float | torch.Tensor,
    cy: float | torch.Tensor,
    xi: float | torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
    y_down: bool = True,
) -> torch.Tensor:
    """
    使用 create_grid 生成像素坐标，再用 UCM 模型严格反投影到相机系方向。
    - 所有参数为 float → 返回 [H, W, 3]
    - 只要有任何参数为 Tensor[B] → 返回 [B, H, W, 3]（即使 B==1）
    """
    # 保存原始输入用于判断是否全是 float
    fx_, fy_, cx_, cy_, xi_ = fx, fy, cx, cy, xi

    def to_tensor1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)  # float → [1]

    fx, fy, cx, cy, xi = map(to_tensor1d, (fx, fy, cx, cy, xi))
    B = fx.shape[0]

    # 创建网格：[B, H, W, 2]
    grid = create_grid(height=height, width=width, batch=B, dtype=dtype, device=device)
    u = grid[..., 0]
    v = grid[..., 1]

    fx = fx[:, None, None]
    fy = fy[:, None, None]
    cx = cx[:, None, None]
    cy = cy[:, None, None]
    xi = xi[:, None, None]

    x = (u - cx) / fx
    y = (v - cy) / fy
    if not y_down:
        y = -y

    r2 = x * x + y * y
    alpha = xi + torch.sqrt(1 + (1 - xi * xi) * r2)
    gamma = alpha / (1 + r2)

    X = gamma * x
    Y = gamma * y
    Z = gamma - xi

    d_cam = torch.stack([X, Y, Z], dim=-1)  # [B, H, W, 3]

    # ✅ 只有当所有输入都是 float 时才 squeeze
    is_scalar_input = all(not torch.is_tensor(p) for p in (fx_, fy_, cx_, cy_, xi_))
    if is_scalar_input:
        return d_cam[0]  # → [H, W, 3]
    else:
        return d_cam       # → [B, H, W, 3]


@lru_cache(maxsize=128)
def create_cam2global_matrix(
    height: int,
    width: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    K = create_intrinsic_matrix(
        height=height,
        width=width,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )
    g2c_rot = create_global2camera_rotation_matrix(dtype=dtype, device=device)

    return g2c_rot @ K.inverse()


def prep_matrices(
    height: int,
    width: int,
    batch: int,
    fov_x: float,
    skew: float = 0.0,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    m = create_grid(
        height=height, width=width, batch=batch, dtype=dtype, device=device
    )
    m = m.unsqueeze(-1)
    G = create_cam2global_matrix(
        height=height,
        width=width,
        fov_x=fov_x,
        skew=skew,
        dtype=dtype,
        device=device,
    )

    return m, G


def matmul(m: torch.Tensor, G: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    M = torch.matmul(torch.matmul(R, G)[:, None, None, ...], m)
    M = M.squeeze(-1)

    return M

def rotate_rays(d_cam: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    d_cam: [B, H, W, 3] 相机坐标系下的方向向量
    R:     [B, 3, 3]    (由 yaw/pitch/roll 构造的旋转，与你现在 create_rotation_matrices 一致)
    返回:  [B, H, W, 3]  旋转到“全景/全局”坐标系下的方向
    """
    # 把方向当作列向量右乘，或当作行向量左乘，保持一致即可
    # 这里用 (B,3,3) @ (B,H,W,3,1) 的广播方式：
    B, H, W, _ = d_cam.shape
    d = d_cam.view(B, H * W, 3).transpose(1, 2)   # [B, 3, HW]
    d_rot = torch.bmm(R, d)                        # [B, 3, HW]
    d_rot = d_rot.transpose(1, 2).view(B, H, W, 3) # [B, H, W, 3]
    return d_rot


def convert_grid_ucm(M: torch.Tensor, h_equi: int, w_equi: int, method: str = "robust") -> torch.Tensor:
    """
    M: [B, H, W, 3]  全景/全局坐标系下的方向向量
    返回: [B, 2, H, W]  (j,i) 网格（与你原先的排列一致）
    """
    X, Y, Z = M[..., 0], M[..., 1], M[..., 2]
    # 球坐标
    # phi ∈ [-pi/2, pi/2], theta ∈ [-pi, pi]
    norm = torch.sqrt(X * X + Y * Y + Z * Z)
    phi = torch.asin(torch.clamp(Z / norm, -1.0, 1.0))
    theta = torch.atan2(Y, X)

    # equirectangular 像素坐标
    ui = (theta - pi) * w_equi / (2 * pi) + 0.5
    uj = (phi   - pi/2) * h_equi / pi      + 0.5

    if method == "robust":
        ui = ui % w_equi
        uj = uj % h_equi
    elif method == "faster":
        ui = torch.where(ui < 0, ui + w_equi, ui)
        ui = torch.where(ui >= w_equi, ui - w_equi, ui)
        uj = torch.where(uj < 0, uj + h_equi, uj)
        uj = torch.where(uj >= h_equi, uj - h_equi, uj)
    else:
        raise ValueError(f"ERR: {method} is not supported")

    grid = torch.stack((uj, ui), dim=1)  # [B, 2, H, W]  与原先 stack((uj,ui), dim=-3) 等价
    return grid


def convert_grid(
    M: torch.Tensor, h_equi: int, w_equi: int, method: str = "robust"
) -> torch.Tensor:
    # convert to rotation
    phi = torch.asin(M[..., 2] / torch.norm(M, dim=-1))
    theta = torch.atan2(M[..., 1], M[..., 0])

    if method == "robust":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui %= w_equi
        uj %= h_equi
    elif method == "faster":
        ui = (theta - pi) * w_equi / (2 * pi)
        uj = (phi - pi / 2) * h_equi / pi
        ui += 0.5
        uj += 0.5
        ui = torch.where(ui < 0, ui + w_equi, ui)
        ui = torch.where(ui >= w_equi, ui - w_equi, ui)
        uj = torch.where(uj < 0, uj + h_equi, uj)
        uj = torch.where(uj >= h_equi, uj - h_equi, uj)
    else:
        raise ValueError(f"ERR: {method} is not supported")

    # stack the pixel maps into a grid
    grid = torch.stack((uj, ui), dim=-3)

    return grid


def minfocal(u0, v0, xi, xref=1, yref=1):
    """compute the minimum focal for the image to be catadioptric given xi"""
    value = -(1 - xi * xi) * ((xref - u0) * (xref - u0) + (yref - v0) * (yref - v0))

    if value < 0:
        return 0
    else:
        return np.sqrt(value) * 1.0001

def diskradius(xi, f, eps=1e-6):  # compute the disk radius when the image is catadioptric
    if abs(1 - xi * xi) < eps or xi <= 1:
        return np.inf
    return np.sqrt(-(f * f) / (1 - xi * xi))


def run(
    equi: torch.Tensor,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
    mode: str,
    clip_output: bool = True,
    backend: str = "native",
    xi: float = 0.0,
) -> torch.Tensor:
    """Run Equi2Pers

    params:
    - equi (torch.Tensor): 4 dims (b, c, h, w)
    - rots (List[dict]): dict of ('yaw', 'pitch', 'roll')
    - height, width (int): height and width of perspective view
    - fov_x (float): fov of horizontal axis in degrees
    - skew (float): skew of the camera
    - z_down (bool)
    - mode (str): sampling mode for grid_sample
    - backend (str): backend of torch `grid_sample` (default: `native`)
    - xi (float): xi parameter for catadioptric camera model (default: 0.0)

    returns:
    - out (torch.Tensor)

    NOTE: `backend` can be either `native` or `pure`

    """

    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        rots
    ), f"ERR: length of equi and rot differs: {len(equi)} vs {len(rots)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if equi.device.type == "cuda":
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )
    if backend == "native" and equi_dtype == torch.uint8:
        # FIXME: hacky way of dealing with images that are uint8 when using
        # torch.grid_sample
        equi = equi.type(torch.float32)

    bs, c, h_equi, w_equi = equi.shape
    img_device = get_device(equi)

    # initialize output tensor
    if backend == "native":
        # NOTE: don't need to initialize for `native`
        out = None
    else:
        out = torch.empty(
            (bs, c, height, width), dtype=dtype, device=img_device
        )

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if equi.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # === 1. 根据 fov_x + xi 计算 fx, fy ===
    theta = torch.deg2rad(torch.tensor(fov_x/2, dtype=tmp_dtype))
    fx = (width / 2) * (torch.cos(theta) + xi) / torch.sin(theta)
    fx = fx.item()   # 变成 python float 便于后面广播
    fy = fx          # 若像素是方形可直接等于 fx

    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5

    # === 2. 用 UCM 严格反投影像素网格到相机系 ===
    d_cam = ucm_unproject_grid(
        height=height, width=width,
        fx=fx, fy=fy, cx=cx, cy=cy,
        xi=xi, dtype=tmp_dtype, device=tmp_device,
        y_down=True  # 与现有坐标系一致
    )  # [H,W,3]
    d_cam = repeat(d_cam, "H W C -> B H W C", B=bs)

    # === 3. 旋转方向向量 ===
    g2c = create_global2camera_rotation_matrix(dtype=tmp_dtype, device=tmp_device)
    R = create_rotation_matrices(rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device)
    R_total = torch.matmul(R, g2c.unsqueeze(0).expand_as(R))  # [B,3,3]
    M = rotate_rays(d_cam, R_total)  # [B,H,W,3]

    # === 4. 转换成全景图采样 grid ===
    grid = convert_grid_ucm(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    # if backend == "native":
    #     grid = grid.to(img_device)
    # FIXME: putting `grid` to device since `pure`'s bilinear interpolation requires it
    # FIXME: better way of forcing `grid` to be the same dtype?
    if equi.dtype != grid.dtype:
        grid = grid.type(equi.dtype)
    if equi.device != grid.device:
        grid = grid.to(equi.device)

    # grid sample
    out = torch_grid_sample(
        img=equi,
        grid=grid,
        out=out,  # FIXME: is this necessary?
        mode=mode,
        backend=backend,
    )

    # === 在 grid_sample 之后加遮罩 ===
    fmin = minfocal(width/2, height/2, xi)  # 这里假设光心在 (cx, cy) = (W/2, H/2)
    if fx < fmin:
        # 使用 create_grid 创建像素坐标
        grid_mask = create_grid(height=height, width=width, batch=None,
                                dtype=out.dtype, device=out.device)  # [H,W,3]
        # grid[...,0]=u (x), grid[...,1]=v (y)
        u = grid_mask[..., 0]
        v = grid_mask[..., 1]

        # 以图像中心为圆心计算半径
        cx = (width - 1) * 0.5
        cy = (height - 1) * 0.5
        r = torch.sqrt((u - cx) ** 2 + (v - cy) ** 2)

        mask = (r < diskradius(xi, fx)).float()  # 内部=1，外部=0
        # 扩展维度与 out 匹配
        if out.ndim == 4:      # [B,C,H,W]
            mask = mask.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        elif out.ndim == 3:    # [C,H,W]
            mask = mask.unsqueeze(0)
        else:                  # [H,W]
            pass
        out = out * mask

    # NOTE: we assume that `out` keeps it's dtype

    out = (
        out.type(equi_dtype)
        if equi_dtype == torch.uint8 or not clip_output
        else torch.clip(out, torch.min(equi), torch.max(equi))
    )

    return out


def get_bounding_fov(
    equi: torch.Tensor,
    rots: List[Dict[str, float]],
    height: int,
    width: int,
    fov_x: float,
    skew: float,
    z_down: bool,
) -> torch.Tensor:
    assert (
        len(equi.shape) == 4
    ), f"ERR: input `equi` should be 4-dim (b, c, h, w), but got {len(equi.shape)}"
    assert len(equi) == len(
        rots
    ), f"ERR: length of equi and rot differs: {len(equi)} vs {len(rots)}"

    equi_dtype = equi.dtype
    assert equi_dtype in (
        torch.uint8,
        torch.float16,
        torch.float32,
        torch.float64,
    ), (
        f"ERR: input equirectangular image has dtype of {equi_dtype}which is\n"
        f"incompatible: try {(torch.uint8, torch.float16, torch.float32, torch.float64)}"
    )

    # NOTE: we don't want to use uint8 as output array's dtype yet since
    # floating point calculations (matmul, etc) are faster
    # NOTE: we are also assuming that uint8 is in range of 0-255 (obviously)
    # and float is in range of 0.0-1.0; later we will refine it
    # NOTE: for the sake of consistency, we will try to use the same dtype as equi
    if equi.device.type == "cuda":
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float16, torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float16, torch.float32, torch.float64)}"
        )
    else:
        # NOTE: for cpu, it can't use half-precision
        dtype = torch.float32 if equi_dtype == torch.uint8 else equi_dtype
        assert dtype in (torch.float32, torch.float64), (
            f"ERR: argument `dtype` is {dtype} which is incompatible:\n"
            f"try {(torch.float32, torch.float64)}"
        )

    bs, c, h_equi, w_equi = equi.shape

    # FIXME: for now, calculate the grid in cpu
    # I need to benchmark performance of it when grid is created on cuda
    tmp_device = torch.device("cpu")
    if equi.device.type == "cuda" and dtype == torch.float16:
        tmp_dtype = torch.float32
    else:
        tmp_dtype = dtype

    # create grid and transform matrix
    m, G = prep_matrices(
        height=height,
        width=width,
        batch=bs,
        fov_x=fov_x,
        skew=skew,
        dtype=tmp_dtype,
        device=tmp_device,
    )

    # create batched rotation matrices
    R = create_rotation_matrices(
        rots=rots, z_down=z_down, dtype=tmp_dtype, device=tmp_device
    )

    # rotate and transform the grid
    M = matmul(m, G, R)

    # create a pixel map grid
    grid = convert_grid(M=M, h_equi=h_equi, w_equi=w_equi, method="robust")

    bboxs = []

    # top row
    for out_x in range(width):
        bboxs.append(grid[:, :, 0, out_x])

    # right column
    for out_y in range(height):
        if out_y > 0:  # exclude first
            bboxs.append(grid[:, :, out_y, width - 1])

    # bottom row
    for out_x in range(width - 2, 0, -1):
        bboxs.append(grid[:, :, height - 1, out_x])

    # left column
    for out_y in range(height - 1, 0, -1):
        bboxs.append(grid[:, :, out_y, 0])

    assert len(bboxs) == width * 2 + (height - 2) * 2

    bboxs = torch.stack(bboxs, dim=1)

    bboxs = bboxs.numpy()
    bboxs = np.rint(bboxs).astype(np.int64)

    return bboxs
