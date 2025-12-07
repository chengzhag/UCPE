import torch
from torch import nn
from equilib.equi2pers.torch import ucm_unproject_grid  # 确保正确 import
from thirdparty.prope.torch import PropeDotProductAttention
from diffsynth.models.wan_video_dit import flash_attention
from einops import rearrange, repeat, einsum
import torch.nn.functional as F
from typing import Tuple
from equilib.torch_utils import create_grid


def patch_dit(pipe, method, height, width, attn_compress=1, adaptation_method="parallel"):
    keywords = []
    if method.startswith("recam"):
        if method == "recammaster":
            emb_dim = 14
        elif method == "recam_plucker":
            emb_dim = 6
        else:
            raise ValueError(f"Unknown method: {method}")

        dim = pipe.dit.blocks[0].self_attn.q.weight.shape[0]
        for block in pipe.dit.blocks:
            block.cam_encoder = nn.Linear(emb_dim, dim)
            block.projector = nn.Linear(dim, dim)
            block.cam_encoder.weight.data.zero_()
            block.cam_encoder.bias.data.zero_()
            block.projector.weight = nn.Parameter(torch.eye(dim))
            block.projector.bias = nn.Parameter(torch.zeros(dim))
        keywords.extend(["cam_encoder", "projector", "self_attn"])

    if method == "plucker":
        from diffsynth.models.wan_video_camera_controller import SimpleAdapter
        pipe.dit.control_adapter = SimpleAdapter(
            24,
            pipe.dit.dim,
            kernel_size=[2, 2],
            stride=[2, 2],
            downscale_factor=pipe.vae.upsampling_factor,
        )
        pipe.dit.control_adapter.conv.weight.data.zero_()
        pipe.dit.control_adapter.conv.bias.data.zero_()
        for block in pipe.dit.control_adapter.residual_blocks:
            block.conv2.weight.data.zero_()
            block.conv2.bias.data.zero_()
        keywords = "*"
    elif any(k in method for k in ("gta", "prope", "relray")):
        patch_factor = pipe.vae.upsampling_factor * 2
        patches_x = width // patch_factor
        patches_y = height // patch_factor

        if "abs" in method:
            if "absc2w" in method or "absray" in method:
                emb_dim = 12
            elif "absmap" in method:
                emb_dim = 3
            else:
                raise ValueError(f"Unknown method: {method}")
        else:
            emb_dim = None

        for block in pipe.dit.blocks:
            block.cam_self_attn = UcpeSelfAttention(
                pipe.dit.dim,
                pipe.dit.dim // attn_compress,
                block.num_heads // attn_compress,
                patches_x=patches_x,
                patches_y=patches_y,
                image_width=width,
                image_height=height,
                emb_dim=emb_dim,
                adaptation_method=adaptation_method,
            )
        keywords.append("cam_self_attn")

    pipe.dit.camera_condition = method
    return keywords


def enable_grad(pipe, keywords):
    pipe.eval()
    pipe.requires_grad_(False)
    if keywords == "*":
        pipe.dit.train()
        pipe.dit.requires_grad_(True)
    else:
        for name, module in pipe.dit.named_modules():
            if any(keyword in name for keyword in keywords):
                print(f"Trainable: {name}")
                module.train()
                module.requires_grad_(True)

    trainable_params = 0
    seen_params = set()
    for name, module in pipe.dit.named_modules():
        for param in module.parameters():
            if param.requires_grad and param not in seen_params:
                trainable_params += param.numel()
                seen_params.add(param)
    print(f"Total number of trainable parameters: {trainable_params}")


def compute_fx_from_fov_xi(
    x_fov: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    根据水平视场角 (x_fov) 和 UCM 参数 (xi) 计算相机焦距 fx。

    Args:
        x_fov: float 或 [B] Tensor，水平视场角（单位：度）
        xi: float 或 [B] Tensor，UCM 镜面参数
        width: 图像宽度（像素）
        device: torch.device
        dtype: torch.dtype

    Returns:
        fx: [B] Tensor，焦距（像素单位）
    """
    # --- 转为 Tensor ---
    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    x_fov = to_tensor_1d(x_fov)
    xi = to_tensor_1d(xi)

    # --- 自动广播 ---
    B = max(x_fov.shape[0], xi.shape[0])
    x_fov = x_fov.view(-1).expand(B)
    xi = xi.view(-1).expand(B)

    # --- 计算 fx ---
    theta = torch.deg2rad(0.5 * x_fov)
    eps = torch.finfo(dtype).eps
    denom = torch.sin(theta).clamp_min(eps)
    fx = (width * 0.5) * (torch.cos(theta) + xi) / denom
    return fx


def compute_fov_from_fx_xi(
    fx: torch.Tensor | float,
    xi: torch.Tensor | float,
    width: int,
    device="cpu",
    dtype=torch.float32,
):
    """
    根据 UCM 模型参数 fx, xi 计算水平 FOV（度）

    Args:
        fx: float 或 [B] Tensor, 焦距
        xi: float 或 [B] Tensor, UCM xi 参数
        width: 图像宽度
    Returns:
        x_fov: [B], 单位 degree
    """
    def to_tensor_1d(x):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype)
        return torch.tensor([x], dtype=dtype, device=device)

    fx  = to_tensor_1d(fx).view(-1)
    xi  = to_tensor_1d(xi).view(-1)
    B = max(fx.shape[0], xi.shape[0])
    fx  = fx.expand(B)
    xi  = xi.expand(B)

    # A = 2 fx / W
    A = 2.0 * fx / width

    # phi = arctan(1/A)
    phi = torch.atan(1.0 / A)

    # sin(theta - phi) = xi / sqrt(A^2 + 1)
    denom = torch.sqrt(A * A + 1.0)
    ratio = (xi / denom).clamp(-1.0, 1.0)
    theta = torch.asin(ratio) + phi

    # x_fov = 2 * theta (rad → deg)
    x_fov = torch.rad2deg(2.0 * theta)
    return x_fov


def ucm_unproject_grid_fov(
    x_fov: float | torch.Tensor,
    xi: float | torch.Tensor,
    height: int,
    width: int,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    计算每个样本的相机方向向量 (UCM model, 用视场角定义)。
    支持 float 或 [B] Tensor 的混合输入。
    - 若全为 float → 返回 [H, W, 3]
    - 若任意为 [B] → 返回 [B, H, W, 3]
    """
    # --- 判断是否 batched ---
    is_batched = any(torch.is_tensor(p) and p.ndim == 1 for p in [x_fov, xi])

    # --- 计算 fx, fy ---
    fx = compute_fx_from_fov_xi(x_fov, xi, width, device, dtype)
    fy = fx

    # --- 调用 ucm_unproject_grid ---
    d_cam = ucm_unproject_grid(
        height=height,
        width=width,
        fx=fx,
        fy=fy,
        cx=width / 2,
        cy=height / 2,
        xi=xi if torch.is_tensor(xi) else torch.tensor([xi], dtype=dtype, device=device),
        dtype=dtype,
        device=device,
        y_down=True,
    )

    # --- 输出 shape 控制 ---
    if not is_batched:
        d_cam = d_cam[0]  # [H, W, 3]

    return d_cam


def project_ucm_points_fov(X, Y, Z, x_fov, xi, height, width):
    """
    Project 3D points in camera frame to UCM image plane using fov-based intrinsics.

    Args:
        X, Y, Z: torch.Tensor [..., 3D coordinates in camera frame]
        x_fov: float or [B] —— horizontal field of view in degrees
        xi: float or [B] —— UCM mirror parameter
        height, width: target image dimensions

    Returns:
        du, dv: projected pixel coordinates [..., 2]
    """
    fx = compute_fx_from_fov_xi(x_fov, xi, width, X.device, X.dtype)
    fy = fx
    cx = width / 2
    cy = height / 2

    return project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi)


def project_ucm_points(X, Y, Z, fx, fy, cx, cy, xi):
    """
    Project 3D points in camera frame to UCM image plane.

    Args:
        X, Y, Z: torch.Tensor [..., 3D coordinates in camera frame]
        fx, fy, cx, cy: intrinsics (scalars or tensors)
        xi: UCM mirror parameter

    Returns:
        du, dv: projected pixel coordinates [..., 2]
    """
    r = torch.sqrt(X * X + Y * Y + Z * Z)
    alpha = Z + xi * r
    du = fx * (X / alpha) + cx
    dv = fy * (Y / alpha) + cy
    return du, dv


def ray_condition_ucm(
    x_fov,      # float or [B] —— same fov as used in equi2pers
    xi,        # float or [B] —— same xi as used in equi2pers
    pose,       # [B, V, 4, 4]
    height, width,      # target height, width
    device,
):
    """
    ✅ UCM-based Plücker embedding, output format: [B, V, H, W, 6]
    🔁 Internally uses your ucm_unproject_grid() for consistent ray geometry.
    
    Only required params:
        fov_x  (degree)
        xi
        c2w    (camera-to-world pose, same as your exported pose)
        H, W   (spatial resolution)
        device
    """

    d_cam = ucm_unproject_grid_fov(
        x_fov, xi, height, width, device, dtype=pose.dtype
    )
    d_cam = repeat(d_cam, "b ... -> b v ...", v=pose.shape[1])  # [B, V, H, W, 3]
    mask = d_cam.isnan().any(-1)

    # --- 4. Transform rays into world coordinates using c2w ---
    R = pose[..., :3, :3]      # [B, V, 3, 3]
    t = pose[..., :3, 3]       # [B, V, 3]

    d_world = torch.einsum("bvij,bvhwj->bvhwi", R.transpose(-1, -2), d_cam)  # [B,V,H,W,3]
    rays_o = t[..., None, None, :].expand_as(d_world)  # [B,V,H,W,3]

    # --- 5. Plücker coordinates: m = o × d ---
    m = torch.cross(rays_o, d_world, dim=-1)  # [B,V,H,W,3]

    # --- 6. Final concat: [m, d] → [B,V,H,W,6]
    plucker = torch.cat([m, d_world], dim=-1)
    plucker[mask] = 0.
    return plucker


def d_cam_to_angles(d_cam: torch.Tensor) -> torch.Tensor:
    """
    将方向向量 [x, y, z] 转换为 [azimuth, elevation]。
    坐标系：z前，x右，y下（符合 UCM 投影输出）
    
    输入: d_cam: [B, H, W, 3]
    输出: angles: [B, H, W, 2] — azimuth, elevation （单位: 弧度）
    """
    d_unit = F.normalize(d_cam, dim=-1)  # [B, H, W, 3]

    x = d_unit[..., 0]  # right
    y = d_unit[..., 1]  # down
    z = d_unit[..., 2]  # forward

    # yaw / azimuth: angle in xz-plane
    azimuth = torch.atan2(x, z)  # ∈ [-π, π]

    # pitch / elevation: angle above xz-plane
    elevation = -torch.asin(y)   # y 向下 → elevation = -asin(y)

    return torch.stack([azimuth, elevation], dim=-1)  # [B, H, W, 2]


def world_to_ray_mats(
    d_cam: torch.Tensor,  # [B, H, W, 3]
    c2w: torch.Tensor,    # [B, T, 4, 4]
) -> torch.Tensor:
    """
    构造每条 ray 的世界到 ray 局部坐标系的变换矩阵 world2ray。
    坐标系定义：
        - z: ray direction
        - x: cam_y × ray_dir
        - y: z × x
    返回:
        raymats: [B, T, H, W, 4, 4]
    """
    B, H, W, _ = d_cam.shape
    T = c2w.shape[1]
    device = d_cam.device
    dtype = d_cam.dtype

    # --- Expand ray dirs across frames ---
    # [B,H,W,3] -> [B,T,H,W,3]
    d_cam = repeat(d_cam, 'b h w c -> b t h w c', t=T)

    # extract camera R,t
    R_cam = c2w[..., :3, :3]       # [B,T,3,3]
    t_cam = c2w[..., :3, 3]        # [B,T,3]
    
    # --- d_world: rotate ray directions into world ---
    d_world = einsum(R_cam, d_cam, 'b t i j, b t h w j -> b t h w i')

    # camera y-axis from each view
    cam_y = R_cam[..., :, 1]       # [B,T,3]
    cam_y = repeat(cam_y, 'b t c -> b t h w c', h=H, w=W)

    # === Construct orthonormal ray-local axes ===
    z_ray = F.normalize(d_world, dim=-1, eps=1e-6)
    x_ray = torch.cross(cam_y, z_ray, dim=-1)
    x_ray = F.normalize(x_ray, dim=-1, eps=1e-6)
    y_ray = torch.cross(z_ray, x_ray, dim=-1)
    y_ray = F.normalize(y_ray, dim=-1, eps=1e-6)
    
    # local->world rotation
    R_l2w = torch.stack([x_ray, y_ray, z_ray], dim=-1)  # [B,T,H,W,3,3]

    # world->local rotation (transpose)
    R_w2l = rearrange(R_l2w, 'b t h w i j -> b t h w j i')  # ✅

    # broadcast camera center
    t_world = repeat(t_cam, 'b t c -> b t h w c', h=H, w=W)

    # world->local translation
    t_w2l = -einsum(R_w2l, t_world, 'b t h w i j, b t h w j -> b t h w i')

    # assemble transform matrix
    raymats = torch.zeros(B, T, H, W, 4, 4, device=device, dtype=dtype)
    raymats[..., :3, :3] = R_w2l
    raymats[..., :3, 3] = t_w2l
    raymats[..., 3, 3] = 1.0

    # NaN handling
    mask = torch.isnan(d_world).any(-1)
    raymats[mask] = torch.eye(4, device=device, dtype=dtype)

    return raymats


def rope_precompute_coeffs(
    positions: torch.Tensor,  # [B, H, W]
    freq_base: float,
    freq_scale: float,
    feat_dim: int,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:  # [B, 1, H*W, D], [B, 1, H*W, D]
    """
    批量计算每个样本对应的 RoPE 系数（cos, sin），用于 patch ray angle embedding。
    输入:
        positions: [B, H, W] —— 每个 patch 的 azimuth 或 elevation（单位弧度）
    输出:
        cos: [B, 1, H*W, feat_dim//2]
        sin: [B, 1, H*W, feat_dim//2]
    """
    # 对 NaN 角度 patch，输出 cos=1, sin=0，即不做旋转，等价于保留原始 token 表示
    mask = positions.isnan()
    positions = positions.clone()
    positions[mask] = 0.0

    B, H, W = positions.shape
    positions_flat = positions.view(B, H * W)  # [B, HW]
    num_freqs = feat_dim // 2

    freqs = freq_scale * (
        freq_base ** (
            -torch.arange(num_freqs, device=positions.device)[None, :]
            / num_freqs
        )  # [1, D]
    )  # [1, D]

    # Expand for batch & positions
    angles = positions_flat[..., None] * freqs[None, :, :]  # [B, HW, D]
    angles = angles.view(B, 1, H * W, num_freqs)

    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def compute_up_lat_map(
    R: torch.Tensor,
    x_fov: torch.Tensor,
    xi: torch.Tensor,
    height: int,
    width: int,
    device: torch.device = torch.device("cpu"),
    delta: float = 0.1,
):
    """
    计算 up_map 和 lat_map。
    
    Args:
        R: [B, T, 3, 3] 相机 c2w 旋转矩阵
        x_fov: [B] 或 [B,T] 水平视场角（度）
        xi:   [B] 或 [B,T] UCM 参数
        height: int，图像/patch 高度
        width:  int，图像/patch 宽度
        device: torch.device
        delta: float，小旋转角度（弧度）
    Returns:
        up_map: [B, T, H, W, 2] 单位向量 map
        lat_map: [B, T, H, W, 1] 纬度 map
    """
    B, T, _, _ = R.shape
    dtype = R.dtype
    R = R.float()

    # Step1：生成每像素射线方向（相机坐标系）
    d_cam = ucm_unproject_grid_fov(
        x_fov=x_fov,
        xi=xi,
        height=height,
        width=width,
        device=device,
        dtype=torch.float32,
    )  # [B, H, W, 3]
    if d_cam.ndim == 3:
        d_cam = d_cam.unsqueeze(0)  # [B, H, W, 3]
    mask = d_cam.isnan().any(dim=-1, keepdim=True)  # [B, H, W, 1]

    # Step2：从相机系旋转到世界系
    d_cam_exp = repeat(d_cam, "B H W C -> B T H W C", T=T)  # [B, T, H, W, 3]
    d_world = torch.einsum('btij,bthwj->bthwi', R, d_cam_exp)
    d_world = d_world / torch.clamp_min(d_world.norm(dim=-1, keepdim=True), 1e-8)

    # Step3：计算纬度 map
    Xw, Yw, Zw = d_world[..., 0], d_world[..., 1], d_world[..., 2]
    lat_map = torch.atan2(-Yw, torch.sqrt(Xw**2 + Zw**2)).unsqueeze(-1)  # [B, T, H, W, 1]

    # Step4：计算 up_map
    v = d_world  # 已归一化
    up_world = torch.tensor([0, -1, 0], device=device, dtype=torch.float32)  # 世界上方方向（+Y 向下设定）
    k = torch.cross(v, up_world.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand_as(v), dim=-1)
    k = k / torch.clamp_min(k.norm(dim=-1, keepdim=True), 1e-8)

    delta = torch.tensor(delta, device=device, dtype=torch.float32)
    cos_eps = torch.cos(delta)
    sin_eps = torch.sin(delta)
    # Rodrigues 公式旋转 v → v_rot
    v_rot = v * cos_eps + torch.cross(k, v, dim=-1) * sin_eps + k * (k * (v * 1).sum(dim=-1, keepdim=True)) * (1 - cos_eps)

    dirs_cam = torch.einsum('btij,bthwj->bthwi', R.transpose(-1, -2), v_rot)
    Xs, Ys, Zs = dirs_cam[..., 0], dirs_cam[..., 1], dirs_cam[..., 2]

    du, dv = project_ucm_points_fov(
        Xs, Ys, Zs,
        x_fov=x_fov.float(),
        xi=xi.float(),
        height=height,
        width=width,
    )
    grid = create_grid(
        height=height,
        width=width,
        batch=B,
        dtype=torch.float32,
        device=device,
    )  # [B, H, W, 3]
    grid_x = grid[..., 0].unsqueeze(1)  # [B,1,H,W]
    grid_y = grid[..., 1].unsqueeze(1)

    up_map = torch.stack((du - grid_x, dv - grid_y), dim=-1)  # [B, T, H, W, 2]
    up_map = up_map / torch.clamp_min(up_map.norm(dim=-1, keepdim=True), 1e-8)

    up_map = up_map.to(dtype=dtype)
    lat_map = lat_map.to(dtype=dtype)

    # 扩 mask 到同 shape
    mask_exp2 = mask.unsqueeze(1).expand(B, T, height, width, 1)
    up_map = up_map.masked_fill(mask_exp2, 0.0)
    lat_map = lat_map.masked_fill(mask_exp2, 0.0)

    return up_map, lat_map


class UcpeSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        attn_dim: int,
        num_heads: int,
        patches_x: int = 8,
        patches_y: int = 8,
        image_width: int = 128,
        image_height: int = 128,
        freq_base: float = 100.0,
        freq_scale: float = 1.0,
        precompute_coeffs: bool = True,
        emb_dim: int | None = None,
        adaptation_method: str = "parallel",
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.patches_x = patches_x
        self.patches_y = patches_y
        self.image_width = image_width
        self.image_height = image_height
        self.freq_base = freq_base
        self.freq_scale = freq_scale
        self.adaptation_method = adaptation_method

        self.q_proj = nn.Linear(dim, attn_dim)
        self.k_proj = nn.Linear(dim, attn_dim)
        self.v_proj = nn.Linear(dim, attn_dim)
        self.out_proj = nn.Linear(attn_dim, dim)
        if emb_dim is not None:
            self.cam_encoder = nn.Linear(emb_dim, dim)

        # 初始化为零以增强 residual 训练稳定性
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # 初始化 PRoPE attention 模块（带 precomputed coeffs）
        self.prope_attn = PropeDotProductAttention(
            head_dim=self.head_dim,
            patches_x=patches_x,
            patches_y=patches_y,
            image_width=image_width,
            image_height=image_height,
            freq_base=freq_base,
            freq_scale=freq_scale,
            precompute_coeffs=precompute_coeffs,
        )

    def forward(self, x: torch.Tensor, control_camera_dit_input: dict):
        """
        Args:
            x: (B, T, D) — input tokens
            control_camera_dit_input: dict with keys:
                - viewmats: (B, N, 4, 4)
                - K: (B, N, 3, 3)
        """
        B, T, D = x.shape
        N = control_camera_dit_input["viewmats"].shape[1]  # number of cameras
        H, W = self.patches_y, self.patches_x
        assert T == N * H * W or T == N, f"Expected token shape ({N}×{H}×{W} or {N}), got {T}"

        if hasattr(self, "cam_encoder") and "cam_emb" in control_camera_dit_input:
            cam_emb = control_camera_dit_input["cam_emb"]
            y = self.cam_encoder(cam_emb)
            if y.shape[1] != T:
                hw = T // cam_emb.shape[1]
                y = repeat(y, "b f d -> b (f hw) d", hw=hw)
            x = x + y

        # Project Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D_head]
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Precompute camera-specific functions (only once per batch)
        self.prope_attn._precompute_and_cache_apply_fns(
            viewmats=control_camera_dit_input["viewmats"],
            Ks=control_camera_dit_input.get("K", None),
            coeffs_x=control_camera_dit_input.get("coeffs_x", None),
            coeffs_y=control_camera_dit_input.get("coeffs_y", None),
        )

        # Apply RoPE-style positional encoding
        q = self.prope_attn._apply_to_q(q)     # [B, H, T, D_head]
        k = self.prope_attn._apply_to_kv(k)
        v = self.prope_attn._apply_to_kv(v)

        # Rearrange to [B, T, D] for flash_attention input
        q = rearrange(q, "b h t d -> b t (h d)")
        k = rearrange(k, "b h t d -> b t (h d)")
        v = rearrange(v, "b h t d -> b t (h d)")

        # Fast attention (Flash/Sage/SDPA fallback)
        out = flash_attention(q, k, v, num_heads=self.num_heads)

        # reshape back
        out = rearrange(out, "b t (h d) -> b h t d", h=self.num_heads)

        # Apply inverse transform for PRoPE
        out = self.prope_attn._apply_to_o(out)

        # Final projection
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(out)
