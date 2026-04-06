"""
Gaussian Splat Renderer — wraps gsplat for Trivima.

Pure CUDA rendering — no EGL, no OpenGL, no context conflicts.
Differentiable — supports optimization against the input photo.
"""

import torch
import numpy as np
import math
from typing import Optional, Tuple
from PIL import Image


class GaussianRenderer:
    """Renders Gaussians from any camera pose using gsplat."""

    def __init__(self, width: int = 1280, height: int = 720, device: str = "cuda"):
        self.W = width
        self.H = height
        self.device = device

    def render(
        self,
        gaussians: dict,
        cam_pos: np.ndarray = None,
        yaw: float = -90,
        pitch: float = 0,
        fov_deg: float = 60,
    ) -> dict:
        """Render Gaussians from a camera pose.

        Args:
            gaussians: dict with positions, scales, rotations, opacities, colors
            cam_pos: (3,) camera position
            yaw, pitch: camera angles in degrees
            fov_deg: field of view

        Returns:
            dict with 'rgb', 'depth', 'alpha' as numpy arrays
        """
        from gsplat import rasterization

        if cam_pos is None:
            cam_pos = np.array([0.0, 0.0, 0.0])

        # Move Gaussians to device
        means = gaussians["positions"].to(self.device).float()
        quats = gaussians["rotations"].to(self.device).float()
        scales = gaussians["scales"].to(self.device).float()
        opacs = gaussians["opacities"].to(self.device).float()
        colors = gaussians["colors"].to(self.device).float()

        # Normalize quaternions
        quats = quats / (quats.norm(dim=-1, keepdim=True) + 1e-8)

        # Build camera matrices
        viewmat = self._view_matrix(cam_pos, yaw, pitch)
        K = self._intrinsics(fov_deg)

        viewmat = viewmat.unsqueeze(0).to(self.device)
        K = K.unsqueeze(0).to(self.device)

        # Render
        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=torch.exp(scales),  # gsplat expects actual scales
            opacities=torch.sigmoid(opacs),
            colors=colors,
            viewmats=viewmat,
            Ks=K,
            width=self.W,
            height=self.H,
            packed=False,
        )

        # Convert to numpy
        rgb = renders[0].clamp(0, 1).cpu().detach().numpy()  # (H, W, 3)
        alpha = alphas[0].cpu().detach().numpy()  # (H, W, 1)

        return {
            "rgb": (rgb * 255).astype(np.uint8),
            "alpha": (alpha[:, :, 0] * 255).astype(np.uint8),
        }

    def _view_matrix(self, pos, yaw, pitch):
        """Build 4x4 view matrix (world-to-camera).

        gsplat convention: camera looks along +Z in camera space.
        Our world: camera at origin looks along -Z (OpenGL convention).
        The view matrix transforms world coords to camera coords.
        """
        ry = math.radians(yaw)
        rp = math.radians(pitch)

        fwd = np.array([math.cos(rp)*math.cos(ry), math.sin(rp), math.cos(rp)*math.sin(ry)])
        right = np.cross(fwd, [0, 1, 0])
        rn = np.linalg.norm(right)
        right = right / rn if rn > 1e-6 else np.array([1.0, 0.0, 0.0])
        up = np.cross(right, fwd)

        # gsplat expects: R maps world to camera, t is camera position in world
        # Camera axes: right=X, up=Y, forward=Z (looking along +Z in cam space)
        # So we map world forward to camera +Z
        R = np.eye(3, dtype=np.float32)
        R[0, :] = right
        R[1, :] = -up  # flip Y for image convention (y down)
        R[2, :] = fwd  # camera looks along fwd → +Z in camera space

        t = -R @ np.array(pos, dtype=np.float32)

        m = np.eye(4, dtype=np.float32)
        m[:3, :3] = R
        m[:3, 3] = t

        return torch.from_numpy(m)

    def _intrinsics(self, fov_deg):
        """Build 3x3 camera intrinsics from FOV."""
        f = self.H / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        K = torch.tensor([
            [f, 0, self.W / 2.0],
            [0, f, self.H / 2.0],
            [0, 0, 1],
        ], dtype=torch.float32)
        return K


def render_multi_view(gaussians: dict, fov_deg: float, output_dir: str,
                      width: int = 1280, height: int = 720):
    """Render Gaussians from multiple viewpoints and save images."""
    import os
    os.makedirs(output_dir, exist_ok=True)

    renderer = GaussianRenderer(width, height)

    views = [
        ("front",           [0.0,  0.0,  0.0], -90,   0),
        ("look_down",       [0.0,  0.0,  0.0], -90, -20),
        ("look_left",       [0.0,  0.0,  0.0], -60,   0),
        ("look_right",      [0.0,  0.0,  0.0],-120,   0),
        ("step_in",         [0.0,  0.0, -0.4], -90,  -5),
        ("deep_in",         [0.0,  0.0, -0.8], -90,  -5),
        ("left_side",      [-0.4,  0.0, -0.5], -80,   0),
        ("right_side",      [0.4,  0.0, -0.5],-100,   0),
        ("center_left",    [-0.1,  0.05,-1.2], -45,  -5),
        ("center_right",   [-0.1,  0.05,-1.2],-135,  -5),
        ("look_back",       [0.0,  0.0, -1.0],  90,   0),
        ("corner",         [-0.5,  0.0, -0.5], -70, -10),
    ]

    print(f"  Rendering {len(views)} views with gsplat...")
    for i, (name, pos, yaw, pitch) in enumerate(views):
        result = renderer.render(
            gaussians, cam_pos=np.array(pos), yaw=yaw, pitch=pitch, fov_deg=fov_deg)

        Image.fromarray(result["rgb"]).save(os.path.join(output_dir, f"{i:02d}_{name}.png"))

        # Coverage: non-black pixels
        alpha = result["alpha"]
        coverage = 100 * (alpha > 10).sum() / (width * height)
        print(f"  [{i:2d}] {name:18s} {coverage:5.1f}%")

    print(f"  Output: {output_dir}/")
