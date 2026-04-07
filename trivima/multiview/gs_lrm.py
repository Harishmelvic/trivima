"""
GS-LRM — direct image → 3D Gaussian Splat model.

Architecture:
    photo (B,3,H,W)
        ↓ DINOv2 ViT-B (frozen)
    patch features (B, 768, H/14, W/14)
        ↓ DPT-style decoder (4 fusion stages)
    per-pixel feature map (B, C, H, W)
        ↓ 1x1 conv Gaussian head
    14 channels per pixel:
        depth (1) + rgb (3) + opacity (1) + log_scale (3) + quat (4) + raw (2)
        ↓ unproject pixels using predicted depth + intrinsics
    3D Gaussians (B, H*W, 14) ready for gsplat rasterization

Output is the 3D representation directly — no diffusion, no multi-view trick.
The 'raw (2)' channels are unused placeholders so the head matches the standard 14-ch layout
of GS-LRM / pixelSplat for future expansion.

Trained with:
    L_rgb   = L1(rendered_rgb, gt_rgb)        — render Gaussians from target poses, compare
    L_depth = L1(rendered_depth * mask, gt_depth * mask)  — masked depth supervision
    total   = L_rgb + 0.1 * L_depth
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# DPT-style decoder
# ---------------------------------------------------------------------------


class ResidualConvUnit(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, 3, padding=1)
        self.act = nn.GELU()

    def forward(self, x):
        h = self.conv1(self.act(x))
        h = self.conv2(self.act(h))
        return x + h


class FeatureFusionBlock(nn.Module):
    """One DPT fusion stage: combine skip features and upsample 2x."""

    def __init__(self, dim: int):
        super().__init__()
        self.rcu1 = ResidualConvUnit(dim)
        self.rcu2 = ResidualConvUnit(dim)
        self.out_conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.out_conv(x)
        return x


class DPTDecoder(nn.Module):
    """Dense Prediction Transformer decoder.

    Takes 4 ViT layer outputs and fuses them at increasing resolutions
    until reaching the input image resolution.
    """

    def __init__(self, vit_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Project ViT tokens to decoder hidden dim at each stage
        self.proj = nn.ModuleList([nn.Conv2d(vit_dim, hidden_dim, 1) for _ in range(4)])

        # Per-stage upsampling to reach 1, 1/2, 1/4, 1/8 of full resolution
        # (DINOv2 patch size 14 → token grid is H/14 x W/14, so we upsample 4 times to reach near-full res)
        # Stage 0 (deepest): keep at native, 1
        # Stage 1: 2x
        # Stage 2: 4x
        # Stage 3 (shallowest): 8x
        # Final fusion outputs at 8x token res, then we bilinear-up to image res
        self.fusion = nn.ModuleList([FeatureFusionBlock(hidden_dim) for _ in range(4)])

    def forward(self, tokens: list[torch.Tensor]) -> torch.Tensor:
        """tokens: list of 4 (B, vit_dim, h, w) feature maps from ViT layers (deep → shallow).

        Returns: (B, hidden_dim, H_out, W_out) where H_out = h * 16 (after 4 upsamples)
        """
        feats = [self.proj[i](tokens[i]) for i in range(4)]
        # Start from the deepest, fuse upward
        x = self.fusion[0](feats[0])  # upsamples to 2x token res
        x = self.fusion[1](x, feats[1])  # 4x
        x = self.fusion[2](x, feats[2])  # 8x
        x = self.fusion[3](x, feats[3])  # 16x
        return x


# ---------------------------------------------------------------------------
# Main GS-LRM model
# ---------------------------------------------------------------------------


class GSLRM(nn.Module):
    """Image → 3D Gaussian Splat predictor.

    Args:
        img_size: input image resolution (square).
        decoder_dim: DPT decoder hidden dim.
        gaussian_channels: per-pixel output channels (14 = depth+rgb+opa+scale+quat+2).
        depth_min, depth_max: depth normalization range. Predictions are sigmoid → [0,1] → rescaled to [depth_min, depth_max].

    Note on depth normalization:
        We use UNIVERSAL depth range and let the model learn to use it.
        Indoor: 0.1m - 10m typical. Outdoor: 1m - 100m. Landmark: 5m - 500m.
        We pick 0.1 - 200m range with INVERSE sigmoid parameterization (more capacity at small depths)
        which is the standard for monocular depth networks.
    """

    def __init__(
        self,
        img_size: int = 256,
        decoder_dim: int = 256,
        gaussian_channels: int = 14,
        depth_min: float = 0.1,
        depth_max: float = 200.0,
    ):
        super().__init__()
        self.img_size = img_size
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.gaussian_channels = gaussian_channels

        # ----- DINOv2 encoder (frozen) -----
        # We import lazily to avoid hard dependency at import time.
        self.encoder = None  # set on first forward
        self.vit_dim = 768

        # ----- DPT decoder -----
        self.decoder = DPTDecoder(vit_dim=self.vit_dim, hidden_dim=decoder_dim)

        # ----- Gaussian head: 1x1 conv to 14 channels -----
        self.gaussian_head = nn.Conv2d(decoder_dim, gaussian_channels, 1)

        # Initialize so initial depth ≈ middle of range and rotations ≈ identity
        with torch.no_grad():
            self.gaussian_head.bias.zero_()
            self.gaussian_head.weight.mul_(0.01)
            # quat channels [10:14] → [1, 0, 0, 0] (identity rotation in wxyz)
            self.gaussian_head.bias[10] = 1.0

    # ------------------------------------------------------------------
    # Encoder lazy init
    # ------------------------------------------------------------------
    def _init_encoder(self, device):
        if self.encoder is not None:
            return
        try:
            # Try torch hub first (no extra dependency)
            enc = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vitb14", pretrained=True, trust_repo=True
            )
        except Exception:
            # Fallback: transformers AutoModel
            from transformers import AutoModel

            enc = AutoModel.from_pretrained("facebook/dinov2-base")
        enc = enc.to(device)
        for p in enc.parameters():
            p.requires_grad_(False)
        enc.eval()
        self.encoder = enc

    @torch.no_grad()
    def _encode_dino(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Run DINOv2 and return 4 intermediate feature maps (deep → shallow)."""
        # DINOv2 expects 224 input by default, but it works at any size that's a multiple of 14
        # For img_size=256 we need to resize to a 14-multiple. Use 252 (18 patches).
        H = W = (self.img_size // 14) * 14  # round down to 14-multiple
        if H != self.img_size:
            x_in = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=False)
        else:
            x_in = x

        # torch hub DINOv2 supports get_intermediate_layers
        n_blocks = 12  # ViT-B
        layers_to_return = (n_blocks - 1, n_blocks - 4, n_blocks - 7, n_blocks - 10)
        try:
            feats = self.encoder.get_intermediate_layers(
                x_in, n=layers_to_return, reshape=True, return_class_token=False
            )
            # feats is a tuple of (B, vit_dim, h, w) ordered shallow→deep in get_intermediate_layers
            # Reverse so deep is first (matches our DPT decoder convention)
            tokens = list(feats)
            tokens.reverse()
        except AttributeError:
            # transformers fallback: returns hidden_states
            out = self.encoder(pixel_values=x_in, output_hidden_states=True)
            hs = out.hidden_states  # tuple of (B, 1+N, D), N = patches
            picks = [hs[i + 1] for i in layers_to_return]  # +1 because hs[0] is embeddings
            picks.reverse()
            B = x_in.shape[0]
            h = w = H // 14
            tokens = []
            for p in picks:
                p = p[:, 1:]  # drop CLS
                p = p.transpose(1, 2).reshape(B, self.vit_dim, h, w)
                tokens.append(p)

        return tokens

    # ------------------------------------------------------------------
    # Forward — image → per-pixel Gaussian params
    # ------------------------------------------------------------------
    def forward(self, image: torch.Tensor) -> dict:
        """
        Args:
            image: (B, 3, H, W) in [0, 1] range
        Returns:
            dict with raw per-pixel outputs:
                'depth':     (B, 1, H, W)  — metric depth in scene units
                'rgb':       (B, 3, H, W)  — color in [0,1]
                'opacity':   (B, 1, H, W)  — in [0,1]
                'log_scale': (B, 3, H, W)  — log-scale for gsplat
                'quat':      (B, 4, H, W)  — normalized quaternion (wxyz)
        """
        device = image.device
        self._init_encoder(device)

        # Encode
        tokens = self._encode_dino(image)

        # Decode
        feat = self.decoder(tokens)  # (B, decoder_dim, H', W')

        # Resize feature map to input image size
        H, W = image.shape[-2:]
        if feat.shape[-2:] != (H, W):
            feat = F.interpolate(feat, size=(H, W), mode="bilinear", align_corners=False)

        # Gaussian head
        out = self.gaussian_head(feat)  # (B, 14, H, W)

        # Split channels
        depth_raw   = out[:, 0:1]
        rgb_raw     = out[:, 1:4]
        opa_raw     = out[:, 4:5]
        scale_raw   = out[:, 5:8]
        quat_raw    = out[:, 8:12]
        # channels 12,13 reserved

        # Activations
        # Inverse-sigmoid depth: more resolution at small depths
        depth_norm = torch.sigmoid(depth_raw)  # [0,1]
        depth = self.depth_min + (self.depth_max - self.depth_min) * depth_norm

        rgb = torch.sigmoid(rgb_raw)
        opacity = torch.sigmoid(opa_raw)
        log_scale = scale_raw - 4.0  # bias toward small Gaussians
        quat = quat_raw / (quat_raw.norm(dim=1, keepdim=True) + 1e-6)

        return {
            "depth": depth,
            "rgb": rgb,
            "opacity": opacity,
            "log_scale": log_scale,
            "quat": quat,
        }


# ---------------------------------------------------------------------------
# Build per-pixel 3D Gaussians from model output
# ---------------------------------------------------------------------------


def build_gaussians(
    out: dict,
    intrinsics: torch.Tensor,
    cam2world: torch.Tensor,
) -> dict:
    """Unproject per-pixel predictions to 3D Gaussians in world coordinates.

    Args:
        out: dict from GSLRM.forward
        intrinsics: (B, 3, 3) camera intrinsics matching image resolution
        cam2world: (B, 4, 4) camera-to-world pose of the input photo

    Returns:
        dict matching trivima/gaussian/renderer.py schema:
            positions: (B, N, 3)  N = H*W
            colors:    (B, N, 3)
            opacities: (B, N)
            scales:    (B, N, 3)  log-scale (gsplat-compatible: render code does exp())
            rotations: (B, N, 4)  unit quaternion wxyz
    """
    B, _, H, W = out["depth"].shape
    device = out["depth"].device

    # Pixel grid (u, v)
    v, u = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing="ij",
    )
    u = u + 0.5
    v = v + 0.5
    uv1 = torch.stack([u, v, torch.ones_like(u)], dim=-1)  # (H, W, 3)
    uv1 = uv1.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 3)

    # Unproject: x_cam = depth * K^-1 @ [u, v, 1]
    K_inv = torch.linalg.inv(intrinsics)  # (B, 3, 3)
    rays_cam = torch.einsum("bij,bhwj->bhwi", K_inv, uv1)  # (B, H, W, 3)
    depth = out["depth"].permute(0, 2, 3, 1)  # (B, H, W, 1)
    pts_cam = rays_cam * depth  # (B, H, W, 3)

    # Camera → world
    pts_cam_h = torch.cat([pts_cam, torch.ones_like(pts_cam[..., :1])], dim=-1)  # (B,H,W,4)
    pts_world_h = torch.einsum("bij,bhwj->bhwi", cam2world, pts_cam_h)
    pts_world = pts_world_h[..., :3]  # (B, H, W, 3)

    # Flatten H*W → N
    N = H * W
    positions = pts_world.reshape(B, N, 3)
    colors = out["rgb"].permute(0, 2, 3, 1).reshape(B, N, 3)
    opacities = out["opacity"].reshape(B, N)
    scales = out["log_scale"].permute(0, 2, 3, 1).reshape(B, N, 3)
    rotations = out["quat"].permute(0, 2, 3, 1).reshape(B, N, 4)

    return {
        "positions": positions,
        "colors": colors,
        "opacities": opacities,
        "scales": scales,
        "rotations": rotations,
    }


# ---------------------------------------------------------------------------
# Differentiable rendering wrapper for training
# ---------------------------------------------------------------------------


def render_gaussians(
    gaussians: dict,
    viewmat: torch.Tensor,
    intrinsics: torch.Tensor,
    width: int,
    height: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable gsplat render. Returns (rgb, depth).

    Args:
        gaussians: dict from build_gaussians (per-batch)
        viewmat: (B, 4, 4) world-to-camera matrix
        intrinsics: (B, 3, 3)
    Returns:
        rgb: (B, height, width, 3) in [0,1]
        depth: (B, height, width) in scene units
    """
    from gsplat import rasterization

    B = viewmat.shape[0]
    rgb_list = []
    depth_list = []
    for b in range(B):
        means = gaussians["positions"][b]
        quats = gaussians["rotations"][b]
        # gsplat expects unnormalized scales (it does NOT exp)
        scales = torch.exp(gaussians["scales"][b])
        opacs = gaussians["opacities"][b]
        colors = gaussians["colors"][b]

        renders, alphas, meta = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacs,
            colors=colors,
            viewmats=viewmat[b : b + 1],
            Ks=intrinsics[b : b + 1],
            width=width,
            height=height,
            packed=False,
            render_mode="RGB+ED",  # render expected depth too
        )
        # renders: (1, H, W, 4) — last channel is depth when render_mode='RGB+ED'
        rgb_list.append(renders[0, ..., :3])
        depth_list.append(renders[0, ..., 3])

    rgb = torch.stack(rgb_list, dim=0)
    depth = torch.stack(depth_list, dim=0)
    return rgb, depth


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def gs_lrm_loss(
    pred_rgb: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_rgb: torch.Tensor,
    gt_depth: torch.Tensor,
    depth_mask: torch.Tensor,
    depth_weight: float = 0.1,
) -> dict:
    """Compute training loss.

    Args:
        pred_rgb:   (B, H, W, 3)
        pred_depth: (B, H, W)
        gt_rgb:     (B, H, W, 3)
        gt_depth:   (B, H, W)
        depth_mask: (B, H, W) bool — where depth supervision is valid
    """
    rgb_loss = F.l1_loss(pred_rgb, gt_rgb)

    if depth_mask.any():
        diff = (pred_depth - gt_depth).abs() * depth_mask.float()
        depth_loss = diff.sum() / depth_mask.float().sum().clamp_min(1)
    else:
        depth_loss = pred_depth.new_zeros(())

    total = rgb_loss + depth_weight * depth_loss
    return {"total": total, "rgb": rgb_loss, "depth": depth_loss}
