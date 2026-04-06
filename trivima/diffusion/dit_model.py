"""
Diffusion Transformer (DiT) for photo → voxel grid generation.

Architecture:
  Photo Encoder: ViT (DINOv2) → visual feature tokens
  Diffusion Transformer: conditions on photo features, denoises voxel grid
  Output: N×N×N×C voxel grid (color, density, normal, label)

The voxel grid IS the cell grid — same data, different generation method.
Instead of Depth Pro + SAM + Qwen + fill (pipeline), one forward pass
through 20-50 diffusion steps produces the complete 3D world.

Training:
  Phase 1: 3D-FRONT synthetic rooms (200K+ pairs)
  Phase 2: Real photos processed by existing pipeline
  Phase 3: User feedback flywheel
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


# ============================================================
# Timestep embedding
# ============================================================
class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding for diffusion."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(emb)


# ============================================================
# 3D Patch Embedding — converts voxel grid to patch tokens
# ============================================================
class VoxelPatchEmbed(nn.Module):
    """Convert N×N×N×C voxel grid into sequence of patch tokens."""

    def __init__(self, grid_size: int = 64, patch_size: int = 4,
                 in_channels: int = 8, embed_dim: int = 768):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.n_patches = (grid_size // patch_size) ** 3

        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, N, N, N) → (B, embed_dim, N/P, N/P, N/P)
        x = self.proj(x)
        # Flatten spatial dims → (B, embed_dim, n_patches)
        x = x.flatten(2).transpose(1, 2)  # (B, n_patches, embed_dim)
        return x


# ============================================================
# 3D Unpatch — converts patch tokens back to voxel grid
# ============================================================
class VoxelUnpatch(nn.Module):
    """Convert patch tokens back to N×N×N×C voxel grid."""

    def __init__(self, grid_size: int = 64, patch_size: int = 4,
                 out_channels: int = 8, embed_dim: int = 768):
        super().__init__()
        self.grid_size = grid_size
        self.patch_size = patch_size
        self.patches_per_dim = grid_size // patch_size

        self.proj = nn.ConvTranspose3d(embed_dim, out_channels,
                                       kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        p = self.patches_per_dim
        # (B, N, C) → (B, C, p, p, p)
        x = x.transpose(1, 2).reshape(B, C, p, p, p)
        # (B, C, p, p, p) → (B, out_channels, grid_size, grid_size, grid_size)
        x = self.proj(x)
        return x


# ============================================================
# DiT Block — transformer block with adaptive layer norm
# ============================================================
class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm (adaLN-Zero).

    Conditions on timestep + photo features via scale/shift modulation.
    """

    def __init__(self, dim: int, n_heads: int = 12, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

        # AdaLN modulation: 6 parameters (scale1, shift1, gate1, scale2, shift2, gate2)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim),
        )
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c: conditioning (timestep + photo features), (B, dim)
        shift1, scale1, gate1, shift2, scale2, gate2 = self.adaLN(c).chunk(6, dim=-1)

        # Self-attention with adaLN
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + gate1.unsqueeze(1) * h

        # MLP with adaLN
        h = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        h = self.mlp(h)
        x = x + gate2.unsqueeze(1) * h

        return x


# ============================================================
# Cross-Attention Block — attends to photo features
# ============================================================
class CrossAttentionBlock(nn.Module):
    """Cross-attention: voxel tokens attend to photo feature tokens."""

    def __init__(self, dim: int, n_heads: int = 12):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: voxel tokens (B, N_voxel, dim)
        # context: photo tokens (B, N_photo, dim)
        h = self.cross_attn(
            self.norm_q(x),
            self.norm_kv(context),
            self.norm_kv(context),
        )[0]
        return x + h


# ============================================================
# Photo Encoder — extracts visual features from the input photo
# ============================================================
class PhotoEncoder(nn.Module):
    """Encodes the input photo into feature tokens for conditioning.

    Uses a lightweight ViT or can be swapped for DINOv2 features.
    """

    def __init__(self, img_size: int = 256, patch_size: int = 16,
                 embed_dim: int = 768):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size,
                                     stride=patch_size)
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.blocks = nn.ModuleList([
            DiTBlock(embed_dim, n_heads=12) for _ in range(4)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Dummy conditioning for self-attention blocks
        self.dummy_c = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        # img: (B, 3, H, W) → (B, n_patches, embed_dim)
        x = self.patch_embed(img).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        c = self.dummy_c.unsqueeze(0).expand(img.shape[0], -1)
        for block in self.blocks:
            x = block(x, c)

        return self.norm(x)  # (B, n_patches, embed_dim)


# ============================================================
# Voxel Diffusion Transformer — the main model
# ============================================================
class VoxelDiT(nn.Module):
    """Diffusion Transformer for photo → voxel grid.

    Input:
      photo: (B, 3, 256, 256) — room photo
      noisy_grid: (B, C, N, N, N) — noisy voxel grid
      timestep: (B,) — diffusion timestep

    Output:
      predicted_noise: (B, C, N, N, N) — noise prediction for denoising

    Voxel channels (C=8):
      0-2: RGB color
      3:   density (0=empty, 1=solid)
      4-6: surface normal (nx, ny, nz)
      7:   semantic label (normalized)
    """

    def __init__(
        self,
        grid_size: int = 64,       # N×N×N voxel grid
        patch_size: int = 4,       # 3D patch size
        voxel_channels: int = 8,   # channels per voxel
        embed_dim: int = 768,      # transformer dimension
        depth: int = 12,           # number of DiT blocks
        n_heads: int = 12,
        img_size: int = 256,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.voxel_channels = voxel_channels

        # Photo encoder
        self.photo_encoder = PhotoEncoder(img_size=img_size, embed_dim=embed_dim)

        # Photo feature projection (match dimensions)
        self.photo_proj = nn.Linear(embed_dim, embed_dim)

        # Timestep embedding
        self.time_embed = TimestepEmbedding(embed_dim)

        # Voxel patch embedding
        self.patch_embed = VoxelPatchEmbed(grid_size, patch_size,
                                           voxel_channels, embed_dim)

        # Positional embedding for 3D patches
        n_patches = (grid_size // patch_size) ** 3
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))

        # DiT blocks with interleaved cross-attention
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(DiTBlock(embed_dim, n_heads))
            # Cross-attention every 3 blocks
            if (i + 1) % 3 == 0:
                self.blocks.append(CrossAttentionBlock(embed_dim, n_heads))

        # Final norm + unpatch
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatch = VoxelUnpatch(grid_size, patch_size,
                                     voxel_channels, embed_dim)

    def forward(self, photo: torch.Tensor, noisy_grid: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            photo: (B, 3, 256, 256)
            noisy_grid: (B, C, N, N, N)
            timestep: (B,) float in [0, 1000]

        Returns:
            predicted_noise: (B, C, N, N, N)
        """
        # Encode photo
        photo_tokens = self.photo_encoder(photo)  # (B, n_photo, dim)
        photo_tokens = self.photo_proj(photo_tokens)

        # Timestep embedding
        t_emb = self.time_embed(timestep)  # (B, dim)

        # Patchify noisy voxel grid
        x = self.patch_embed(noisy_grid)  # (B, n_patches, dim)
        x = x + self.pos_embed

        # Conditioning = timestep + global photo feature
        c = t_emb + photo_tokens.mean(dim=1)  # (B, dim)

        # Transformer blocks
        for block in self.blocks:
            if isinstance(block, CrossAttentionBlock):
                x = block(x, photo_tokens)
            else:
                x = block(x, c)

        # Final norm + unpatch
        x = self.final_norm(x)
        noise_pred = self.unpatch(x)  # (B, C, N, N, N)

        return noise_pred

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ============================================================
# DDPM Scheduler — noise schedule for training and inference
# ============================================================
class DDPMScheduler:
    """Simple DDPM noise scheduler."""

    def __init__(self, n_steps: int = 1000, beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        self.n_steps = n_steps
        self.betas = torch.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, noise: torch.Tensor,
                  timestep: torch.Tensor) -> torch.Tensor:
        """Add noise to clean data at given timestep."""
        alpha_t = self.alpha_cumprod[timestep.long()].view(-1, 1, 1, 1, 1)
        return torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise

    def step(self, noise_pred: torch.Tensor, timestep: int,
             x_t: torch.Tensor) -> torch.Tensor:
        """One denoising step."""
        alpha_t = self.alpha_cumprod[timestep]
        alpha_prev = self.alpha_cumprod[timestep - 1] if timestep > 0 else torch.tensor(1.0)
        beta_t = self.betas[timestep]

        pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        pred_x0 = pred_x0.clamp(-1, 1)

        mean = (torch.sqrt(alpha_prev) * beta_t / (1 - alpha_t)) * pred_x0 + \
               (torch.sqrt(alpha_t) * (1 - alpha_prev) / (1 - alpha_t)) * x_t

        if timestep > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            return mean + sigma * noise
        return mean

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_cumprod = self.alpha_cumprod.to(device)
        return self


# ============================================================
# Training loop
# ============================================================
class VoxelDiTTrainer:
    """Training wrapper for the Voxel DiT."""

    def __init__(self, grid_size: int = 64, device: str = "cuda",
                 lr: float = 1e-4):
        self.device = device
        self.model = VoxelDiT(grid_size=grid_size).to(device)
        self.scheduler = DDPMScheduler().to(device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr,
                                            weight_decay=0.01)

        print(f"VoxelDiT: {self.model.count_params() / 1e6:.1f}M parameters")
        print(f"Grid size: {grid_size}^3, voxel channels: 8")

    def train_step(self, photo: torch.Tensor,
                   voxel_grid: torch.Tensor) -> dict:
        """One training step.

        Args:
            photo: (B, 3, 256, 256) — room photo
            voxel_grid: (B, 8, N, N, N) — ground truth voxel grid

        Returns:
            dict with loss values
        """
        B = photo.shape[0]

        # Random timestep
        t = torch.randint(0, self.scheduler.n_steps, (B,), device=self.device)

        # Add noise
        noise = torch.randn_like(voxel_grid)
        noisy = self.scheduler.add_noise(voxel_grid, noise, t)

        # Predict noise
        noise_pred = self.model(photo, noisy, t.float())

        # MSE loss on noise prediction
        loss = F.mse_loss(noise_pred, noise)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}

    @torch.no_grad()
    def generate(self, photo: torch.Tensor,
                 n_steps: int = 50) -> torch.Tensor:
        """Generate voxel grid from photo.

        Args:
            photo: (B, 3, 256, 256)
            n_steps: number of denoising steps

        Returns:
            voxel_grid: (B, 8, N, N, N)
        """
        self.model.eval()
        B = photo.shape[0]
        N = self.model.grid_size

        # Start from pure noise
        x = torch.randn(B, 8, N, N, N, device=self.device)

        # Denoise step by step
        step_size = self.scheduler.n_steps // n_steps
        for i in range(self.scheduler.n_steps - 1, -1, -step_size):
            t = torch.full((B,), i, device=self.device, dtype=torch.float)
            noise_pred = self.model(photo, x, t)
            x = self.scheduler.step(noise_pred, i, x)

        self.model.train()
        return x

    def save(self, path: str):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
