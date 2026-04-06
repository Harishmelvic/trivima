"""
Multi-View Diffusion Transformer — generates 20 consistent room views from 1 photo.

Key difference from standard DiT: cross-view attention.
Each generated view attends to all other views during denoising,
ensuring they depict the SAME room from different angles.

Based on MVDiffusion's architecture with DiT backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class TimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        emb = torch.cat([torch.cos(t[:, None] * freqs), torch.sin(t[:, None] * freqs)], dim=-1)
        return self.mlp(emb)


class CameraPoseEmbedding(nn.Module):
    """Encode 4x4 camera pose matrix into a conditioning vector."""
    def __init__(self, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(16, dim),  # 4x4 = 16 values
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, pose):
        # pose: (B, 4, 4)
        flat = pose.reshape(pose.shape[0], -1)  # (B, 16)
        return self.mlp(flat)  # (B, dim)


class PatchEmbed2D(nn.Module):
    """Convert image to patch tokens."""
    def __init__(self, img_size=256, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)  # (B, n_patches, dim)


class Unpatch2D(nn.Module):
    """Convert patch tokens back to image."""
    def __init__(self, img_size=256, patch_size=16, out_ch=3, embed_dim=768):
        super().__init__()
        self.patches_per_dim = img_size // patch_size
        self.proj = nn.ConvTranspose2d(embed_dim, out_ch, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, N, C = x.shape
        p = self.patches_per_dim
        x = x.transpose(1, 2).reshape(B, C, p, p)
        return self.proj(x)


class DiTBlock(nn.Module):
    """Transformer block with adaptive layer norm."""
    def __init__(self, dim, n_heads=12, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        s1, sh1, g1, s2, sh2, g2 = self.adaLN(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + s1.unsqueeze(1)) + sh1.unsqueeze(1)
        h, _ = self.attn(h, h, h)
        x = x + g1.unsqueeze(1) * h
        h = self.norm2(x) * (1 + s2.unsqueeze(1)) + sh2.unsqueeze(1)
        x = x + g2.unsqueeze(1) * self.mlp(h)
        return x


class CrossViewAttention(nn.Module):
    """Cross-attention between different views.

    Each view's tokens attend to all other views' tokens.
    This enforces multi-view consistency — views see each other.
    """
    def __init__(self, dim, n_heads=12):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)

    def forward(self, views_tokens):
        """
        views_tokens: list of (B, N, dim) tensors, one per view
        Returns: list of (B, N, dim) tensors with cross-view information
        """
        n_views = len(views_tokens)
        B, N, D = views_tokens[0].shape

        # Concatenate all other views as context for each view
        result = []
        for i in range(n_views):
            query = self.norm(views_tokens[i])
            # Context = all other views concatenated
            context_list = [views_tokens[j] for j in range(n_views) if j != i]
            context = torch.cat(context_list, dim=1)  # (B, (n_views-1)*N, dim)
            context = self.norm(context)
            out, _ = self.cross_attn(query, context, context)
            result.append(views_tokens[i] + out)
        return result


class PhotoEncoder(nn.Module):
    """Encodes the input photo into feature tokens."""
    def __init__(self, img_size=256, patch_size=16, embed_dim=768, depth=4):
        super().__init__()
        self.patch_embed = PatchEmbed2D(img_size, patch_size, 3, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))
        self.blocks = nn.ModuleList([DiTBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.dummy_c = nn.Parameter(torch.zeros(embed_dim))

    def forward(self, img):
        x = self.patch_embed(img) + self.pos_embed
        c = self.dummy_c.unsqueeze(0).expand(img.shape[0], -1)
        for block in self.blocks:
            x = block(x, c)
        return self.norm(x)


class MultiViewDiT(nn.Module):
    """Diffusion Transformer that generates N consistent views simultaneously.

    Input:
      photo: (B, 3, 256, 256) — single room photo
      noisy_views: (B, N, 3, 256, 256) — N noisy target views
      timestep: (B,) — diffusion timestep
      poses: (B, N, 4, 4) — camera poses for each target view

    Output:
      noise_pred: (B, N, 3, 256, 256) — predicted noise for each view
    """

    def __init__(
        self,
        num_views=20,
        img_size=256,
        patch_size=16,
        embed_dim=768,
        depth=12,
        n_heads=12,
    ):
        super().__init__()
        self.num_views = num_views
        self.img_size = img_size

        # Photo encoder
        self.photo_encoder = PhotoEncoder(img_size, patch_size, embed_dim, depth=4)

        # Per-view patch embedding
        self.patch_embed = PatchEmbed2D(img_size, patch_size, 3, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, embed_dim))

        # Timestep + pose conditioning
        self.time_embed = TimestepEmbedding(embed_dim)
        self.pose_embed = CameraPoseEmbedding(embed_dim)
        self.photo_proj = nn.Linear(embed_dim, embed_dim)

        # DiT blocks + cross-view attention every 3 blocks
        self.blocks = nn.ModuleList()
        self.cross_view_blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(DiTBlock(embed_dim, n_heads))
            if (i + 1) % 3 == 0:
                self.cross_view_blocks.append(CrossViewAttention(embed_dim, n_heads))
            else:
                self.cross_view_blocks.append(None)

        # Cross-attention to photo features
        self.photo_cross_attn = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
            for _ in range(depth // 3)
        ])
        self.photo_cross_norm = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(depth // 3)
        ])

        # Output
        self.final_norm = nn.LayerNorm(embed_dim)
        self.unpatch = Unpatch2D(img_size, patch_size, 3, embed_dim)

    def forward(self, photo, noisy_views, timestep, poses):
        B, N = noisy_views.shape[:2]

        # Encode photo
        photo_tokens = self.photo_encoder(photo)  # (B, P, dim)
        photo_tokens = self.photo_proj(photo_tokens)

        # Timestep embedding
        t_emb = self.time_embed(timestep)  # (B, dim)

        # Patchify each view and add pose conditioning
        views_tokens = []
        for i in range(N):
            tokens = self.patch_embed(noisy_views[:, i]) + self.pos_embed
            pose_emb = self.pose_embed(poses[:, i])  # (B, dim)
            # Conditioning = timestep + pose + photo global
            c = t_emb + pose_emb + photo_tokens.mean(dim=1)
            views_tokens.append((tokens, c))

        # Process through blocks
        cross_attn_idx = 0
        for block_idx, (dit_block, cross_view) in enumerate(zip(self.blocks, self.cross_view_blocks)):
            # Self-attention per view
            new_tokens = []
            for tokens, c in views_tokens:
                tokens = dit_block(tokens, c)
                new_tokens.append((tokens, c))
            views_tokens = new_tokens

            # Cross-view attention every 3 blocks
            if cross_view is not None:
                token_list = [t for t, c in views_tokens]
                token_list = cross_view(token_list)

                # Also cross-attend to photo features
                photo_attn = self.photo_cross_attn[cross_attn_idx]
                photo_norm = self.photo_cross_norm[cross_attn_idx]
                new_views = []
                for i, (tokens, c) in enumerate(views_tokens):
                    tokens = token_list[i]
                    # Photo cross-attention
                    q = photo_norm(tokens)
                    out, _ = photo_attn(q, photo_tokens, photo_tokens)
                    tokens = tokens + out
                    new_views.append((tokens, c))
                views_tokens = new_views
                cross_attn_idx += 1

        # Output: unpatch each view
        outputs = []
        for tokens, c in views_tokens:
            tokens = self.final_norm(tokens)
            img = self.unpatch(tokens)  # (B, 3, H, W)
            outputs.append(img)

        return torch.stack(outputs, dim=1)  # (B, N, 3, H, W)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
