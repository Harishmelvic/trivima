"""
Unified .torch dataset for indoor + outdoor + landmark scenes.

All packers (Replica, KITTI, MegaDepth, ...) write to this schema:

    {
        'images':     list[torch.Tensor]  # N RGB jpegs as byte tensors
        'depths':     torch.Tensor[N, H, W]   # depth in meters (NaN/0 = invalid)
        'depth_mask': torch.Tensor[N, H, W]   # bool: True where depth is valid
        'cameras':    torch.Tensor[N, 18]     # [fx fy cx cy near far ext_3x4]
        'domain':     str                     # 'indoor' / 'outdoor' / 'landmark'
        'scene_scale': float                  # rough scene size in meters
    }

The dataset yields per-batch tuples of:
    input_image (B, 3, H, W)
    input_clip  (B, 3, 224, 224)  -- only if a CLIP encoder is needed; we don't here
    targets     (B, K, 3, H, W)   -- K = num target views
    target_depths (B, K, H, W)
    target_masks  (B, K, H, W) bool
    target_intrinsics (B, K, 3, 3)
    target_cam2world  (B, K, 4, 4)
    input_intrinsics (B, 3, 3)
    input_cam2world  (B, 4, 4)
    domain (list[str])
"""

from __future__ import annotations

import glob
import io
import random
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def _decode_rgb(jpg_bytes: torch.Tensor, target_size: int) -> torch.Tensor:
    """Decode JPEG bytes → tensor (3, H, W) in [0, 1] resized to target_size."""
    if isinstance(jpg_bytes, torch.Tensor):
        jpg_bytes = jpg_bytes.numpy().tobytes()
    elif isinstance(jpg_bytes, (bytes, bytearray)):
        pass
    else:
        jpg_bytes = bytes(jpg_bytes)
    img = Image.open(io.BytesIO(jpg_bytes)).convert("RGB")
    img = img.resize((target_size, target_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)


def _resize_depth(depth: torch.Tensor, mask: torch.Tensor, target_size: int):
    """Resize depth + mask. Mask is nearest, depth is also nearest to avoid invalid blending."""
    if depth.ndim == 2:
        depth = depth.unsqueeze(0).unsqueeze(0)
        mask = mask.unsqueeze(0).unsqueeze(0)
    depth = TF.resize(
        depth.float(),
        [target_size, target_size],
        interpolation=TF.InterpolationMode.NEAREST,
    )
    mask = TF.resize(
        mask.float(),
        [target_size, target_size],
        interpolation=TF.InterpolationMode.NEAREST,
    )
    return depth.squeeze(0).squeeze(0), mask.squeeze(0).squeeze(0).bool()


def _scale_intrinsics(K_18: torch.Tensor, src_h: int, src_w: int, dst: int) -> torch.Tensor:
    """Build a 3x3 intrinsics matrix scaled to dst×dst from the 18-vec camera entry.

    Standard pixelSplat schema: K_18 = [fx, fy, cx, cy, near, far, R(9), t(3)]
    where fx,fy,cx,cy may be either pixel-space or normalized [0,1] coords.

    We assume normalized (pixelSplat convention). If your packer writes pixel coords,
    set the source resolution accordingly.
    """
    fx, fy, cx, cy = K_18[0].item(), K_18[1].item(), K_18[2].item(), K_18[3].item()
    # If values look normalized (<=2), treat as fraction of image
    if max(abs(fx), abs(fy), abs(cx), abs(cy)) <= 2.0:
        fx_px = fx * dst
        fy_px = fy * dst
        cx_px = cx * dst
        cy_px = cy * dst
    else:
        # Pixel-space, scale by ratio
        sx = dst / src_w
        sy = dst / src_h
        fx_px = fx * sx
        fy_px = fy * sy
        cx_px = cx * sx
        cy_px = cy * sy
    K = torch.zeros(3, 3, dtype=torch.float32)
    K[0, 0] = fx_px
    K[1, 1] = fy_px
    K[0, 2] = cx_px
    K[1, 2] = cy_px
    K[2, 2] = 1.0
    return K


def _ext_to_cam2world(K_18: torch.Tensor) -> torch.Tensor:
    """Extract 4x4 cam2world from the 18-vec.

    pixelSplat stores world-to-camera (R|t) in indices 6..18.
    We invert it to get cam2world.
    """
    ext = K_18[6:18].reshape(3, 4).float()
    w2c = torch.eye(4, dtype=torch.float32)
    w2c[:3, :4] = ext
    c2w = torch.linalg.inv(w2c)
    return c2w


class PointCloudScene(Dataset):
    """Loads a single .torch scene file."""

    def __init__(self, path: str, img_size: int = 256, num_target_views: int = 4):
        self.path = path
        self.img_size = img_size
        self.K = num_target_views
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.images = d["images"]
        self.depths = d.get("depths")  # may be None for unsupervised data
        self.depth_mask = d.get("depth_mask")
        self.cameras = d["cameras"]  # tensor (N, 18)
        self.domain = d.get("domain", "unknown")
        self.scene_scale = float(d.get("scene_scale", 5.0))
        self.N = len(self.images)
        # Estimate src resolution from first depth (if present) or default
        if self.depths is not None and len(self.depths) > 0:
            src = self.depths[0]
            self.src_h, self.src_w = int(src.shape[-2]), int(src.shape[-1])
        else:
            # Decode first image to get its native size
            tmp = Image.open(io.BytesIO(_to_bytes(self.images[0])))
            self.src_w, self.src_h = tmp.size

    def __len__(self):
        # Yield each input frame once per epoch
        return max(0, self.N - 1)

    def __getitem__(self, idx):
        if self.N < 2:
            raise IndexError("Scene has fewer than 2 frames")
        i_in = idx % self.N
        # K target indices distinct from input
        candidates = [j for j in range(self.N) if j != i_in]
        if len(candidates) < self.K:
            target_idxs = (candidates * ((self.K // len(candidates)) + 1))[: self.K]
        else:
            target_idxs = random.sample(candidates, self.K)

        # Decode input image
        input_img = _decode_rgb(self.images[i_in], self.img_size)

        # Targets
        target_imgs = []
        target_depths = []
        target_masks = []
        target_K = []
        target_c2w = []
        for j in target_idxs:
            target_imgs.append(_decode_rgb(self.images[j], self.img_size))
            if self.depths is not None:
                d_j = self.depths[j].float()
                m_j = (
                    self.depth_mask[j].bool()
                    if self.depth_mask is not None
                    else (d_j > 0)
                )
                d_r, m_r = _resize_depth(d_j, m_j, self.img_size)
            else:
                d_r = torch.zeros(self.img_size, self.img_size)
                m_r = torch.zeros(self.img_size, self.img_size, dtype=torch.bool)
            target_depths.append(d_r)
            target_masks.append(m_r)
            target_K.append(_scale_intrinsics(self.cameras[j], self.src_h, self.src_w, self.img_size))
            target_c2w.append(_ext_to_cam2world(self.cameras[j]))

        return {
            "input_image": input_img,                                     # (3, H, W)
            "input_intrinsics": _scale_intrinsics(
                self.cameras[i_in], self.src_h, self.src_w, self.img_size
            ),  # (3, 3)
            "input_cam2world": _ext_to_cam2world(self.cameras[i_in]),     # (4, 4)
            "targets": torch.stack(target_imgs),                          # (K, 3, H, W)
            "target_depths": torch.stack(target_depths),                  # (K, H, W)
            "target_masks": torch.stack(target_masks),                    # (K, H, W)
            "target_intrinsics": torch.stack(target_K),                   # (K, 3, 3)
            "target_cam2world": torch.stack(target_c2w),                  # (K, 4, 4)
            "domain": self.domain,
            "scene_scale": self.scene_scale,
        }


def _to_bytes(x):
    if isinstance(x, torch.Tensor):
        return x.numpy().tobytes()
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    return bytes(x)


class MultiDomainDataset(Dataset):
    """Wraps multiple PointCloudScene files. Yields balanced batches per domain.

    Pass a list of (glob_pattern, domain_label, weight) tuples.
    Sampling is weighted-random across all loaded scenes; the loader will balance domains
    if you set per-domain weights.

    For simplicity, this just uniform-samples across all scenes from all .torch files.
    For true domain balancing, use a `WeightedRandomSampler` with weights computed from
    the `domain` field of each scene.
    """

    def __init__(
        self,
        sources: list[tuple[str, Optional[str]]],
        img_size: int = 256,
        num_target_views: int = 4,
        max_scenes_per_source: Optional[int] = None,
    ):
        self.scenes: list[PointCloudScene] = []
        self.scene_domains: list[str] = []
        for pattern, domain in sources:
            paths = sorted(glob.glob(pattern, recursive=True))
            if max_scenes_per_source:
                paths = paths[:max_scenes_per_source]
            for p in paths:
                try:
                    s = PointCloudScene(p, img_size=img_size, num_target_views=num_target_views)
                    if len(s) > 0:
                        self.scenes.append(s)
                        self.scene_domains.append(domain or s.domain)
                except Exception as e:
                    print(f"  WARN: failed to load {p}: {e}")
        if not self.scenes:
            raise RuntimeError("No scenes loaded — check source patterns")
        # Per-scene index range
        self.cum_lens = []
        total = 0
        for s in self.scenes:
            total += len(s)
            self.cum_lens.append(total)
        self.total = total
        print(f"MultiDomainDataset: {len(self.scenes)} scenes, {self.total} samples")
        # Domain breakdown
        from collections import Counter
        c = Counter(self.scene_domains)
        for k, v in c.items():
            print(f"  {k}: {v} scenes")

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        # Binary search for which scene this index belongs to
        lo, hi = 0, len(self.cum_lens)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.cum_lens[mid] > idx:
                hi = mid
            else:
                lo = mid + 1
        scene = self.scenes[lo]
        local = idx if lo == 0 else idx - self.cum_lens[lo - 1]
        return scene[local]


def collate_fn(batch: list[dict]) -> dict:
    """Standard collate that stacks tensors and keeps domain as a list."""
    out = {}
    for key in batch[0]:
        if key == "domain":
            out[key] = [b[key] for b in batch]
        elif key == "scene_scale":
            out[key] = torch.tensor([b[key] for b in batch], dtype=torch.float32)
        else:
            out[key] = torch.stack([b[key] for b in batch], dim=0)
    return out
