"""
Convert depth-backprojected points to 3D Gaussians.

Each pixel with valid depth becomes a Gaussian:
  position = backprojected 3D point
  scale = pixel footprint at that depth
  rotation = quaternion from surface normal
  opacity = depth confidence
  color = RGB from photo
  label = SAM segment ID
"""

import numpy as np
import torch
import math
from typing import Tuple, Optional


def normal_to_quaternion(normals: np.ndarray) -> np.ndarray:
    """Convert surface normals to quaternions that orient Gaussians.

    The Gaussian's thin axis (z-local) aligns with the surface normal.
    """
    n = len(normals)
    quats = np.zeros((n, 4), dtype=np.float32)

    # Default orientation: z-axis = [0, 0, 1]
    # We want to rotate z-axis to match each normal
    z = np.array([0, 0, 1], dtype=np.float32)

    for i in range(n):
        nrm = normals[i]
        nm = np.linalg.norm(nrm)
        if nm < 1e-6:
            quats[i] = [1, 0, 0, 0]  # identity
            continue
        nrm = nrm / nm

        # Rotation axis = cross(z, normal)
        axis = np.cross(z, nrm)
        axis_len = np.linalg.norm(axis)

        if axis_len < 1e-6:
            # Parallel or anti-parallel
            if np.dot(z, nrm) > 0:
                quats[i] = [1, 0, 0, 0]  # identity
            else:
                quats[i] = [0, 1, 0, 0]  # 180 deg around x
            continue

        axis = axis / axis_len
        angle = math.acos(np.clip(np.dot(z, nrm), -1, 1))
        half = angle / 2
        quats[i] = [math.cos(half), *(axis * math.sin(half))]

    return quats


def points_to_gaussians(
    image: np.ndarray,
    depth: np.ndarray,
    focal: float,
    pixel_labels: Optional[np.ndarray] = None,
    subsample: int = 2,
    min_depth: float = 0.1,
) -> dict:
    """Convert a depth map + image to initial Gaussians.

    Args:
        image: (H, W, 3) uint8 RGB
        depth: (H, W) float32 metric depth (smoothed)
        focal: focal length in pixels
        pixel_labels: (H, W) int32 SAM labels (optional)
        subsample: take every Nth pixel (1=all, 2=quarter)
        min_depth: minimum valid depth

    Returns:
        dict with torch tensors on CPU:
          positions: (N, 3)
          scales: (N, 3)
          rotations: (N, 4) quaternions
          opacities: (N,)
          colors: (N, 3) RGB [0,1]
          labels: (N,) int
    """
    h, w = depth.shape

    # Subsample grid
    ys = np.arange(0, h, subsample)
    xs = np.arange(0, w, subsample)
    u_grid, v_grid = np.meshgrid(xs, ys)
    u_flat = u_grid.ravel()
    v_flat = v_grid.ravel()

    # Get depth at sampled positions
    d = depth[v_flat, u_flat]
    valid = d > min_depth

    u_valid = u_flat[valid].astype(np.float32)
    v_valid = v_flat[valid].astype(np.float32)
    d_valid = d[valid]

    # Backproject to 3D (OpenGL convention: camera looks along -Z)
    cx, cy = w / 2.0, h / 2.0
    px = (u_valid - cx) * d_valid / focal
    py = -(v_valid - cy) * d_valid / focal
    pz = -d_valid

    positions = np.stack([px, py, pz], axis=-1).astype(np.float32)

    # Colors from image
    colors = image[v_flat[valid].astype(int), u_flat[valid].astype(int)].astype(np.float32) / 255.0

    # Compute per-pixel normals from depth gradient
    try:
        import cv2
        dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
        dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
    except ImportError:
        from scipy.ndimage import convolve
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        dz_dx = convolve(depth, sobel_x)
        dz_dy = convolve(depth, sobel_x.T)

    nx = -dz_dx / focal
    ny = dz_dy / focal
    nz = np.ones_like(depth)
    norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
    nx /= norm; ny /= norm; nz /= norm

    normals = np.stack([
        nx[v_flat[valid].astype(int), u_flat[valid].astype(int)],
        ny[v_flat[valid].astype(int), u_flat[valid].astype(int)],
        -nz[v_flat[valid].astype(int), u_flat[valid].astype(int)],
    ], axis=-1).astype(np.float32)

    # Rotations from normals
    rotations = normal_to_quaternion(normals)

    # Scale: pixel footprint at that depth
    # At depth d, pixel spacing = d / focal * subsample
    pixel_scale = d_valid / focal * subsample
    # Gaussian scale: cover the pixel footprint in xy, thin in z
    scales = np.stack([
        pixel_scale,       # x: pixel width
        pixel_scale,       # y: pixel height
        pixel_scale * 0.3, # z: thin (surface)
    ], axis=-1).astype(np.float32)

    # Take log for gsplat (it exponentiates internally)
    log_scales = np.log(np.clip(scales, 1e-6, None))

    # Opacity: full for valid depth
    opacities = np.ones(len(positions), dtype=np.float32) * 0.95

    # Labels
    if pixel_labels is not None:
        labels = pixel_labels[v_flat[valid].astype(int), u_flat[valid].astype(int)].astype(np.int32)
    else:
        labels = np.zeros(len(positions), dtype=np.int32)

    n = len(positions)
    print(f"  Created {n:,} Gaussians from {h}x{w} image (subsample={subsample})")
    print(f"  Position range: X=[{positions[:,0].min():.2f}, {positions[:,0].max():.2f}]")
    print(f"  Position range: Z=[{positions[:,2].min():.2f}, {positions[:,2].max():.2f}]")
    print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")

    return {
        "positions": torch.from_numpy(positions),
        "scales": torch.from_numpy(log_scales),
        "rotations": torch.from_numpy(rotations),
        "opacities": torch.from_numpy(opacities),
        "colors": torch.from_numpy(colors),
        "labels": torch.from_numpy(labels),
        "normals": torch.from_numpy(normals),
    }
