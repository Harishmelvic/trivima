"""
Bilateral depth smoothing — the single most impactful preprocessing step for cell quality.

Smooths depth noise within uniform surfaces (where RGB colors are similar)
while preserving sharp depth transitions at object boundaries (where colors differ).

Parameters (from single_image_precision_theory.md, Chapter 9.2):
  - Spatial sigma: 3-4 pixels (NOT 5-7, to preserve surface detail like brick mortar)
  - Color sigma: 20-30 intensity units on 0-255 scale
  - At 2MP resolution, spatial sigma 3-4px spans ~6-12mm at 2m distance

The bilateral filter and 5x5 Sobel gradient kernel address separate concerns:
  - Depth should be as accurate as possible (mild bilateral smoothing)
  - Gradients should be as clean as possible (smooth during differentiation)

Quality impact: gradient noise reduced by 40-65%.
Processing time: ~5ms for 2MP image.
"""

import numpy as np
from typing import Tuple, Optional


def bilateral_depth_smooth(
    depth: np.ndarray,
    rgb_guide: np.ndarray,
    spatial_sigma: float = 3.5,
    color_sigma: float = 25.0,
    kernel_radius: Optional[int] = None,
) -> np.ndarray:
    """Apply RGB-guided bilateral filter to a depth map.

    Smooths depth within surfaces (similar RGB) while preserving edges
    (different RGB). This is ESSENTIAL for cell-based representations —
    without it, finite-difference gradients amplify per-pixel depth noise.

    Args:
        depth: (H, W) float32 metric depth map in meters
        rgb_guide: (H, W, 3) uint8 or float32 RGB image (guide for edge preservation)
        spatial_sigma: Gaussian sigma in pixels for spatial weighting (3-4 recommended)
        color_sigma: sigma for color similarity weighting (20-30 on 0-255 scale)
        kernel_radius: filter radius in pixels (default: ceil(2 * spatial_sigma))

    Returns:
        (H, W) float32 smoothed depth map
    """
    try:
        import cv2
        return _bilateral_cv2(depth, rgb_guide, spatial_sigma, color_sigma)
    except ImportError:
        return _bilateral_numpy(depth, rgb_guide, spatial_sigma, color_sigma, kernel_radius)


def _bilateral_cv2(
    depth: np.ndarray,
    rgb_guide: np.ndarray,
    spatial_sigma: float,
    color_sigma: float,
) -> np.ndarray:
    """OpenCV bilateral filter — fast C++ implementation."""
    import cv2

    # Convert guide to float32 to match depth dtype (ximgproc requires same depth)
    if rgb_guide.dtype == np.uint8:
        guide_f32 = rgb_guide.astype(np.float32)
    elif rgb_guide.max() <= 1.0:
        guide_f32 = (rgb_guide * 255.0).astype(np.float32)
    else:
        guide_f32 = rgb_guide.astype(np.float32)

    try:
        smoothed = cv2.ximgproc.jointBilateralFilter(
            joint=guide_f32,
            src=depth,
            d=-1,
            sigmaColor=color_sigma,
            sigmaSpace=spatial_sigma,
        )
        return smoothed
    except (AttributeError, cv2.error):
        # Fallback: standard bilateral on depth (no RGB guide)
        # Less effective but still reduces noise
        d = int(np.ceil(spatial_sigma * 2)) * 2 + 1
        smoothed = cv2.bilateralFilter(
            depth,
            d=d,
            sigmaColor=color_sigma / 255.0 * (depth.max() - depth.min()),  # scale to depth range
            sigmaSpace=spatial_sigma,
        )
        return smoothed


def _bilateral_numpy(
    depth: np.ndarray,
    rgb_guide: np.ndarray,
    spatial_sigma: float,
    color_sigma: float,
    kernel_radius: Optional[int] = None,
) -> np.ndarray:
    """Pure NumPy bilateral filter — slower but no dependencies."""
    h, w = depth.shape
    if kernel_radius is None:
        kernel_radius = int(np.ceil(2 * spatial_sigma))

    # Ensure guide is float 0-255 range
    if rgb_guide.dtype == np.uint8:
        guide = rgb_guide.astype(np.float32)
    elif rgb_guide.max() <= 1.0:
        guide = rgb_guide * 255.0
    else:
        guide = rgb_guide.astype(np.float32)

    smoothed = np.zeros_like(depth)
    valid = depth > 0

    # Precompute spatial kernel
    y_offsets, x_offsets = np.mgrid[-kernel_radius:kernel_radius+1, -kernel_radius:kernel_radius+1]
    spatial_kernel = np.exp(-(x_offsets**2 + y_offsets**2) / (2 * spatial_sigma**2))

    for y in range(h):
        for x in range(w):
            if not valid[y, x]:
                continue

            # Extract neighborhood
            y0 = max(0, y - kernel_radius)
            y1 = min(h, y + kernel_radius + 1)
            x0 = max(0, x - kernel_radius)
            x1 = min(w, x + kernel_radius + 1)

            # Spatial weights (from precomputed kernel, clipped to image bounds)
            ky0 = y0 - (y - kernel_radius)
            ky1 = ky0 + (y1 - y0)
            kx0 = x0 - (x - kernel_radius)
            kx1 = kx0 + (x1 - x0)
            w_spatial = spatial_kernel[ky0:ky1, kx0:kx1]

            # Color weights (RGB L2 distance)
            center_color = guide[y, x]
            patch_colors = guide[y0:y1, x0:x1]
            color_diff = np.sqrt(np.sum((patch_colors - center_color)**2, axis=2))
            w_color = np.exp(-color_diff**2 / (2 * color_sigma**2))

            # Valid depth mask
            w_valid = (depth[y0:y1, x0:x1] > 0).astype(np.float32)

            # Combined weights
            weights = w_spatial * w_color * w_valid
            w_sum = weights.sum()

            if w_sum > 1e-8:
                smoothed[y, x] = (weights * depth[y0:y1, x0:x1]).sum() / w_sum
            else:
                smoothed[y, x] = depth[y, x]

    return smoothed


def compute_depth_local_variance(
    depth: np.ndarray,
    kernel_size: int = 7,
) -> np.ndarray:
    """Compute local depth variance — used for per-pixel confidence estimation.

    Low variance → smooth surface → high confidence.
    High variance → noisy or boundary → lower confidence.

    Args:
        depth: (H, W) float32 depth map (before or after smoothing)
        kernel_size: size of the local window

    Returns:
        (H, W) float32 local variance map
    """
    try:
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(depth, size=kernel_size)
        mean_sq = uniform_filter(depth**2, size=kernel_size)
        variance = np.maximum(mean_sq - mean**2, 0)
        return variance.astype(np.float32)
    except ImportError:
        # Manual box filter
        h, w = depth.shape
        r = kernel_size // 2
        padded = np.pad(depth, r, mode='reflect')
        variance = np.zeros_like(depth)
        for y in range(h):
            for x in range(w):
                patch = padded[y:y+kernel_size, x:x+kernel_size]
                variance[y, x] = patch.var()
        return variance
