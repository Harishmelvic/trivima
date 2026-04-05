"""
Perception pipeline orchestrator.

Single-image flow:
  1. Depth Pro → metric depth + focal length + depth smoothness confidence
  2. Bilateral smoothing (sigma 3-4px spatial, 20-30 color)
  3. SAM 3 (or Grounded SAM 2) → semantic labels
  4. Failure mode detection → modified depth + per-pixel confidence
  5. Scale calibration → corrected depth
  6. Backproject → labeled 3D point cloud with per-point confidence

Confidence derivation (from theory doc Section 2.5):
  - Depth smoothness: low local variance → high confidence
  - Point density: propagated during cell construction (more points → higher)
  - Semantic penalty: glass/mirror/transparent → low confidence
  - DUSt3R agreement: (multi-image only) model disagreement → low confidence

Models are run sequentially with torch.cuda.empty_cache() between each to avoid OOM.
"""

import numpy as np
import time
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class PerceptionOutput:
    """Output of the full perception pipeline."""
    positions: np.ndarray      # (N, 3) float32 — 3D point positions
    colors: np.ndarray         # (N, 3) float32 — RGB colors [0,1]
    normals: np.ndarray        # (N, 3) float32 — surface normals
    labels: np.ndarray         # (N,) int32 — semantic label per point
    confidence: np.ndarray     # (N,) float32 — per-point confidence [0,1]
    label_names: Dict[int, str]

    # Metadata
    focal_length: float
    scale_factor: float
    scale_confidence: float
    num_points: int
    processing_time_s: float


class PerceptionPipeline:
    """Orchestrates all perception models for cell grid construction.

    Usage:
        pipeline = PerceptionPipeline(device="cuda")
        pipeline.load_models()
        result = pipeline.run("room.jpg")
        # result.positions, result.colors, result.normals, result.labels, result.confidence
    """

    def __init__(
        self,
        device: str = "cuda",
        bilateral_spatial_sigma: float = 2.5,
        bilateral_color_sigma: float = 25.0,
    ):
        # Bilateral sigma default is 2.5 (not 3.5) because the 5x5 Sobel kernel
        # in gradient computation adds its own implicit smoothing. Combined,
        # sigma=3.5 + Sobel can over-smooth fine surface detail (brick mortar
        # grooves at ~5-10mm, wood grain). With sigma=2.5, the bilateral handles
        # per-pixel noise while the Sobel handles gradient noise — separation of
        # concerns. If Week 4 testing shows noisy gradients, increase to 3.0-3.5.
        self.device = device
        self.bilateral_spatial_sigma = bilateral_spatial_sigma
        self.bilateral_color_sigma = bilateral_color_sigma

        self._depth_model = None
        self._sam_model = None

    def load_models(self):
        """Load all perception models."""
        from .depth_pro import DepthProEstimator
        from .sam import SAMSegmenter

        self._depth_model = DepthProEstimator(device=self.device)
        self._depth_model.load()

        self._sam_model = SAMSegmenter(device=self.device)
        self._sam_model.load()

    def run(self, image_path: str) -> PerceptionOutput:
        """Run the full perception pipeline on a single image.

        Args:
            image_path: path to RGB image

        Returns:
            PerceptionOutput with labeled 3D point cloud + per-point confidence
        """
        from PIL import Image
        import torch

        t_start = time.time()
        image = np.array(Image.open(image_path).convert("RGB"))
        h, w = image.shape[:2]

        # --- Step 1: Depth Pro ---
        if self._depth_model._model is None:
            self._depth_model.load()
        depth_result = self._depth_model.estimate(image)
        depth_raw = depth_result["depth"]
        focal_length = depth_result["focal_length"]
        depth_confidence = depth_result["confidence_proxy"]

        # Free GPU memory before next model (re-loads on next call if needed)
        self._depth_model.unload()
        torch.cuda.empty_cache()

        # --- Step 2: Bilateral smoothing ---
        from .depth_smoothing import bilateral_depth_smooth
        depth_smooth = bilateral_depth_smooth(
            depth_raw, image,
            spatial_sigma=self.bilateral_spatial_sigma,
            color_sigma=self.bilateral_color_sigma,
        )

        # --- Step 3: SAM segmentation ---
        if self._sam_model._model is None:
            self._sam_model.load()
        labels_2d, label_names = self._sam_model.segment(image)
        self._sam_model.unload()
        torch.cuda.empty_cache()

        # --- Step 4: Failure mode detection + mitigation ---
        from .failure_modes import detect_failure_modes, apply_failure_mitigations
        failure_report = detect_failure_modes(image, labels_2d, label_names)
        depth_mitigated, failure_confidence = apply_failure_mitigations(
            depth_smooth, failure_report
        )

        # --- Step 5: Scale calibration ---
        from .scale_calibration import calibrate_depth_scale, apply_scale_correction
        scale_factor, scale_conf = calibrate_depth_scale(
            depth_mitigated, labels_2d, label_names, focal_length, h
        )
        depth_final = apply_scale_correction(depth_mitigated, scale_factor)

        # --- Step 6: Backproject to 3D ---
        positions, colors, normals, point_labels, point_confidence = self._backproject(
            depth_final, image, labels_2d, depth_confidence, failure_confidence,
            focal_length * scale_factor, h, w
        )

        t_elapsed = time.time() - t_start

        return PerceptionOutput(
            positions=positions,
            colors=colors,
            normals=normals,
            labels=point_labels,
            confidence=point_confidence,
            label_names=label_names,
            focal_length=focal_length,
            scale_factor=scale_factor,
            scale_confidence=scale_conf,
            num_points=len(positions),
            processing_time_s=t_elapsed,
        )

    def _backproject(
        self,
        depth: np.ndarray,
        image: np.ndarray,
        labels: np.ndarray,
        depth_confidence: np.ndarray,
        failure_confidence: np.ndarray,
        focal_length: float,
        h: int, w: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Backproject depth map to 3D point cloud with per-point confidence.

        Confidence = min(depth_smoothness_confidence, failure_mode_confidence).
        This ensures glass/mirror cells get low confidence even if depth appears smooth.
        """
        # Create pixel grid
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        u, v = np.meshgrid(u, v)

        # Valid depth mask (exclude sky and zero-depth pixels)
        valid = depth > 0.1  # minimum 10cm depth

        # Backproject: (u, v, depth) → (X, Y, Z)
        cx, cy = w / 2.0, h / 2.0
        x = (u - cx) * depth / focal_length
        y = (v - cy) * depth / focal_length
        z = depth

        # Extract valid points
        positions = np.stack([x[valid], y[valid], z[valid]], axis=-1)
        colors = image[valid].astype(np.float32) / 255.0
        point_labels = labels[valid]

        # Combined confidence: minimum of depth smoothness and failure mode signals
        combined_confidence = np.minimum(depth_confidence, failure_confidence)
        point_confidence = combined_confidence[valid]

        # Compute normals from depth gradient (cross product of partial derivatives)
        normals = self._compute_normals_from_depth(depth, focal_length, valid)

        return positions, colors, normals, point_labels, point_confidence

    def _compute_normals_from_depth(
        self,
        depth: np.ndarray,
        focal_length: float,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute surface normals from depth map gradients.

        Uses 5x5 Sobel kernel (not simple 2-point finite differences) for
        cleaner normals. See theory doc Section 9.2.
        """
        try:
            import cv2
            # 5x5 Sobel kernel for gradient computation
            dz_dx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=5)
            dz_dy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=5)
        except ImportError:
            # Fallback: 3x3 Sobel via numpy convolution
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
            sobel_y = sobel_x.T
            from scipy.ndimage import convolve
            dz_dx = convolve(depth, sobel_x)
            dz_dy = convolve(depth, sobel_y)

        # Normal from depth gradient: n = normalize(-dz/dx, -dz/dy, 1)
        nx = -dz_dx
        ny = -dz_dy
        nz = np.ones_like(depth)

        norm = np.sqrt(nx**2 + ny**2 + nz**2) + 1e-8
        nx /= norm
        ny /= norm
        nz /= norm

        # Extract valid normals
        normals = np.stack([nx[valid_mask], ny[valid_mask], nz[valid_mask]], axis=-1)
        return normals.astype(np.float32)

    def unload(self):
        """Free GPU memory from all perception models."""
        if self._depth_model is not None:
            self._depth_model.unload()
        if self._sam_model is not None:
            self._sam_model.unload()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
