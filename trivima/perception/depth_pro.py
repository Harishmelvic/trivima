"""
Depth Pro wrapper — Apple's metric depth estimation model.

Produces: metric depth map (meters), estimated focal length, boundary mask.
Precision: 5-8% AbsRel average on NYU-V2, 3-5% on favorable surfaces.
Speed: ~0.3s per 2.25MP image on A100.

Boundary precision note (from theory doc Section 2.3):
  - Boundary LOCALIZATION: 2-5mm (where edges are in 2D) — F1 > 0.9
  - Boundary DEPTH accuracy: 5-15mm (depth VALUES at edges) — worse due to mixed pixels
  These are different measurements. Cell density gradients at boundaries
  depend on depth accuracy, not just localization.
"""

import numpy as np
from typing import Tuple, Optional
from pathlib import Path


class DepthProEstimator:
    """Wrapper for Apple's Depth Pro metric depth estimation."""

    def __init__(self, device: str = "cuda", model_path: Optional[str] = None):
        self.device = device
        self._model = None
        self._transform = None
        self._model_path = model_path

    def load(self):
        """Load Depth Pro model. Call once before inference."""
        try:
            import depth_pro

            model, transform = depth_pro.create_model_and_transforms(device=self.device)
            model.eval()
            self._model = model
            self._transform = transform
            print("[DepthPro] Model loaded successfully")

        except ImportError:
            print("[DepthPro] depth_pro not installed.")
            print("  Clone: git clone https://github.com/apple/ml-depth-pro")
            print("  Install: pip install -e ml-depth-pro/")
            raise

    def estimate(self, image: np.ndarray) -> dict:
        """Estimate metric depth from a single RGB image.

        Args:
            image: (H, W, 3) uint8 RGB image

        Returns:
            dict with keys:
              depth: (H, W) float32 metric depth in meters
              focal_length: estimated focal length in pixels
              confidence_proxy: (H, W) float32 depth smoothness (for confidence)
        """
        if self._model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        import torch
        from PIL import Image

        # Convert to PIL for Depth Pro's transform
        pil_image = Image.fromarray(image)
        input_tensor = self._transform(pil_image).to(self.device)

        with torch.no_grad():
            prediction = self._model.infer(input_tensor)

        depth = prediction["depth"].squeeze().cpu().numpy().astype(np.float32)
        focal_length = float(prediction.get("focallength_px", 577.0))

        # Compute confidence proxy: local depth smoothness
        # Low variance → high confidence, high variance → low confidence
        from .depth_smoothing import compute_depth_local_variance
        variance = compute_depth_local_variance(depth, kernel_size=7)
        # Normalize: variance → confidence (inverse relationship)
        max_var = np.percentile(variance[variance > 0], 95) if (variance > 0).any() else 1.0
        confidence_proxy = 1.0 - np.clip(variance / (max_var + 1e-8), 0, 1)

        return {
            "depth": depth,
            "focal_length": focal_length,
            "confidence_proxy": confidence_proxy.astype(np.float32),
        }

    def unload(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            import torch
            torch.cuda.empty_cache()
