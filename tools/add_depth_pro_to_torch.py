"""
Add Depth Pro pseudo-depth to existing .torch scene files.

Reads each scene file, runs each frame through Depth Pro (or DepthAnythingV2 as
fallback), writes back the same .torch with `depths` and `depth_mask` populated.

Depth Pro produces metric depth in absolute meters, which is what GS-LRM needs.
DepthAnythingV2 (the fallback) produces relative depth — we rescale per-scene
using the camera baselines as a metric anchor.

Usage in Colab:
    !pip install -q depth_pro || pip install -q git+https://github.com/apple/ml-depth-pro
    python tools/add_depth_pro_to_torch.py --in_dir /content/data/torch_packed
"""

from __future__ import annotations

import argparse
import glob
import io
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _bytes(x):
    if isinstance(x, torch.Tensor):
        return x.numpy().tobytes()
    if isinstance(x, (bytes, bytearray)):
        return bytes(x)
    return bytes(x)


# ---------------------------------------------------------------------------
# Depth backend selection
# ---------------------------------------------------------------------------


class DepthBackend:
    """Lightweight wrapper around whichever depth model we can load."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.model = None
        self.transform = None
        self.kind = None
        self._init()

    def _init(self):
        # Try Depth Pro first (metric depth, best quality)
        try:
            import depth_pro

            print("  Loading Depth Pro...")
            self.model, self.transform = depth_pro.create_model_and_transforms(
                device=self.device, precision=torch.float16
            )
            self.model.eval()
            self.kind = "depth_pro"
            print("  Depth Pro loaded")
            return
        except Exception as e:
            print(f"  Depth Pro unavailable ({e}), trying DepthAnythingV2...")

        # Fallback: DepthAnythingV2 via transformers
        try:
            from transformers import AutoImageProcessor, AutoModelForDepthEstimation

            print("  Loading DepthAnythingV2-Base...")
            self.processor = AutoImageProcessor.from_pretrained("depth-anything/Depth-Anything-V2-Base-hf")
            self.model = AutoModelForDepthEstimation.from_pretrained(
                "depth-anything/Depth-Anything-V2-Base-hf"
            ).to(self.device).eval()
            self.kind = "depth_anything"
            print("  DepthAnythingV2 loaded")
            return
        except Exception as e:
            print(f"  DepthAnythingV2 unavailable ({e})")
            raise RuntimeError("No depth backend available. Install depth_pro or transformers")

    @torch.inference_mode()
    def predict(self, pil_image: Image.Image) -> np.ndarray:
        """Return a HxW float32 depth map in meters (or relative units for DepthAnything)."""
        if self.kind == "depth_pro":
            import depth_pro

            inp = self.transform(pil_image).to(self.device)
            pred = self.model.infer(inp)
            d = pred["depth"].squeeze().float().cpu().numpy()
            return d
        else:
            # DepthAnythingV2
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.autocast(self.device, dtype=torch.float16):
                outputs = self.model(**inputs)
            d = outputs.predicted_depth.squeeze().float().cpu().numpy()
            # DepthAnything returns INVERSE depth — convert to metric-ish
            # Pixel value is "disparity" / "inverse depth"; bigger = closer
            d = d - d.min() + 1e-3
            d = 1.0 / d  # convert to depth-like
            return d


# ---------------------------------------------------------------------------
# Per-scene metric calibration
# ---------------------------------------------------------------------------


def calibrate_to_metric(rel_depth: np.ndarray, camera_baseline: float) -> np.ndarray:
    """For relative depth (DepthAnything), rescale so the median equals scene_baseline.

    This is a coarse heuristic — for production use Depth Pro which is already metric.
    """
    med = float(np.median(rel_depth))
    if med < 1e-3:
        return rel_depth
    scale = camera_baseline / med
    return rel_depth * scale


# ---------------------------------------------------------------------------
# Process one .torch file
# ---------------------------------------------------------------------------


def add_depth_to_scene(path: str, backend: DepthBackend, target_size: int = 256, force: bool = False):
    pkg = torch.load(path, map_location="cpu", weights_only=False)
    if pkg.get("depths") is not None and not force:
        print(f"  {os.path.basename(path)}: already has depths, skipping")
        return False

    images = pkg["images"]
    cameras = pkg["cameras"]

    # Estimate scene baseline from camera path (median translation between consecutive frames)
    ext = cameras[:, 6:18].reshape(-1, 3, 4)
    translations = ext[:, :, 3].numpy()
    if len(translations) > 1:
        diffs = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        baseline = max(0.5, float(np.median(diffs)) * 5.0)  # crude
    else:
        baseline = float(pkg.get("scene_scale", 5.0))

    depths = []
    masks = []

    for i, img_bytes in enumerate(images):
        img = Image.open(io.BytesIO(_bytes(img_bytes))).convert("RGB")
        d = backend.predict(img)  # HxW float

        if backend.kind != "depth_pro":
            d = calibrate_to_metric(d, baseline)

        # Resize to target_size to match what the dataset loader resizes to
        d_t = torch.from_numpy(d.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        d_resized = F.interpolate(d_t, size=(target_size, target_size), mode="nearest").squeeze()

        # Mask: positive depths
        m = d_resized > 0
        depths.append(d_resized)
        masks.append(m)

    pkg["depths"] = torch.stack(depths)
    pkg["depth_mask"] = torch.stack(masks)

    # Update scene_scale to depth-based estimate
    valid = pkg["depths"][pkg["depth_mask"]]
    if valid.numel() > 0:
        pkg["scene_scale"] = float(valid.quantile(0.95).item())

    torch.save(pkg, path)
    print(
        f"  {os.path.basename(path)}: added {len(depths)} depths, "
        f"range=[{valid.min():.2f}, {valid.max():.2f}], scale={pkg['scene_scale']:.2f}m"
    )
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True, help="Directory of packed .torch files")
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument("--force", action="store_true", help="Re-process even if depths already exist")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "*.torch")))
    if not files:
        raise SystemExit(f"No .torch files in {args.in_dir}")
    print(f"Found {len(files)} scene files")

    backend = DepthBackend(device=args.device)

    n_done = 0
    for f in files:
        try:
            if add_depth_to_scene(f, backend, target_size=args.target_size, force=args.force):
                n_done += 1
        except Exception as e:
            print(f"  {os.path.basename(f)}: FAILED — {e}")

    print(f"\nProcessed: {n_done}/{len(files)} files")


if __name__ == "__main__":
    main()
