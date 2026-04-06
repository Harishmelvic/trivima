#!/usr/bin/env python3
"""
Trivima — Gaussian Splatting Pipeline
=======================================
Photo → Depth Pro → SAM → Gaussians → gsplat render

No EGL. No OpenGL. No cells. No cubes. Pure CUDA.

Usage:
    python pipeline_gaussian.py test_room.jpg output_gaussian
"""

import os, sys, math, time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "test_room.jpg"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "output_gaussian"
    subsample = int(sys.argv[3]) if len(sys.argv) > 3 else 2

    print("=" * 60)
    print("  Trivima — Gaussian Splatting Pipeline")
    print(f"  Image: {image_path}")
    print(f"  Subsample: {subsample} (every {subsample}th pixel)")
    print("=" * 60)

    t_total = time.time()

    # ============================================================
    # Step 1: Depth Pro
    # ============================================================
    import torch
    from trivima.perception.depth_pro import DepthProEstimator
    from trivima.perception.depth_smoothing import bilateral_depth_smooth

    print("\n[1/4] Depth Pro...")
    t0 = time.time()
    model = DepthProEstimator(device="cuda")
    model.load()
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    result = model.estimate(image)
    depth = result["depth"]
    focal = result["focal_length"]
    model.unload()
    torch.cuda.empty_cache()

    smoothed = bilateral_depth_smooth(depth, image, spatial_sigma=3.0, color_sigma=25.0)
    fov_deg = math.degrees(2.0 * math.atan(h / (2.0 * focal)))
    print(f"  {w}x{h}, focal={focal:.0f}, FOV={fov_deg:.1f}")
    print(f"  Depth: {smoothed[smoothed>0.1].min():.2f} - {smoothed.max():.2f}m")
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 2: SAM segmentation
    # ============================================================
    print("\n[2/4] SAM segmentation...")
    t0 = time.time()
    from ultralytics import SAM
    sam_model = SAM("sam2.1_l.pt")
    sam_results = sam_model(image_path)
    sam_r = sam_results[0]

    pixel_labels = np.zeros((h, w), dtype=np.int32)
    if sam_r.masks:
        masks = sam_r.masks.data.cpu().numpy()
        if masks.shape[1:] != (h, w):
            resized = []
            for m in masks:
                rm = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((w, h), Image.NEAREST)) > 127
                resized.append(rm)
            masks = np.array(resized)
        areas = [(i, m.sum()) for i, m in enumerate(masks)]
        areas.sort(key=lambda x: -x[1])
        for idx, area in areas:
            pixel_labels[masks[idx]] = idx + 1
        print(f"  {len(masks)} segments")
    else:
        print(f"  No segments found")

    del sam_model
    torch.cuda.empty_cache()
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 3: Convert to Gaussians
    # ============================================================
    print("\n[3/4] Points to Gaussians...")
    t0 = time.time()
    from trivima.gaussian.point_to_gaussians import points_to_gaussians

    gaussians = points_to_gaussians(
        image, smoothed, focal,
        pixel_labels=pixel_labels,
        subsample=subsample,
    )
    n_gaussians = len(gaussians["positions"])
    print(f"  Time: {time.time()-t0:.1f}s")

    # ============================================================
    # Step 4: Render with gsplat
    # ============================================================
    print("\n[4/4] Rendering with gsplat...")
    t0 = time.time()
    from trivima.gaussian.renderer import render_multi_view

    render_multi_view(gaussians, fov_deg, output_dir)
    print(f"  Time: {time.time()-t0:.1f}s")

    # Summary
    dt = time.time() - t_total
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Gaussians:    {n_gaussians:,}")
    print(f"  Total time:   {dt:.1f}s")
    print(f"  Output:       {output_dir}/")


if __name__ == "__main__":
    main()
