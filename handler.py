"""
RunPod Serverless Handler for Trivima.

Accepts a job with an image (base64 or URL), runs the full pipeline,
returns cell grid stats + optional PLY/preview outputs.

Input:
  {
    "input": {
      "image_base64": "<base64 encoded jpg/png>",   # OR
      "image_url": "https://example.com/room.jpg",   # one of these
      "cell_size": 0.05,                              # optional
      "bilateral_sigma": 2.5,                         # optional
      "output_ply": true,                             # optional
      "output_preview": true,                         # optional
      "synthetic_test": true                          # run synthetic test instead of real image
    }
  }

Output:
  {
    "stats": { ... cell grid statistics ... },
    "preview_base64": "<base64 png>",    # if output_preview=true
    "ply_base64": "<base64 ply>",        # if output_ply=true
    "timings": { ... }
  }
"""

import runpod
import base64
import time
import tempfile
import numpy as np
from pathlib import Path


def handler(job):
    """RunPod serverless handler."""
    job_input = job.get("input", {})

    t_start = time.time()

    # --- Synthetic test mode ---
    if job_input.get("synthetic_test", False):
        return run_synthetic_test()

    # --- Real image processing ---
    image_path = get_image(job_input)
    if not image_path:
        return {"error": "Provide image_base64 or image_url"}

    cell_size = job_input.get("cell_size", 0.05)
    bilateral_sigma = job_input.get("bilateral_sigma", 2.5)
    device = "cuda"

    try:
        result = run_pipeline(image_path, device, cell_size, bilateral_sigma)
    except Exception as e:
        return {"error": str(e), "stage": "pipeline"}

    # Build response
    response = {
        "stats": result["stats"],
        "timings": result["timings"],
    }

    # Optional PLY output
    if job_input.get("output_ply", False):
        ply_data = export_ply_string(result["grid_data"], cell_size)
        response["ply_base64"] = base64.b64encode(ply_data.encode()).decode()

    # Optional preview image
    if job_input.get("output_preview", False):
        preview_b64 = render_preview_base64(result["grid_data"], cell_size)
        response["preview_base64"] = preview_b64

    response["total_time_s"] = round(time.time() - t_start, 2)
    return response


def get_image(job_input: dict) -> str:
    """Download or decode the input image to a temp file. Returns path."""
    if "image_base64" in job_input:
        img_bytes = base64.b64decode(job_input["image_base64"])
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(img_bytes)
        tmp.close()
        return tmp.name

    if "image_url" in job_input:
        import urllib.request
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        urllib.request.urlretrieve(job_input["image_url"], tmp.name)
        return tmp.name

    return None


def run_pipeline(image_path: str, device: str, cell_size: float,
                 bilateral_sigma: float) -> dict:
    """Run the full Trivima pipeline."""
    from trivima.perception.pipeline import PerceptionPipeline
    from trivima.construction.point_to_grid import (
        build_cell_grid, apply_failure_mode_density_forcing
    )

    timings = {}

    # Perception
    t0 = time.time()
    pipeline = PerceptionPipeline(
        device=device,
        bilateral_spatial_sigma=bilateral_sigma,
    )
    pipeline.load_models()
    result = pipeline.run(image_path)
    timings["perception_s"] = round(time.time() - t0, 2)

    # Cell grid construction
    t0 = time.time()
    grid_data, build_stats = build_cell_grid(
        result.positions, result.colors, result.normals,
        result.labels, result.confidence,
        cell_size=cell_size,
    )
    apply_failure_mode_density_forcing(
        grid_data, None, result.label_names,
        result.positions, cell_size,
    )
    timings["grid_build_s"] = round(time.time() - t0, 2)

    # Shell extension
    t0 = time.time()
    grid_data = shell_extend(grid_data, cell_size)
    timings["shell_extension_s"] = round(time.time() - t0, 2)

    # Stats
    confidences = [c["confidence"] for c in grid_data.values()]
    stats = {
        "total_cells": len(grid_data),
        "solid_cells": build_stats.solid_cells,
        "surface_cells": build_stats.surface_cells,
        "avg_confidence": round(float(np.mean(confidences)), 3),
        "low_confidence_cells": sum(1 for c in confidences if c < 0.5),
        "points_extracted": result.num_points,
        "focal_length": round(result.focal_length, 1),
        "scale_factor": round(result.scale_factor, 4),
        "memory_mb": round(len(grid_data) * 512 / 1024 / 1024, 1),
    }

    return {"grid_data": grid_data, "stats": stats, "timings": timings}


def shell_extend(grid_data: dict, cell_size: float) -> dict:
    """Simple floor extension."""
    floor_cells = [k for k, v in grid_data.items() if v["normal"][1] > 0.9]
    if not floor_cells:
        return grid_data

    from collections import Counter
    floor_y = Counter(k[1] for k in floor_cells).most_common(1)[0][0]

    all_xs = [k[0] for k in grid_data]
    all_zs = [k[2] for k in grid_data]
    x_min, x_max = min(all_xs) - 5, max(all_xs) + 5
    z_min, z_max = min(all_zs) - 5, max(all_zs) + 5

    floor_albedos = [grid_data[k]["albedo"] for k in floor_cells if k in grid_data]
    floor_albedo = np.mean(floor_albedos, axis=0) if floor_albedos else np.array([0.4, 0.35, 0.3])

    for ix in range(x_min, x_max + 1):
        for iz in range(z_min, z_max + 1):
            key = (ix, floor_y, iz)
            if key not in grid_data:
                grid_data[key] = {
                    "density": 1.0, "cell_type": 2,
                    "albedo": floor_albedo.copy(),
                    "normal": np.array([0.0, 1.0, 0.0]),
                    "label": 0, "confidence": 0.6,
                    "collision_margin": 0.0,
                    "density_integral": cell_size ** 3,
                    "density_gradient": np.zeros(3),
                    "albedo_gradient": np.zeros(3),
                    "normal_gradient": np.zeros(3),
                    "neighbors": [{"type": 0, "density": 0, "normal_y": 0, "light_luma": 0}] * 6,
                }
    return grid_data


def run_synthetic_test() -> dict:
    """Run the synthetic test — validates pipeline without ML models."""
    import sys
    sys.path.insert(0, "/workspace/trivima")
    from scripts.run_demo import (
        create_synthetic_scene, test_grid_construction,
        test_lod_subdivision, test_collision
    )

    t0 = time.time()
    grid_data = create_synthetic_scene(0.05)
    test_grid_construction(grid_data, 0.05)
    test_lod_subdivision(grid_data)
    test_collision(grid_data, 0.05)

    confidences = [c["confidence"] for c in grid_data.values()]

    return {
        "status": "all_tests_passed",
        "total_cells": len(grid_data),
        "avg_confidence": round(float(np.mean(confidences)), 3),
        "low_confidence_cells": sum(1 for c in confidences if c < 0.5),
        "time_s": round(time.time() - t0, 2),
    }


def export_ply_string(grid_data: dict, cell_size: float) -> str:
    """Export cell grid as PLY string."""
    lines = [
        "ply", "format ascii 1.0",
        f"element vertex {len(grid_data)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    for (ix, iy, iz), cell in grid_data.items():
        x = (ix + 0.5) * cell_size
        y = (iy + 0.5) * cell_size
        z = (iz + 0.5) * cell_size
        a = cell["albedo"]
        r, g, b = int(a[0]*255), int(a[1]*255), int(a[2]*255)
        lines.append(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}")
    return "\n".join(lines)


def render_preview_base64(grid_data: dict, cell_size: float) -> str:
    """Render top-down preview, return as base64 PNG."""
    from PIL import Image
    import io

    keys = list(grid_data.keys())
    xs = [k[0] for k in keys]
    zs = [k[2] for k in keys]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)

    w = x_max - x_min + 1
    h = z_max - z_min + 1
    scale = max(1, 512 // max(w, h))

    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)
    for (ix, iy, iz), cell in grid_data.items():
        px = (ix - x_min) * scale
        py = (iz - z_min) * scale
        a = cell["albedo"]
        color = (np.clip(a * 255, 0, 255)).astype(np.uint8)
        if cell.get("confidence", 1.0) < 0.5:
            color = (np.array([255, 128, 0]) * 0.6 + color * 0.4).astype(np.uint8)
        img[py:py+scale, px:px+scale] = color

    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# RunPod serverless entry point
runpod.serverless.start({"handler": handler})
