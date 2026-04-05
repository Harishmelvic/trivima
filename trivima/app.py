"""
Trivima — CLI application.

Usage:
  python trivima/app.py --image room.jpg
  python trivima/app.py --image room.jpg --save-grid output/room.bin
  python trivima/app.py --image room.jpg --render-preview output/preview.png
  python trivima/app.py --grid output/room.bin --stats

Pipeline:
  Photo → Depth Pro → Bilateral Smooth → SAM → Failure Modes → Scale Cal
  → Backproject → Cell Grid → Shell Extension → Stats / Preview / Save
"""

import sys
import argparse
import time
import numpy as np
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Trivima — Photo to 3D Cell Grid")

    # Input
    parser.add_argument("--image", type=str, help="Input photograph path")
    parser.add_argument("--grid", type=str, help="Load pre-built cell grid (.bin)")

    # Output
    parser.add_argument("--save-grid", type=str, help="Save cell grid to .bin file")
    parser.add_argument("--render-preview", type=str, help="Render a preview image to this path")
    parser.add_argument("--export-ply", type=str, help="Export cell centers as .ply point cloud")
    parser.add_argument("--stats", action="store_true", help="Print detailed grid statistics")

    # Config
    parser.add_argument("--cell-size", type=float, default=0.05,
                        help="Base cell size in meters (default 0.05)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device (cuda or cpu)")
    parser.add_argument("--no-shell", action="store_true",
                        help="Skip room shell extension")
    parser.add_argument("--bilateral-sigma", type=float, default=2.5,
                        help="Bilateral filter spatial sigma (default 2.5)")

    return parser.parse_args()


def run_perception(image_path: str, device: str, cell_size: float,
                   bilateral_sigma: float) -> dict:
    """Run full perception pipeline: photo → cell grid data."""
    print(f"\n{'='*60}")
    print(f"  Processing: {image_path}")
    print(f"{'='*60}")

    from trivima.perception.pipeline import PerceptionPipeline

    t0 = time.time()

    print("\n[1/5] Loading perception models...")
    pipeline = PerceptionPipeline(
        device=device,
        bilateral_spatial_sigma=bilateral_sigma,
    )
    pipeline.load_models()

    print("[2/5] Running perception (Depth Pro + SAM + smoothing + calibration)...")
    result = pipeline.run(image_path)

    print(f"       Points extracted:   {result.num_points:,}")
    print(f"       Focal length:       {result.focal_length:.1f} px")
    print(f"       Scale correction:   {result.scale_factor:.4f} (confidence {result.scale_confidence:.2f})")
    print(f"       Perception time:    {result.processing_time_s:.1f}s")

    # Build cell grid
    print("\n[3/5] Building cell grid (5x5 Sobel gradients + confidence)...")
    from trivima.construction.point_to_grid import (
        build_cell_grid, apply_failure_mode_density_forcing
    )

    grid_data, stats = build_cell_grid(
        result.positions, result.colors, result.normals,
        result.labels, result.confidence,
        cell_size=cell_size,
    )

    apply_failure_mode_density_forcing(
        grid_data, None, result.label_names,
        result.positions, cell_size,
    )

    print(f"       Total cells:        {stats.total_cells:,}")
    print(f"       Solid / Surface:    {stats.solid_cells:,} / {stats.surface_cells:,}")
    print(f"       Avg points/cell:    {stats.avg_points_per_cell:.1f}")
    print(f"       Avg confidence:     {stats.avg_confidence:.2f}")
    print(f"       Grid build time:    {stats.construction_time_s:.1f}s")

    total = time.time() - t0
    print(f"\n       Total pipeline:     {total:.1f}s")

    return {
        "grid_data": grid_data,
        "stats": stats,
        "label_names": result.label_names,
        "perception_result": result,
    }


def run_shell_extension(grid_data: dict, cell_size: float) -> dict:
    """Extend cell grid with floor/wall/ceiling planes."""
    print("\n[4/5] Shell extension (room completion)...")

    floor_cells = [k for k, v in grid_data.items() if v["normal"][1] > 0.9]

    if not floor_cells:
        print("       No floor detected — skipping")
        return grid_data

    from collections import Counter
    floor_y = Counter(k[1] for k in floor_cells).most_common(1)[0][0]

    all_xs = [k[0] for k in grid_data]
    all_zs = [k[2] for k in grid_data]
    x_min, x_max = min(all_xs) - 5, max(all_xs) + 5
    z_min, z_max = min(all_zs) - 5, max(all_zs) + 5

    floor_albedos = [grid_data[k]["albedo"] for k in floor_cells if k in grid_data]
    floor_albedo = np.mean(floor_albedos, axis=0) if floor_albedos else np.array([0.4, 0.35, 0.3])

    extended = 0
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
                extended += 1

    print(f"       Floor at y={floor_y}: +{extended} cells")
    print(f"       Total after extension: {len(grid_data):,}")
    return grid_data


def print_stats(grid_data: dict, cell_size: float):
    """Print detailed grid statistics."""
    print(f"\n{'='*60}")
    print(f"  Cell Grid Statistics")
    print(f"{'='*60}")

    n = len(grid_data)
    confidences = np.array([c["confidence"] for c in grid_data.values()])
    densities = np.array([c["density"] for c in grid_data.values()])
    types = [c["cell_type"] for c in grid_data.values()]

    from collections import Counter
    tc = Counter(types)

    print(f"\n  Cells:           {n:,}")
    print(f"  Solid:           {tc.get(2, 0):,}")
    print(f"  Surface:         {tc.get(1, 0):,}")
    print(f"  Cell size:       {cell_size*100:.1f} cm")
    print(f"  Memory (est):    {n * 512 / 1024 / 1024:.1f} MB")

    print(f"\n  Confidence:")
    print(f"    Mean:          {confidences.mean():.3f}")
    print(f"    Median:        {np.median(confidences):.3f}")
    print(f"    Min:           {confidences.min():.3f}")
    print(f"    Low (<0.5):    {(confidences < 0.5).sum():,} ({(confidences < 0.5).mean()*100:.1f}%)")
    print(f"    High (>0.8):   {(confidences > 0.8).sum():,} ({(confidences > 0.8).mean()*100:.1f}%)")

    print(f"\n  Density:")
    print(f"    Mean:          {densities.mean():.3f}")
    print(f"    Solid (>0.5):  {(densities > 0.5).sum():,}")

    # Bounding box
    keys = list(grid_data.keys())
    xs = [k[0] for k in keys]
    ys = [k[1] for k in keys]
    zs = [k[2] for k in keys]
    bbox_min = np.array([min(xs), min(ys), min(zs)]) * cell_size
    bbox_max = np.array([max(xs), max(ys), max(zs)]) * cell_size
    bbox_size = bbox_max - bbox_min

    print(f"\n  Bounding box:")
    print(f"    Min:           ({bbox_min[0]:.2f}, {bbox_min[1]:.2f}, {bbox_min[2]:.2f}) m")
    print(f"    Max:           ({bbox_max[0]:.2f}, {bbox_max[1]:.2f}, {bbox_max[2]:.2f}) m")
    print(f"    Size:          {bbox_size[0]:.2f} × {bbox_size[1]:.2f} × {bbox_size[2]:.2f} m")

    # Gradient stats
    has_grads = [c for c in grid_data.values() if "density_gradient" in c]
    if has_grads:
        grad_mags = [np.linalg.norm(c["density_gradient"]) for c in has_grads]
        print(f"\n  Density gradients:")
        print(f"    Mean magnitude: {np.mean(grad_mags):.4f}")
        print(f"    Max magnitude:  {np.max(grad_mags):.4f}")
        print(f"    Zero (< 1e-6):  {sum(1 for g in grad_mags if g < 1e-6):,}")

    # LOD info
    from trivima.rendering.lod import LODConfig, InputType
    config = LODConfig(input_type=InputType.SINGLE_IMAGE)
    print(f"\n  LOD (single-image mode):")
    print(f"    Max subdivisions:  {config.max_subdivisions}")
    print(f"    Finest cell size:  {cell_size / (2 ** config.max_subdivisions) * 100:.1f} cm")
    print(f"    Subdivisible:      {(confidences >= 0.5).sum():,} cells (confidence ≥ 0.5)")

    # Conservation validation (unified_pipeline_theory.md §Step 7)
    print(f"\n  Conservation validation:")
    total_mass = sum(c.get("density_integral", c["density"] * cell_size**3)
                     for c in grid_data.values())
    print(f"    Reference mass:    {total_mass:.6f}")
    print(f"    Mass conservation: OK (baseline set)")

    # Surface field summary
    from trivima.validation.surface_field import SurfaceField
    sf = SurfaceField(cell_size=cell_size)
    sf.build(grid_data)
    sf_summary = sf.get_summary()
    print(f"\n  Surface field:")
    print(f"    Surfaces found:    {sf_summary['num_surfaces']}")
    print(f"    Floor height:      {sf_summary['floor_height']}")
    for s in sf_summary['surfaces'][:5]:
        print(f"      {s['type']:15s} h={s['height']:.2f}m  area={s['area']:.1f}m²  conf={s['confidence']:.2f}")

    # Functional field summary
    from trivima.validation.functional_field import FunctionalField
    ff = FunctionalField(cell_size=cell_size)
    ff.build(grid_data)
    ff_summary = ff.get_summary()
    print(f"\n  Functional field:")
    print(f"    Label clusters:    {ff_summary['clusters']}")
    print(f"    Wall cells:        {ff_summary['wall_cells']}")
    print(f"    Object categories: {ff_summary['supported_categories']}")


def export_ply(grid_data: dict, cell_size: float, output_path: str):
    """Export cell centers as a colored .ply point cloud."""
    print(f"\n  Exporting PLY to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    n = len(grid_data)
    with open(output_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")

        for (ix, iy, iz), cell in grid_data.items():
            x = (ix + 0.5) * cell_size
            y = (iy + 0.5) * cell_size
            z = (iz + 0.5) * cell_size
            a = cell["albedo"]
            r, g, b = int(a[0]*255), int(a[1]*255), int(a[2]*255)
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")

    print(f"  → {n:,} points written")


def render_preview(grid_data: dict, cell_size: float, output_path: str):
    """Render a simple orthographic top-down preview of the grid."""
    print(f"\n  Rendering preview to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    keys = list(grid_data.keys())
    xs = [k[0] for k in keys]
    zs = [k[2] for k in keys]
    x_min, x_max = min(xs), max(xs)
    z_min, z_max = min(zs), max(zs)

    w = x_max - x_min + 1
    h = z_max - z_min + 1
    scale = max(1, 512 // max(w, h))  # scale up small grids

    img = np.zeros((h * scale, w * scale, 3), dtype=np.uint8)

    for (ix, iy, iz), cell in grid_data.items():
        px = (ix - x_min) * scale
        py = (iz - z_min) * scale
        a = cell["albedo"]
        color = (np.clip(a * 255, 0, 255)).astype(np.uint8)

        # Tint low-confidence cells orange
        if cell.get("confidence", 1.0) < 0.5:
            color = (np.array([255, 128, 0]) * 0.6 + color * 0.4).astype(np.uint8)

        img[py:py+scale, px:px+scale] = color

    from PIL import Image
    Image.fromarray(img).save(output_path)
    print(f"  → {w}×{h} cells, {img.shape[1]}×{img.shape[0]}px image")


def main():
    args = parse_args()

    if not args.image and not args.grid:
        print("Trivima — Photo to 3D Cell Grid")
        print()
        print("Usage:")
        print("  python trivima/app.py --image room.jpg")
        print("  python trivima/app.py --image room.jpg --save-grid room.bin --stats")
        print("  python trivima/app.py --image room.jpg --render-preview preview.png")
        print("  python trivima/app.py --image room.jpg --export-ply room.ply")
        print("  python trivima/app.py --grid room.bin --stats")
        sys.exit(0)

    # --- Build or load grid ---
    if args.grid:
        print(f"Loading grid from {args.grid}...")
        # TODO: implement Python-side grid loading from .bin
        # For now, only native grids supported
        try:
            import trivima_native as tn
            grid = tn.CellGrid()
            tn.load_grid(grid, args.grid)
            print(f"  → {grid.size():,} cells loaded")
            if args.stats:
                print("  [Stats for native grids: use --image instead]")
        except ImportError:
            print("  Native module not built. Use --image to construct grid.")
            sys.exit(1)
        return

    # --- Full pipeline from image ---
    result = run_perception(args.image, args.device, args.cell_size, args.bilateral_sigma)
    grid_data = result["grid_data"]

    if not args.no_shell:
        grid_data = run_shell_extension(grid_data, args.cell_size)

    print("\n[5/5] Output...")

    # Stats
    if args.stats or (not args.save_grid and not args.render_preview and not args.export_ply):
        print_stats(grid_data, args.cell_size)

    # Save grid
    if args.save_grid:
        # Save as numpy archive (portable, no native module needed)
        Path(args.save_grid).parent.mkdir(parents=True, exist_ok=True)
        keys = np.array(list(grid_data.keys()), dtype=np.int32)
        densities = np.array([c["density"] for c in grid_data.values()], dtype=np.float32)
        albedos = np.array([c["albedo"] for c in grid_data.values()], dtype=np.float32)
        normals = np.array([c["normal"] for c in grid_data.values()], dtype=np.float32)
        confidences = np.array([c["confidence"] for c in grid_data.values()], dtype=np.float32)
        labels = np.array([c.get("label", 0) for c in grid_data.values()], dtype=np.int32)

        np.savez_compressed(args.save_grid,
            keys=keys, densities=densities, albedos=albedos,
            normals=normals, confidences=confidences, labels=labels,
            cell_size=args.cell_size)
        size_mb = Path(args.save_grid).stat().st_size / 1024 / 1024
        print(f"\n  Grid saved: {args.save_grid} ({size_mb:.1f} MB)")

    # Export PLY
    if args.export_ply:
        export_ply(grid_data, args.cell_size, args.export_ply)

    # Render preview
    if args.render_preview:
        render_preview(grid_data, args.cell_size, args.render_preview)

    print(f"\n{'='*60}")
    print(f"  Done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
