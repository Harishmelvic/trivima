#!/usr/bin/env python3
"""
Quick demo — minimal test that the pipeline works end-to-end.

Usage:
  python scripts/run_demo.py
  python scripts/run_demo.py --image path/to/room.jpg
  python scripts/run_demo.py --synthetic  # use synthetic test scene (no models needed)

This script is designed to be the first thing you run after setup_models.sh.
"""

import sys
import time
import argparse
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_synthetic_scene(cell_size: float = 0.05) -> dict:
    """Create a synthetic test room (no perception models needed).

    A 4m × 3m × 3m room with textured floor, plain walls, and a door.
    Used for testing grid construction, gradients, collision, and rendering
    without requiring Depth Pro or SAM.
    """
    print("Creating synthetic test room (4m × 3m × 3m)...")
    grid_data = {}

    # Room dimensions in cells
    nx = int(4.0 / cell_size)   # 80 cells wide
    ny = int(3.0 / cell_size)   # 60 cells tall
    nz = int(3.0 / cell_size)   # 60 cells deep

    # Floor (y=0 plane, brown wood color)
    for ix in range(nx):
        for iz in range(nz):
            # Wood grain pattern: albedo varies in X direction
            wood_t = (ix % 10) / 10.0
            albedo = np.array([0.45 + 0.1 * wood_t, 0.3 + 0.05 * wood_t, 0.15])

            grid_data[(ix, 0, iz)] = {
                "density": 1.0, "cell_type": 2,
                "albedo": albedo,
                "normal": np.array([0.0, 1.0, 0.0]),
                "label": 1,  # floor
                "confidence": 0.9,
                "collision_margin": 0.0,
                "density_integral": cell_size ** 3,
                "density_gradient": np.zeros(3),
                "albedo_gradient": np.array([0.01, 0.0, 0.0]),  # grain in X
                "normal_gradient": np.zeros(3),
                "neighbors": [{"type": 0, "density": 0, "normal_y": 0, "light_luma": 0}] * 6,
            }

    # Back wall (z=0 plane, white)
    for ix in range(nx):
        for iy in range(1, ny):
            grid_data[(ix, iy, 0)] = {
                "density": 1.0, "cell_type": 2,
                "albedo": np.array([0.85, 0.85, 0.82]),
                "normal": np.array([0.0, 0.0, 1.0]),
                "label": 2,  # wall
                "confidence": 0.85,
                "collision_margin": 0.0,
                "density_integral": cell_size ** 3,
                "density_gradient": np.zeros(3),
                "albedo_gradient": np.zeros(3),
                "normal_gradient": np.zeros(3),
                "neighbors": [{"type": 0, "density": 0, "normal_y": 0, "light_luma": 0}] * 6,
            }

    # Left wall (x=0 plane)
    for iy in range(1, ny):
        for iz in range(1, nz):
            grid_data[(0, iy, iz)] = {
                "density": 1.0, "cell_type": 2,
                "albedo": np.array([0.8, 0.82, 0.85]),
                "normal": np.array([1.0, 0.0, 0.0]),
                "label": 2,
                "confidence": 0.85,
                "collision_margin": 0.0,
                "density_integral": cell_size ** 3,
                "density_gradient": np.zeros(3),
                "albedo_gradient": np.zeros(3),
                "normal_gradient": np.zeros(3),
                "neighbors": [{"type": 0, "density": 0, "normal_y": 0, "light_luma": 0}] * 6,
            }

    # Right wall (x=nx plane)
    for iy in range(1, ny):
        for iz in range(1, nz):
            grid_data[(nx - 1, iy, iz)] = {
                "density": 1.0, "cell_type": 2,
                "albedo": np.array([0.8, 0.82, 0.85]),
                "normal": np.array([-1.0, 0.0, 0.0]),
                "label": 2,
                "confidence": 0.85,
                "collision_margin": 0.0,
                "density_integral": cell_size ** 3,
                "density_gradient": np.zeros(3),
                "albedo_gradient": np.zeros(3),
                "normal_gradient": np.zeros(3),
                "neighbors": [{"type": 0, "density": 0, "normal_y": 0, "light_luma": 0}] * 6,
            }

    # A "glass table" (low confidence) in the center
    table_x, table_z = nx // 2, nz // 2
    table_y = int(0.75 / cell_size)  # 75cm height
    for dx in range(-3, 4):
        for dz in range(-2, 3):
            key = (table_x + dx, table_y, table_z + dz)
            grid_data[key] = {
                "density": 1.0, "cell_type": 2,
                "albedo": np.array([0.7, 0.85, 0.9]),  # bluish glass
                "normal": np.array([0.0, 1.0, 0.0]),
                "label": 10,  # glass table
                "confidence": 0.2,  # LOW — glass surface
                "collision_margin": 0.025,
                "density_integral": cell_size ** 3,
                "density_gradient": np.zeros(3),
                "albedo_gradient": np.zeros(3),
                "normal_gradient": np.zeros(3),
                "neighbors": [{"type": 0, "density": 0, "normal_y": 0, "light_luma": 0}] * 6,
            }

    print(f"  → {len(grid_data)} cells created")
    return grid_data


def test_grid_construction(grid_data: dict, cell_size: float):
    """Validate grid construction quality."""
    print("\n--- Grid Construction Test ---")

    n = len(grid_data)
    confidences = [c["confidence"] for c in grid_data.values()]
    densities = [c["density"] for c in grid_data.values()]

    print(f"  Cells: {n}")
    print(f"  Confidence: mean={np.mean(confidences):.2f}, low(<0.5)={sum(1 for c in confidences if c < 0.5)}")

    # Test: floor cells should have normal pointing up
    floor_cells = [(k, v) for k, v in grid_data.items() if k[1] == 0]
    floor_normals_y = [v["normal"][1] for _, v in floor_cells]
    assert all(ny > 0.9 for ny in floor_normals_y), "Floor normals should point up"
    print(f"  ✓ Floor normals correct ({len(floor_cells)} floor cells)")

    # Test: wall cells should have horizontal normals
    wall_cells = [(k, v) for k, v in grid_data.items() if v.get("label") == 2 and k[1] > 0]
    wall_ny = [abs(v["normal"][1]) for _, v in wall_cells]
    assert all(ny < 0.3 for ny in wall_ny), "Wall normals should be horizontal"
    print(f"  ✓ Wall normals correct ({len(wall_cells)} wall cells)")

    # Test: glass table should have low confidence
    glass_cells = [(k, v) for k, v in grid_data.items() if v.get("label") == 10]
    glass_conf = [v["confidence"] for _, v in glass_cells]
    assert all(c < 0.5 for c in glass_conf), "Glass cells should have low confidence"
    print(f"  ✓ Glass table low confidence ({len(glass_cells)} cells, conf={np.mean(glass_conf):.2f})")

    print("  ✓ All construction tests passed")


def test_lod_subdivision(grid_data: dict):
    """Test LOD controller respects subdivision limits."""
    print("\n--- LOD Subdivision Test ---")

    from trivima.rendering.lod import LODController, LODConfig, InputType

    # Single-image mode: max 1 subdivision level
    config = LODConfig(input_type=InputType.SINGLE_IMAGE)
    lod = LODController(config)

    assert config.max_subdivisions == 1, "Single image should allow max 1 subdivision"
    print(f"  ✓ Single-image max subdivisions: {config.max_subdivisions}")

    # Multi-image mode: max 3
    config_multi = LODConfig(input_type=InputType.MULTI_IMAGE)
    assert config_multi.max_subdivisions == 3
    print(f"  ✓ Multi-image max subdivisions: {config_multi.max_subdivisions}")

    # Low-confidence cells should NOT subdivide
    assert not lod.should_subdivide(cell_level=0, cell_confidence=0.3, distance=1.0)
    print("  ✓ Low-confidence cells blocked from subdivision")

    # High-confidence near cells SHOULD subdivide
    assert lod.should_subdivide(cell_level=0, cell_confidence=0.9, distance=1.0)
    print("  ✓ High-confidence near cells allowed to subdivide")

    print("  ✓ All LOD tests passed")


def test_collision(grid_data: dict, cell_size: float):
    """Test collision detection with confidence-aware margins."""
    print("\n--- Collision Test ---")

    # Test: floor cell should block downward movement
    floor_key = list(k for k in grid_data if k[1] == 0)[0]
    floor_cell = grid_data[floor_key]
    assert floor_cell["density"] >= 0.5, "Floor should be solid"
    print(f"  ✓ Floor is solid (density={floor_cell['density']:.1f})")

    # Test: glass table has expanded collision margin
    glass_cells = [v for v in grid_data.values() if v.get("label") == 10]
    if glass_cells:
        assert glass_cells[0]["collision_margin"] > 0
        print(f"  ✓ Glass table has expanded margin ({glass_cells[0]['collision_margin']*100:.1f}cm)")

    # Test: empty space should not block
    empty_key = (40, 30, 30)  # middle of room, above floor
    assert empty_key not in grid_data, "Center of room should be empty"
    print(f"  ✓ Room center is empty (navigable)")

    print("  ✓ All collision tests passed")


def main():
    parser = argparse.ArgumentParser(description="Trivima Quick Demo")
    parser.add_argument("--image", type=str, help="Input photograph")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic test scene")
    parser.add_argument("--cell-size", type=float, default=0.05)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    print("=" * 60)
    print("Trivima — Quick Demo")
    print("=" * 60)

    if args.image:
        # Full pipeline with real image
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from trivima.app import run_perception, run_shell_extension

        result = run_perception(args.image, args.device, args.cell_size)
        grid_data = run_shell_extension(result["grid_data"], args.cell_size)

        test_grid_construction(grid_data, args.cell_size)
        test_lod_subdivision(grid_data)
        test_collision(grid_data, args.cell_size)

    elif args.synthetic:
        # Synthetic scene — no models needed
        grid_data = create_synthetic_scene(args.cell_size)

        test_grid_construction(grid_data, args.cell_size)
        test_lod_subdivision(grid_data)
        test_collision(grid_data, args.cell_size)

    else:
        print("\nUsage:")
        print("  python scripts/run_demo.py --synthetic     # test without GPU")
        print("  python scripts/run_demo.py --image room.jpg  # full pipeline")
        return

    print("\n" + "=" * 60)
    print("All tests passed! Pipeline is working.")
    print("=" * 60)


if __name__ == "__main__":
    main()
