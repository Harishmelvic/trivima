"""
Point-to-cell grid conversion with 5x5 Sobel gradients and confidence propagation.

Converts labeled 3D point cloud (from perception pipeline) into the sparse cell grid.

For each cell:
  1. Bin points → compute density, albedo, normal, label (averaging)
  2. Compute confidence from point density + propagated per-point confidence
  3. Compute gradients via 5x5 Sobel-like kernel on cell neighbors (not 2-point finite diff)
  4. Compute second derivatives, integrals, neighbor summaries

Gradient computation uses the 5x5 Sobel approach from theory doc Section 9.2:
  - Gradients should be as clean as possible (smooth during differentiation)
  - The larger kernel provides implicit smoothing, reducing noise without
    requiring more aggressive depth smoothing that erases surface detail
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class GridConstructionStats:
    """Statistics from the grid construction process."""
    total_points: int
    total_cells: int
    surface_cells: int
    solid_cells: int
    empty_cells: int
    avg_points_per_cell: float
    avg_confidence: float
    construction_time_s: float


def build_cell_grid(
    positions: np.ndarray,
    colors: np.ndarray,
    normals: np.ndarray,
    labels: np.ndarray,
    confidence: np.ndarray,
    cell_size: float = 0.05,
    min_points_for_solid: int = 3,
) -> Tuple[dict, GridConstructionStats]:
    """Convert a labeled point cloud to the sparse cell grid.

    Args:
        positions: (N, 3) float32 — 3D point positions
        colors: (N, 3) float32 — RGB [0,1]
        normals: (N, 3) float32 — surface normals
        labels: (N,) int32 — semantic labels
        confidence: (N,) float32 — per-point confidence [0,1]
        cell_size: base cell size in meters (default 5cm)
        min_points_for_solid: minimum points for a cell to be classified solid

    Returns:
        (grid_data, stats)
        grid_data: dict ready for CellGridCPU insertion, keyed by (ix, iy, iz) tuples
        stats: GridConstructionStats
    """
    import time
    t0 = time.time()

    n_points = len(positions)

    # Step 1: Bin points into cells
    cell_indices = np.floor(positions / cell_size).astype(np.int32)
    bins = {}  # (ix, iy, iz) → list of point indices

    for i in range(n_points):
        key = (int(cell_indices[i, 0]), int(cell_indices[i, 1]), int(cell_indices[i, 2]))
        if key not in bins:
            bins[key] = []
        bins[key].append(i)

    # Step 2: Compute per-cell properties
    grid_data = {}
    for cell_key, point_ids in bins.items():
        point_ids = np.array(point_ids)
        n = len(point_ids)

        cell = {}

        # Density: proportional to point count (normalized by expected density)
        expected_points = 20  # typical points per solid cell at 5cm
        cell["density"] = min(1.0, n / expected_points)

        # Cell type classification
        if n >= min_points_for_solid:
            cell["cell_type"] = 2  # Solid
        elif n >= 1:
            cell["cell_type"] = 1  # Surface
        else:
            continue  # skip empty

        # Albedo: average color
        cell["albedo"] = colors[point_ids].mean(axis=0)

        # Normal: average (then normalize)
        avg_normal = normals[point_ids].mean(axis=0)
        norm = np.linalg.norm(avg_normal)
        cell["normal"] = avg_normal / norm if norm > 1e-6 else np.array([0, 1, 0])

        # Semantic label: majority vote
        cell["label"] = int(np.bincount(labels[point_ids].astype(np.int64)).argmax())

        # Confidence: combine point density signal with propagated per-point confidence
        # point_density_confidence: more points → more reliable
        density_conf = min(1.0, n / 10.0)  # saturates at 10 points
        # propagated_confidence: min of per-point confidences (worst case)
        # Using mean instead of min to avoid single outlier tanking the cell
        propagated_conf = float(confidence[point_ids].mean())
        # Combined: geometric mean (both must be high for high confidence)
        cell["confidence"] = float(np.sqrt(density_conf * propagated_conf))

        # Collision margin: larger for low-confidence cells
        if cell["confidence"] < 0.5:
            cell["collision_margin"] = 0.025  # 2.5cm extra
        else:
            cell["collision_margin"] = 0.0

        # Density integral
        cell["density_integral"] = cell["density"] * cell_size ** 3

        # Albedo integral
        cell["albedo_integral"] = float(cell["albedo"].mean() * cell_size ** 3)

        grid_data[cell_key] = cell

    # Step 3: Compute gradients using weighted Sobel-like kernel
    _compute_gradients_sobel(grid_data, cell_size)

    # Step 4: Compute neighbor summaries
    _compute_neighbor_summaries(grid_data)

    # Stats
    n_cells = len(grid_data)
    n_surface = sum(1 for c in grid_data.values() if c["cell_type"] == 1)
    n_solid = sum(1 for c in grid_data.values() if c["cell_type"] == 2)
    avg_conf = np.mean([c["confidence"] for c in grid_data.values()]) if grid_data else 0

    stats = GridConstructionStats(
        total_points=n_points,
        total_cells=n_cells,
        surface_cells=n_surface,
        solid_cells=n_solid,
        empty_cells=0,
        avg_points_per_cell=n_points / max(n_cells, 1),
        avg_confidence=float(avg_conf),
        construction_time_s=time.time() - t0,
    )

    return grid_data, stats


def _compute_gradients_sobel(grid_data: dict, cell_size: float):
    """Compute gradients using a 5x5 Sobel-like kernel on cell neighborhoods.

    Instead of simple 2-point finite differences (which amplify noise),
    we use a weighted kernel that samples from a 5×5 neighborhood of cells.
    This provides implicit smoothing during differentiation.

    The 5x5 Sobel kernel weights (1D, for each axis):
      [-1, -2, 0, 2, 1] / (8 * cell_size)

    This is equivalent to a Sobel derivative with a Gaussian smoothing kernel.
    See: single_image_precision_theory.md, Chapter 9.2.
    """
    # Sobel weights for 5-cell span: [-1, -2, 0, 2, 1] normalized
    sobel_weights = np.array([-1, -2, 0, 2, 1], dtype=np.float32)
    sobel_norm = 8.0 * cell_size  # normalization factor

    for cell_key, cell in grid_data.items():
        ix, iy, iz = cell_key

        for prop_name, prop_key in [("density", "density"), ("albedo", "albedo")]:
            grads = np.zeros(3, dtype=np.float32)

            for axis in range(3):  # X, Y, Z
                weighted_sum = 0.0
                weight_total = 0.0

                for offset, weight in zip([-2, -1, 0, 1, 2], sobel_weights):
                    if axis == 0:
                        neighbor_key = (ix + offset, iy, iz)
                    elif axis == 1:
                        neighbor_key = (ix, iy + offset, iz)
                    else:
                        neighbor_key = (ix, iy, iz + offset)

                    if neighbor_key in grid_data:
                        neighbor = grid_data[neighbor_key]
                        if prop_name == "density":
                            val = neighbor["density"]
                        else:
                            val = neighbor["albedo"].mean()  # scalar luminance
                        weighted_sum += weight * val
                        weight_total += abs(weight)

                if weight_total > 0:
                    grads[axis] = weighted_sum / sobel_norm
                else:
                    grads[axis] = 0.0

            if prop_name == "density":
                cell["density_gradient"] = grads
            else:
                cell["albedo_gradient"] = grads

        # Normal gradient (curvature) — same Sobel approach on normal components
        normal_grads = np.zeros(3, dtype=np.float32)
        for axis in range(3):
            weighted_sum = 0.0
            for offset, weight in zip([-2, -1, 0, 1, 2], sobel_weights):
                if axis == 0:
                    nk = (ix + offset, iy, iz)
                elif axis == 1:
                    nk = (ix, iy + offset, iz)
                else:
                    nk = (ix, iy, iz + offset)

                if nk in grid_data:
                    # Use the primary normal component for this axis
                    weighted_sum += weight * grid_data[nk]["normal"][axis]

            normal_grads[axis] = weighted_sum / sobel_norm

        cell["normal_gradient"] = normal_grads

        # Set gradients to zero for low-confidence cells (unreliable)
        if cell["confidence"] < 0.3:
            cell["density_gradient"] = np.zeros(3)
            cell["albedo_gradient"] = np.zeros(3)
            cell["normal_gradient"] = np.zeros(3)


def _compute_neighbor_summaries(grid_data: dict):
    """Populate neighbor summaries for each cell."""
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]

    for cell_key, cell in grid_data.items():
        ix, iy, iz = cell_key
        neighbors = []

        for dx, dy, dz in directions:
            nk = (ix + dx, iy + dy, iz + dz)
            if nk in grid_data:
                n = grid_data[nk]
                neighbors.append({
                    "type": n["cell_type"],
                    "density": n["density"],
                    "normal_y": n["normal"][1],
                    "light_luma": 0.0,  # filled during AI texturing
                })
            else:
                neighbors.append({
                    "type": 0,  # empty
                    "density": 0.0,
                    "normal_y": 0.0,
                    "light_luma": 0.0,
                })

        cell["neighbors"] = neighbors


def apply_failure_mode_density_forcing(
    grid_data: dict,
    labels_2d: np.ndarray,
    label_names: Dict[int, str],
    positions: np.ndarray,
    cell_size: float = 0.05,
):
    """Force density=1.0 for glass and mirror cells after grid construction.

    This is the density forcing step that the failure_modes.py confidence
    assignment alone doesn't cover. Glass/mirror cells must be impassable
    barriers regardless of what Depth Pro reported.

    See: theory doc Chapter 7 — mirror density=1.0, glass density=1.0.
    """
    from .failure_modes import MIRROR_LABELS, GLASS_LABELS

    # Build set of cell keys that contain mirror or glass points
    mirror_glass_cells = set()

    # This requires knowing which points mapped to which cells
    # In practice, the pipeline passes this information through labels
    for cell_key, cell in grid_data.items():
        label_idx = cell.get("label", 0)
        name = label_names.get(label_idx, "").lower()

        is_mirror = name in MIRROR_LABELS or "mirror" in name
        is_glass = name in GLASS_LABELS or "glass" in name

        if is_mirror or is_glass:
            cell["density"] = 1.0
            cell["cell_type"] = 2  # Solid
            cell["density_integral"] = cell_size ** 3  # full volume
            # Gradients zeroed — we don't trust the depth data for these
            cell["density_gradient"] = np.zeros(3)
            cell["albedo_gradient"] = np.zeros(3)
            cell["normal_gradient"] = np.zeros(3)
