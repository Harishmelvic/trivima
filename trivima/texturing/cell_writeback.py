"""
Cell write-back — maps AI model's 2D output pixels back to 3D cells.

The inverse rendering step:
  1. AI model produces a photorealistic 2D image from buffer inputs
  2. The cell_ids buffer tells us which cell each pixel corresponds to
  3. For each cell, we accumulate the AI output weighted by view angle
     (face-on pixels contribute more than grazing-angle pixels)
  4. The accumulated light values are written back into cell.light_*

GPU path: CUDA kernel with atomicAdd for parallel accumulation
CPU path: NumPy vectorized implementation (this module)
"""

import numpy as np
from typing import Optional
from .buffer_renderer import RenderBuffers


def writeback_light_to_cells(
    ai_output: np.ndarray,
    buffers: RenderBuffers,
    camera_pos: np.ndarray,
    grid,
    smoothing_alpha: float = 1.0,
    confidence_boost_low: bool = False,
) -> dict:
    """Write AI texturing output back to cell light values.

    Args:
        ai_output: (H, W, 3) float32 — photorealistic RGB from AI model
        buffers: RenderBuffers from the buffer renderer (contains cell_ids, normals)
        camera_pos: (3,) camera position for view-angle weighting
        grid: CellGrid (native or Python wrapper)
        smoothing_alpha: temporal blend factor. 1.0 = full replace, <1.0 = blend with previous
        confidence_boost_low: if True, low-confidence cells get MORE weight from
            the AI output (because their gradient-based shading is unreliable).
            High-confidence cells blend AI output with existing gradient shading.

    Returns:
        dict with stats: num_cells_updated, avg_weight, max_weight
    """
    h, w = buffers.height, buffers.width
    cell_ids = buffers.cell_ids       # (H, W) int32
    normals = buffers.normals         # (H, W, 3)

    # Compute per-pixel view directions
    # For each pixel, view_dir = normalize(pixel_world_pos - camera_pos)
    # Approximate: use the depth buffer + camera ray to get world positions
    # Simplified: use normals directly for view-angle weighting
    # weight = max(dot(view_dir, cell_normal), 0.01)
    # Since we don't have per-pixel world positions readily available,
    # approximate view_dir as -camera_forward for all pixels (acceptable for prototype)

    # For better accuracy, compute per-pixel view direction from depth buffer
    # TODO: use actual per-pixel ray directions from the camera model

    # Accumulate per cell: weighted sum of AI output colors
    # Using vectorized NumPy operations instead of per-pixel loop

    valid_mask = cell_ids >= 0  # (H, W)
    valid_ids = cell_ids[valid_mask]  # (N,)
    valid_colors = ai_output[valid_mask]  # (N, 3)
    valid_normals = normals[valid_mask]  # (N, 3)

    if len(valid_ids) == 0:
        return {"num_cells_updated": 0, "avg_weight": 0, "max_weight": 0}

    # View-angle weight: dot(view_dir, normal)
    # Approximate view_dir as (0, 0, -1) for forward-facing camera
    # TODO: compute actual per-pixel view directions
    # For now, use normal_z as proxy (works for forward-facing scenes)
    weights = np.abs(valid_normals[:, 2])  # |dot(forward, normal)|
    weights = np.maximum(weights, 0.01)     # floor to avoid zero weight

    # Weighted accumulation per cell
    unique_ids = np.unique(valid_ids)
    num_updated = 0

    # Pre-compute per-cell accumulators
    light_accum = {}    # cell_idx -> (weighted_color_sum, weight_sum)

    for i in range(len(valid_ids)):
        cid = valid_ids[i]
        w = weights[i]
        c = valid_colors[i] * w

        if cid in light_accum:
            light_accum[cid][0] += c
            light_accum[cid][1] += w
        else:
            light_accum[cid] = [c.copy(), w]

    # Write back to grid
    all_weights = []
    for cid, (color_sum, weight_sum) in light_accum.items():
        new_light = color_sum / weight_sum  # weighted average

        # Confidence-weighted AI texturing (theory doc Section 2.5):
        # Low-confidence cells get MORE weight from AI output (their gradients are unreliable).
        # High-confidence cells can blend AI output with gradient-based shading.
        if confidence_boost_low and hasattr(grid, 'get_geo'):
            geo = grid.get_geo(cid)
            cell_conf = getattr(geo, 'confidence', 1.0)

            if cell_conf < 0.5:
                # Low confidence: AI output dominates (alpha → 1.0)
                effective_alpha = 1.0
            else:
                # High confidence: blend AI with existing (gradient-based) lighting
                # Higher confidence → less AI override, more trust in existing shading
                effective_alpha = max(smoothing_alpha, 1.0 - cell_conf * 0.5)
        else:
            effective_alpha = smoothing_alpha

        # Temporal blending with confidence-adjusted alpha
        if effective_alpha < 1.0:
            vis = grid.get_vis(cid)
            old_light = np.array([vis.light_r, vis.light_g, vis.light_b])
            new_light = effective_alpha * new_light + (1.0 - effective_alpha) * old_light

        # Write to cell
        grid.set_cell_light(cid, new_light[0], new_light[1], new_light[2])
        num_updated += 1
        all_weights.append(weight_sum)

    return {
        "num_cells_updated": num_updated,
        "avg_weight": float(np.mean(all_weights)) if all_weights else 0,
        "max_weight": float(np.max(all_weights)) if all_weights else 0,
    }


def compute_light_gradients(grid, updated_cell_ids: list):
    """Compute light gradients via finite differences for updated cells.

    After AI write-back updates cell light values, we need to recompute
    the light gradient for those cells (and their neighbors).

    Args:
        grid: CellGrid
        updated_cell_ids: list of cell indices that were updated
    """
    for idx in updated_cell_ids:
        key = grid.get_key(idx)

        # Finite differences in each axis
        gx, gy, gz = 0.0, 0.0, 0.0
        cell_size = grid.cell_size_at(key)

        for axis in range(3):  # X, Y, Z
            # Get neighbors in +/- direction along this axis
            plus_idx = grid.find_neighbor(key, axis * 2)      # +X, +Y, +Z
            minus_idx = grid.find_neighbor(key, axis * 2 + 1)  # -X, -Y, -Z

            if plus_idx is not None and minus_idx is not None:
                # Central difference
                plus_light = grid.get_cell_light_luma(plus_idx)
                minus_light = grid.get_cell_light_luma(minus_idx)
                grad = (plus_light - minus_light) / (2.0 * cell_size)
            elif plus_idx is not None:
                plus_light = grid.get_cell_light_luma(plus_idx)
                center_light = grid.get_cell_light_luma(idx)
                grad = (plus_light - center_light) / cell_size
            elif minus_idx is not None:
                center_light = grid.get_cell_light_luma(idx)
                minus_light = grid.get_cell_light_luma(minus_idx)
                grad = (center_light - minus_light) / cell_size
            else:
                grad = 0.0

            if axis == 0:
                gx = grad
            elif axis == 1:
                gy = grad
            else:
                gz = grad

        grid.set_cell_light_gradient(idx, gx, gy, gz)
