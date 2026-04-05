"""
Cell-based collision detection with confidence-aware margins.

Core algorithm:
  1. Lookup cell at proposed camera position
  2. If cell is solid/surface with density > threshold → blocked
  3. Sub-cell precision via density gradient: local_density = cell.density + dot(gradient, offset)
  4. Wall-sliding: if blocked, project movement onto surface plane

Confidence-aware collision:
  - Low-confidence cells get expanded collision margins
  - Glass/mirror cells (forced to density=1.0 by failure mode mitigation)
    are treated as solid barriers even though their depth data is unreliable
  - This prevents walking through glass tables that Depth Pro missed

See: single_image_precision_theory.md, Section 2.5 (Per-Cell Confidence)
"""

import numpy as np
from typing import Optional, Tuple

# Default thresholds
DENSITY_THRESHOLD = 0.5
BASE_COLLISION_MARGIN = 0.0           # meters, added to cell half-size
LOW_CONFIDENCE_EXTRA_MARGIN = 0.025   # 2.5cm extra margin for uncertain cells


def check_collision(
    grid,
    position: np.ndarray,
    cell_size: float = 0.05,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Check if a position collides with the cell grid.

    Args:
        grid: CellGrid (native or Python wrapper)
        position: (3,) proposed position in world space
        cell_size: base cell size in meters

    Returns:
        (is_blocked, surface_normal) — if blocked, surface_normal is the
        normal of the blocking cell (for wall-sliding). If not blocked,
        surface_normal is None.
    """
    idx = grid.find_at_position(position[0], position[1], position[2])

    if idx is None or idx < 0:
        return False, None  # empty space

    geo = grid.get_geo(idx)

    if geo.is_empty():
        return False, None

    if not geo.is_solid():
        return False, None

    # Sub-cell precision via density gradient
    cell_center = grid.get_cell_center(idx)
    offset = position - cell_center
    local_density = geo.density_at_offset(offset[0], offset[1], offset[2])

    # Confidence-aware threshold:
    # Low-confidence cells use a LOWER density threshold (more conservative —
    # block earlier because we're less sure where the surface actually is)
    effective_threshold = DENSITY_THRESHOLD
    if geo.confidence < 0.5:
        effective_threshold = max(0.2, DENSITY_THRESHOLD - 0.2)

    if local_density > effective_threshold:
        normal = np.array([geo.normal_x, geo.normal_y, geo.normal_z])
        norm = np.linalg.norm(normal)
        if norm > 1e-6:
            normal = normal / norm
        else:
            normal = np.array([0, 1, 0])  # default up
        return True, normal

    return False, None


def check_collision_with_margin(
    grid,
    position: np.ndarray,
    radius: float = 0.2,  # camera capsule radius
    cell_size: float = 0.05,
) -> Tuple[bool, Optional[np.ndarray]]:
    """Check collision with a margin around the position.

    Tests the position and 6 offset points (±radius in each axis).
    Uses per-cell collision margins for low-confidence cells.

    Args:
        grid: CellGrid
        position: (3,) proposed position
        radius: collision check radius around position
        cell_size: base cell size

    Returns:
        (is_blocked, surface_normal)
    """
    # Check center first
    blocked, normal = check_collision(grid, position, cell_size)
    if blocked:
        return blocked, normal

    # Check 6 axis-aligned offsets
    offsets = [
        np.array([radius, 0, 0]),
        np.array([-radius, 0, 0]),
        np.array([0, 0, radius]),
        np.array([0, 0, -radius]),
        np.array([0, radius, 0]),
        np.array([0, -radius, 0]),
    ]

    for off in offsets:
        blocked, normal = check_collision(grid, position + off, cell_size)
        if blocked:
            return blocked, normal

    return False, None


def query_clearance(
    grid,
    position: np.ndarray,
    cell_size: float = 0.05,
    max_steps: int = 40,
) -> float:
    """Compute minimum distance from position to nearest non-empty cell via BFS.

    From unified_pipeline_theory.md §1.3:
    BFS outward through empty cells until hitting a solid/surface cell.
    Returns clearance_distance in meters. Used for spacing scores —
    positions farther from existing objects score higher.

    Args:
        grid: CellGrid (Python dict or native)
        position: (3,) world position
        cell_size: base cell size in meters
        max_steps: max BFS radius in cells (default 40 = 2m at 5cm)

    Returns:
        Distance in meters to nearest non-empty cell. Returns max_steps * cell_size
        if no object found within search radius.
    """
    from collections import deque

    ix = int(np.floor(position[0] / cell_size))
    iy = int(np.floor(position[1] / cell_size))
    iz = int(np.floor(position[2] / cell_size))

    # Check if starting position is already inside an object
    start_key = (ix, iy, iz)
    if isinstance(grid, dict):
        if start_key in grid and grid[start_key].get("density", 0) > 0.3:
            return 0.0
    else:
        idx = grid.find_at_position(position[0], position[1], position[2])
        if idx is not None and idx >= 0:
            geo = grid.get_geo(idx)
            if geo.density > 0.3:
                return 0.0

    # BFS in 6 directions (±X, ±Y, ±Z)
    directions = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    visited = {start_key}
    queue = deque([(ix, iy, iz, 0)])  # (x, y, z, distance_in_cells)

    while queue:
        cx, cy, cz, dist = queue.popleft()
        if dist >= max_steps:
            continue

        for dx, dy, dz in directions:
            nx, ny, nz = cx + dx, cy + dy, cz + dz
            nkey = (nx, ny, nz)

            if nkey in visited:
                continue
            visited.add(nkey)

            # Check if this cell is occupied
            if isinstance(grid, dict):
                if nkey in grid and grid[nkey].get("density", 0) > 0.3:
                    return (dist + 1) * cell_size
            else:
                nidx = grid.find_at_cell_coords(0, nx, ny, nz)
                if nidx is not None and nidx >= 0:
                    geo = grid.get_geo(nidx)
                    if geo.density > 0.3:
                        return (dist + 1) * cell_size

            queue.append((nx, ny, nz, dist + 1))

    return max_steps * cell_size  # nothing found within radius


def slide_along_wall(
    movement: np.ndarray,
    wall_normal: np.ndarray,
) -> np.ndarray:
    """Project movement vector onto wall surface for wall-sliding.

    When the camera hits a wall, instead of stopping dead, we slide
    along the wall by removing the component of movement that goes
    into the wall.

    Args:
        movement: (3,) desired movement vector
        wall_normal: (3,) normal of the blocking surface (pointing away from wall)

    Returns:
        (3,) adjusted movement vector (parallel to wall)
    """
    # Remove the component of movement that points into the wall
    into_wall = np.dot(movement, wall_normal)
    if into_wall >= 0:
        return movement  # moving away from wall, no adjustment needed

    return movement - into_wall * wall_normal


def floor_follow(
    grid,
    position: np.ndarray,
    eye_height: float = 1.6,
    cell_size: float = 0.05,
    max_scan_depth: int = 100,
    smoothing_alpha: float = 0.3,
    prev_floor_y: Optional[float] = None,
) -> float:
    """Find the floor height below a position and return camera Y.

    Scans downward through the cell grid to find the highest surface cell
    below the camera. Uses density gradient for sub-cell floor position.

    Args:
        grid: CellGrid
        position: (3,) current XZ position (Y is ignored, we're finding it)
        eye_height: camera height above floor in meters
        cell_size: base cell size
        max_scan_depth: max cells to scan downward
        smoothing_alpha: lerp factor for height smoothing (0 = no change, 1 = instant)
        prev_floor_y: previous frame's floor Y for smoothing

    Returns:
        Camera Y position (floor height + eye_height)
    """
    # Start from current Y and scan downward
    ix = int(np.floor(position[0] / cell_size))
    iz = int(np.floor(position[2] / cell_size))
    start_iy = int(np.floor(position[1] / cell_size))

    for dy in range(max_scan_depth):
        iy = start_iy - dy
        if iy < 0:
            break

        idx = grid.find_at_cell_coords(0, ix, iy, iz)
        if idx is None or idx < 0:
            continue

        geo = grid.get_geo(idx)
        if geo.is_solid() and geo.normal_y > 0.7:  # upward-facing surface = floor
            # Sub-cell floor position via density gradient
            floor_y = (iy + 0.5) * cell_size

            if abs(geo.density_gy) > 1e-6:
                surface_offset = (0.5 - geo.density) / max(abs(geo.density_gy), 1e-6)
                surface_offset = np.clip(surface_offset, -0.5, 0.5)
                floor_y += surface_offset * cell_size

            target_y = floor_y + eye_height

            # Smooth transition (prevents snapping on stairs/slopes)
            if prev_floor_y is not None:
                target_y = prev_floor_y + smoothing_alpha * (target_y - prev_floor_y)

            return target_y

    # No floor found — maintain current height
    return position[1] if prev_floor_y is None else prev_floor_y
