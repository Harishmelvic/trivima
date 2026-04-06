"""
Room shell extension — detect planes and extend cells to enclose the room.

Takes the front-facing cells from Depth Pro and generates new cells for:
  - Floor (extend to room bounds)
  - Ceiling (estimated from room height)
  - Left/right walls (from detected wall planes)
  - Back wall (behind camera, inferred from room dimensions)

Uses RANSAC plane fitting on existing cell positions + normals.
Generated cells get flat albedo extrapolated from nearest observed cells,
correct normals, and density=1.0.

The quality seam between observed and generated cells is expected —
the AI texturing GAN handles this in the rendering pipeline.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class DetectedPlane:
    """A plane detected by RANSAC."""
    normal: np.ndarray      # (3,) unit normal
    point: np.ndarray       # (3,) a point on the plane
    d: float                # plane equation: n.x + d = 0
    inlier_count: int
    avg_albedo: np.ndarray  # (3,) average color of inlier cells
    label: str              # 'floor', 'ceiling', 'wall_left', 'wall_right', 'wall_back'


def fit_plane_ransac(
    positions: np.ndarray,
    normals: np.ndarray,
    normal_filter: np.ndarray,
    normal_threshold: float = 0.7,
    distance_threshold: float = 0.05,
    max_iterations: int = 200,
    min_inliers: int = 20,
) -> Optional[Tuple[np.ndarray, float, np.ndarray]]:
    """RANSAC plane fitting filtered by expected normal direction.

    Args:
        positions: (N, 3) cell positions
        normals: (N, 3) cell normals
        normal_filter: (3,) expected normal direction (e.g. [0,1,0] for floor)
        normal_threshold: min dot product with normal_filter to be a candidate
        distance_threshold: max distance from plane to count as inlier
        max_iterations: RANSAC iterations
        min_inliers: minimum inliers for a valid plane

    Returns:
        (plane_normal, plane_d, inlier_mask) or None
    """
    # Filter candidates by normal direction
    dots = np.abs(normals @ normal_filter)
    candidates = np.where(dots > normal_threshold)[0]

    if len(candidates) < min_inliers:
        return None

    cand_pos = positions[candidates]
    best_inliers = 0
    best_result = None

    rng = np.random.RandomState(42)

    for _ in range(max_iterations):
        # Pick 3 random candidates
        idx = rng.choice(len(cand_pos), size=3, replace=False)
        p0, p1, p2 = cand_pos[idx[0]], cand_pos[idx[1]], cand_pos[idx[2]]

        # Fit plane
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        nm = np.linalg.norm(n)
        if nm < 1e-8:
            continue
        n = n / nm

        # Ensure normal points in expected direction
        if np.dot(n, normal_filter) < 0:
            n = -n

        d = -np.dot(n, p0)

        # Count inliers from ALL positions (not just candidates)
        distances = np.abs(positions @ n + d)
        inlier_mask = distances < distance_threshold
        n_inliers = inlier_mask.sum()

        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_result = (n.copy(), d, inlier_mask.copy())

    if best_result is None or best_inliers < min_inliers:
        return None

    return best_result


def detect_room_planes(
    cell_pos: np.ndarray,
    cell_col: np.ndarray,
    cell_nrm: np.ndarray,
    cell_size: float = 0.02,
) -> List[DetectedPlane]:
    """Detect floor, ceiling, and wall planes from observed cells.

    Args:
        cell_pos: (N, 3) cell positions
        cell_col: (N, 3) cell colors [0,1]
        cell_nrm: (N, 3) cell normals
        cell_size: cell size for distance threshold

    Returns:
        List of DetectedPlane
    """
    planes = []
    dist_thresh = cell_size * 3  # 3x cell size tolerance

    # Floor: normal pointing up (+Y)
    result = fit_plane_ransac(cell_pos, cell_nrm, np.array([0, 1, 0]),
                              normal_threshold=0.6, distance_threshold=dist_thresh)
    if result:
        n, d, mask = result
        planes.append(DetectedPlane(
            normal=n, point=cell_pos[mask].mean(axis=0), d=d,
            inlier_count=int(mask.sum()),
            avg_albedo=cell_col[mask].mean(axis=0),
            label='floor',
        ))

    # Ceiling: normal pointing down (-Y)
    result = fit_plane_ransac(cell_pos, cell_nrm, np.array([0, -1, 0]),
                              normal_threshold=0.6, distance_threshold=dist_thresh)
    if result:
        n, d, mask = result
        planes.append(DetectedPlane(
            normal=n, point=cell_pos[mask].mean(axis=0), d=d,
            inlier_count=int(mask.sum()),
            avg_albedo=cell_col[mask].mean(axis=0),
            label='ceiling',
        ))

    # Back wall: normal pointing toward camera (+Z in our convention)
    result = fit_plane_ransac(cell_pos, cell_nrm, np.array([0, 0, 1]),
                              normal_threshold=0.6, distance_threshold=dist_thresh)
    if result:
        n, d, mask = result
        planes.append(DetectedPlane(
            normal=n, point=cell_pos[mask].mean(axis=0), d=d,
            inlier_count=int(mask.sum()),
            avg_albedo=cell_col[mask].mean(axis=0),
            label='wall_back',
        ))

    # Left wall: normal pointing right (+X)
    result = fit_plane_ransac(cell_pos, cell_nrm, np.array([1, 0, 0]),
                              normal_threshold=0.6, distance_threshold=dist_thresh)
    if result:
        n, d, mask = result
        planes.append(DetectedPlane(
            normal=n, point=cell_pos[mask].mean(axis=0), d=d,
            inlier_count=int(mask.sum()),
            avg_albedo=cell_col[mask].mean(axis=0),
            label='wall_left',
        ))

    # Right wall: normal pointing left (-X)
    result = fit_plane_ransac(cell_pos, cell_nrm, np.array([-1, 0, 0]),
                              normal_threshold=0.6, distance_threshold=dist_thresh)
    if result:
        n, d, mask = result
        planes.append(DetectedPlane(
            normal=n, point=cell_pos[mask].mean(axis=0), d=d,
            inlier_count=int(mask.sum()),
            avg_albedo=cell_col[mask].mean(axis=0),
            label='wall_right',
        ))

    return planes


def extend_shell(
    cell_pos: np.ndarray,
    cell_col: np.ndarray,
    cell_nrm: np.ndarray,
    cell_size: float = 0.02,
    room_height: float = 2.7,
    extend_behind: float = 1.0,
    extend_sides: float = 0.5,
    image_path: Optional[str] = None,
    focal_length: Optional[float] = None,
    device: str = "cuda",
    use_vlm: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extend observed cells with generated shell cells to enclose the room.

    Args:
        cell_pos: (N, 3) observed cell positions
        cell_col: (N, 3) observed cell colors
        cell_nrm: (N, 3) observed cell normals
        cell_size: cell size in meters
        room_height: estimated room height in meters
        extend_behind: how far behind camera to extend (meters)
        extend_sides: extra extension beyond observed bounds (meters)

    Returns:
        (all_pos, all_col, all_nrm) — observed + generated cells concatenated
    """
    n_observed = len(cell_pos)

    # Use VLM to estimate room dimensions if available
    if use_vlm and image_path and focal_length:
        try:
            from trivima.vlm.room_estimator import estimate_room_dimensions
            obs_bounds = {
                'x_min': float(cell_pos[:, 0].min()),
                'x_max': float(cell_pos[:, 0].max()),
                'y_min': float(cell_pos[:, 1].min()),
                'y_max': float(cell_pos[:, 1].max()),
                'z_min': float(cell_pos[:, 2].min()),
                'z_max': float(cell_pos[:, 2].max()),
            }
            estimate = estimate_room_dimensions(image_path, obs_bounds, focal_length, device)
            room_height = estimate.ceiling_height_m
            extend_behind = estimate.behind_camera_m
            extend_sides = (estimate.width_m - (obs_bounds['x_max'] - obs_bounds['x_min'])) / 2.0
            print(f"  VLM estimate: {estimate.room_type}, {estimate.width_m:.1f}x{estimate.depth_m:.1f}m, "
                  f"ceiling={estimate.ceiling_height_m:.1f}m, behind={estimate.behind_camera_m:.1f}m "
                  f"(conf={estimate.confidence:.0%})")
            print(f"  Reasoning: {estimate.reasoning}")
        except Exception as e:
            print(f"  VLM estimation failed ({e}), using default dimensions")

    # Detect planes from observed cells
    planes = detect_room_planes(cell_pos, cell_col, cell_nrm, cell_size)
    plane_labels = {p.label: p for p in planes}

    print(f"  Shell extension: detected {len(planes)} planes: {[p.label for p in planes]}")
    for p in planes:
        print(f"    {p.label}: {p.inlier_count} inliers, normal={p.normal.round(2)}, "
              f"albedo={p.avg_albedo.round(2)}")

    # Compute room bounds from observed cells
    x_min, x_max = cell_pos[:, 0].min(), cell_pos[:, 0].max()
    y_min, y_max = cell_pos[:, 1].min(), cell_pos[:, 1].max()
    z_min, z_max = cell_pos[:, 2].min(), cell_pos[:, 2].max()

    # Extend bounds
    x_min -= extend_sides
    x_max += extend_sides
    z_max_ext = min(z_max + extend_behind, 0.0)  # Don't go past camera at z=0

    # Detect floor height
    if 'floor' in plane_labels:
        floor_y = -plane_labels['floor'].d / max(abs(plane_labels['floor'].normal[1]), 1e-6)
    else:
        floor_y = y_min

    # Estimate ceiling
    ceiling_y = floor_y + room_height

    # Default colors for generated surfaces
    floor_color = plane_labels['floor'].avg_albedo if 'floor' in plane_labels else np.array([0.4, 0.35, 0.3])
    wall_color = np.array([0.75, 0.73, 0.70])  # light gray default
    ceiling_color = np.array([0.85, 0.83, 0.80])  # slightly lighter

    # Pick wall color from detected walls if available
    for label in ['wall_back', 'wall_left', 'wall_right']:
        if label in plane_labels:
            wall_color = plane_labels[label].avg_albedo
            break

    generated = []  # list of (pos, col, nrm)

    step = cell_size

    # Use coarser step for shell cells (5x observed cell size) to reduce cell count.
    # Shell cells are flat-colored anyway — finer resolution adds no visual detail.
    shell_step = max(step * 5, 0.05)

    # Normals face INWARD (toward room center) so lighting works when
    # the camera is inside the room.

    # --- Floor extension ---
    # Floor normal faces up INTO the room
    for x in np.arange(x_min, x_max + shell_step, shell_step):
        for z in np.arange(z_min, z_max_ext + shell_step, shell_step):
            generated.append(([x, floor_y, z], floor_color, [0, 1, 0]))

    # --- Ceiling ---
    # Ceiling normal faces down INTO the room
    for x in np.arange(x_min, x_max + shell_step, shell_step):
        for z in np.arange(z_min, z_max_ext + shell_step, shell_step):
            generated.append(([x, ceiling_y, z], ceiling_color, [0, 1, 0]))

    # --- Left wall (x = x_min) ---
    # Normal faces right (+X) INTO the room
    for y in np.arange(floor_y, ceiling_y + shell_step, shell_step):
        for z in np.arange(z_min, z_max_ext + shell_step, shell_step):
            generated.append(([x_min, y, z], wall_color, [1, 0, 0]))

    # --- Right wall (x = x_max) ---
    # Normal faces left (-X) INTO the room
    for y in np.arange(floor_y, ceiling_y + shell_step, shell_step):
        for z in np.arange(z_min, z_max_ext + shell_step, shell_step):
            generated.append(([x_max, y, z], wall_color, [-1, 0, 0]))

    # --- Back wall (behind camera, z = z_max_ext) ---
    # Normal faces forward (-Z) INTO the room
    for x in np.arange(x_min, x_max + shell_step, shell_step):
        for y in np.arange(floor_y, ceiling_y + shell_step, shell_step):
            generated.append(([x, y, z_max_ext], wall_color, [0, 0, -1]))

    # --- Front wall (far wall, z = z_min) — only fill gaps ---
    # Normal faces backward (+Z) INTO the room
    occupied = set()
    for i in range(n_observed):
        key = (round(cell_pos[i, 0] / step), round(cell_pos[i, 2] / step))
        occupied.add(key)

    for x in np.arange(x_min, x_max + shell_step, shell_step):
        for y in np.arange(floor_y, ceiling_y + shell_step, shell_step):
            key = (round(x / step), round(z_min / step))
            if key not in occupied:
                generated.append(([x, y, z_min], wall_color, [0, 0, 1]))

    if not generated:
        return cell_pos, cell_col, cell_nrm

    n_generated = len(generated)
    gen_pos = np.array([g[0] for g in generated], dtype=np.float32)
    gen_col = np.array([g[1] for g in generated], dtype=np.float32)
    gen_nrm = np.array([g[2] for g in generated], dtype=np.float32)

    # Remove generated cells that overlap with observed cells
    # Build spatial hash of observed cells
    obs_keys = set()
    for i in range(n_observed):
        key = tuple(np.floor(cell_pos[i] / step).astype(int))
        obs_keys.add(key)

    keep = []
    for i in range(n_generated):
        key = tuple(np.floor(gen_pos[i] / step).astype(int))
        if key not in obs_keys:
            keep.append(i)

    gen_pos = gen_pos[keep]
    gen_col = gen_col[keep]
    gen_nrm = gen_nrm[keep]

    print(f"  Generated {len(gen_pos):,} shell cells (from {n_generated:,} candidates, "
          f"{n_generated - len(gen_pos):,} overlapped)")

    # Concatenate observed + generated
    all_pos = np.concatenate([cell_pos, gen_pos], axis=0)
    all_col = np.concatenate([cell_col, gen_col], axis=0)
    all_nrm = np.concatenate([cell_nrm, gen_nrm], axis=0)

    return all_pos, all_col, all_nrm
